"""
Label Generation v2 — Triple Barrier Method

Problem with the previous "max return over horizon" approach:
  Stock drops -5% day 1 (stop-loss hit), recovers to +10% day 7.
  Old code labels this BUY. Real trader lost money on day 1.

Triple Barrier traces the CHRONOLOGICAL price path:
  Upper barrier (profit target): tp_mult × ATR above entry
  Lower barrier (stop loss):     sl_mult × ATR below entry
  Time barrier:                  horizon bars

  → Profit target hit first  → BUY
  → Stop loss hit first      → SELL
  → Neither hit              → HOLD

This ensures the model learns what a real trader would see,
not an idealized "could have been profitable" view.
"""
import numpy as np
import pandas as pd
from loguru import logger

BUY  =  1
HOLD =  0
SELL = -1

# ── Barrier config per asset class ───────────────────────────────────────────
# CRYPTO FIX: was sl=2.0 tp=3.0 on 3-day horizon → 92% HOLD (majority class cheat)
# Fix: tighter barriers + longer horizon → realistic 20-30% BUY/SELL distribution
RR_CONFIG = {
    "index":   {"sl_mult": 1.5, "tp_mult": 2.0, "horizon_default": 7},
    "equity":  {"sl_mult": 1.5, "tp_mult": 2.0, "horizon_default": 7},
    "futures": {"sl_mult": 1.5, "tp_mult": 2.5, "horizon_default": 7},
    "crypto":  {"sl_mult": 1.0, "tp_mult": 1.5, "horizon_default": 7},  # FIXED: was 2.0/3.0/3d
}


def make_labels_v2(
    df:          pd.DataFrame,
    features:    pd.DataFrame = None,
    horizon:     int   = 5,
    asset_class: str   = "equity",
    buy_pct:     float = 0.30,
    sell_pct:    float = 0.30,
    min_move:    float = 0.003,
    use_regime:  bool  = True,
    smooth:      bool  = True,   # ignored — kept for compatibility
) -> tuple[pd.Series, pd.Series]:
    """
    Triple Barrier labels with balanced sample weights.

    Returns:
        y:       pd.Series of {-1, 0, 1}
        weights: pd.Series of sample weights
    """
    if df is None or len(df) < horizon + 10:
        return pd.Series(dtype=int), pd.Series(dtype=float)

    close = df["close"].copy()
    high  = df["high"].copy()  if "high"  in df.columns else close
    low   = df["low"].copy()   if "low"   in df.columns else close
    n     = len(close)

    rr      = RR_CONFIG.get(asset_class, RR_CONFIG["equity"])
    sl_mult = rr["sl_mult"]
    tp_mult = rr["tp_mult"]

    # ATR for dynamic barriers
    atr_series = None
    if features is not None and not features.empty and "atr_14" in features.columns:
        atr_series = features["atr_14"].reindex(close.index)
    else:
        try:
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low  - close.shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr_series = tr.rolling(14, min_periods=5).mean()
        except Exception:
            pass

    # Triple barrier loop
    labels        = []
    valid_indices = []

    for i in range(n - horizon):
        entry = float(close.iloc[i])
        if entry <= 0:
            continue

        if atr_series is not None and not pd.isna(atr_series.iloc[i]):
            atr = float(atr_series.iloc[i])
        else:
            atr = entry * 0.015

        upper = entry + atr * tp_mult
        lower = entry - atr * sl_mult

        # Minimum move enforcement
        if (upper - entry) / entry < min_move:
            upper = entry * (1 + min_move * 2)
        if (entry - lower) / entry < min_move:
            lower = entry * (1 - min_move * 1.5)

        label = HOLD
        for j in range(1, horizon + 1):
            if i + j >= n:
                break
            bar_h = float(high.iloc[i + j])
            bar_l = float(low.iloc[i + j])
            if bar_h >= upper:
                label = BUY
                break
            if bar_l <= lower:
                label = SELL
                break

        labels.append(label)
        valid_indices.append(close.index[i])

    if not labels:
        return pd.Series(dtype=int), pd.Series(dtype=float)

    y = pd.Series(labels, index=valid_indices, dtype=int)

    # Regime gating — only for non-crypto (crypto has no exchange session)
    if use_regime and features is not None and not features.empty and asset_class != "crypto":
        try:
            fa = features.reindex(y.index)
            adx     = fa.get("adx",      pd.Series(20, index=y.index))
            di_diff = fa.get("di_diff",   pd.Series(0,  index=y.index))
            atr_r   = fa.get("atr_ratio", pd.Series(1,  index=y.index))
            y[(y == BUY)  & (adx > 25) & (di_diff < 0)] = HOLD
            y[(y == SELL) & (adx > 25) & (di_diff > 0)] = HOLD
            y[atr_r > 3.5] = HOLD
        except Exception:
            pass

    # Crypto-specific: RSI gates only (no DI/ADX gating — crypto is 24x7)
    if asset_class == "crypto" and features is not None:
        try:
            rsi = features.reindex(y.index).get("rsi_14", pd.Series(50, index=y.index))
            y[(y == BUY)  & (rsi > 80)] = HOLD
            y[(y == SELL) & (rsi < 20)] = HOLD
        except Exception:
            pass

    n_buy  = (y == BUY).sum()
    n_sell = (y == SELL).sum()
    n_hold = (y == HOLD).sum()
    total  = len(y)

    logger.info(
        f"Triple Barrier | {asset_class} | horizon={horizon} | "
        f"SL={sl_mult}×ATR TP={tp_mult}×ATR | "
        f"BUY={n_buy}({n_buy/total:.0%}) "
        f"HOLD={n_hold}({n_hold/total:.0%}) "
        f"SELL={n_sell}({n_sell/total:.0%})"
    )

    # Warn if HOLD > 75% — likely barrier config issue
    if n_hold / total > 0.75:
        logger.warning(
            f"⚠️  {asset_class} HOLD={n_hold/total:.0%} is very high. "
            f"Consider tightening barriers (current: SL={sl_mult}×ATR TP={tp_mult}×ATR) "
            f"or extending horizon."
        )

    # Sample weights — upweight minority classes
    if n_buy == 0 or n_sell == 0:
        weights = pd.Series(1.0, index=y.index)
    else:
        bw = min(total / (3 * n_buy),  5.0)
        sw = min(total / (3 * n_sell), 5.0)
        hw = min(total / (3 * n_hold), 2.0)
        weights = pd.Series(hw, index=y.index, dtype=float)
        weights[y == BUY]  = bw
        weights[y == SELL] = sw

    return y, weights


def make_labels_v2_compatible(df, features=None, horizon=5, asset_class="equity"):
    """Drop-in for make_labels() — returns y only."""
    y, _ = make_labels_v2(df, features, horizon, asset_class)
    return y