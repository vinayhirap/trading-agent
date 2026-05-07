# trading-agent/src/features/regime_features.py
"""
Regime Interaction Features — fixes the biggest accuracy problem.

Problem: RSI=65 means different things in different markets.
  - In TRENDING market: RSI=65 = momentum continuation → BUY
  - In RANGING market:  RSI=65 = approaching overbought → SELL

Raw features confuse the model because it sees RSI=65 associated
with both BUY and SELL outcomes. This kills accuracy.

Solution: Create regime-interaction features that encode the context.
  rsi_in_trend = RSI × (ADX > 25)      # RSI only matters in trend
  rsi_in_range = RSI × (ADX < 18)      # RSI only matters in range
  macd_trend_conf = MACD_hist × adx_norm   # MACD strength × trend strength
  bb_squeeze_flag = 1 if BB is very tight   # squeeze before breakout

These 8 features give XGBoost the regime context it needs to
learn RSI=65-in-trend ≠ RSI=65-in-range.

Usage:
    from src.features.regime_features import add_regime_features
    featured_df = add_regime_features(featured_df)
"""
import numpy as np
import pandas as pd


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 8 regime-interaction features to an existing feature DataFrame.
    Safe to call on DataFrames that may or may not have all source columns.

    New columns added:
      rsi_in_trend      — RSI value when trending, 50 otherwise
      rsi_in_range      — RSI value when ranging, 50 otherwise
      macd_trend_conf   — MACD histogram × ADX strength
      bb_squeeze_flag   — 1 if Bollinger bandwidth is very tight
      vol_trend_conf    — Volume ratio × ADX (volume confirms trend)
      momentum_regime   — EMA momentum sign × ADX (signed trend strength)
      mean_rev_signal   — RSI extreme flag in low-ADX regime only
      breakout_signal   — Price near BB band in squeeze regime
    """
    out = df.copy()

    # Safe getters with fallback values
    def _get(col, default):
        return out[col] if col in out.columns else pd.Series(default, index=out.index)

    rsi      = _get("rsi_14",       50.0)
    adx      = _get("adx",          20.0)
    macd_h   = _get("macd_hist",     0.0)
    bb_pct   = _get("bb_pct_b",      0.5)
    vol_r    = _get("vol_ratio",      1.0)
    e9_pct   = _get("ema9_pct",       0.0)
    atr_r    = _get("atr_ratio",      1.0)
    bb_width = _get("bb_width",        0.04)   # BB width as fraction of price

    # ── Regime flags ──────────────────────────────────────────────────────────
    is_trending = (adx > 25).astype(float)
    is_ranging  = (adx < 18).astype(float)
    adx_norm    = (adx - 20) / 20   # normalized: -1 = no trend, +1 = strong trend

    # 1. rsi_in_trend: RSI value only when trending, neutral (50) in range
    out["rsi_in_trend"] = rsi * is_trending + 50.0 * (1 - is_trending)

    # 2. rsi_in_range: RSI value only when ranging, neutral (50) in trend
    out["rsi_in_range"] = rsi * is_ranging + 50.0 * (1 - is_ranging)

    # 3. macd_trend_conf: MACD momentum × trend strength
    #    Positive strong value = strong bullish trend confirmation
    out["macd_trend_conf"] = macd_h * adx_norm

    # 4. bb_squeeze_flag: 1 if BB width is in bottom 20% (squeeze before breakout)
    #    Use rolling percentile — tight bands precede explosive moves
    try:
        bb_width_pctile = bb_width.rolling(50, min_periods=20).rank(pct=True)
        out["bb_squeeze_flag"] = (bb_width_pctile < 0.20).astype(float)
    except Exception:
        out["bb_squeeze_flag"] = 0.0

    # 5. vol_trend_conf: Volume spike × trend direction
    #    High volume in a trend = institutional confirmation
    vol_trend_dir = (e9_pct > 0).astype(float) - (e9_pct < 0).astype(float)  # +1 or -1
    out["vol_trend_conf"] = (vol_r - 1.0) * vol_trend_dir * is_trending

    # 6. momentum_regime: EMA momentum sign weighted by ADX
    #    Strong value = price well above EMA in strong trend
    out["momentum_regime"] = e9_pct * adx_norm

    # 7. mean_rev_signal: RSI extreme signal BUT ONLY in ranging market
    #    RSI<35 in range = strong BUY; RSI>65 in range = strong SELL
    #    In trend this would be wrong
    rsi_extreme = ((rsi - 50) / 50)   # -1 to +1, oversold=negative, overbought=positive
    out["mean_rev_signal"] = -rsi_extreme * is_ranging   # flip sign: oversold → positive

    # 8. breakout_signal: Price near BB extreme in squeeze regime
    #    Price at upper band after squeeze = bullish breakout
    #    Price at lower band after squeeze = bearish breakout
    bb_direction = (bb_pct - 0.5) * 2   # -1 to +1 (lower=-1, upper=+1)
    out["breakout_signal"] = bb_direction * out["bb_squeeze_flag"]

    return out


def get_regime_feature_names() -> list[str]:
    """Return list of all regime feature column names."""
    return [
        "rsi_in_trend",
        "rsi_in_range",
        "macd_trend_conf",
        "bb_squeeze_flag",
        "vol_trend_conf",
        "momentum_regime",
        "mean_rev_signal",
        "breakout_signal",
    ]