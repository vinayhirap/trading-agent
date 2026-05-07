# trading-agent/src/models/regime_filter.py
"""
Regime-Aware Signal Filter v1

The problem this solves:
  The ensemble model and rule engine generate raw signals (BUY/SELL/HOLD).
  But a BUY signal in a TRENDING_UP market is very different from a BUY signal
  in a VOLATILE market. Same signal, very different expected outcome.

  This module gates and adjusts signals based on the current market regime:

  TRENDING_UP   → Favour BUY signals, raise confidence on trend-following entries
  TRENDING_DOWN → Favour SELL signals, suppress BUY signals (against trend)
  RANGING       → Mean-reversion mode: buy oversold, sell overbought, tighter SL
  VOLATILE      → Reduce position size, require higher confidence, use tighter SL

Usage:
    from src.models.regime_filter import RegimeFilter

    rf = RegimeFilter()

    # In signal generation loop:
    filtered = rf.filter(
        symbol   = "NIFTY50",
        bias     = "BUY",
        confidence = 0.62,
        regime   = "TRENDING_UP",
        regime_conf = 0.81,
        features = feat_series,   # pd.Series with indicators
    )

    print(filtered.action)       # ENTER / WAIT / AVOID
    print(filtered.adj_confidence) # confidence after regime adjustment
    print(filtered.adj_bias)       # possibly demoted e.g. BUY → HOLD
    print(filtered.sl_multiplier)  # how much wider/tighter the SL should be
    print(filtered.size_multiplier)# position size adjustment (1.0 = normal)
    print(filtered.reasons)        # human-readable explanation list

Wire into app.py Signal Scanner:
    from src.models.regime_filter import RegimeFilter
    _rf = RegimeFilter()

    # After getting sig from get_signal_with_ensemble():
    filtered = _rf.filter(
        symbol      = sym,
        bias        = sig["bias"],
        confidence  = sig["confidence"],
        regime      = sig.get("regime", "RANGING") or "RANGING",
        regime_conf = sig.get("regime_conf") or 0.5,
        features    = ft.iloc[-1] if not ft.empty else None,
    )
    sig["action"]          = filtered.action
    sig["adj_confidence"]  = filtered.adj_confidence
    sig["adj_bias"]        = filtered.adj_bias
    sig["sl_multiplier"]   = filtered.sl_multiplier
    sig["size_multiplier"] = filtered.size_multiplier
    sig["regime_notes"]    = filtered.reasons
"""
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


# ── Thresholds ─────────────────────────────────────────────────────────────────

# Minimum confidence to ENTER (after regime adjustment)
ENTER_THRESHOLD = 0.55

# Regime confidence below this → treat regime as uncertain → be conservative
REGIME_TRUST_THRESHOLD = 0.55

# RSI extremes for ranging mode
RSI_OVERSOLD  = 35
RSI_OVERBOUGHT= 65


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class RegimeFilterResult:
    """Output of RegimeFilter.filter()."""

    # Original inputs
    symbol:        str
    orig_bias:     str      # e.g. "BUY"
    orig_conf:     float    # 0-1
    regime:        str      # TRENDING_UP / TRENDING_DOWN / RANGING / VOLATILE

    # Adjusted outputs
    adj_bias:      str      # possibly demoted: "STRONG BUY" → "BUY", or "BUY" → "HOLD"
    adj_confidence:float    # confidence after regime boost/penalty
    action:        str      # ENTER / WAIT / AVOID

    # Position sizing adjustments (multipliers applied by risk manager)
    sl_multiplier:   float = 1.0   # >1 = wider SL, <1 = tighter
    size_multiplier: float = 1.0   # >1 = larger position, <1 = smaller

    # Strategy hint
    strategy_mode:   str   = "TREND"   # TREND / MEAN_REVERSION / DEFENSIVE
    hold_type:       str   = "SWING"   # INTRADAY / SWING / POSITIONAL

    # Explanation
    reasons:         list  = field(default_factory=list)
    warnings:        list  = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol":          self.symbol,
            "orig_bias":       self.orig_bias,
            "adj_bias":        self.adj_bias,
            "orig_conf":       round(self.orig_conf, 3),
            "adj_confidence":  round(self.adj_confidence, 3),
            "action":          self.action,
            "regime":          self.regime,
            "sl_multiplier":   round(self.sl_multiplier, 2),
            "size_multiplier": round(self.size_multiplier, 2),
            "strategy_mode":   self.strategy_mode,
            "hold_type":       self.hold_type,
            "reasons":         self.reasons,
            "warnings":        self.warnings,
        }


# ── Main filter ────────────────────────────────────────────────────────────────

class RegimeFilter:
    """
    Applies regime-specific signal adjustments.

    Rules by regime:

    TRENDING_UP:
      - BUY / STRONG BUY → confidence boost +5-10%
      - SELL / STRONG SELL → demote to HOLD, action = WAIT
        (counter-trend sells allowed only at >70% confidence)
      - SL: wider (×1.3) — give trend room to breathe
      - Size: normal to larger (×1.0-1.2)
      - Hold type: SWING or POSITIONAL

    TRENDING_DOWN:
      - SELL / STRONG SELL → confidence boost +5-10%
      - BUY / STRONG BUY → demote to HOLD, action = WAIT
        (counter-trend buys blocked below 70% confidence)
      - SL: wider (×1.3)
      - Size: normal (×1.0)
      - Hold type: SWING

    RANGING:
      - Mean-reversion mode
      - BUY only if RSI < RSI_OVERSOLD (35)
      - SELL only if RSI > RSI_OVERBOUGHT (65)
      - BUY with RSI 35-50 → WAIT
      - SELL with RSI 50-65 → WAIT
      - SL: tighter (×0.7) — quick exits on mean reversion
      - Size: smaller (×0.8) — ranging = lower conviction
      - Hold type: INTRADAY

    VOLATILE:
      - All signals require confidence ≥ 65% to ENTER
      - SL: tighter (×0.8)
      - Size: significantly smaller (×0.5)
      - Hold type: INTRADAY
      - Confidence penalty: -10%

    UNCERTAIN REGIME (regime_conf < REGIME_TRUST_THRESHOLD):
      - Act as RANGING: conservative, require RSI confirmation
    """

    def filter(
        self,
        symbol:       str,
        bias:         str,           # raw signal: BUY / SELL / STRONG BUY / STRONG SELL / NEUTRAL / HOLD
        confidence:   float,         # 0-1
        regime:       str,           # TRENDING_UP / TRENDING_DOWN / RANGING / VOLATILE
        regime_conf:  float = 0.7,   # how confident we are about the regime
        features:     Optional[pd.Series] = None,
    ) -> RegimeFilterResult:
        """
        Apply regime-aware filtering to a raw signal.
        Returns RegimeFilterResult with adjusted confidence, action, and sizing.
        """
        reasons  = []
        warnings = []

        # Normalise inputs
        bias       = (bias or "NEUTRAL").strip().upper()
        regime     = (regime or "RANGING").strip().upper()
        regime_conf= float(regime_conf or 0.5)

        # Extract indicator values from features
        rsi    = float(features.get("rsi_14", 50))    if features is not None else 50.0
        adx    = float(features.get("adx", 20))       if features is not None else 20.0
        atr_r  = float(features.get("atr_ratio", 1.0))if features is not None else 1.0
        bb_pct = float(features.get("bb_pct_b", 0.5)) if features is not None else 0.5

        is_bullish = "BUY"  in bias
        is_bearish = "SELL" in bias
        is_strong  = "STRONG" in bias

        adj_conf = float(confidence)
        adj_bias = bias
        sl_mult  = 1.0
        sz_mult  = 1.0
        mode     = "TREND"
        hold     = "SWING"
        action   = "WAIT"

        # ── Uncertain regime → conservative ───────────────────────────────────
        if regime_conf < REGIME_TRUST_THRESHOLD:
            warnings.append(f"Regime confidence low ({regime_conf:.0%}) — treating as RANGING")
            regime = "RANGING"
            reasons.append("Regime uncertain → mean-reversion mode")

        # ── TRENDING_UP ───────────────────────────────────────────────────────
        if regime == "TRENDING_UP":
            mode = "TREND"
            hold = "SWING"
            sl_mult = 1.3   # wider SL — let trend breathe

            if is_bullish:
                boost = 0.08 if is_strong else 0.05
                adj_conf = min(0.95, adj_conf + boost)
                sz_mult  = 1.15 if is_strong else 1.0
                reasons.append(
                    f"TRENDING_UP: {bias} signal aligned with trend "
                    f"→ confidence +{boost:.0%}, SL widened ×{sl_mult}"
                )
                if adj_conf >= ENTER_THRESHOLD:
                    action = "ENTER"

            elif is_bearish:
                # Counter-trend sell — only allow with very high confidence
                if confidence >= 0.70:
                    warnings.append("Counter-trend SELL in uptrend — high risk, use small size")
                    adj_bias = "SELL"   # demote STRONG SELL → SELL
                    adj_conf = confidence - 0.08   # penalty for going against trend
                    sz_mult  = 0.6
                    sl_mult  = 0.9
                    mode     = "DEFENSIVE"
                    reasons.append(
                        f"TRENDING_UP + {bias}: counter-trend allowed at "
                        f">{confidence:.0%} conf with reduced size"
                    )
                    if adj_conf >= ENTER_THRESHOLD:
                        action = "ENTER"
                    else:
                        action = "WAIT"
                else:
                    adj_bias = "HOLD"
                    adj_conf = 0.40
                    action   = "AVOID"
                    reasons.append(
                        f"TRENDING_UP: {bias} suppressed (counter-trend below 70% conf)"
                    )
            else:
                action = "WAIT"
                reasons.append("TRENDING_UP: neutral signal — wait for directional setup")

        # ── TRENDING_DOWN ─────────────────────────────────────────────────────
        elif regime == "TRENDING_DOWN":
            mode = "TREND"
            hold = "SWING"
            sl_mult = 1.3

            if is_bearish:
                boost = 0.08 if is_strong else 0.05
                adj_conf = min(0.95, adj_conf + boost)
                sz_mult  = 1.1 if is_strong else 1.0
                reasons.append(
                    f"TRENDING_DOWN: {bias} aligned with trend "
                    f"→ confidence +{boost:.0%}, SL widened ×{sl_mult}"
                )
                if adj_conf >= ENTER_THRESHOLD:
                    action = "ENTER"

            elif is_bullish:
                if confidence >= 0.70:
                    warnings.append("Counter-trend BUY in downtrend — high risk")
                    adj_bias = "BUY"
                    adj_conf = confidence - 0.08
                    sz_mult  = 0.6
                    sl_mult  = 0.9
                    mode     = "DEFENSIVE"
                    reasons.append(
                        f"TRENDING_DOWN + {bias}: counter-trend allowed at "
                        f">{confidence:.0%} conf with reduced size"
                    )
                    action = "ENTER" if adj_conf >= ENTER_THRESHOLD else "WAIT"
                else:
                    adj_bias = "HOLD"
                    adj_conf = 0.40
                    action   = "AVOID"
                    reasons.append(
                        f"TRENDING_DOWN: {bias} suppressed (counter-trend, <70% conf)"
                    )
            else:
                action = "WAIT"
                reasons.append("TRENDING_DOWN: neutral — wait for breakdown continuation")

        # ── RANGING ───────────────────────────────────────────────────────────
        elif regime == "RANGING":
            mode    = "MEAN_REVERSION"
            hold    = "INTRADAY"
            sl_mult = 0.75   # tighter SL — quick exits on mean reversion
            sz_mult = 0.85   # smaller size — lower conviction in range

            if is_bullish:
                if rsi <= RSI_OVERSOLD:
                    # Good: oversold in range → mean-reversion BUY
                    reasons.append(
                        f"RANGING: {bias} at RSI {rsi:.0f} (oversold) "
                        f"→ mean-reversion entry, tight SL ×{sl_mult}"
                    )
                    adj_bias = "BUY"
                    # Small boost for RSI confirmation
                    adj_conf = min(0.90, adj_conf + 0.04)
                    action   = "ENTER" if adj_conf >= ENTER_THRESHOLD else "WAIT"

                elif rsi <= 45:
                    # Mildly oversold — wait for more extreme setup
                    reasons.append(
                        f"RANGING: {bias} but RSI {rsi:.0f} not oversold enough — WAIT"
                    )
                    adj_bias = "HOLD"
                    adj_conf = 0.42
                    action   = "WAIT"

                else:
                    # BUY signal in ranging market but not oversold → AVOID
                    adj_bias = "HOLD"
                    adj_conf = 0.35
                    action   = "AVOID"
                    reasons.append(
                        f"RANGING: {bias} suppressed — RSI {rsi:.0f} not oversold "
                        f"(need <{RSI_OVERSOLD} for range BUY)"
                    )

            elif is_bearish:
                if rsi >= RSI_OVERBOUGHT:
                    # Good: overbought in range → mean-reversion SELL
                    reasons.append(
                        f"RANGING: {bias} at RSI {rsi:.0f} (overbought) "
                        f"→ mean-reversion entry, tight SL ×{sl_mult}"
                    )
                    adj_bias = "SELL"
                    adj_conf = min(0.90, adj_conf + 0.04)
                    action   = "ENTER" if adj_conf >= ENTER_THRESHOLD else "WAIT"

                elif rsi >= 55:
                    reasons.append(
                        f"RANGING: {bias} but RSI {rsi:.0f} not overbought enough — WAIT"
                    )
                    adj_bias = "HOLD"
                    adj_conf = 0.42
                    action   = "WAIT"

                else:
                    adj_bias = "HOLD"
                    adj_conf = 0.35
                    action   = "AVOID"
                    reasons.append(
                        f"RANGING: {bias} suppressed — RSI {rsi:.0f} not overbought "
                        f"(need >{RSI_OVERBOUGHT} for range SELL)"
                    )
            else:
                action = "WAIT"
                reasons.append("RANGING: neutral — no clear mean-reversion setup")

            # Bollinger Band confirmation for ranging trades
            if action == "ENTER":
                if is_bullish and bb_pct > 0.3:
                    warnings.append(
                        f"RANGING BUY: BB %B={bb_pct:.2f} — price not near lower band "
                        f"(best <0.10)"
                    )
                elif is_bearish and bb_pct < 0.7:
                    warnings.append(
                        f"RANGING SELL: BB %B={bb_pct:.2f} — price not near upper band "
                        f"(best >0.90)"
                    )

        # ── VOLATILE ──────────────────────────────────────────────────────────
        elif regime == "VOLATILE":
            mode    = "DEFENSIVE"
            hold    = "INTRADAY"
            sl_mult = 0.80   # tighter SL — volatile markets can gap
            sz_mult = 0.50   # half size always in volatile regime
            adj_conf = adj_conf - 0.10   # penalty for volatility uncertainty

            warnings.append(
                f"VOLATILE regime (ATR ratio={atr_r:.1f}×) — "
                f"confidence penalised -10%, size halved"
            )

            VOLATILE_THRESHOLD = 0.65
            if adj_conf >= VOLATILE_THRESHOLD and (is_bullish or is_bearish):
                adj_bias = bias
                action   = "ENTER"
                reasons.append(
                    f"VOLATILE: {bias} allowed at adjusted conf "
                    f"{adj_conf:.0%} ≥ {VOLATILE_THRESHOLD:.0%} — "
                    f"half size, tight SL"
                )
            else:
                adj_bias = "HOLD"
                action   = "AVOID" if adj_conf < 0.45 else "WAIT"
                reasons.append(
                    f"VOLATILE: {bias} requires >{VOLATILE_THRESHOLD:.0%} confidence — "
                    f"currently {adj_conf:.0%} — {action}"
                )

        # ── Unknown regime ────────────────────────────────────────────────────
        else:
            # Unknown/null regime — apply mild conservatism
            adj_conf = adj_conf - 0.03
            reasons.append(f"Unknown regime '{regime}' — mild confidence penalty")
            if adj_conf >= ENTER_THRESHOLD and (is_bullish or is_bearish):
                action = "ENTER"
            elif adj_conf >= 0.45:
                action = "WAIT"
            else:
                action = "AVOID"

        # ── Final guard: never ENTER below ENTER_THRESHOLD ────────────────────
        adj_conf = max(0.0, min(1.0, adj_conf))
        if action == "ENTER" and adj_conf < ENTER_THRESHOLD:
            action = "WAIT"
            reasons.append(
                f"Final gate: adjusted conf {adj_conf:.0%} < "
                f"{ENTER_THRESHOLD:.0%} → demoted to WAIT"
            )

        # ── ADX sanity check ─────────────────────────────────────────────────
        if regime in ("TRENDING_UP", "TRENDING_DOWN") and adx < 18:
            warnings.append(
                f"ADX={adx:.0f} is low for a trend regime — "
                f"regime detector may be lagging"
            )

        return RegimeFilterResult(
            symbol         = symbol,
            orig_bias      = bias,
            orig_conf      = confidence,
            regime         = regime,
            adj_bias       = adj_bias,
            adj_confidence = round(adj_conf, 3),
            action         = action,
            sl_multiplier  = sl_mult,
            size_multiplier= sz_mult,
            strategy_mode  = mode,
            hold_type      = hold,
            reasons        = reasons,
            warnings       = warnings,
        )

    def batch_filter(
        self,
        signals: list[dict],
        regime:  str,
        regime_conf: float = 0.7,
        features_map: dict = None,   # {symbol: pd.Series}
    ) -> list[dict]:
        """
        Filter a list of signal dicts in one call.
        Each dict must have: symbol, bias, confidence.
        Returns enriched dicts with regime filter outputs merged in.
        """
        results = []
        for sig in signals:
            sym  = sig.get("symbol", "")
            feat = (features_map or {}).get(sym)
            fr   = self.filter(
                symbol      = sym,
                bias        = sig.get("bias", "NEUTRAL"),
                confidence  = sig.get("confidence", 0.5),
                regime      = sig.get("regime", regime),
                regime_conf = sig.get("regime_conf", regime_conf),
                features    = feat,
            )
            merged = {**sig, **fr.to_dict()}
            results.append(merged)
        return results


# Module-level singleton
regime_filter = RegimeFilter()


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd

    rf = RegimeFilter()

    test_cases = [
        ("NIFTY50",   "STRONG BUY",  0.78, "TRENDING_UP",   0.85),
        ("BANKNIFTY", "SELL",         0.62, "TRENDING_UP",   0.80),
        ("RELIANCE",  "BUY",          0.62, "RANGING",       0.75),
        ("TCS",       "STRONG SELL",  0.78, "RANGING",       0.75),
        ("GOLD",      "BUY",          0.65, "VOLATILE",      0.70),
        ("CRUDEOIL",  "SELL",         0.55, "TRENDING_DOWN", 0.82),
        ("BTC",       "BUY",          0.72, "TRENDING_DOWN", 0.79),
        ("SBIN",      "BUY",          0.60, "RANGING",       0.40),  # uncertain regime
    ]

    # Create fake features
    feat_overbought = pd.Series({"rsi_14": 71, "adx": 16, "atr_ratio": 1.0, "bb_pct_b": 0.92})
    feat_oversold   = pd.Series({"rsi_14": 29, "adx": 16, "atr_ratio": 1.0, "bb_pct_b": 0.05})
    feat_neutral    = pd.Series({"rsi_14": 52, "adx": 28, "atr_ratio": 1.2, "bb_pct_b": 0.55})

    feat_map = {
        "NIFTY50":   feat_neutral,
        "BANKNIFTY": feat_neutral,
        "RELIANCE":  feat_oversold,
        "TCS":       feat_overbought,
        "GOLD":      feat_neutral,
        "CRUDEOIL":  feat_neutral,
        "BTC":       feat_neutral,
        "SBIN":      feat_neutral,
    }

    print(f"{'Symbol':<12} {'Regime':<14} {'Bias':<14} {'Conf':>5} "
          f"→ {'Action':<6} {'Adj Bias':<14} {'AdjConf':>7} "
          f"{'SL×':>4} {'Sz×':>4} {'Mode'}")
    print("─" * 110)

    for sym, bias, conf, regime, rconf in test_cases:
        r = rf.filter(
            symbol      = sym,
            bias        = bias,
            confidence  = conf,
            regime      = regime,
            regime_conf = rconf,
            features    = feat_map.get(sym),
        )
        print(
            f"{sym:<12} {regime:<14} {bias:<14} {conf:>5.0%} "
            f"→ {r.action:<6} {r.adj_bias:<14} {r.adj_confidence:>7.0%} "
            f"{r.sl_multiplier:>4.1f} {r.size_multiplier:>4.1f} {r.strategy_mode}"
        )
        for reason in r.reasons:
            print(f"  ✓ {reason}")
        for warning in r.warnings:
            print(f"  ⚠ {warning}")
        print()