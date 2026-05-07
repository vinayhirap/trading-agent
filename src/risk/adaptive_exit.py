# trading-agent/src/risk/adaptive_exit.py
"""
Adaptive Exit Engine — replaces rigid ATR-based stop/target.

The core problem with fixed stops in Indian markets:
- Nifty can gap down 200 points at open, recover 350 by 2pm → stop triggers at bottom
- Volatile stocks spike through stops then reverse → we exit at the worst price
- MCX crude oil can move ±5% intraday on global news

This engine decides:
1. WHAT type of exit to use (stop, trail, time, structure, none)
2. HOW WIDE the stop should be (regime-adjusted)
3. WHEN to skip the stop entirely (high-conviction swing trades)
4. WHEN to use time-based exits instead of price-based

Exit modes:
    NONE        — hold with no stop (high conviction, clear structure)
    FIXED_ATR   — classic 2× ATR stop (low volatility trending markets)
    TRAIL_ATR   — trailing ATR stop (trending markets, protect profits)
    STRUCTURE   — exit below/above key support/resistance level
    TIME_BASED  — exit at end of session regardless of price
    TIGHT       — 1× ATR stop (overextended moves, risky entries)
    WIDE        — 3× ATR stop (high volatility, give trade room to breathe)
"""
import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass
from loguru import logger

from src.risk.models import OrderSide


class ExitMode(str, Enum):
    NONE       = "none"        # no stop — hold until signal reverses
    FIXED_ATR  = "fixed_atr"   # standard 2× ATR
    TRAIL_ATR  = "trail_atr"   # trailing ATR
    STRUCTURE  = "structure"   # support/resistance level
    TIME_BASED = "time_based"  # session close exit
    TIGHT      = "tight"       # 1× ATR (high risk entry)
    WIDE       = "wide"        # 3× ATR (volatile market)


@dataclass
class ExitPlan:
    """Complete exit plan for a trade."""
    mode:           ExitMode
    stop_loss:      float | None     # None if mode is NONE or TIME_BASED
    target:         float | None     # None if no target set
    atr_multiplier: float
    reasoning:      str              # human-readable explanation
    use_trailing:   bool = False
    time_exit_bar:  int  = 0         # bar index to exit (for TIME_BASED)

    def __str__(self):
        sl  = f"₹{self.stop_loss:.2f}" if self.stop_loss else "NONE"
        tgt = f"₹{self.target:.2f}"   if self.target    else "OPEN"
        return (
            f"[{self.mode.value.upper()}] SL={sl} TGT={tgt} | "
            f"trail={self.use_trailing} | {self.reasoning}"
        )


class AdaptiveExitEngine:
    """
    Decides the optimal exit strategy based on:
    1. Market volatility regime (VIX proxy via ATR ratio)
    2. Trend strength (ADX)
    3. Trade type (intraday vs swing)
    4. Confidence level of the signal
    5. Current market structure (support/resistance proximity)
    """

    def __init__(
        self,
        base_atr_mult:   float = 2.0,
        min_rr:          float = 1.5,    # minimum R:R to take the trade
        high_vol_threshold: float = 1.5, # ATR ratio above this = high vol regime
        low_vol_threshold:  float = 0.7, # ATR ratio below this = squeeze
    ):
        self.base_atr_mult        = base_atr_mult
        self.min_rr               = min_rr
        self.high_vol_threshold   = high_vol_threshold
        self.low_vol_threshold    = low_vol_threshold

    def plan(
        self,
        df:          pd.DataFrame,    # OHLCV + features DataFrame
        entry_price: float,
        side:        OrderSide,
        confidence:  float,           # model confidence 0-1
        trade_type:  str = "swing",   # "intraday" or "swing"
        adx:         float = 20.0,
        atr_ratio:   float = 1.0,     # current ATR / 20-bar avg ATR
    ) -> ExitPlan:
        """
        Main method — returns the optimal ExitPlan for this trade.
        """
        atr = self._get_atr(df)
        if atr <= 0 or entry_price <= 0:
            return self._fallback_plan(entry_price, side, atr)

        # ── Determine volatility regime ──────────────────────────────────────
        regime = self._classify_regime(atr_ratio, adx)

        # ── Intraday: always use some form of stop ───────────────────────────
        if trade_type == "intraday":
            return self._intraday_plan(entry_price, side, atr, regime, confidence, df)

        # ── Swing: can skip stop in right conditions ─────────────────────────
        return self._swing_plan(entry_price, side, atr, regime, confidence, adx, df)

    def _intraday_plan(
        self, entry, side, atr, regime, confidence, df
    ) -> ExitPlan:
        """Intraday always exits same day — time-based is fallback."""

        if regime == "high_vol":
            # Wide stop — market is swinging, tight stop will get hit
            mult = 2.5
            mode = ExitMode.WIDE
            reason = (
                f"High volatility regime (ATR ratio {regime}) — "
                f"using wider {mult}×ATR stop to avoid whipsaw"
            )
        elif regime == "trending":
            # Trail ATR — lock in profits as trend continues
            mult   = 1.5
            mode   = ExitMode.TRAIL_ATR
            reason = (
                f"Strong trend (ADX>{20}) — trailing {mult}×ATR stop "
                f"to let profits run"
            )
        elif regime == "squeeze":
            # Tight stop — squeeze breakout either works fast or fails fast
            mult   = 1.2
            mode   = ExitMode.TIGHT
            reason = (
                "Volatility squeeze — breakout play, tight stop, "
                "exit quickly if wrong"
            )
        else:
            # Normal regime
            mult   = self.base_atr_mult
            mode   = ExitMode.FIXED_ATR
            reason = f"Normal regime — standard {mult}×ATR stop"

        stop, target = self._compute_levels(entry, side, atr, mult, mult * self.min_rr)

        return ExitPlan(
            mode=mode, stop_loss=stop, target=target,
            atr_multiplier=mult, reasoning=reason,
            use_trailing=(mode == ExitMode.TRAIL_ATR),
        )

    def _swing_plan(
        self, entry, side, atr, regime, confidence, adx, df
    ) -> ExitPlan:
        """
        Swing trades can run for days. Key question: should we even set a stop?

        Skip stop when:
        - Very high confidence (>70%) + strong trend (ADX>30)
        - Price at major structure (support/resistance)
        - Position is small enough that full loss is acceptable

        Always use stop when:
        - Low confidence (<60%)
        - High volatility regime
        - Intraday-style entry (chasing a move)
        """

        # Find key structure levels
        support, resistance = self._find_structure(df)

        # ── Condition: high confidence strong trend → no fixed stop ─────────
        if confidence >= 0.68 and adx >= 28 and regime == "trending":
            # Use structure-based exit or trailing stop only
            if side == OrderSide.BUY and support > 0:
                stop = support - atr * 0.5    # just below support
                target = entry + (entry - stop) * 2.5
                return ExitPlan(
                    mode=ExitMode.STRUCTURE,
                    stop_loss=round(stop, 2),
                    target=round(target, 2),
                    atr_multiplier=0,
                    reasoning=(
                        f"High confidence ({confidence:.0%}) + strong trend "
                        f"(ADX={adx:.0f}) → structure stop at ₹{stop:.2f} "
                        f"(below support ₹{support:.2f})"
                    ),
                    use_trailing=True,
                )
            elif side == OrderSide.SELL and resistance > 0:
                stop = resistance + atr * 0.5
                target = entry - (stop - entry) * 2.5
                return ExitPlan(
                    mode=ExitMode.STRUCTURE,
                    stop_loss=round(stop, 2),
                    target=round(target, 2),
                    atr_multiplier=0,
                    reasoning=(
                        f"Structure stop above resistance ₹{resistance:.2f}"
                    ),
                    use_trailing=True,
                )

        # ── Condition: very high confidence, small position → no stop ────────
        if confidence >= 0.72 and regime not in ("high_vol",):
            return ExitPlan(
                mode=ExitMode.NONE,
                stop_loss=None,
                target=None,
                atr_multiplier=0,
                reasoning=(
                    f"Very high confidence ({confidence:.0%}), normal regime → "
                    f"no fixed stop. Exit on signal reversal or RSI extremes. "
                    f"RISK: position must be sized for full loss tolerance."
                ),
                use_trailing=False,
            )

        # ── High volatility → wide stop ──────────────────────────────────────
        if regime == "high_vol":
            mult = 3.0
            stop, target = self._compute_levels(entry, side, atr, mult, mult * self.min_rr)
            return ExitPlan(
                mode=ExitMode.WIDE,
                stop_loss=stop, target=target, atr_multiplier=mult,
                reasoning=(
                    f"High volatility swing — {mult}×ATR stop gives room for "
                    f"100–300pt intraday swings without triggering"
                ),
            )

        # ── Default: trailing ATR for swing ──────────────────────────────────
        mult = self.base_atr_mult
        stop, target = self._compute_levels(entry, side, atr, mult, mult * self.min_rr)
        return ExitPlan(
            mode=ExitMode.TRAIL_ATR,
            stop_loss=stop, target=target, atr_multiplier=mult,
            reasoning=f"Swing trade — trailing {mult}×ATR, target {mult*self.min_rr:.1f}×ATR",
            use_trailing=True,
        )

    def update_exit(
        self,
        plan:          ExitPlan,
        current_price: float,
        entry_price:   float,
        side:          OrderSide,
        df:            pd.DataFrame,
    ) -> tuple[ExitPlan, str]:
        """
        Re-evaluate the exit plan every bar.
        Returns (updated_plan, action) where action is:
        'hold', 'exit_stop', 'exit_target', 'exit_reversal', 'exit_time'
        """
        atr = self._get_atr(df)

        # Check if stop hit (if we have one)
        if plan.stop_loss is not None:
            if side == OrderSide.BUY and current_price <= plan.stop_loss:
                return plan, "exit_stop"
            if side == OrderSide.SELL and current_price >= plan.stop_loss:
                return plan, "exit_stop"

        # Check if target hit
        if plan.target is not None:
            if side == OrderSide.BUY and current_price >= plan.target:
                return plan, "exit_target"
            if side == OrderSide.SELL and current_price <= plan.target:
                return plan, "exit_target"

        # Update trailing stop
        if plan.use_trailing and plan.stop_loss is not None:
            new_stop = self._trail_stop(
                plan.stop_loss, current_price, atr, plan.atr_multiplier or 2.0, side
            )
            if new_stop != plan.stop_loss:
                plan.stop_loss = new_stop
                plan.reasoning += f" | Trailed to ₹{new_stop:.2f}"

        # For NONE mode — check for signal reversal via RSI
        if plan.mode == ExitMode.NONE:
            rsi = df["rsi_14"].iloc[-1] if "rsi_14" in df.columns else 50
            if side == OrderSide.BUY and rsi > 78:
                return plan, "exit_reversal"   # overbought — take profits
            if side == OrderSide.SELL and rsi < 22:
                return plan, "exit_reversal"   # oversold — cover short

        return plan, "hold"

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_atr(self, df: pd.DataFrame) -> float:
        if "atr_14" in df.columns:
            val = df["atr_14"].iloc[-1]
            return float(val) if not np.isnan(val) else 0.0
        if all(c in df.columns for c in ["high","low","close"]):
            tr = pd.concat([
                df["high"] - df["low"],
                (df["high"] - df["close"].shift()).abs(),
                (df["low"]  - df["close"].shift()).abs(),
            ], axis=1).max(axis=1)
            return float(tr.ewm(span=14).mean().iloc[-1])
        return 0.0

    def _classify_regime(self, atr_ratio: float, adx: float) -> str:
        if atr_ratio >= self.high_vol_threshold:
            return "high_vol"
        if adx >= 25 and atr_ratio < self.high_vol_threshold:
            return "trending"
        if atr_ratio <= self.low_vol_threshold:
            return "squeeze"
        return "normal"

    def _compute_levels(
        self, entry, side, atr, stop_mult, target_mult
    ) -> tuple[float, float]:
        if side == OrderSide.BUY:
            stop   = entry - stop_mult   * atr
            target = entry + target_mult * atr
        else:
            stop   = entry + stop_mult   * atr
            target = entry - target_mult * atr
        return round(stop, 2), round(target, 2)

    def _find_structure(self, df: pd.DataFrame) -> tuple[float, float]:
        """Find nearest support and resistance from recent swing highs/lows."""
        if len(df) < 20:
            return 0.0, 0.0
        recent = df.tail(50)
        # Swing lows = local minima (support)
        lows   = recent["low"].rolling(5, center=True).min()
        support = float(lows.dropna().tail(10).max())
        # Swing highs = local maxima (resistance)
        highs  = recent["high"].rolling(5, center=True).max()
        resistance = float(highs.dropna().tail(10).min())
        return support, resistance

    def _trail_stop(
        self, current_stop, current_price, atr, mult, side
    ) -> float:
        if side == OrderSide.BUY:
            new = current_price - mult * atr
            return round(max(current_stop, new), 2)
        else:
            new = current_price + mult * atr
            return round(min(current_stop, new), 2)

    def _fallback_plan(self, entry, side, atr) -> ExitPlan:
        fallback_atr = max(atr, entry * 0.015)
        stop, target = self._compute_levels(entry, side, fallback_atr, 2.0, 3.0)
        return ExitPlan(
            mode=ExitMode.FIXED_ATR, stop_loss=stop, target=target,
            atr_multiplier=2.0,
            reasoning="Fallback: insufficient data for regime detection",
        )