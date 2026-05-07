# trading-agent/src/analysis/intraday_engine.py
"""
Intraday Engine — time-aware signal enhancement.

Combines:
  - BehaviorModel (session classification + quality scores)
  - Technical signals (from ensemble or rule-based)
  - Event context (from EventEngine)

Output: time-adjusted signal with:
  - Session-appropriate confidence adjustment
  - ATR-based SL width recommendation (wider in chaotic sessions)
  - Intraday vs positional trade recommendation
  - Specific entry timing advice

Key behaviors modelled:
  1. Opening range breakout (ORB): 9:30-10:00 range defines the day
  2. FII activity ramp: signals more reliable as institutions enter
  3. Lunch reversal: mean-reversion window 12:00-13:00
  4. Closing momentum: trend continuation 14:30-15:00
  5. Expiry pin risk: prices gravitate to max-pain on expiry Thursdays
"""
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Optional
from zoneinfo import ZoneInfo
from loguru import logger

from src.analysis.behavior_model import (
    BehaviorModel, BehaviorReading, MarketSession,
    StrategyType, SESSION_CHARACTERISTICS, behavior_model,
)

IST = ZoneInfo("Asia/Kolkata")


@dataclass
class IntradaySignal:
    """
    A trading signal with full intraday behavioral context.
    """
    symbol:               str
    original_bias:        str
    original_confidence:  float

    # Behavior adjustments
    session:              MarketSession
    session_label:        str
    behavior_reading:     BehaviorReading
    adjusted_confidence:  float      # original × session quality
    behavior_multiplier:  float

    # Time-specific recommendations
    entry_timing:         str        # "Enter now" / "Wait for 10:30" / "Avoid"
    strategy_fit:         str        # "GOOD" / "MARGINAL" / "POOR"
    sl_multiplier:        float      # multiply your ATR SL by this
    recommended_sl_pct:   float      # suggested SL % from entry
    hold_type:            str        # "INTRADAY" / "SWING" / "POSITIONAL"

    # Narrative
    timing_advice:        str
    session_warnings:     list[str] = field(default_factory=list)
    session_notes:        list[str] = field(default_factory=list)

    # MCX-specific (if commodity)
    is_mcx_symbol:        bool = False
    mcx_session_note:     str = ""


class IntradayEngine:
    """
    Produces time-aware trading signals by combining technical signals
    with intraday behavioral patterns.

    Usage:
        engine = IntradayEngine()
        signal = engine.enhance_signal(
            symbol="NIFTY50",
            bias="BUY",
            confidence=0.72,
            atr_pct=0.015,
        )
        # signal.adjusted_confidence at 9:16 AM ≈ 0.72 × 0.25 = 0.18 (don't trade)
        # signal.adjusted_confidence at 10:45 AM ≈ 0.72 × 0.80 = 0.58 (trade)
    """

    MCX_SYMBOLS = {"GOLD", "SILVER", "CRUDEOIL", "COPPER", "NATURALGAS"}

    def __init__(self):
        self._bm = behavior_model

    def enhance_signal(
        self,
        symbol:     str,
        bias:       str,
        confidence: float,
        atr_pct:    float = 0.015,  # current ATR as % of price
        dt:         datetime = None,
    ) -> IntradaySignal:
        """
        Enhance a technical signal with intraday behavioral context.

        Args:
            symbol:     trading symbol
            bias:       BUY / SELL / HOLD / STRONG BUY etc.
            confidence: model confidence 0-1
            atr_pct:    ATR as fraction of price (0.015 = 1.5%)
            dt:         datetime (IST), defaults to now
        """
        ist_dt  = (dt or datetime.now(IST)).astimezone(IST)
        reading = self._bm.get_reading(ist_dt)
        chars   = reading.characteristics
        is_mcx  = symbol in self.MCX_SYMBOLS

        # Adjust for MCX: use MCX session quality if applicable
        if is_mcx and reading.session == MarketSession.MCX_EVENING:
            mcx_chars = SESSION_CHARACTERISTICS[MarketSession.MCX_EVENING]
            base_mult = mcx_chars.signal_quality
        else:
            base_mult = chars.signal_quality

        # Apply behavior multiplier
        adj_confidence = round(confidence * base_mult * reading.signal_multiplier, 3)
        # Cap to 0.99
        adj_confidence = min(0.99, adj_confidence)

        # Strategy fit check
        bias_upper = bias.upper()
        if "BUY" in bias_upper or "SELL" in bias_upper:
            strategy = StrategyType.TREND_FOLLOWING
        else:
            strategy = StrategyType.MEAN_REVERSION

        if strategy in chars.best_strategy:
            strategy_fit = "GOOD"
        elif strategy in chars.avoid_strategy:
            strategy_fit = "POOR"
        else:
            strategy_fit = "MARGINAL"

        # SL multiplier: wider stops in chaotic sessions
        sl_mult    = chars.sl_multiplier
        sl_rec_pct = round(atr_pct * sl_mult * 100, 2)  # as percentage

        # Entry timing advice
        entry_timing, timing_advice = self._get_entry_advice(
            reading, bias, adj_confidence, strategy_fit
        )

        # Hold type: intraday vs positional
        if reading.session in (MarketSession.TREND_WINDOW, MarketSession.INSTITUTIONAL):
            hold_type = "INTRADAY" if atr_pct > 0.02 else "SWING"
        elif reading.session == MarketSession.CLOSED:
            hold_type = "POSITIONAL"
        else:
            hold_type = "INTRADAY"

        # Warnings and notes
        warnings = []
        notes    = list(reading.special_notes)

        if reading.session == MarketSession.OPENING_CHAOS:
            warnings.append("🚨 Opening chaos — do NOT enter. Wait until 9:30 AM minimum.")
        if reading.session == MarketSession.CLOSING_CHAOS:
            warnings.append("⚠️ Closing chaos — F&O squaring off. Avoid new entries.")
        if reading.session == MarketSession.CLOSING_AUCTION:
            warnings.append("🚨 Closing auction — EXIT positions only. No new entries.")
        if reading.is_expiry_day and reading.session in (
            MarketSession.CLOSING_DRIVE, MarketSession.CLOSING_CHAOS
        ):
            warnings.append(
                "⚠️ Weekly expiry near close — options pins cause erratic price action. "
                "Widen SL by 50% or exit before 3:15 PM."
            )
        if strategy_fit == "POOR":
            warnings.append(
                f"⚠️ {strategy.value} strategy is not recommended during "
                f"{reading.session.value} session."
            )

        # MCX-specific notes
        mcx_note = ""
        if is_mcx:
            t = ist_dt.time()
            if time(9, 0) <= t < time(11, 30):
                mcx_note = "MCX morning session: follows overnight WTI/Comex prices."
            elif time(17, 0) <= t < time(19, 30):
                mcx_note = "MCX evening session: high volume. Watch US pre-market for direction."
            elif time(19, 30) <= t < time(22, 0):
                mcx_note = "US markets open — MCX crude at peak volume. Best session for commodity signals."
            elif time(22, 0) <= t <= time(23, 30):
                mcx_note = "Late MCX session — volume declining. Tighten SL before close."

        return IntradaySignal(
            symbol               = symbol,
            original_bias        = bias,
            original_confidence  = confidence,
            session              = reading.session,
            session_label        = chars.label,
            behavior_reading     = reading,
            adjusted_confidence  = adj_confidence,
            behavior_multiplier  = round(base_mult * reading.signal_multiplier, 2),
            entry_timing         = entry_timing,
            strategy_fit         = strategy_fit,
            sl_multiplier        = sl_mult,
            recommended_sl_pct   = sl_rec_pct,
            hold_type            = hold_type,
            timing_advice        = timing_advice,
            session_warnings     = warnings,
            session_notes        = notes,
            is_mcx_symbol        = is_mcx,
            mcx_session_note     = mcx_note,
        )

    def get_best_entry_time(self, strategy: StrategyType) -> str:
        """Return the best next entry window for a strategy type."""
        ist_now  = datetime.now(IST)
        t_now    = ist_now.time()
        windows  = {
            StrategyType.TREND_FOLLOWING:  [(time(9,30), "9:30"), (time(10,30), "10:30"), (time(14,30), "14:30")],
            StrategyType.MEAN_REVERSION:   [(time(12,0), "12:00"), (time(13,30), "13:30")],
            StrategyType.BREAKOUT:         [(time(9,30), "9:30"), (time(10,30), "10:30")],
            StrategyType.MOMENTUM:         [(time(10,30), "10:30"), (time(14,0), "14:00")],
        }
        for window_time, label in windows.get(strategy, []):
            if t_now < window_time:
                return f"Next best window: {label} IST"
        return "Tomorrow at 9:30 AM IST (current session ending)"

    def get_daily_schedule(self) -> list[dict]:
        """Return today's intraday trading schedule with quality scores."""
        schedule = []
        sessions = [
            (time(9, 0),  time(9, 15),  MarketSession.PRE_OPEN),
            (time(9, 15), time(9, 30),  MarketSession.OPENING_CHAOS),
            (time(9, 30), time(10, 30), MarketSession.INSTITUTIONAL),
            (time(10,30), time(12, 0),  MarketSession.TREND_WINDOW),
            (time(12, 0), time(13,30),  MarketSession.LUNCH_LULL),
            (time(13,30), time(14,30),  MarketSession.US_PREMARKET),
            (time(14,30), time(15, 0),  MarketSession.CLOSING_DRIVE),
            (time(15, 0), time(15,20),  MarketSession.CLOSING_CHAOS),
            (time(15,20), time(15,30),  MarketSession.CLOSING_AUCTION),
        ]
        ist_now = datetime.now(IST).time()
        for start, end, session in sessions:
            chars   = SESSION_CHARACTERISTICS[session]
            is_now  = start <= ist_now < end
            schedule.append({
                "time":    f"{start.strftime('%H:%M')}–{end.strftime('%H:%M')}",
                "session": chars.label,
                "quality": chars.signal_quality,
                "vol":     chars.volatility_mult,
                "action":  "TRADE" if chars.signal_quality >= 0.65 else
                           "CAUTION" if chars.signal_quality >= 0.45 else "AVOID",
                "is_now":  is_now,
                "note":    chars.behavioral_note[:80],
            })
        return schedule

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_entry_advice(
        self,
        reading:    BehaviorReading,
        bias:       str,
        adj_conf:   float,
        fit:        str,
    ) -> tuple[str, str]:
        """Returns (entry_timing, timing_advice)."""
        session = reading.session

        if session == MarketSession.OPENING_CHAOS:
            return (
                "WAIT — opening chaos",
                "Do not enter now. Wait until at least 9:30 AM. "
                "Opening range (9:15-9:30) high/low will be key levels."
            )
        if session == MarketSession.PRE_OPEN:
            return (
                "WAIT — pre-open",
                "Market hasn't opened yet. Set your alerts. "
                "Plan your entry for after 9:30 AM."
            )
        if session in (MarketSession.CLOSING_CHAOS, MarketSession.CLOSING_AUCTION):
            return (
                "EXIT ONLY",
                "Too late to enter new positions. Focus on managing existing positions. "
                "Exit intraday trades before 3:20 PM."
            )
        if session == MarketSession.LUNCH_LULL:
            if "BUY" in bias.upper() or "SELL" in bias.upper():
                return (
                    "CAUTION — lunch lull",
                    "Low-volume session. Trend signals unreliable. "
                    "If you must trade, reduce size by 50% and use tighter SL."
                )
            return ("WAIT", "Lunch lull is better for mean-reversion. Wait for 13:30 PM.")

        if session == MarketSession.TREND_WINDOW and fit == "GOOD":
            return (
                "ENTER NOW",
                "Optimal entry window. Institutional flow is active, spreads tight, "
                "trend signals are most reliable. Enter at market or use 0.1% limit."
            )

        if session == MarketSession.INSTITUTIONAL and adj_conf >= 0.55:
            return (
                "ENTER — institutional window",
                "Institutional flow window. Signal is good. "
                "Enter with normal size. Watch for opening range breakout confirmation."
            )

        if session == MarketSession.CLOSING_DRIVE:
            return (
                "ENTER — closing drive",
                "Closing drive is active. Real direction confirming. "
                "Use for intraday momentum trades only — exit by 3:15 PM."
            )

        if adj_conf < 0.45:
            return (
                "WAIT — low confidence",
                f"Session quality is low (multiplier: {reading.signal_multiplier:.0%}). "
                f"Adjusted confidence {adj_conf:.0%} is below threshold. "
                f"Wait for better conditions."
            )

        return (
            "MARGINAL ENTRY",
            f"Session ({reading.session.value}) has moderate quality. "
            f"Enter with reduced size (50% of normal) and wider SL."
        )


# Module-level singleton
intraday_engine = IntradayEngine()