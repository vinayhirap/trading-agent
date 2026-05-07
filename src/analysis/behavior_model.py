# trading-agent/src/analysis/behavior_model.py
"""
Trader Behavior Model

Models intraday market behavior patterns for Indian markets (NSE + MCX).

Key insight: A BUY signal at 9:16 AM is very different from the same signal
at 10:45 AM. The first is in a chaotic auction period with wide spreads and
stop-hunt risk. The second is in a more predictable institutional flow period.

This module provides:
  1. Session classification (which behavioral phase the market is in)
  2. Per-session signal quality scores (how reliable signals are right now)
  3. Intraday volatility expectations (expected ATR for current session)
  4. Optimal entry/exit windows per strategy type
  5. Day-of-week behavioral biases

All times are IST (Asia/Kolkata).

Sources of behavioral patterns:
  - NSE data studies on intraday volatility distribution
  - FII/DII activity patterns (FII more active in afternoon)
  - Options expiry effects (Thursday for weekly, last Thursday for monthly)
  - Settlement day effects (T+2)
"""
from dataclasses import dataclass, field
from datetime import datetime, time, date, timedelta
from enum import Enum
from typing import Optional
from zoneinfo import ZoneInfo
from loguru import logger

IST = ZoneInfo("Asia/Kolkata")


# ── Session taxonomy ──────────────────────────────────────────────────────────

class MarketSession(str, Enum):
    PRE_OPEN       = "PRE_OPEN"        # 9:00–9:15 AM
    OPENING_CHAOS  = "OPENING_CHAOS"   # 9:15–9:30 AM (highest volatility, worst entries)
    INSTITUTIONAL  = "INSTITUTIONAL"   # 9:30–10:30 AM (institutional order flow)
    TREND_WINDOW   = "TREND_WINDOW"    # 10:30–12:00 PM (most predictable)
    LUNCH_LULL     = "LUNCH_LULL"      # 12:00–13:30 PM (choppy, low volume)
    US_PREMARKET   = "US_PREMARKET"    # 13:30–14:30 PM (US pre-market influences)
    CLOSING_DRIVE  = "CLOSING_DRIVE"   # 14:30–15:00 PM (institutional positioning)
    CLOSING_CHAOS  = "CLOSING_CHAOS"   # 15:00–15:20 PM (squaring off, volatile)
    CLOSING_AUCTION= "CLOSING_AUCTION" # 15:20–15:30 PM (avoid new entries)
    CLOSED         = "CLOSED"
    MCX_EVENING    = "MCX_EVENING"     # 17:00–23:30 (MCX evening session)
    MCX_MORNING    = "MCX_MORNING"     # 09:00–17:00 (MCX day session)


class StrategyType(str, Enum):
    TREND_FOLLOWING  = "TREND_FOLLOWING"
    MEAN_REVERSION   = "MEAN_REVERSION"
    BREAKOUT         = "BREAKOUT"
    MOMENTUM         = "MOMENTUM"
    SCALPING         = "SCALPING"


@dataclass
class SessionCharacteristics:
    """What we know about a specific market session."""
    session:            MarketSession
    label:              str
    volatility_mult:    float    # multiplier vs average (1.0 = average)
    volume_mult:        float    # volume relative to daily average
    trend_reliability:  float    # 0-1: how reliable trend signals are
    mean_rev_quality:   float    # 0-1: how reliable mean-reversion is
    signal_quality:     float    # 0-1: overall signal quality
    entry_quality:      float    # 0-1: how good fills/spreads are
    best_strategy:      list[StrategyType]
    avoid_strategy:     list[StrategyType]
    behavioral_note:    str      # human-readable explanation
    sl_multiplier:      float    # multiply ATR SL by this (wider in chaotic sessions)


@dataclass
class BehaviorReading:
    """Complete behavioral reading for current market state."""
    session:             MarketSession
    characteristics:     SessionCharacteristics
    ist_time:            str
    day_of_week:         str
    day_bias:            str          # BULLISH / BEARISH / NEUTRAL tendency
    day_bias_strength:   float        # 0-1
    is_expiry_day:       bool
    is_expiry_week:      bool
    is_settlement_day:   bool
    special_notes:       list[str]    # any special behavioral notes
    signal_multiplier:   float        # multiply signal confidence by this
    recommended_action:  str          # TRADE / CAUTION / WAIT / AVOID


# ── Session definitions ───────────────────────────────────────────────────────

SESSION_CHARACTERISTICS: dict[MarketSession, SessionCharacteristics] = {

    MarketSession.PRE_OPEN: SessionCharacteristics(
        session           = MarketSession.PRE_OPEN,
        label             = "Pre-Open (9:00-9:15)",
        volatility_mult   = 0.5,
        volume_mult       = 0.1,
        trend_reliability = 0.2,
        mean_rev_quality  = 0.2,
        signal_quality    = 0.2,
        entry_quality     = 0.1,
        best_strategy     = [],
        avoid_strategy    = [StrategyType.TREND_FOLLOWING, StrategyType.BREAKOUT],
        behavioral_note   = "Order accumulation phase. No real price discovery yet. Don't trade.",
        sl_multiplier     = 2.0,
    ),

    MarketSession.OPENING_CHAOS: SessionCharacteristics(
        session           = MarketSession.OPENING_CHAOS,
        label             = "Opening Chaos (9:15-9:30)",
        volatility_mult   = 2.8,
        volume_mult       = 2.5,
        trend_reliability = 0.25,
        mean_rev_quality  = 0.3,
        signal_quality    = 0.25,
        entry_quality     = 0.2,
        best_strategy     = [],
        avoid_strategy    = [StrategyType.TREND_FOLLOWING, StrategyType.SCALPING],
        behavioral_note   = (
            "Highest volatility of the day. Wide spreads. Stop-hunting common. "
            "Retail panic and institutional manipulation. Wait for dust to settle."
        ),
        sl_multiplier     = 2.5,
    ),

    MarketSession.INSTITUTIONAL: SessionCharacteristics(
        session           = MarketSession.INSTITUTIONAL,
        label             = "Institutional Flow (9:30-10:30)",
        volatility_mult   = 1.8,
        volume_mult       = 1.9,
        trend_reliability = 0.65,
        mean_rev_quality  = 0.35,
        signal_quality    = 0.65,
        entry_quality     = 0.6,
        best_strategy     = [StrategyType.TREND_FOLLOWING, StrategyType.BREAKOUT],
        avoid_strategy    = [StrategyType.MEAN_REVERSION],
        behavioral_note   = (
            "Institutional algorithms executing morning orders. "
            "Trends established here tend to hold for the day. "
            "Watch for breakout of opening range."
        ),
        sl_multiplier     = 1.5,
    ),

    MarketSession.TREND_WINDOW: SessionCharacteristics(
        session           = MarketSession.TREND_WINDOW,
        label             = "Trend Window (10:30-12:00)",
        volatility_mult   = 1.2,
        volume_mult       = 1.1,
        trend_reliability = 0.75,
        mean_rev_quality  = 0.45,
        signal_quality    = 0.80,
        entry_quality     = 0.85,
        best_strategy     = [StrategyType.TREND_FOLLOWING, StrategyType.MOMENTUM],
        avoid_strategy    = [],
        behavioral_note   = (
            "Best period for trend-following. Volatility normalised. "
            "Institutional flow still active. Spreads tightest. "
            "Best risk/reward entries here."
        ),
        sl_multiplier     = 1.0,
    ),

    MarketSession.LUNCH_LULL: SessionCharacteristics(
        session           = MarketSession.LUNCH_LULL,
        label             = "Lunch Lull (12:00-13:30)",
        volatility_mult   = 0.7,
        volume_mult       = 0.55,
        trend_reliability = 0.40,
        mean_rev_quality  = 0.65,
        signal_quality    = 0.50,
        entry_quality     = 0.60,
        best_strategy     = [StrategyType.MEAN_REVERSION],
        avoid_strategy    = [StrategyType.BREAKOUT, StrategyType.MOMENTUM],
        behavioral_note   = (
            "Low volume, choppy price action. Breakouts often false. "
            "Trend signals unreliable — small institutions gone for lunch. "
            "Mean-reversion within day's range works better here."
        ),
        sl_multiplier     = 1.2,
    ),

    MarketSession.US_PREMARKET: SessionCharacteristics(
        session           = MarketSession.US_PREMARKET,
        label             = "US Pre-Market Influence (13:30-14:30)",
        volatility_mult   = 1.1,
        volume_mult       = 0.9,
        trend_reliability = 0.55,
        mean_rev_quality  = 0.45,
        signal_quality    = 0.60,
        entry_quality     = 0.65,
        best_strategy     = [StrategyType.TREND_FOLLOWING],
        avoid_strategy    = [],
        behavioral_note   = (
            "US futures pre-market activity influences Indian FIIs. "
            "Watch S&P futures direction. If strong gap in US pre-market, "
            "often drags Nifty in same direction. FII block deals common."
        ),
        sl_multiplier     = 1.1,
    ),

    MarketSession.CLOSING_DRIVE: SessionCharacteristics(
        session           = MarketSession.CLOSING_DRIVE,
        label             = "Closing Drive (14:30-15:00)",
        volatility_mult   = 1.4,
        volume_mult       = 1.5,
        trend_reliability = 0.70,
        mean_rev_quality  = 0.30,
        signal_quality    = 0.65,
        entry_quality     = 0.55,
        best_strategy     = [StrategyType.TREND_FOLLOWING, StrategyType.MOMENTUM],
        avoid_strategy    = [StrategyType.MEAN_REVERSION],
        behavioral_note   = (
            "Institutional positioning before close. Large blocks traded. "
            "Real direction of the day gets confirmed here. "
            "If trend was up all day, closing drive usually continues it."
        ),
        sl_multiplier     = 1.3,
    ),

    MarketSession.CLOSING_CHAOS: SessionCharacteristics(
        session           = MarketSession.CLOSING_CHAOS,
        label             = "Closing Chaos (15:00-15:20)",
        volatility_mult   = 2.2,
        volume_mult       = 2.0,
        trend_reliability = 0.30,
        mean_rev_quality  = 0.25,
        signal_quality    = 0.25,
        entry_quality     = 0.20,
        best_strategy     = [],
        avoid_strategy    = [StrategyType.TREND_FOLLOWING, StrategyType.SCALPING],
        behavioral_note   = (
            "F&O position squaring. High volatility, erratic moves. "
            "Intraday traders closing positions — moves often reverse. "
            "Avoid new entries. Focus on exits."
        ),
        sl_multiplier     = 2.0,
    ),

    MarketSession.CLOSING_AUCTION: SessionCharacteristics(
        session           = MarketSession.CLOSING_AUCTION,
        label             = "Closing Auction (15:20-15:30)",
        volatility_mult   = 1.8,
        volume_mult       = 1.3,
        trend_reliability = 0.20,
        mean_rev_quality  = 0.20,
        signal_quality    = 0.15,
        entry_quality     = 0.10,
        best_strategy     = [],
        avoid_strategy    = list(StrategyType),
        behavioral_note   = "Closing auction. Do not enter new positions. Exit only.",
        sl_multiplier     = 3.0,
    ),

    MarketSession.CLOSED: SessionCharacteristics(
        session           = MarketSession.CLOSED,
        label             = "Market Closed",
        volatility_mult   = 0.0,
        volume_mult       = 0.0,
        trend_reliability = 0.0,
        mean_rev_quality  = 0.0,
        signal_quality    = 0.5,    # daily signals still valid
        entry_quality     = 0.0,
        best_strategy     = [],
        avoid_strategy    = list(StrategyType),
        behavioral_note   = "Market closed. Use this time for analysis, not execution.",
        sl_multiplier     = 1.0,
    ),

    MarketSession.MCX_EVENING: SessionCharacteristics(
        session           = MarketSession.MCX_EVENING,
        label             = "MCX Evening Session (17:00-23:30)",
        volatility_mult   = 1.5,
        volume_mult       = 1.8,
        trend_reliability = 0.65,
        mean_rev_quality  = 0.40,
        signal_quality    = 0.70,
        entry_quality     = 0.70,
        best_strategy     = [StrategyType.TREND_FOLLOWING, StrategyType.MOMENTUM],
        avoid_strategy    = [],
        behavioral_note   = (
            "MCX's most active session. US markets open at 7:30 PM IST. "
            "Crude, gold volume peaks 7-10 PM. Follow WTI/Brent direction."
        ),
        sl_multiplier     = 1.2,
    ),
}

# ── Day-of-week behavioral biases (Indian markets) ────────────────────────────
# Based on academic studies and practitioner observation
DAY_BIASES = {
    0: ("BULLISH",  0.35, "Monday often sees FII buying after weekend analysis. Gap-up bias."),
    1: ("NEUTRAL",  0.20, "Tuesday: continuation of Monday trend or mild reversal."),
    2: ("BULLISH",  0.30, "Wednesday: mid-week often sees institutional accumulation."),
    3: ("VOLATILE", 0.50, "Thursday: weekly F&O expiry. Extreme volatility near 3:30 PM."),
    4: ("BEARISH",  0.25, "Friday: profit booking before weekend. Some FII selling."),
    5: ("CLOSED",   0.0,  "Saturday: market closed."),
    6: ("CLOSED",   0.0,  "Sunday: market closed."),
}


class BehaviorModel:
    """
    Models intraday trader behavior for Indian markets.

    Usage:
        bm = BehaviorModel()
        reading = bm.get_current_reading()
        # reading.signal_multiplier = 0.25 at 9:16 AM
        # reading.signal_multiplier = 1.0 at 10:45 AM

        # Get session for a specific time
        session = bm.get_session(datetime(2026, 4, 3, 10, 45))
        # → MarketSession.TREND_WINDOW
    """

    def get_current_reading(self) -> BehaviorReading:
        """Get behavioral reading for right now."""
        return self.get_reading(datetime.now(IST))

    def get_reading(self, dt: datetime) -> BehaviorReading:
        """Get behavioral reading for a specific datetime."""
        ist_dt  = dt.astimezone(IST) if dt.tzinfo else dt.replace(tzinfo=IST)
        session = self.get_session(ist_dt)
        chars   = SESSION_CHARACTERISTICS.get(session, SESSION_CHARACTERISTICS[MarketSession.CLOSED])

        # Day-of-week bias
        dow         = ist_dt.weekday()
        day_name    = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][dow]
        day_bias, day_strength, day_note = DAY_BIASES.get(dow, ("NEUTRAL", 0.0, ""))

        # Expiry checks
        is_expiry_day  = self._is_weekly_expiry(ist_dt)
        is_expiry_week = self._is_expiry_week(ist_dt)
        is_settlement  = self._is_settlement_day(ist_dt)

        # Build special notes
        notes = []
        if is_expiry_day:
            notes.append("⚠️ Weekly F&O expiry today — extreme volatility near close")
        if is_expiry_week and not is_expiry_day:
            notes.append("📅 Expiry week — elevated volatility Thursday")
        if is_settlement:
            notes.append("📊 T+2 settlement day — some forced selling possible")
        if day_bias == "VOLATILE":
            notes.append("⚡ Thursday expiry effect — signals less reliable near 3:30 PM")
        notes.append(day_note)

        # Compute overall signal multiplier
        # Base from session quality
        mult = chars.signal_quality
        # Expiry penalty
        if is_expiry_day and session in (MarketSession.CLOSING_CHAOS, MarketSession.CLOSING_DRIVE):
            mult *= 0.5
        # Day-of-week adjustment (small effect, ±15%)
        if day_bias == "BULLISH":
            mult = min(1.0, mult * 1.08)
        elif day_bias == "BEARISH":
            mult *= 0.92
        elif day_bias == "VOLATILE":
            mult *= 0.80

        # Recommended action
        if session in (MarketSession.OPENING_CHAOS, MarketSession.PRE_OPEN,
                       MarketSession.CLOSING_AUCTION):
            action = "AVOID"
        elif session in (MarketSession.CLOSING_CHAOS,):
            action = "EXIT_ONLY"
        elif mult < 0.40:
            action = "WAIT"
        elif mult < 0.60:
            action = "CAUTION"
        else:
            action = "TRADE"

        return BehaviorReading(
            session             = session,
            characteristics     = chars,
            ist_time            = ist_dt.strftime("%H:%M IST"),
            day_of_week         = day_name,
            day_bias            = day_bias,
            day_bias_strength   = day_strength,
            is_expiry_day       = is_expiry_day,
            is_expiry_week      = is_expiry_week,
            is_settlement_day   = is_settlement,
            special_notes       = notes,
            signal_multiplier   = round(mult, 2),
            recommended_action  = action,
        )

    def get_session(self, dt: datetime) -> MarketSession:
        """Classify the current market session."""
        ist_dt = dt.astimezone(IST) if dt.tzinfo else dt.replace(tzinfo=IST)
        t      = ist_dt.time()
        dow    = ist_dt.weekday()

        # Weekend
        if dow >= 5:
            return MarketSession.CLOSED

        # MCX evening (only relevant for commodity trading)
        if time(17, 0) <= t <= time(23, 30):
            return MarketSession.MCX_EVENING

        # NSE/BSE sessions
        if t < time(9, 0):              return MarketSession.CLOSED
        if t < time(9, 15):             return MarketSession.PRE_OPEN
        if t < time(9, 30):             return MarketSession.OPENING_CHAOS
        if t < time(10, 30):            return MarketSession.INSTITUTIONAL
        if t < time(12, 0):             return MarketSession.TREND_WINDOW
        if t < time(13, 30):            return MarketSession.LUNCH_LULL
        if t < time(14, 30):            return MarketSession.US_PREMARKET
        if t < time(15, 0):             return MarketSession.CLOSING_DRIVE
        if t < time(15, 20):            return MarketSession.CLOSING_CHAOS
        if t <= time(15, 30):           return MarketSession.CLOSING_AUCTION
        return MarketSession.CLOSED

    def get_best_entry_windows(self, strategy: StrategyType) -> list[str]:
        """Return best time windows for a given strategy type."""
        windows = []
        for session, chars in SESSION_CHARACTERISTICS.items():
            if strategy in chars.best_strategy and chars.signal_quality >= 0.60:
                windows.append(chars.label)
        return windows

    def get_intraday_volatility_pattern(self) -> list[dict]:
        """
        Returns expected volatility curve for the day.
        Useful for visualising the U-shaped intraday volatility pattern.
        """
        return [
            {"time": "09:00", "vol": 0.5,  "session": "Pre-open"},
            {"time": "09:15", "vol": 2.8,  "session": "Opening chaos"},
            {"time": "09:30", "vol": 1.8,  "session": "Institutional"},
            {"time": "10:00", "vol": 1.4,  "session": "Institutional"},
            {"time": "10:30", "vol": 1.2,  "session": "Trend window"},
            {"time": "11:00", "vol": 1.0,  "session": "Trend window"},
            {"time": "12:00", "vol": 0.7,  "session": "Lunch lull"},
            {"time": "13:00", "vol": 0.6,  "session": "Lunch lull"},
            {"time": "13:30", "vol": 0.9,  "session": "US pre-market"},
            {"time": "14:00", "vol": 1.0,  "session": "US pre-market"},
            {"time": "14:30", "vol": 1.4,  "session": "Closing drive"},
            {"time": "15:00", "vol": 2.2,  "session": "Closing chaos"},
            {"time": "15:20", "vol": 1.8,  "session": "Closing auction"},
        ]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _is_weekly_expiry(self, dt: datetime) -> bool:
        """Thursday = weekly Nifty/BankNifty F&O expiry."""
        return dt.weekday() == 3   # Thursday

    def _is_expiry_week(self, dt: datetime) -> bool:
        """Is this the week of monthly expiry (last Thursday of month)?"""
        today  = dt.date()
        # Find last Thursday of current month
        # Start from end of month and work back
        month_end = date(today.year, today.month % 12 + 1, 1) - timedelta(days=1) \
                    if today.month < 12 \
                    else date(today.year + 1, 1, 1) - timedelta(days=1)
        # Find last Thursday
        days_back = (month_end.weekday() - 3) % 7
        last_thu  = month_end - timedelta(days=days_back)
        # Is today within 5 days of last Thursday?
        diff = (last_thu - today).days
        return -2 <= diff <= 5

    def _is_settlement_day(self, dt: datetime) -> bool:
        """T+2 settlement. Roughly Tuesday = Monday's settlement, etc."""
        # This is a simplification — real T+2 ignores holidays
        return dt.weekday() == 1   # Tuesday (settlement of Friday trades)


# Module-level singleton
behavior_model = BehaviorModel()