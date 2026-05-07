# trading-agent/src/utils/market_hours.py
"""
Market Hours — Complete Indian Market Calendar

Handles:
- NSE/BSE: 9:15 AM - 3:30 PM IST (Mon-Fri, except holidays)
- MCX: 9:00 AM - 11:30 PM IST on normal days
        5:00 PM - 11:30 PM IST on NSE holidays (evening session only)
        CLOSED on MCX-specific holidays

Key concept: NSE and MCX have different holiday lists.
On NSE holidays (like Mahavir Jayanti), MCX still opens at 5 PM.
Only on certain holidays (Diwali Muhurat, etc.) is MCX also closed.

Mahavir Jayanti 2026 = March 31, 2026
  NSE/BSE: CLOSED all day
  MCX: Evening session 5:00 PM - 11:30 PM IST
"""
from datetime import datetime, time, date, timedelta

from zoneinfo import ZoneInfo
from enum import Enum
from dataclasses import dataclass

IST = ZoneInfo("Asia/Kolkata")
UTC = ZoneInfo("UTC")


class MarketSession(str, Enum):
    PRE_OPEN   = "pre_open"
    OPEN       = "open"
    POST_CLOSE = "post_close"
    CLOSED     = "closed"
    MCX_EVENING = "mcx_evening"   # NSE holiday but MCX evening open


# ── NSE/BSE Holidays (confirmed from NSE circulars) ──────────────────────────
NSE_HOLIDAYS_2025: set[date] = {
    date(2025,  1, 26),   # Republic Day
    date(2025,  2, 19),   # Chhatrapati Shivaji Maharaj Jayanti
    date(2025,  3, 14),   # Holi
    date(2025,  3, 31),   # Id-ul-Fitr (Ramzan Id) — tentative
    date(2025,  4, 10),   # Dr. Ambedkar Jayanti
    date(2025,  4, 14),   # Ram Navami
    date(2025,  4, 18),   # Good Friday
    date(2025,  5,  1),   # Maharashtra Day
    date(2025,  8, 15),   # Independence Day
    date(2025,  8, 27),   # Ganesh Chaturthi
    date(2025, 10,  2),   # Mahatma Gandhi Jayanti
    date(2025, 10, 21),   # Diwali (Laxmi Puja)
    date(2025, 10, 22),   # Diwali (Balipratipada)
    date(2025, 11,  5),   # Guru Nanak Jayanti
    date(2025, 12, 25),   # Christmas
}

NSE_HOLIDAYS_2026: set[date] = {
    date(2026,  1, 26),   # Republic Day
    date(2026,  3, 20),   # Holi
    date(2026,  3, 31),   # Mahavir Jayanti ← TODAY (user confirmed)
    date(2026,  4,  3),   # Good Friday
    date(2026,  4, 14),   # Dr. Ambedkar Jayanti
    date(2026,  5,  1),   # Maharashtra Day
    date(2026,  8, 15),   # Independence Day
    date(2026,  9, 16),   # Ganesh Chaturthi (approx)
    date(2026, 10,  2),   # Gandhi Jayanti
    date(2026, 10, 19),   # Diwali Laxmi Puja (approx)
    date(2026, 10, 20),   # Diwali Balipratipada (approx)
    date(2026, 11, 24),   # Guru Nanak Jayanti (approx)
    date(2026, 12, 25),   # Christmas
}

NSE_HOLIDAYS: set[date] = NSE_HOLIDAYS_2025 | NSE_HOLIDAYS_2026

# ── MCX-specific full closures (MCX closed even for evening session) ──────────
# These are holidays where MCX also does NOT open at 5pm
MCX_FULL_CLOSURES: set[date] = {
    date(2025,  1, 26),   # Republic Day
    date(2025,  8, 15),   # Independence Day
    date(2025, 10,  2),   # Gandhi Jayanti
    date(2025, 10, 21),   # Diwali Laxmi Puja (MCX Muhurat trading only)
    date(2025, 12, 25),   # Christmas
    date(2026,  1, 26),   # Republic Day
    date(2026,  8, 15),   # Independence Day
    date(2026, 10,  2),   # Gandhi Jayanti
    date(2026, 12, 25),   # Christmas
}

# ── MCX normal session ────────────────────────────────────────────────────────
MCX_NORMAL_START    = time(9,  0)
MCX_NORMAL_END      = time(23, 30)
MCX_EVENING_START   = time(17,  0)   # 5:00 PM on NSE holidays
MCX_EVENING_END     = time(23, 30)


@dataclass
class MarketStatus:
    """Complete market status for all exchanges."""
    ist_time:          str
    nse_session:       str       # open / closed / pre_open / post_close
    nse_tradeable:     bool
    mcx_session:       str       # open / closed / evening_only
    mcx_tradeable:     bool
    is_nse_holiday:    bool
    is_mcx_full_close: bool
    holiday_name:      str
    secs_to_nse_open:  int
    secs_to_nse_close: int
    secs_to_mcx_open:  int
    secs_to_mcx_close: int


# Human-readable holiday names
HOLIDAY_NAMES = {
    date(2026, 3, 31): "Mahavir Jayanti",
    date(2026, 4,  3): "Good Friday",
    date(2026, 1, 26): "Republic Day",
    date(2026, 8, 15): "Independence Day",
    date(2025, 1, 26): "Republic Day",
    date(2025, 3, 14): "Holi",
    date(2025, 4, 18): "Good Friday",
    date(2025, 8, 15): "Independence Day",
    date(2025, 10,21): "Diwali",
    date(2025, 12,25): "Christmas",
    date(2026, 12,25): "Christmas",
}


class MarketHours:
    """
    Complete market hours handler for NSE, BSE, and MCX.

    Key rules:
    1. NSE: 9:15 AM - 3:30 PM, closed on NSE holidays + weekends
    2. MCX normal: 9:00 AM - 11:30 PM, closed on weekends
    3. MCX on NSE holidays: EVENING SESSION ONLY 5:00 PM - 11:30 PM
       (EXCEPT on full MCX closures like Republic Day, Independence Day)
    4. Crypto (CoinSwitch): 24×7, always tradeable
    """

    def now_ist(self) -> datetime:
        return datetime.now(IST)

    def is_weekend(self, dt: datetime = None) -> bool:
        dt = dt or self.now_ist()
        return dt.weekday() >= 5   # Sat=5, Sun=6

    def is_nse_holiday(self, dt: datetime = None) -> bool:
        dt = dt or self.now_ist()
        return dt.date() in NSE_HOLIDAYS

    def is_mcx_full_closure(self, dt: datetime = None) -> bool:
        dt = dt or self.now_ist()
        return dt.date() in MCX_FULL_CLOSURES

    def get_holiday_name(self, dt: datetime = None) -> str:
        dt = dt or self.now_ist()
        return HOLIDAY_NAMES.get(dt.date(), "Public Holiday")

    # ── NSE Status ────────────────────────────────────────────────────────────

    def get_nse_session(self, dt: datetime = None) -> str:
        dt = dt or self.now_ist()
        if self.is_weekend(dt) or self.is_nse_holiday(dt):
            return "closed"
        t = dt.time()
        if time(9, 0)  <= t < time(9, 15):   return "pre_open"
        if time(9, 15) <= t < time(15, 30):   return "open"
        if time(15, 30)<= t < time(16, 0):    return "post_close"
        return "closed"

    def is_nse_tradeable(self, dt: datetime = None) -> bool:
        return self.get_nse_session(dt) == "open"

    # ── MCX Status ────────────────────────────────────────────────────────────

    def get_mcx_session(self, dt: datetime = None) -> str:
        dt = dt or self.now_ist()
        if self.is_weekend(dt):
            return "closed"
        if self.is_mcx_full_closure(dt):
            return "closed"

        t = dt.time()

        # On NSE holidays → evening session only
        if self.is_nse_holiday(dt):
            if MCX_EVENING_START <= t <= MCX_EVENING_END:
                return "open (evening)"
            elif t < MCX_EVENING_START:
                return "closed (opens 5pm)"
            else:
                return "closed"

        # Normal weekday
        if MCX_NORMAL_START <= t <= MCX_NORMAL_END:
            return "open"
        return "closed"

    def is_mcx_tradeable(self, dt: datetime = None) -> bool:
        sess = self.get_mcx_session(dt)
        return "open" in sess

    # ── Countdown helpers ─────────────────────────────────────────────────────

    def secs_to_nse_open(self, dt: datetime = None) -> int:
        dt = dt or self.now_ist()
        if self.is_nse_tradeable(dt): return 0
        # Find next trading day 9:15 AM
        candidate = dt.replace(hour=9, minute=15, second=0, microsecond=0)
        if dt.time() >= time(15, 30) or self.is_nse_holiday(dt) or self.is_weekend(dt):
            candidate += timedelta(days=1)
        iters = 0
        while (self.is_nse_holiday(candidate) or candidate.weekday() >= 5) and iters < 14:
            candidate += timedelta(days=1)
            iters += 1
        diff = (candidate - dt).total_seconds()
        return max(0, int(diff))

    def secs_to_nse_close(self, dt: datetime = None) -> int:
        dt = dt or self.now_ist()
        if not self.is_nse_tradeable(dt): return 0
        close = dt.replace(hour=15, minute=30, second=0, microsecond=0)
        return max(0, int((close - dt).total_seconds()))

    def secs_to_mcx_open(self, dt: datetime = None) -> int:
        dt = dt or self.now_ist()
        if self.is_mcx_tradeable(dt): return 0
        t = dt.time()
        if self.is_nse_holiday(dt) and not self.is_mcx_full_closure(dt):
            # Opens at 5pm today
            evening = dt.replace(hour=17, minute=0, second=0, microsecond=0)
            if dt < evening:
                return int((evening - dt).total_seconds())
        # Next normal day 9am
        candidate = dt.replace(hour=9, minute=0, second=0, microsecond=0)
        candidate += timedelta(days=1)
        iters = 0
        while (candidate.weekday() >= 5 or self.is_mcx_full_closure(candidate)) and iters < 14:
            candidate += timedelta(days=1)
            iters += 1
        return max(0, int((candidate - dt).total_seconds()))

    def secs_to_mcx_close(self, dt: datetime = None) -> int:
        dt = dt or self.now_ist()
        if not self.is_mcx_tradeable(dt): return 0
        close = dt.replace(hour=23, minute=30, second=0, microsecond=0)
        return max(0, int((close - dt).total_seconds()))

    # ── Full status ───────────────────────────────────────────────────────────

    def get_full_status(self, dt: datetime = None) -> MarketStatus:
        dt = dt or self.now_ist()
        on_holiday    = self.is_nse_holiday(dt)
        full_closure  = self.is_mcx_full_closure(dt)
        holiday_name  = self.get_holiday_name(dt) if on_holiday else ""

        return MarketStatus(
            ist_time          = dt.strftime("%Y-%m-%d %H:%M:%S IST"),
            nse_session       = self.get_nse_session(dt),
            nse_tradeable     = self.is_nse_tradeable(dt),
            mcx_session       = self.get_mcx_session(dt),
            mcx_tradeable     = self.is_mcx_tradeable(dt),
            is_nse_holiday    = on_holiday,
            is_mcx_full_close = full_closure,
            holiday_name      = holiday_name,
            secs_to_nse_open  = self.secs_to_nse_open(dt),
            secs_to_nse_close = self.secs_to_nse_close(dt),
            secs_to_mcx_open  = self.secs_to_mcx_open(dt),
            secs_to_mcx_close = self.secs_to_mcx_close(dt),
        )

    # ── Legacy compatibility (keep existing callers working) ─────────────────

    def get_status(self, dt: datetime = None) -> dict:
        """Backward-compatible dict output."""
        dt   = dt or self.now_ist()
        full = self.get_full_status(dt)
        return {
            "ist_time":       full.ist_time,
            "session":        full.nse_session,
            "tradeable":      full.nse_tradeable,
            "is_trading_day": not full.is_nse_holiday and not self.is_weekend(dt),
            "secs_to_open":   full.secs_to_nse_open,
            "secs_to_close":  full.secs_to_nse_close,
            "mcx_session":    full.mcx_session,
            "mcx_tradeable":  full.mcx_tradeable,
            "is_holiday":     full.is_nse_holiday,
            "holiday_name":   full.holiday_name,
        }

    def is_tradeable(self, dt: datetime = None) -> bool:
        return self.is_nse_tradeable(dt)

    def format_status(self, dt: datetime = None) -> str:
        dt   = dt or self.now_ist()
        full = self.get_full_status(dt)
        parts = [f"[{dt.strftime('%H:%M:%S IST')}]"]

        if full.is_nse_holiday:
            parts.append(f"NSE CLOSED ({full.holiday_name})")
        elif full.nse_tradeable:
            parts.append(f"NSE OPEN | closes in {full.secs_to_nse_close//60}m")
        else:
            h, m = divmod(full.secs_to_nse_open // 60, 60)
            parts.append(f"NSE CLOSED | opens in {h}h {m}m")

        if full.mcx_tradeable:
            parts.append(f"MCX OPEN ({full.mcx_session})")
        elif full.is_nse_holiday and not full.is_mcx_full_close:
            h, m = divmod(full.secs_to_mcx_open // 60, 60)
            parts.append(f"MCX opens 5pm ({h}h {m}m)")
        else:
            parts.append("MCX CLOSED")

        return "  |  ".join(parts)


# Module-level singleton
market_hours = MarketHours()
