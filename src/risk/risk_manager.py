# trading-agent/src/risk/risk_manager.py
"""
Ideal Risk Manager

Enforces 4 hard rules on every trade decision:
  1. Daily loss limit  — stops all trading if day P&L < -2%
  2. Position concentration — max 2 open positions in same sector
  3. Minimum confidence gate — rejects signals below 55%
  4. Trailing stop logic — moves SL up as trade profits

Usage (in paper_broker or live_broker execute()):
    from src.risk.risk_manager import RiskManager
    rm = RiskManager(capital=10000)
    ok, reason = rm.approve(order, portfolio, current_prices)
    if not ok:
        return reject(reason)
"""
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional
import json
from pathlib import Path
from loguru import logger


# ── Sector map for concentration check ────────────────────────────────────────
SECTOR_MAP = {
    # Banking & Finance
    "HDFCBANK": "banking", "ICICIBANK": "banking", "SBIN": "banking",
    "AXISBANK": "banking", "KOTAKBANK": "banking", "BAJFINANCE": "banking",
    "BAJAJFINSV": "banking",
    # IT
    "TCS": "it", "INFY": "it", "WIPRO": "it", "TECHM": "it",
    "HCLTECH": "it", "NIFTYIT": "it",
    # Energy & Oil
    "RELIANCE": "energy", "ONGC": "energy", "CRUDEOIL": "energy",
    "NATURALGAS": "energy", "BPCL": "energy",
    # Metals
    "TATASTEEL": "metals", "JSWSTEEL": "metals", "HINDALCO": "metals",
    "COPPER": "metals", "ALUMINIUM": "metals",
    # Commodities (precious)
    "GOLD": "precious", "SILVER": "precious",
    # Crypto
    "BTC": "crypto", "ETH": "crypto", "SOL": "crypto",
    "BNB": "crypto", "XRP": "crypto",
    # Indices — no concentration limit (broad market)
    "NIFTY50": "index", "BANKNIFTY": "index", "NIFTYIT": "index",
    "SENSEX": "index", "NIFTYMID": "index",
    # Forex
    "USDINR": "forex", "EURUSD": "forex", "GBPUSD": "forex",
}
MAX_SECTOR_POSITIONS = 2   # max open positions in same sector


@dataclass
class RiskDecision:
    approved:    bool
    reason:      str
    adjusted_sl: Optional[float] = None   # trailing-adjusted stop loss
    adjusted_qty:Optional[int]   = None   # reduced quantity if needed


@dataclass
class DailyStats:
    date:        str   = ""
    starting_capital: float = 0.0
    realised_pnl:float = 0.0
    n_trades:    int   = 0
    n_winners:   int   = 0
    n_losers:    int   = 0
    trading_halted: bool = False


class RiskManager:
    """
    Stateful risk manager. Persists daily stats to disk so
    daily loss limits survive dashboard restarts.
    """

    # ── Hard limits ────────────────────────────────────────────────────────────
    MIN_CONFIDENCE      = 0.55   # reject anything below 55%
    DAILY_LOSS_LIMIT    = 0.02   # halt trading if day loss > 2% of capital
    MAX_SECTOR_POS      = MAX_SECTOR_POSITIONS
    TRAILING_TRIGGER    = 0.01   # start trailing after +1% profit
    TRAILING_STEP       = 0.005  # move SL up by 0.5% increments

    def __init__(
        self,
        capital:         float = 10_000,
        data_dir:        str   = "data",
        daily_loss_pct:  float = None,
        min_confidence:  float = None,
    ):
        self.capital          = capital
        self._data_path       = Path(data_dir) / "daily_risk_stats.json"
        self._data_path.parent.mkdir(exist_ok=True)

        if daily_loss_pct:
            self.DAILY_LOSS_LIMIT = daily_loss_pct
        if min_confidence:
            self.MIN_CONFIDENCE = min_confidence

        self._stats = self._load_stats()

    # ── Public API ─────────────────────────────────────────────────────────────

    def approve(
        self,
        symbol:         str,
        signal:         str,          # "BUY" or "SELL"
        confidence:     float,
        stop_loss:      float,
        entry_price:    float,
        quantity:       int,
        open_positions: dict,         # {symbol: position_dict}
        day_pnl:        float = 0.0,
    ) -> RiskDecision:
        """
        Main gate. Returns RiskDecision(approved=True/False, reason=...).
        Call this before every trade execution.
        """

        # ── Rule 1: Daily loss limit ───────────────────────────────────────────
        if self._stats.trading_halted:
            return RiskDecision(
                approved=False,
                reason=f"Trading halted: daily loss limit hit "
                       f"(P&L: ₹{self._stats.realised_pnl:+,.0f})"
            )

        total_day_loss = self._stats.realised_pnl + day_pnl
        limit_amount   = self.capital * self.DAILY_LOSS_LIMIT
        if total_day_loss < -limit_amount:
            self._stats.trading_halted = True
            self._save_stats()
            logger.warning(
                f"RiskManager: Daily loss limit hit! "
                f"P&L ₹{total_day_loss:+,.0f} < -₹{limit_amount:.0f}. "
                f"Trading halted for today."
            )
            return RiskDecision(
                approved=False,
                reason=f"Daily loss limit hit: ₹{total_day_loss:+,.0f} "
                       f"(limit -₹{limit_amount:.0f}). No more trades today."
            )

        # ── Rule 2: Minimum confidence ─────────────────────────────────────────
        if confidence < self.MIN_CONFIDENCE:
            return RiskDecision(
                approved=False,
                reason=f"Confidence {confidence:.0%} below minimum {self.MIN_CONFIDENCE:.0%}"
            )

        # ── Rule 3: Sector concentration ───────────────────────────────────────
        sector = SECTOR_MAP.get(symbol.upper(), "other")
        if sector not in ("index", "forex", "other"):
            same_sector = [
                sym for sym, pos in open_positions.items()
                if SECTOR_MAP.get(sym.upper(), "other") == sector
                and sym.upper() != symbol.upper()
            ]
            if len(same_sector) >= self.MAX_SECTOR_POS:
                return RiskDecision(
                    approved=False,
                    reason=f"Sector concentration limit: already "
                           f"{len(same_sector)} open in {sector} sector "
                           f"({', '.join(same_sector)})"
                )

        # ── Rule 4: Don't add to already losing position ───────────────────────
        if symbol in open_positions:
            existing = open_positions[symbol]
            existing_pnl_pct = (
                (entry_price - existing.get("entry_price", entry_price))
                / existing.get("entry_price", entry_price)
            )
            if existing_pnl_pct < -0.005:   # existing position down >0.5%
                return RiskDecision(
                    approved=False,
                    reason=f"Averaging down denied: {symbol} already "
                           f"down {existing_pnl_pct:.1%}"
                )

        # ── All rules passed ───────────────────────────────────────────────────
        return RiskDecision(
            approved=True,
            reason="All risk checks passed",
        )

    def compute_trailing_sl(
        self,
        entry_price:   float,
        current_price: float,
        current_sl:    float,
        side:          str = "BUY",
    ) -> float:
        """
        Compute updated trailing stop loss.
        Moves SL up (for BUY) in 0.5% steps once trade is +1% profitable.
        Never moves SL down.
        """
        if side == "BUY":
            profit_pct = (current_price - entry_price) / entry_price
            if profit_pct < self.TRAILING_TRIGGER:
                return current_sl   # not enough profit yet, keep existing SL

            # How many 0.5% steps have we moved?
            steps = int(profit_pct / self.TRAILING_STEP)
            new_sl = entry_price * (1 + (steps - 1) * self.TRAILING_STEP)
            new_sl = round(new_sl, 2)
            return max(new_sl, current_sl)   # never move SL down

        else:   # SELL / short
            profit_pct = (entry_price - current_price) / entry_price
            if profit_pct < self.TRAILING_TRIGGER:
                return current_sl
            steps  = int(profit_pct / self.TRAILING_STEP)
            new_sl = entry_price * (1 - (steps - 1) * self.TRAILING_STEP)
            new_sl = round(new_sl, 2)
            return min(new_sl, current_sl)   # never move SL up (for shorts)

    def record_trade_result(self, pnl: float, is_winner: bool):
        """Call after every trade close to update daily stats."""
        today = date.today().isoformat()
        if self._stats.date != today:
            self._reset_day()

        self._stats.realised_pnl += pnl
        self._stats.n_trades     += 1
        if is_winner:
            self._stats.n_winners += 1
        else:
            self._stats.n_losers  += 1
        self._save_stats()

    def get_daily_stats(self) -> DailyStats:
        today = date.today().isoformat()
        if self._stats.date != today:
            self._reset_day()
        return self._stats

    def reset_halt(self):
        """Manually override halt (e.g. for testing)."""
        self._stats.trading_halted = False
        self._save_stats()
        logger.info("RiskManager: trading halt manually cleared")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _reset_day(self):
        self._stats = DailyStats(
            date             = date.today().isoformat(),
            starting_capital = self.capital,
        )
        self._save_stats()
        logger.info("RiskManager: new trading day — stats reset")

    def _load_stats(self) -> DailyStats:
        try:
            if self._data_path.exists():
                d = json.loads(self._data_path.read_text())
                s = DailyStats(**d)
                # Auto-reset if it's a new day
                if s.date != date.today().isoformat():
                    return DailyStats(
                        date             = date.today().isoformat(),
                        starting_capital = self.capital,
                    )
                return s
        except Exception:
            pass
        return DailyStats(
            date             = date.today().isoformat(),
            starting_capital = self.capital,
        )

    def _save_stats(self):
        try:
            self._data_path.write_text(
                json.dumps(self._stats.__dict__, indent=2)
            )
        except Exception as e:
            logger.debug(f"RiskManager: could not save stats: {e}")