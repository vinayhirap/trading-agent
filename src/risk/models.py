# trading-agent/src/risk/models.py
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class OrderSide(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT  = "LIMIT"
    SL     = "SL"        # stop-loss market
    SL_M   = "SL-M"     # stop-loss limit


class OrderStatus(str, Enum):
    PENDING   = "PENDING"
    APPROVED  = "APPROVED"
    REJECTED  = "REJECTED"
    PLACED    = "PLACED"
    FILLED    = "FILLED"
    CANCELLED = "CANCELLED"


class RejectionReason(str, Enum):
    MARKET_CLOSED      = "market_closed"
    DAILY_LOSS_CAP     = "daily_loss_cap_hit"
    MAX_POSITIONS      = "max_positions_reached"
    LOW_CONFIDENCE     = "model_confidence_too_low"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    POSITION_EXISTS    = "position_already_open"
    ZERO_QUANTITY      = "calculated_qty_is_zero"
    RISK_TOO_HIGH      = "risk_per_trade_exceeded"


@dataclass
class TradeOrder:
    """
    A fully specified trade order ready for execution.
    Every field is set before this reaches the execution layer — no ambiguity.
    """
    # Identity
    order_id:       str
    symbol:         str
    created_at:     datetime

    # Trade parameters
    side:           OrderSide
    quantity:       int              # number of shares/lots
    order_type:     OrderType = OrderType.MARKET
    limit_price:    Optional[float] = None

    # Risk parameters (must be set before approval)
    entry_price:    float = 0.0
    stop_loss:      float = 0.0      # hard stop — price where we exit if wrong
    target_price:   float = 0.0     # profit target
    trailing_stop:  bool  = True     # activate trailing stop after +1 ATR

    # Sizing metadata
    risk_amount:    float = 0.0     # ₹ at risk on this trade
    risk_pct:       float = 0.0     # % of capital at risk

    # State
    status:         OrderStatus = OrderStatus.PENDING
    rejection_reason: Optional[RejectionReason] = None
    confidence:     float = 0.0     # model confidence that triggered this

    # Computed R:R
    @property
    def reward_risk_ratio(self) -> float:
        if self.stop_loss == 0 or self.entry_price == 0:
            return 0.0
        risk   = abs(self.entry_price - self.stop_loss)
        reward = abs(self.target_price - self.entry_price)
        return round(reward / risk, 2) if risk > 0 else 0.0

    def __str__(self):
        return (
            f"[{self.status.value}] {self.side.value} {self.quantity}x {self.symbol} "
            f"@ ₹{self.entry_price:.2f} | SL=₹{self.stop_loss:.2f} "
            f"TGT=₹{self.target_price:.2f} | R:R={self.reward_risk_ratio} "
            f"| risk=₹{self.risk_amount:.0f} ({self.risk_pct:.1%})"
        )


@dataclass
class Position:
    """A currently open position in the portfolio."""
    symbol:       str
    side:         OrderSide
    quantity:     int
    entry_price:  float
    entry_time:   datetime
    stop_loss:    float
    target_price: float
    current_price: float = 0.0
    trailing_stop_price: float = 0.0

    @property
    def unrealised_pnl(self) -> float:
        if self.side == OrderSide.BUY:
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity

    @property
    def unrealised_pnl_pct(self) -> float:
        cost = self.entry_price * self.quantity
        return self.unrealised_pnl / cost if cost > 0 else 0.0

    @property
    def is_stop_hit(self) -> bool:
        if self.side == OrderSide.BUY:
            return self.current_price <= self.stop_loss
        return self.current_price >= self.stop_loss

    @property
    def is_target_hit(self) -> bool:
        if self.side == OrderSide.BUY:
            return self.current_price >= self.target_price
        return self.current_price <= self.target_price


@dataclass
class PortfolioState:
    """Snapshot of the portfolio at a point in time."""
    capital:          float
    cash_available:   float
    positions:        dict[str, Position] = field(default_factory=dict)
    daily_pnl:        float = 0.0
    total_pnl:        float = 0.0
    trades_today:     int   = 0
    trading_halted:   bool  = False
    halt_reason:      str   = ""

    @property
    def open_position_count(self) -> int:
        return len(self.positions)

    @property
    def daily_pnl_pct(self) -> float:
        return self.daily_pnl / self.capital if self.capital > 0 else 0.0