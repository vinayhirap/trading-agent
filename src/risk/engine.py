# trading-agent/src/risk/engine.py
"""
The Risk Engine is the final authority before any order goes to execution.
Every trade must pass through this gate. No exceptions.
"""
import uuid
from datetime import datetime, timezone
from loguru import logger

from src.risk.models import (
    TradeOrder, PortfolioState, Position,
    OrderSide, OrderType, OrderStatus, RejectionReason,
)
from src.risk.position_sizer import PositionSizer
from src.risk.stop_calculator import StopCalculator
from src.prediction.model import PredictionResult, Signal
from src.utils.market_hours import market_hours
from config.settings import settings


class RiskEngine:
    """
    Single entry point for all trade decisions.

    Sequence:
    1. Gate checks (market hours, daily loss, position count, confidence)
    2. Stop-loss and target calculation
    3. Position sizing
    4. Order construction
    5. Return approved or rejected TradeOrder
    """

    def __init__(
        self,
        portfolio: PortfolioState = None,
        min_confidence:    float = 0.55,
        atr_mult_stop:     float = 2.0,
        atr_mult_target:   float = 3.0,
    ):
        self.portfolio = portfolio or PortfolioState(
            capital=settings.INITIAL_CAPITAL,
            cash_available=settings.INITIAL_CAPITAL,
        )
        self.min_confidence = min_confidence
        self.sizer     = PositionSizer(capital=self.portfolio.capital)
        self.stop_calc = StopCalculator(atr_mult_stop, atr_mult_target)
        self.max_positions  = settings.MAX_OPEN_POSITIONS
        self.max_daily_loss = settings.MAX_DAILY_LOSS_PCT

    # ── Main entry point ─────────────────────────────────────────────────────

    def evaluate(
        self,
        symbol:      str,
        prediction:  PredictionResult,
        entry_price: float,
        atr:         float,
        check_market_hours: bool = True,
    ) -> TradeOrder:
        """
        Evaluate a prediction signal and return a TradeOrder.
        The order will be APPROVED or REJECTED — caller checks order.status.
        """
        order = TradeOrder(
            order_id   = str(uuid.uuid4())[:8],
            symbol     = symbol,
            created_at = datetime.now(timezone.utc),
            side       = OrderSide.BUY if prediction.signal == Signal.BUY else OrderSide.SELL,
            quantity   = 0,
            entry_price= entry_price,
            confidence = prediction.confidence,
        )

        # HOLD signal → no trade
        if prediction.signal == Signal.HOLD:
            return self._reject(order, RejectionReason.LOW_CONFIDENCE,
                                "Signal is HOLD — no trade")

        # ── Gate 1: Market hours ─────────────────────────────────────────────
        if check_market_hours and not market_hours.is_tradeable():
            return self._reject(order, RejectionReason.MARKET_CLOSED,
                                f"NSE not in OPEN session: {market_hours.format_status()}")

        # ── Gate 2: Daily loss cap ────────────────────────────────────────────
        if self.portfolio.trading_halted:
            return self._reject(order, RejectionReason.DAILY_LOSS_CAP,
                                f"Trading halted: {self.portfolio.halt_reason}")

        if self.portfolio.daily_pnl_pct <= -self.max_daily_loss:
            self.portfolio.trading_halted = True
            self.portfolio.halt_reason = (
                f"Daily loss cap hit: {self.portfolio.daily_pnl_pct:.2%} "
                f"(limit={self.max_daily_loss:.2%})"
            )
            logger.critical(f"TRADING HALTED — {self.portfolio.halt_reason}")
            return self._reject(order, RejectionReason.DAILY_LOSS_CAP,
                                self.portfolio.halt_reason)

        # ── Gate 3: Max open positions ────────────────────────────────────────
        if self.portfolio.open_position_count >= self.max_positions:
            return self._reject(order, RejectionReason.MAX_POSITIONS,
                                f"Already at max positions ({self.max_positions})")

        # ── Gate 4: Already in this symbol ───────────────────────────────────
        if symbol in self.portfolio.positions:
            return self._reject(order, RejectionReason.POSITION_EXISTS,
                                f"Position in {symbol} already open")

        # ── Gate 5: Model confidence ──────────────────────────────────────────
        if prediction.confidence < self.min_confidence:
            return self._reject(order, RejectionReason.LOW_CONFIDENCE,
                                f"Confidence {prediction.confidence:.1%} < "
                                f"threshold {self.min_confidence:.1%}")

        # ── Calculate stops ───────────────────────────────────────────────────
        stops = self.stop_calc.calculate(entry_price, atr, order.side)
        if not stops["valid"]:
            return self._reject(order, RejectionReason.RISK_TOO_HIGH,
                                f"Invalid R:R ratio: {stops.get('rr_ratio', 0):.2f}")

        order.stop_loss    = stops["stop_loss"]
        order.target_price = stops["target_price"]

        # ── Calculate position size ───────────────────────────────────────────
        sizing = self.sizer.calculate(
            entry_price    = entry_price,
            stop_loss      = order.stop_loss,
            side           = order.side,
            cash_available = self.portfolio.cash_available,
            confidence     = prediction.confidence,
        )

        if sizing["quantity"] == 0:
            return self._reject(order, RejectionReason.ZERO_QUANTITY,
                                sizing.get("reason", "quantity = 0"))

        order.quantity    = sizing["quantity"]
        order.risk_amount = sizing["risk_amount"]
        order.risk_pct    = sizing["risk_pct"]
        order.status      = OrderStatus.APPROVED

        logger.info(f"Order APPROVED: {order}")
        return order

    # ── Portfolio update methods ──────────────────────────────────────────────

    def record_fill(self, order: TradeOrder) -> None:
        """Call this when an order is confirmed filled by the broker."""
        position = Position(
            symbol       = order.symbol,
            side         = order.side,
            quantity     = order.quantity,
            entry_price  = order.entry_price,
            entry_time   = order.created_at,
            stop_loss    = order.stop_loss,
            target_price = order.target_price,
            current_price= order.entry_price,
            trailing_stop_price = order.stop_loss,
        )
        self.portfolio.positions[order.symbol] = position
        self.portfolio.cash_available -= order.quantity * order.entry_price
        self.portfolio.trades_today   += 1
        logger.info(f"Position opened: {order.symbol} | "
                    f"cash remaining=INR{self.portfolio.cash_available:,.0f}")

    def update_position_price(self, symbol: str, current_price: float, atr: float) -> dict:
        """
        Update a position with current market price.
        Checks stops and updates trailing stop.
        Returns action: 'hold', 'stop_hit', 'target_hit'.
        """
        if symbol not in self.portfolio.positions:
            return {"action": "no_position"}

        pos = self.portfolio.positions[symbol]
        pos.current_price = current_price

        # Update trailing stop
        new_trail = self.stop_calc.update_trailing_stop(
            pos.trailing_stop_price, current_price, atr, pos.side
        )
        if new_trail != pos.trailing_stop_price:
            logger.debug(f"Trailing stop updated {symbol}: "
                         f"INR{pos.trailing_stop_price:.2f} → INR{new_trail:.2f}")
            pos.trailing_stop_price = new_trail
            pos.stop_loss = new_trail    # trailing stop IS the new stop

        # Check exits
        if pos.is_stop_hit:
            pnl = pos.unrealised_pnl
            self._close_position(symbol, current_price, pnl)
            return {"action": "stop_hit", "pnl": pnl, "price": current_price}

        if pos.is_target_hit:
            pnl = pos.unrealised_pnl
            self._close_position(symbol, current_price, pnl)
            return {"action": "target_hit", "pnl": pnl, "price": current_price}

        return {"action": "hold", "unrealised_pnl": pos.unrealised_pnl}

    def reset_daily_state(self) -> None:
        """Call at start of each trading day."""
        self.portfolio.daily_pnl    = 0.0
        self.portfolio.trades_today = 0
        self.portfolio.trading_halted = False
        self.portfolio.halt_reason  = ""
        logger.info("Daily state reset — ready to trade")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _reject(self, order: TradeOrder, reason: RejectionReason, msg: str) -> TradeOrder:
        order.status          = OrderStatus.REJECTED
        order.rejection_reason = reason
        logger.warning(f"Order REJECTED [{reason.value}]: {msg}")
        return order

    def _close_position(self, symbol: str, exit_price: float, pnl: float) -> None:
        if symbol in self.portfolio.positions:
            pos = self.portfolio.positions.pop(symbol)
            proceeds = pos.quantity * exit_price
            self.portfolio.cash_available += proceeds
            self.portfolio.daily_pnl      += pnl
            self.portfolio.total_pnl      += pnl
            logger.info(f"Position closed: {symbol} @ INR{exit_price:.2f} | "
                        f"PnL=INR{pnl:+.2f} | daily=INR{self.portfolio.daily_pnl:+.2f}")