# trading-agent/src/risk/position_sizer.py
"""
Position sizing — the mathematical heart of risk management.

The core formula:
    quantity = risk_per_trade_INR / stop_distance_per_share

Example:
    Capital = INR1,00,000
    Risk per trade = 1% = INR1,000
    Entry price = INR500
    Stop loss = INR490  (stop distance = INR10 per share)
    → quantity = INR1,000 / INR10 = 100 shares
    → total position value = 100 × INR500 = INR50,000 (50% of capital)
    → actual risk = 100 × INR10 = INR1,000 ✓
"""
import math
from loguru import logger
from src.risk.models import OrderSide
from config.settings import settings


class PositionSizer:

    def __init__(
        self,
        capital: float = None,
        max_risk_per_trade: float = None,
        max_position_pct: float = 0.30,   # never put >30% in one stock
    ):
        self.capital            = capital or settings.INITIAL_CAPITAL
        self.max_risk_per_trade = max_risk_per_trade or settings.MAX_RISK_PER_TRADE
        self.max_position_pct   = max_position_pct

    def calculate(
        self,
        entry_price:  float,
        stop_loss:    float,
        side:         OrderSide,
        cash_available: float = None,
        confidence:   float = 1.0,
        lot_size:     int   = 1,
    ) -> dict:
        """
        Calculate position size using fixed-fractional risk.

        Parameters
        ----------
        entry_price    : expected fill price
        stop_loss      : hard stop-loss price
        side           : BUY or SELL
        cash_available : current available cash
        confidence     : model confidence (0-1) — scales risk down
        lot_size       : minimum tradeable unit (1 for equity, > 1 for F&O)

        Returns
        -------
        dict with quantity, risk_amount, risk_pct, stop_distance, position_value
        """
        cash = cash_available or self.capital

        # Stop distance (always positive)
        if side == OrderSide.BUY:
            stop_distance = entry_price - stop_loss
        else:
            stop_distance = stop_loss - entry_price

        if stop_distance <= 0:
            logger.error(f"Invalid stop distance: {stop_distance:.2f} — stop must be on the loss side")
            return self._zero_result("invalid stop distance")

        # Risk amount: scale by confidence (higher confidence → full risk)
        # Minimum confidence scaling: 0.5 → risk_pct × 0.5
        confidence_scale = max(0.5, min(1.0, confidence))
        risk_pct         = self.max_risk_per_trade * confidence_scale
        risk_amount      = self.capital * risk_pct

        # Raw quantity
        raw_qty = risk_amount / stop_distance

        # Round down to lot size
        quantity = max(0, int(math.floor(raw_qty / lot_size)) * lot_size)

        if quantity == 0:
            logger.warning(
                f"Position size = 0 | risk=INR{risk_amount:.0f} "
                f"stop_dist=INR{stop_distance:.2f} | increase capital or widen stop"
            )
            return self._zero_result("quantity rounds to zero")

        position_value = quantity * entry_price

        # Cap: never use more than max_position_pct of capital
        max_value = self.capital * self.max_position_pct
        if position_value > max_value:
            quantity      = max(0, int(math.floor(max_value / entry_price / lot_size)) * lot_size)
            position_value = quantity * entry_price
            logger.info(f"Position capped at {self.max_position_pct:.0%} of capital: {quantity} shares")

        # Cap: never exceed available cash
        if position_value > cash:
            quantity      = max(0, int(math.floor(cash / entry_price / lot_size)) * lot_size)
            position_value = quantity * entry_price
            logger.info(f"Position capped by available cash: {quantity} shares")

        actual_risk = quantity * stop_distance
        actual_risk_pct = actual_risk / self.capital

        logger.info(
            f"Position size: {quantity} shares @ INR{entry_price:.2f} | "
            f"value=INR{position_value:,.0f} | "
            f"risk=INR{actual_risk:.0f} ({actual_risk_pct:.2%}) | "
            f"SL=INR{stop_loss:.2f} (dist=INR{stop_distance:.2f})"
        )

        return {
            "quantity":       quantity,
            "risk_amount":    round(actual_risk, 2),
            "risk_pct":       round(actual_risk_pct, 4),
            "stop_distance":  round(stop_distance, 2),
            "position_value": round(position_value, 2),
            "risk_pct_input": round(risk_pct, 4),
        }

    def _zero_result(self, reason: str) -> dict:
        return {
            "quantity": 0, "risk_amount": 0, "risk_pct": 0,
            "stop_distance": 0, "position_value": 0,
            "risk_pct_input": 0, "reason": reason,
        }