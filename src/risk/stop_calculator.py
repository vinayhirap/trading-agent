# trading-agent/src/risk/stop_calculator.py
"""
ATR-based stop-loss and target price calculator.

Why ATR-based stops?
- Fixed % stops (e.g. "always 2% below entry") ignore volatility.
  A 2% stop on Reliance is too tight. A 2% stop on a penny stock is too wide.
- ATR (Average True Range) measures how much a stock typically moves per day.
  Setting the stop at 2×ATR gives the trade room to breathe without
  excessive risk of getting shaken out by normal volatility.
"""
from src.risk.models import OrderSide
from loguru import logger


class StopCalculator:

    def __init__(
        self,
        atr_multiplier_stop:   float = 2.0,   # stop at 2×ATR from entry
        atr_multiplier_target: float = 3.0,   # target at 3×ATR (R:R = 1.5)
        min_rr_ratio:          float = 1.5,   # reject trade if R:R < 1.5
    ):
        self.atr_mult_stop   = atr_multiplier_stop
        self.atr_mult_target = atr_multiplier_target
        self.min_rr_ratio    = min_rr_ratio

    def calculate(
        self,
        entry_price: float,
        atr:         float,
        side:        OrderSide,
    ) -> dict:
        """
        Calculate stop-loss and target price from ATR.

        Returns
        -------
        dict: stop_loss, target, stop_distance, target_distance, rr_ratio, valid
        """
        if atr <= 0:
            logger.error("ATR must be positive")
            return {"valid": False, "reason": "invalid ATR"}

        stop_distance   = self.atr_mult_stop   * atr
        target_distance = self.atr_mult_target * atr

        if side == OrderSide.BUY:
            stop_loss    = entry_price - stop_distance
            target_price = entry_price + target_distance
        else:
            stop_loss    = entry_price + stop_distance
            target_price = entry_price - target_distance

        rr_ratio = target_distance / stop_distance

        # Validate minimum R:R
        valid = rr_ratio >= self.min_rr_ratio
        if not valid:
            logger.warning(f"R:R={rr_ratio:.2f} below minimum {self.min_rr_ratio}")

        result = {
            "stop_loss":       round(stop_loss, 2),
            "target_price":    round(target_price, 2),
            "stop_distance":   round(stop_distance, 2),
            "target_distance": round(target_distance, 2),
            "rr_ratio":        round(rr_ratio, 2),
            "atr_used":        round(atr, 2),
            "valid":           valid,
        }

        logger.debug(
            f"Stops [{side.value}]: entry=INR{entry_price:.2f} "
            f"SL=INR{stop_loss:.2f} TGT=INR{target_price:.2f} "
            f"R:R={rr_ratio:.1f}"
        )
        return result

    def update_trailing_stop(
        self,
        current_stop: float,
        current_price: float,
        atr: float,
        side: OrderSide,
    ) -> float:
        """
        Move stop-loss in favour of the trade as price advances.
        Only ever moves the stop in the profitable direction — never widens it.
        """
        new_stop = current_price - self.atr_mult_stop * atr \
            if side == OrderSide.BUY \
            else current_price + self.atr_mult_stop * atr

        if side == OrderSide.BUY:
            # Only move stop up, never down
            return max(current_stop, round(new_stop, 2))
        else:
            # Only move stop down, never up
            return min(current_stop, round(new_stop, 2))