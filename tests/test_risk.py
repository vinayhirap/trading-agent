# trading-agent/tests/test_risk.py
"""
Run: python -m pytest tests/test_risk.py -v -s
"""
import pytest
from datetime import datetime, timezone
from src.risk.models import (
    OrderSide, OrderStatus, RejectionReason,
    PortfolioState, TradeOrder,
)
from src.risk.position_sizer import PositionSizer
from src.risk.stop_calculator import StopCalculator
from src.risk.engine import RiskEngine
from src.prediction.model import PredictionResult, Signal


def make_prediction(signal=Signal.BUY, confidence=0.65) -> PredictionResult:
    return PredictionResult(
        signal=signal, confidence=confidence,
        buy_prob=confidence if signal==Signal.BUY else 0.1,
        hold_prob=0.1,
        sell_prob=confidence if signal==Signal.SELL else 0.1,
        top_features={},
        passes_threshold=confidence >= 0.55,
    )


class TestPositionSizer:
    def setup_method(self):
        self.sizer = PositionSizer(capital=100_000, max_risk_per_trade=0.01)

    def test_basic_sizing(self):
        result = self.sizer.calculate(
            entry_price=500, stop_loss=490,
            side=OrderSide.BUY, cash_available=100_000,
        )
        # With 30% cap:
        # max position = ₹30,000 → 60 shares
        # actual risk = 60 × 10 = ₹600
        assert result["quantity"] == 60
        assert abs(result["risk_amount"] - 600) < 1
        print(f"\nSizing: {result}")

    def test_confidence_scales_risk(self):
        high_conf = self.sizer.calculate(500, 490, OrderSide.BUY, confidence=1.0)
        low_conf  = self.sizer.calculate(500, 490, OrderSide.BUY, confidence=0.5)
        assert high_conf["quantity"] >= low_conf["quantity"]

    def test_cash_cap(self):
        # Only INR5000 cash available → can't buy 100 shares at INR500
        result = self.sizer.calculate(500, 490, OrderSide.BUY, cash_available=5_000)
        assert result["position_value"] <= 5_000

    def test_invalid_stop(self):
        # Stop above entry for BUY — invalid
        result = self.sizer.calculate(500, 510, OrderSide.BUY)
        assert result["quantity"] == 0


class TestStopCalculator:
    def setup_method(self):
        self.calc = StopCalculator(atr_multiplier_stop=2.0, atr_multiplier_target=3.0)

    def test_buy_stops(self):
        result = self.calc.calculate(500, atr=10, side=OrderSide.BUY)
        assert result["stop_loss"]    == 480.0   # 500 - 2×10
        assert result["target_price"] == 530.0   # 500 + 3×10
        assert result["rr_ratio"]     == 1.5
        assert result["valid"]
        print(f"\nBUY stops: {result}")

    def test_sell_stops(self):
        result = self.calc.calculate(500, atr=10, side=OrderSide.SELL)
        assert result["stop_loss"]    == 520.0   # 500 + 2×10
        assert result["target_price"] == 470.0   # 500 - 3×10

    def test_trailing_stop_moves_up(self):
        stop = self.calc.update_trailing_stop(
            current_stop=480, current_price=520, atr=10, side=OrderSide.BUY
        )
        assert stop > 480, "Trailing stop must move up as price rises"
        assert stop == 500.0   # 520 - 2×10

    def test_trailing_stop_never_moves_down(self):
        # Price falls back after rising — stop must stay at high watermark
        stop = self.calc.update_trailing_stop(
            current_stop=495, current_price=490, atr=10, side=OrderSide.BUY
        )
        assert stop == 495, "Trailing stop must never move against the trade"


class TestRiskEngine:
    def setup_method(self):
        self.portfolio = PortfolioState(
            capital=100_000, cash_available=100_000
        )
        self.engine = RiskEngine(
            portfolio=self.portfolio,
            min_confidence=0.55,
        )

    def test_approved_trade(self):
        order = self.engine.evaluate(
            symbol="RELIANCE",
            prediction=make_prediction(Signal.BUY, 0.65),
            entry_price=1350, atr=18,
            check_market_hours=False,   # bypass for testing
        )
        assert order.status == OrderStatus.APPROVED
        assert order.quantity > 0
        assert order.stop_loss < 1350
        assert order.target_price > 1350
        assert order.reward_risk_ratio >= 1.5
        print(f"\nApproved order: {order}")

    def test_hold_signal_rejected(self):
        order = self.engine.evaluate(
            "RELIANCE", make_prediction(Signal.HOLD, 0.4),
            1350, 18, check_market_hours=False
        )
        assert order.status == OrderStatus.REJECTED

    def test_low_confidence_rejected(self):
        order = self.engine.evaluate(
            "RELIANCE", make_prediction(Signal.BUY, 0.45),
            1350, 18, check_market_hours=False
        )
        assert order.status == OrderStatus.REJECTED
        assert order.rejection_reason == RejectionReason.LOW_CONFIDENCE

    def test_max_positions_gate(self):
        # Fill portfolio to max
        for sym in ["RELIANCE", "TCS", "INFY"]:
            pred = make_prediction(Signal.BUY, 0.65)
            order = self.engine.evaluate(sym, pred, 1000, 15,
                                         check_market_hours=False)
            if order.status == OrderStatus.APPROVED:
                self.engine.record_fill(order)

        # 4th trade should be rejected
        order = self.engine.evaluate(
            "HDFCBANK", make_prediction(Signal.BUY, 0.70),
            1600, 20, check_market_hours=False
        )
        assert order.status == OrderStatus.REJECTED
        assert order.rejection_reason == RejectionReason.MAX_POSITIONS

    def test_daily_loss_cap(self):
        self.portfolio.daily_pnl = -2500   # -2.5% loss on INR1L capital
        order = self.engine.evaluate(
            "RELIANCE", make_prediction(Signal.BUY, 0.70),
            1350, 18, check_market_hours=False
        )
        assert order.status == OrderStatus.REJECTED
        assert order.rejection_reason == RejectionReason.DAILY_LOSS_CAP
        assert self.portfolio.trading_halted

    def test_trailing_stop_on_position(self):
        # Open a position
        pred  = make_prediction(Signal.BUY, 0.65)
        order = self.engine.evaluate("RELIANCE", pred, 1350, 18,
                                      check_market_hours=False)
        self.engine.record_fill(order)

        # Price rises — trailing stop moves up
        result = self.engine.update_position_price("RELIANCE", 1420, 18)
        pos = self.portfolio.positions.get("RELIANCE")
        assert pos is None or pos.stop_loss > order.stop_loss, \
            "Stop must have moved up after price advance"
        print(f"\nTrailing stop result: {result}")

    def test_stop_hit_closes_position(self):
        pred  = make_prediction(Signal.BUY, 0.65)
        order = self.engine.evaluate("RELIANCE", pred, 1350, 18,
                                      check_market_hours=False)
        self.engine.record_fill(order)

        # Price crashes below stop
        result = self.engine.update_position_price(
            "RELIANCE", order.stop_loss - 1, 18
        )
        assert result["action"] == "stop_hit"
        assert "RELIANCE" not in self.portfolio.positions
        print(f"\nStop hit: {result}")