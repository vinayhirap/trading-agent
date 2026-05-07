# trading-agent/tests/test_execution.py
"""
Run: python -m pytest tests/test_execution.py -v -s
"""
import pytest
from pathlib import Path
from datetime import datetime, timezone
from src.execution.paper_broker import PaperBroker
from src.risk.models import TradeOrder, OrderStatus, OrderSide


def make_approved_order(symbol="RELIANCE", price=1350.0, qty=10) -> TradeOrder:
    import uuid
    order = TradeOrder(
        order_id    = str(uuid.uuid4())[:8],
        symbol      = symbol,
        created_at  = datetime.now(timezone.utc),
        side        = OrderSide.BUY,
        quantity    = qty,
        entry_price = price,
        stop_loss   = price - 36,
        target_price= price + 54,
        risk_amount = 360.0,
        risk_pct    = 0.01,
        confidence  = 0.65,
        status      = OrderStatus.APPROVED,
    )
    return order


class TestPaperBroker:
    def setup_method(self, tmp_path=None):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp())
        PaperBroker.LOG_PATH = self.tmp / "paper_trades.json"
        self.broker = PaperBroker(initial_capital=100_000, slippage_pct=0.0005)

    def test_initial_state(self):
        summary = self.broker.get_portfolio_summary()
        assert summary["cash"] == 100_000
        assert summary["n_positions"] == 0
        assert summary["total_pnl"] == 0

    def test_execute_buy(self):
        order  = make_approved_order(price=1350, qty=10)
        record = self.broker.execute(order, current_price=1350)
        assert record.fill_price > 1350, "Buy should fill above requested (slippage)"
        assert record.total_charges > 0
        summary = self.broker.get_portfolio_summary()
        assert summary["n_positions"] == 1
        assert summary["cash"] < 100_000
        print(f"\nBuy fill: INR{record.fill_price:.2f} | "
              f"charges=INR{record.total_charges:.2f} | "
              f"slippage=INR{record.slippage:.2f}")

    def test_charges_are_realistic(self):
        order  = make_approved_order(price=1350, qty=10)
        record = self.broker.execute(order, current_price=1350)
        trade_value = 10 * record.fill_price
        charges_pct = record.total_charges / trade_value
        # Zerodha-equivalent charges should be 0.1–0.5% of trade value
        assert 0.0001 < charges_pct < 0.01, \
            f"Charges {charges_pct:.3%} outside realistic range"
        print(f"\nCharges: INR{record.total_charges:.2f} = {charges_pct:.3%} of trade value")

    def test_portfolio_summary_with_prices(self):
        order = make_approved_order(price=1350, qty=10)
        self.broker.execute(order, current_price=1350)
        # Price moves up — unrealised P&L should be positive
        summary = self.broker.get_portfolio_summary(prices={"RELIANCE": 1400})
        pos = next(p for p in summary["positions"] if p["symbol"] == "RELIANCE")
        assert pos["pnl"] > 0, "Rising price should show positive unrealised P&L"
        print(f"\nUnrealised P&L at INR1400: INR{pos['pnl']:.2f}")

    def test_persistence(self):
        order = make_approved_order(price=1350, qty=10)
        self.broker.execute(order, current_price=1350)
        # Create new broker instance — should load saved state
        PaperBroker.LOG_PATH = self.tmp / "paper_trades.json"
        broker2 = PaperBroker(initial_capital=100_000)
        summary = broker2.get_portfolio_summary()
        assert summary["n_positions"] == 1, "Position should persist across restarts"
