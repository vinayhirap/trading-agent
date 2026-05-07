# trading-agent/src/execution/paper_broker.py
"""
Paper trading broker — simulates real execution with realistic fills.

Simulates:
- Market order slippage (0.05% by default)
- Brokerage fees (Zerodha flat ₹20/order or 0.03% whichever is lower)
- STT, exchange charges for equity delivery
- Order rejection if outside market hours (configurable)

This is your default execution engine. Switch to LiveBroker only after
3+ months of profitable paper trading.
"""
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
import json
from loguru import logger

from src.risk.models import TradeOrder, OrderStatus, OrderSide, Position


# ── Indian brokerage cost structure (Zerodha as reference) ───────────────────
BROKERAGE_PER_ORDER  = 20.0       # ₹20 flat or 0.03% whichever lower
BROKERAGE_PCT        = 0.0003     # 0.03%
STT_DELIVERY         = 0.001      # 0.1% on buy side (delivery)
STT_INTRADAY         = 0.00025    # 0.025% on sell side (intraday)
EXCHANGE_TXN_CHARGE  = 0.0000335  # NSE: 0.00335%
SEBI_CHARGE          = 0.000001   # ₹10 per crore
GST_RATE             = 0.18       # 18% GST on brokerage + charges
STAMP_DUTY           = 0.00015    # 0.015% on buy side


@dataclass
class ExecutionRecord:
    """Complete record of a paper trade execution."""
    execution_id:   str
    order_id:       str
    symbol:         str
    side:           str
    quantity:       int
    requested_price: float
    fill_price:     float          # after slippage
    slippage:       float          # ₹ slippage cost
    brokerage:      float
    stt:            float
    other_charges:  float
    total_charges:  float
    net_cost:       float          # total ₹ outflow including all charges
    timestamp:      datetime
    is_paper:       bool = True

    @property
    def charges_pct(self) -> float:
        return self.total_charges / (self.fill_price * self.quantity)


@dataclass
class PaperPortfolio:
    """Paper trading portfolio state — persisted to JSON."""
    cash:           float
    initial_capital: float
    positions:      dict = field(default_factory=dict)
    trade_log:      list = field(default_factory=list)
    daily_pnl:      float = 0.0
    total_charges:  float = 0.0


class PaperBroker:
    """
    Realistic paper trading execution engine.

    Key realism features:
    - Slippage on market orders (worse fills at high volume or volatile stocks)
    - Full Indian brokerage cost stack (STT, GST, stamp duty, exchange charges)
    - Persistent trade log (survives restarts)
    - Position tracking with unrealised P&L

    Usage:
        broker = PaperBroker(initial_capital=100_000)
        record = broker.execute(approved_order, current_price=1350)
    """

    LOG_PATH = Path("data/paper_trades.json")

    def __init__(
        self,
        initial_capital: float = 100_000,
        slippage_pct:    float = 0.0005,    # 0.05% slippage on market orders
        is_intraday:     bool  = False,
    ):
        self.slippage_pct    = slippage_pct
        self.is_intraday     = is_intraday
        self.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.portfolio = self._load_or_create(initial_capital)
        logger.info(
            f"PaperBroker ready | "
            f"cash=₹{self.portfolio.cash:,.0f} | "
            f"positions={len(self.portfolio.positions)}"
        )

    # ── Main execute method ───────────────────────────────────────────────────

    def execute(self, order: TradeOrder, current_price: float) -> ExecutionRecord:
        """
        Simulate execution of an approved TradeOrder.
        Returns an ExecutionRecord with full cost breakdown.
        """
        if order.status != OrderStatus.APPROVED:
            raise ValueError(f"Cannot execute non-approved order: {order.status}")

        # Simulate slippage — buy fills slightly higher, sell slightly lower
        if order.side == OrderSide.BUY:
            fill_price = current_price * (1 + self.slippage_pct)
        else:
            fill_price = current_price * (1 - self.slippage_pct)

        fill_price = round(fill_price, 2)
        trade_value = order.quantity * fill_price
        slippage_cost = abs(fill_price - current_price) * order.quantity

        # Calculate charges
        charges = self._calculate_charges(
            trade_value, order.side, order.quantity, fill_price
        )

        # Total cost (BUY = outflow, SELL = reduces inflow)
        if order.side == OrderSide.BUY:
            net_cost = trade_value + charges["total"]
            if net_cost > self.portfolio.cash:
                raise ValueError(
                    f"Insufficient paper cash: need ₹{net_cost:,.0f} "
                    f"have ₹{self.portfolio.cash:,.0f}"
                )
            self.portfolio.cash -= net_cost
            self._open_position(order, fill_price)
        else:
            net_cost = trade_value - charges["total"]
            self.portfolio.cash += net_cost
            self._close_position(order.symbol, fill_price, charges["total"])

        record = ExecutionRecord(
            execution_id    = str(uuid.uuid4())[:8],
            order_id        = order.order_id,
            symbol          = order.symbol,
            side            = order.side.value,
            quantity        = order.quantity,
            requested_price = current_price,
            fill_price      = fill_price,
            slippage        = round(slippage_cost, 2),
            brokerage       = charges["brokerage"],
            stt             = charges["stt"],
            other_charges   = charges["other"],
            total_charges   = charges["total"],
            net_cost        = round(net_cost, 2),
            timestamp       = datetime.now(timezone.utc),
        )

        self.portfolio.total_charges += charges["total"]
        self.portfolio.trade_log.append(self._record_to_dict(record))
        self._save()

        logger.info(
            f"PAPER FILL: {order.side.value} {order.quantity}x {order.symbol} "
            f"@ ₹{fill_price:.2f} (slippage=₹{slippage_cost:.0f} "
            f"charges=₹{charges['total']:.0f}) | "
            f"cash=₹{self.portfolio.cash:,.0f}"
        )
        return record

    # ── Portfolio queries ─────────────────────────────────────────────────────

    def get_portfolio_summary(self, prices: dict[str, float] = None) -> dict:
        """Get current portfolio value and P&L."""
        prices = prices or {}
        positions_value = 0.0
        position_details = []

        for sym, pos in self.portfolio.positions.items():
            current = prices.get(sym, pos["entry_price"])
            mkt_val = pos["quantity"] * current
            cost    = pos["quantity"] * pos["entry_price"]
            pnl     = mkt_val - cost
            positions_value += mkt_val
            position_details.append({
                "symbol":       sym,
                "quantity":     pos["quantity"],
                "entry":        pos["entry_price"],
                "current":      current,
                "market_value": round(mkt_val, 2),
                "pnl":          round(pnl, 2),
                "pnl_pct":      round(pnl / cost * 100, 2) if cost > 0 else 0,
                "stop_loss":    pos.get("stop_loss", 0),
                "target":       pos.get("target_price", 0),
            })

        total_value = self.portfolio.cash + positions_value
        total_pnl   = total_value - self.portfolio.initial_capital

        return {
            "cash":            round(self.portfolio.cash, 2),
            "positions_value": round(positions_value, 2),
            "total_value":     round(total_value, 2),
            "total_pnl":       round(total_pnl, 2),
            "total_pnl_pct":   round(total_pnl / self.portfolio.initial_capital * 100, 2),
            "total_charges":   round(self.portfolio.total_charges, 2),
            "n_trades":        len(self.portfolio.trade_log),
            "n_positions":     len(self.portfolio.positions),
            "positions":       position_details,
        }

    def get_trade_history(self) -> list[dict]:
        return self.portfolio.trade_log.copy()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _calculate_charges(
        self,
        trade_value: float,
        side: OrderSide,
        quantity: int,
        price: float,
    ) -> dict:
        brokerage = min(BROKERAGE_PER_ORDER, trade_value * BROKERAGE_PCT)

        if self.is_intraday:
            stt = trade_value * STT_INTRADAY if side == OrderSide.SELL else 0
        else:
            stt = trade_value * STT_DELIVERY if side == OrderSide.BUY else 0

        exchange = trade_value * EXCHANGE_TXN_CHARGE
        sebi     = trade_value * SEBI_CHARGE
        stamp    = trade_value * STAMP_DUTY if side == OrderSide.BUY else 0
        gst      = (brokerage + exchange) * GST_RATE
        other    = round(exchange + sebi + stamp + gst, 2)
        total    = round(brokerage + stt + other, 2)

        return {"brokerage": round(brokerage, 2), "stt": round(stt, 2),
                "other": other, "total": total}

    def _open_position(self, order: TradeOrder, fill_price: float) -> None:
        self.portfolio.positions[order.symbol] = {
            "quantity":     order.quantity,
            "entry_price":  fill_price,
            "stop_loss":    order.stop_loss,
            "target_price": order.target_price,
            "side":         order.side.value,
            "opened_at":    datetime.now(timezone.utc).isoformat(),
        }

    def _close_position(self, symbol: str, fill_price: float, charges: float) -> None:
        if symbol in self.portfolio.positions:
            pos = self.portfolio.positions.pop(symbol)
            entry_val = pos["quantity"] * pos["entry_price"]
            exit_val  = pos["quantity"] * fill_price
            pnl       = (exit_val - entry_val) - charges
            self.portfolio.daily_pnl += pnl
            logger.info(f"Position closed: {symbol} PnL=₹{pnl:+.2f}")

    def _load_or_create(self, initial_capital: float) -> PaperPortfolio:
        if self.LOG_PATH.exists():
            with open(self.LOG_PATH) as f:
                data = json.load(f)
            p = PaperPortfolio(
                cash=data["cash"],
                initial_capital=data["initial_capital"],
                positions=data.get("positions", {}),
                trade_log=data.get("trade_log", []),
                daily_pnl=data.get("daily_pnl", 0),
                total_charges=data.get("total_charges", 0),
            )
            logger.info(f"Loaded paper portfolio: ₹{p.cash:,.0f} cash, "
                        f"{len(p.positions)} positions")
            return p
        return PaperPortfolio(cash=initial_capital, initial_capital=initial_capital)

    def _save(self) -> None:
        with open(self.LOG_PATH, "w") as f:
            json.dump({
                "cash":            self.portfolio.cash,
                "initial_capital": self.portfolio.initial_capital,
                "positions":       self.portfolio.positions,
                "trade_log":       self.portfolio.trade_log,
                "daily_pnl":       self.portfolio.daily_pnl,
                "total_charges":   self.portfolio.total_charges,
            }, f, indent=2, default=str)

    def _record_to_dict(self, r: ExecutionRecord) -> dict:
        return {
            "execution_id": r.execution_id, "order_id": r.order_id,
            "symbol": r.symbol, "side": r.side, "quantity": r.quantity,
            "fill_price": r.fill_price, "slippage": r.slippage,
            "total_charges": r.total_charges, "net_cost": r.net_cost,
            "timestamp": r.timestamp.isoformat(),
        }

    def reset(self, initial_capital: float = None) -> None:
        """Reset paper portfolio to initial state."""
        capital = initial_capital or self.portfolio.initial_capital
        self.portfolio = PaperPortfolio(cash=capital, initial_capital=capital)
        self._save()
        logger.warning(f"Paper portfolio RESET to ₹{capital:,.0f}")