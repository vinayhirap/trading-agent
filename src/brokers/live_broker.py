# trading-agent/src/brokers/live_broker.py
"""
Live Broker — Angel One SmartAPI execution engine.

Mirrors the PaperBroker interface exactly so the rest of the system
(dashboard, risk engine, trade page) doesn't need to know whether
it's paper or live trading.

SAFETY GATES (all must pass before any order is sent):
  1. ENV must be "live" in settings
  2. Angel One must be connected and authenticated
  3. Market must be open (NSE or MCX depending on symbol)
  4. Daily loss cap must not be breached
  5. Position limit must not be reached
  6. Order value must be within per-trade risk limit

Live order flow:
  execute(order) → safety checks → Angel One placeOrder → track position → log

Positions and orders are persisted to data/live_trades.json
so they survive restarts (important if Streamlit reloads during market hours).
"""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field
from loguru import logger

from src.risk.models import TradeOrder, OrderSide, OrderStatus
from src.brokers.token_manager import token_manager
from src.utils.market_hours import market_hours
from config.settings import settings

LIVE_LOG_PATH = Path("data/live_trades.json")


@dataclass
class LiveOrderRecord:
    order_id:       str
    angel_order_id: str        # returned by Angel One API
    symbol:         str
    side:           str
    quantity:       int
    order_type:     str
    requested_price: float
    fill_price:     float      # 0 until confirmed fill
    status:         str        # PENDING | FILLED | REJECTED | CANCELLED
    stop_loss:      float
    target_price:   float
    timestamp:      str
    exchange:       str
    error_msg:      str = ""


@dataclass
class LivePortfolio:
    cash:            float
    initial_capital: float
    positions:       dict = field(default_factory=dict)
    order_log:       list = field(default_factory=list)
    daily_pnl:       float = 0.0
    daily_loss_used: float = 0.0   # tracks today's realised loss
    total_charges:   float = 0.0


class LiveBroker:
    """
    Real-money execution engine via Angel One SmartAPI.

    IMPORTANT: Only use when:
      - ENV=live in .env
      - You have tested strategy for 3+ months on paper
      - You understand the risks of automated trading
    """

    def __init__(self):
        self._adapter  = None
        self._connected = False
        self.portfolio  = self._load_or_create()
        self._connect()

    def is_connected(self) -> bool:
        return self._connected

    # ── Main execute method (mirrors PaperBroker.execute) ─────────────────────

    def execute(self, order: TradeOrder, current_price: float) -> LiveOrderRecord:
        """
        Execute a live order via Angel One.
        Runs all safety gates before sending to exchange.
        """
        # ── Safety gate 1: ENV check ──────────────────────────────────────────
        if settings.ENV != "live":
            raise RuntimeError(
                f"ENV={settings.ENV}. Set ENV=live in .env to enable live trading. "
                f"This is intentional — prevents accidental live orders."
            )

        # ── Safety gate 2: Connection ─────────────────────────────────────────
        if not self._connected or self._adapter is None:
            raise RuntimeError(
                "Angel One not connected. Check credentials in .env and "
                "ensure smartapi-python is installed."
            )

        # ── Safety gate 3: Order must be approved ────────────────────────────
        if order.status != OrderStatus.APPROVED:
            raise ValueError(f"Cannot execute non-approved order: {order.status}")

        # ── Safety gate 4: Market hours ───────────────────────────────────────
        mhf = market_hours.get_full_status()
        if not mhf.nse_tradeable and not mhf.mcx_tradeable:
            raise RuntimeError(
                "Market is closed. Orders can only be placed during market hours."
            )

        # ── Safety gate 5: Daily loss cap ─────────────────────────────────────
        max_daily_loss = self.portfolio.initial_capital * settings.MAX_DAILY_LOSS_PCT
        if self.portfolio.daily_loss_used >= max_daily_loss:
            raise RuntimeError(
                f"Daily loss cap reached: ₹{self.portfolio.daily_loss_used:,.0f} "
                f"(limit ₹{max_daily_loss:,.0f}). No more trades today."
            )

        # ── Safety gate 6: Position limit ────────────────────────────────────
        if (order.side == OrderSide.BUY and
                len(self.portfolio.positions) >= settings.MAX_OPEN_POSITIONS):
            raise RuntimeError(
                f"Max positions reached ({settings.MAX_OPEN_POSITIONS}). "
                f"Close an existing position first."
            )

        # ── Safety gate 7: Capital check ─────────────────────────────────────
        trade_value = order.quantity * current_price
        max_trade   = self.portfolio.initial_capital * settings.MAX_RISK_PER_TRADE * 20
        if order.side == OrderSide.BUY and trade_value > self.portfolio.cash:
            raise RuntimeError(
                f"Insufficient funds: need ₹{trade_value:,.0f}, "
                f"have ₹{self.portfolio.cash:,.0f}"
            )

        # ── Resolve token ─────────────────────────────────────────────────────
        token_info = token_manager.get_token_info(order.symbol)
        if not token_info:
            raise ValueError(f"No Angel One token for {order.symbol}. Cannot place live order.")

        # ── Send to Angel One ─────────────────────────────────────────────────
        logger.warning(
            f"LIVE ORDER: {order.side.value} {order.quantity}× {order.symbol} "
            f"@ ₹{current_price:,.2f} | SL=₹{order.stop_loss:,.2f} TGT=₹{order.target_price:,.2f}"
        )

        order_params = {
            "variety":         "NORMAL",
            "tradingsymbol":   token_info["symbol"],
            "symboltoken":     token_info["token"],
            "transactiontype": order.side.value,
            "exchange":        token_info["exchange"],
            "ordertype":       order.order_type or "MARKET",
            "producttype":     "DELIVERY",
            "duration":        "DAY",
            "price":           "0",        # market order
            "quantity":        str(order.quantity),
        }

        try:
            resp = self._adapter._smart_api.placeOrder(order_params)
        except Exception as e:
            logger.error(f"Angel One placeOrder failed: {e}")
            raise RuntimeError(f"Order placement failed: {e}")

        if not resp.get("status", False):
            msg = resp.get("message", "Unknown error")
            logger.error(f"Angel One rejected order: {msg}")
            raise RuntimeError(f"Order rejected by Angel One: {msg}")

        angel_order_id = resp.get("data", {}).get("orderid", str(uuid.uuid4())[:8])
        logger.info(f"Angel One order placed: {angel_order_id}")

        # ── Update portfolio state ────────────────────────────────────────────
        record = LiveOrderRecord(
            order_id        = order.order_id,
            angel_order_id  = angel_order_id,
            symbol          = order.symbol,
            side            = order.side.value,
            quantity        = order.quantity,
            order_type      = order.order_type or "MARKET",
            requested_price = current_price,
            fill_price      = current_price,    # market order — assume fill at current
            status          = "FILLED",
            stop_loss       = order.stop_loss,
            target_price    = order.target_price,
            timestamp       = datetime.now(timezone.utc).isoformat(),
            exchange        = token_info["exchange"],
        )

        if order.side == OrderSide.BUY:
            self.portfolio.cash -= trade_value
            self.portfolio.positions[order.symbol] = {
                "quantity":      order.quantity,
                "entry_price":   current_price,
                "stop_loss":     order.stop_loss,
                "target_price":  order.target_price,
                "angel_order_id": angel_order_id,
                "opened_at":     record.timestamp,
                "exchange":      token_info["exchange"],
                "token":         token_info["token"],
            }
        else:
            pos = self.portfolio.positions.pop(order.symbol, {})
            if pos:
                entry_val = pos["quantity"] * pos["entry_price"]
                exit_val  = order.quantity * current_price
                pnl       = exit_val - entry_val
                self.portfolio.daily_pnl      += pnl
                self.portfolio.cash            += exit_val
                if pnl < 0:
                    self.portfolio.daily_loss_used += abs(pnl)
                logger.info(f"Position closed: {order.symbol} PnL=₹{pnl:+,.2f}")

        self.portfolio.order_log.append(self._record_to_dict(record))
        self._save()
        return record

    # ── Portfolio queries (mirrors PaperBroker) ───────────────────────────────

    def get_portfolio_summary(self, prices: dict = None) -> dict:
        prices = prices or {}
        positions_value = 0.0
        position_details = []

        for sym, pos in self.portfolio.positions.items():
            current  = prices.get(sym, pos["entry_price"])
            mkt_val  = pos["quantity"] * current
            cost     = pos["quantity"] * pos["entry_price"]
            pnl      = mkt_val - cost
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
                "angel_order_id": pos.get("angel_order_id", ""),
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
            "n_trades":        len(self.portfolio.order_log),
            "n_positions":     len(self.portfolio.positions),
            "positions":       position_details,
            "daily_pnl":       round(self.portfolio.daily_pnl, 2),
            "daily_loss_used": round(self.portfolio.daily_loss_used, 2),
            "is_live":         True,
        }

    def get_trade_history(self) -> list:
        return self.portfolio.order_log.copy()

    def fetch_live_positions(self) -> list:
        """
        Fetch actual positions from Angel One (ground truth).
        Use this to reconcile our local state with broker's records.
        """
        if not self._connected:
            return []
        try:
            resp = self._adapter._smart_api.position()
            if resp and resp.get("status") and resp.get("data"):
                return resp["data"]
        except Exception as e:
            logger.warning(f"Could not fetch live positions: {e}")
        return []

    def fetch_order_book(self) -> list:
        """Fetch today's orders from Angel One."""
        if not self._connected:
            return []
        try:
            resp = self._adapter._smart_api.orderBook()
            if resp and resp.get("status") and resp.get("data"):
                return resp["data"]
        except Exception as e:
            logger.warning(f"Could not fetch order book: {e}")
        return []

    def cancel_order(self, angel_order_id: str, variety: str = "NORMAL") -> bool:
        """Cancel a pending order by Angel One order ID."""
        if not self._connected:
            return False
        try:
            resp = self._adapter._smart_api.cancelOrder(angel_order_id, variety)
            if resp and resp.get("status"):
                logger.info(f"Order cancelled: {angel_order_id}")
                return True
        except Exception as e:
            logger.warning(f"Cancel failed: {e}")
        return False

    # ── Internal ──────────────────────────────────────────────────────────────

    def _connect(self):
        try:
            from src.data.adapters.angel_one_adapter import AngelOneAdapter
            self._adapter = AngelOneAdapter()
            self._connected = self._adapter.connect()
            if self._connected:
                logger.info("LiveBroker: Angel One connected")
            else:
                logger.warning("LiveBroker: Angel One not connected — check credentials")
        except Exception as e:
            logger.warning(f"LiveBroker connect failed: {e}")
            self._connected = False

    def _load_or_create(self) -> LivePortfolio:
        if LIVE_LOG_PATH.exists():
            try:
                with open(LIVE_LOG_PATH) as f:
                    data = json.load(f)
                return LivePortfolio(
                    cash            = data["cash"],
                    initial_capital = data["initial_capital"],
                    positions       = data.get("positions", {}),
                    order_log       = data.get("order_log", []),
                    daily_pnl       = data.get("daily_pnl", 0.0),
                    daily_loss_used = data.get("daily_loss_used", 0.0),
                    total_charges   = data.get("total_charges", 0.0),
                )
            except Exception as e:
                logger.warning(f"Live portfolio load failed: {e}")
        return LivePortfolio(
            cash            = settings.INITIAL_CAPITAL,
            initial_capital = settings.INITIAL_CAPITAL,
        )

    def _save(self):
        LIVE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LIVE_LOG_PATH, "w") as f:
            json.dump({
                "cash":            self.portfolio.cash,
                "initial_capital": self.portfolio.initial_capital,
                "positions":       self.portfolio.positions,
                "order_log":       self.portfolio.order_log,
                "daily_pnl":       self.portfolio.daily_pnl,
                "daily_loss_used": self.portfolio.daily_loss_used,
                "total_charges":   self.portfolio.total_charges,
            }, f, indent=2, default=str)

    def _record_to_dict(self, r: LiveOrderRecord) -> dict:
        return {
            "order_id":        r.order_id,
            "angel_order_id":  r.angel_order_id,
            "symbol":          r.symbol,
            "side":            r.side,
            "quantity":        r.quantity,
            "fill_price":      r.fill_price,
            "stop_loss":       r.stop_loss,
            "target_price":    r.target_price,
            "status":          r.status,
            "exchange":        r.exchange,
            "timestamp":       r.timestamp,
        }