# trading-agent/src/brokers/order_monitor.py
"""
Order Monitor — background SL/target watcher.

Polls LTP every N seconds for all open positions.
When SL or target is hit, fires the exit order automatically.

Design:
  - Runs as a background thread (not blocking Streamlit)
  - Uses Angel One LTP for live mode, yfinance for paper mode
  - Fires exit via LiveBroker.execute() or PaperBroker.execute()
  - Sends alert via AlertManager when SL/target is triggered
  - Persists state — recovers after restarts

Usage (in app.py):
    from src.brokers.order_monitor import OrderMonitor
    monitor = OrderMonitor(broker=get_broker(), alert_manager=get_alert_manager())
    monitor.start()   # starts background thread
    # monitor.stop()  # call on app shutdown
"""
import threading
import time
from datetime import datetime, timezone
from loguru import logger


class OrderMonitor:
    """
    Background thread that monitors open positions for SL/target hits.

    Polling interval: 30s (matches dashboard refresh)
    In real-world use, switch to WebSocket for <1s latency.
    """

    def __init__(
        self,
        broker,                   # LiveBroker or PaperBroker instance
        alert_manager=None,       # AlertManager for notifications
        poll_interval: int = 30,  # seconds between checks
        is_live: bool = False,    # True → use Angel One LTP
    ):
        self._broker        = broker
        self._alerts        = alert_manager
        self._poll_interval = poll_interval
        self._is_live       = is_live
        self._running       = False
        self._thread: threading.Thread | None = None
        self._triggered: set = set()   # order_ids already exited this session

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info(f"OrderMonitor started (poll={self._poll_interval}s, live={self._is_live})")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("OrderMonitor stopped")

    def check_once(self) -> list[dict]:
        """
        Run one check cycle manually (useful for Streamlit — call on each refresh).
        Returns list of triggered exits.
        """
        return self._check_positions()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _loop(self):
        while self._running:
            try:
                triggered = self._check_positions()
                if triggered:
                    logger.info(f"OrderMonitor: {len(triggered)} exit(s) triggered")
            except Exception as e:
                logger.error(f"OrderMonitor loop error: {e}")
            time.sleep(self._poll_interval)

    def _check_positions(self) -> list[dict]:
        triggered = []
        positions  = self._broker.portfolio.positions

        for sym, pos in list(positions.items()):
            try:
                current_price = self._fetch_price(sym, pos)
                if current_price is None or current_price <= 0:
                    continue

                sl  = pos.get("stop_loss", 0)
                tgt = pos.get("target_price", 0)
                side = pos.get("side", "BUY")

                reason = None

                if side == "BUY" or side == "buy":
                    if sl > 0 and current_price <= sl:
                        reason = f"STOP LOSS HIT: {sym} @ ₹{current_price:,.2f} (SL=₹{sl:,.2f})"
                    elif tgt > 0 and current_price >= tgt:
                        reason = f"TARGET HIT: {sym} @ ₹{current_price:,.2f} (TGT=₹{tgt:,.2f})"
                else:  # SELL position
                    if sl > 0 and current_price >= sl:
                        reason = f"STOP LOSS HIT (short): {sym} @ ₹{current_price:,.2f} (SL=₹{sl:,.2f})"
                    elif tgt > 0 and current_price <= tgt:
                        reason = f"TARGET HIT (short): {sym} @ ₹{current_price:,.2f} (TGT=₹{tgt:,.2f})"

                if reason:
                    exit_result = self._exit_position(sym, pos, current_price, reason)
                    if exit_result:
                        # Calculate P&L
                        pnl = 0
                        if pos["side"] == "BUY":
                            pnl = (current_price - pos["entry_price"]) * pos["quantity"]
                        else:
                            pnl = (pos["entry_price"] - current_price) * pos["quantity"]
                        triggered.append({"symbol": sym, "reason": reason, "price": current_price})
                        self._notify(sym, reason, current_price, pnl)

            except Exception as e:
                logger.warning(f"OrderMonitor: error checking {sym}: {e}")

        return triggered

    def _fetch_price(self, sym: str, pos: dict) -> float | None:
        """Fetch latest price — price store (low latency), Angel One live, then yfinance."""
        from src.streaming.price_store import price_store

        px = price_store.get(sym, fallback=False)
        if px and px > 0:
            return px

        if self._is_live:
            try:
                adapter = getattr(self._broker, "_adapter", None)
                if adapter and adapter._connected:
                    return adapter.fetch_latest_price(sym)
            except Exception:
                pass

        # Final fallback: yfinance
        try:
            import yfinance as yf
            from src.data.models import ALL_SYMBOLS
            info   = ALL_SYMBOLS.get(sym)
            yf_sym = info.symbol if info else f"{sym}.NS"
            px     = yf.Ticker(yf_sym).fast_info.last_price
            return float(px) if px and px > 0 else None
        except Exception:
            return None

    def _exit_position(self, sym: str, pos: dict, price: float, reason: str) -> bool:
        """Fire an exit order for the position."""
        try:
            from src.risk.models import TradeOrder, OrderSide, OrderStatus
            import uuid as _uuid

            order = TradeOrder(
                order_id     = str(_uuid.uuid4())[:8],
                symbol       = sym,
                side         = OrderSide.SELL,
                quantity     = pos["quantity"],
                order_type   = "MARKET",
                status       = OrderStatus.APPROVED,
                stop_loss    = 0,
                target_price = 0,
                signal       = "AUTO_EXIT",
                confidence   = 1.0,
            )
            self._broker.execute(order, current_price=price)
            logger.warning(f"AUTO EXIT executed: {reason}")
            return True
        except Exception as e:
            logger.error(f"Auto exit failed for {sym}: {e}")
            return False

    def _notify(self, sym: str, reason: str, price: float, pnl: float = 0):
        """Send alert notification."""
        if not self._alerts:
            return
        try:
            # Email via AlertFire (existing)
            from src.alerts.alert_manager import AlertFire
            from datetime import timezone
            fire = AlertFire(
                alert_id   = "auto_exit",
                symbol     = sym,
                alert_type = "AUTO_EXIT",
                label      = "Auto Exit",
                message    = reason,
                value      = price,
                threshold  = 0,
                fired_at   = datetime.now(timezone.utc).isoformat(),
            )
            email = getattr(self._alerts, "_email", None)
            if email:
                email.send_alert(fire)

            # Telegram — direct method with P&L context
            tg = getattr(self._alerts, "_telegram", None)
            if tg and tg.is_configured:
                tg.send_auto_exit(
                    symbol  = sym,
                    reason  = reason,
                    price   = price,
                    pnl     = pnl,
                )
        except Exception as e:
            logger.warning(f"Monitor notify failed: {e}")