# trading-agent/src/brokers/angel_one_live.py
"""
Angel One SmartAPI — Full Live Trading Integration

Capabilities:
  1. WebSocket tick feed — real-time price streaming (sub-second)
  2. Order execution — market, limit, SL orders
  3. Position tracking — live P&L, SL monitoring
  4. Order book — fetch all orders for today
  5. Auto-exit — SL hit detection + trailing stop execution
  6. Token management — auto-login with TOTP refresh

Why Angel One:
  - ₹20 flat brokerage per order (not % based)
  - Free SmartAPI (no extra subscription)
  - WebSocket tick data free
  - Supports all NSE/BSE/MCX instruments

Credentials needed in .env:
  ANGEL_API_KEY=zkVKxcuW
  ANGEL_CLIENT_ID=AAAU372187
  ANGEL_PASSWORD=2003
  ANGEL_TOTP_SECRET=7Z26VC7NBWXJRQ32JIK66NEZ5I

NSE instrument tokens:
  Required for WebSocket subscription.
  Fetched from Angel One instrument master file (downloaded on startup).
  Stored in data/cache/angel_tokens.json

Usage:
    from src.brokers.angel_one_live import AngelOneLive
    broker = AngelOneLive()
    if broker.connect():
        # Place order
        order_id = broker.place_order("RELIANCE", "BUY", qty=1, price=0)  # 0 = market
        # Start tick feed
        broker.start_tick_feed(["RELIANCE", "NIFTY50"])
        # Get positions
        positions = broker.get_positions()

Integration with existing system:
    - Replaces PaperBroker when settings.ENV = "live"
    - Same interface as PaperBroker.execute()
    - Feeds prices to price_store for dashboard
    - Emits events to event_bus for signal pipeline
"""
from __future__ import annotations
import json
import time
import threading
import pyotp
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Callable
from loguru import logger

ROOT       = Path(__file__).resolve().parents[2]
TOKEN_CACHE = ROOT / "data" / "cache" / "angel_tokens.json"
TOKEN_CACHE.parent.mkdir(parents=True, exist_ok=True)
SESSION_CACHE = ROOT / "data" / "cache" / "angel_session.json"


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class LiveOrder:
    order_id:     str
    symbol:       str
    exchange:     str          # NSE, BSE, MCX
    side:         str          # BUY, SELL
    quantity:     int
    order_type:   str          # MARKET, LIMIT, SL
    price:        float        # 0 for MARKET
    trigger_price:float        # for SL orders
    status:       str          # OPEN, COMPLETE, CANCELLED, REJECTED
    fill_price:   float = 0.0
    fill_qty:     int   = 0
    placed_at:    str   = ""
    updated_at:   str   = ""
    angel_order_id: str = ""
    product:      str   = "INTRADAY"   # INTRADAY or DELIVERY


@dataclass
class LivePosition:
    symbol:       str
    exchange:     str
    quantity:     int
    entry_price:  float
    current_price:float
    side:         str          # BUY, SELL
    product:      str
    pnl:          float = 0.0
    pnl_pct:      float = 0.0
    stop_loss:    float = 0.0
    target:       float = 0.0
    angel_token:  str   = ""
    sl_order_id:  str   = ""   # SL order placed with broker

    def update_pnl(self, current_price: float):
        self.current_price = current_price
        if self.side == "BUY":
            self.pnl     = (current_price - self.entry_price) * self.quantity
            self.pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:
            self.pnl     = (self.entry_price - current_price) * self.quantity
            self.pnl_pct = (self.entry_price - current_price) / self.entry_price


# ── Angel One Live Broker ─────────────────────────────────────────────────────

class AngelOneLive:
    """
    Full Angel One SmartAPI integration.
    Drop-in replacement for PaperBroker when ENV=live.
    """

    EXCHANGE_MAP = {
        "NIFTY50":   ("NSE", "99926000"),
        "BANKNIFTY": ("NSE", "99926009"),
        "NIFTYIT":   ("NSE", "99926009"),
        "SENSEX":    ("BSE", "1"),
        "RELIANCE":  ("NSE", "2885"),
        "TCS":       ("NSE", "11536"),
        "HDFCBANK":  ("NSE", "1333"),
        "INFY":      ("NSE", "1594"),
        "ICICIBANK": ("NSE", "4963"),
        "SBIN":      ("NSE", "3045"),
        "WIPRO":     ("NSE", "3787"),
        "AXISBANK":  ("NSE", "5900"),
        "KOTAKBANK": ("NSE", "1922"),
        "LT":        ("NSE", "11483"),
        "BAJFINANCE":("NSE", "317"),
        "MARUTI":    ("NSE", "10999"),
        "TATASTEEL": ("NSE", "3505"),
        "GOLD":      ("MCX", "234257"),
        "SILVER":    ("MCX", "234265"),
        "CRUDEOIL":  ("MCX", "234219"),
        "BTC":       ("NSE", ""),     # crypto via CoinSwitch
        "USDINR":    ("NSE", "10"),
    }

    def __init__(self):
        self._api         = None
        self._connected   = False
        self._ws          = None
        self._ws_running  = False
        self._positions:  dict[str, LivePosition] = {}
        self._orders:     dict[str, LiveOrder]     = {}
        self._tick_callbacks: list[Callable] = []
        self._session_token  = None
        self._refresh_token  = None

        # Load settings
        try:
            from config.settings import settings
            self._api_key     = getattr(settings, "ANGEL_API_KEY",     "")
            self._client_id   = getattr(settings, "ANGEL_CLIENT_ID",   "")
            self._password    = getattr(settings, "ANGEL_PASSWORD",     "")
            self._totp_secret = getattr(settings, "ANGEL_TOTP_SECRET",  "")
        except Exception:
            self._api_key = self._client_id = self._password = self._totp_secret = ""

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """
        Login to Angel One SmartAPI.
        Returns True if connected successfully.
        """
        if not self._api_key or not self._client_id:
            logger.warning("Angel One credentials not configured")
            return False

        try:
            from smartapi import SmartConnect
            self._api = SmartConnect(api_key=self._api_key)

            # Generate TOTP
            totp = pyotp.TOTP(self._totp_secret).now()

            data = self._api.generateSession(
                clientCode  = self._client_id,
                password    = self._password,
                totp        = totp,
            )

            if data.get("status"):
                self._session_token = data["data"]["jwtToken"]
                self._refresh_token = data["data"]["refreshToken"]
                self._connected = True

                logger.info(f"Angel One connected: {self._client_id}")
                self._save_session()
                return True
            else:
                logger.error(f"Angel One login failed: {data.get('message','')}")
                return False

        except ImportError:
            logger.error("smartapi-python not installed: pip install smartapi-python")
            return False
        except Exception as e:
            logger.error(f"Angel One connect error: {e}")
            return False

    def is_connected(self) -> bool:
        return self._connected and self._api is not None

    def refresh_session(self) -> bool:
        """Refresh JWT token before it expires (every ~24h)."""
        try:
            if not self._refresh_token:
                return self.connect()
            data = self._api.generateToken(self._refresh_token)
            if data.get("status"):
                self._session_token = data["data"]["jwtToken"]
                self._save_session()
                logger.info("Angel One session refreshed")
                return True
        except Exception as e:
            logger.warning(f"Session refresh failed: {e} — reconnecting")
        return self.connect()

    # ── Order execution ───────────────────────────────────────────────────────

    def place_order(
        self,
        symbol:        str,
        side:          str,          # "BUY" or "SELL"
        quantity:      int,
        price:         float = 0,    # 0 = MARKET order
        trigger_price: float = 0,    # for SL orders
        order_type:    str   = "MARKET",
        product:       str   = "INTRADAY",
        stop_loss:     float = 0,
        target:        float = 0,
    ) -> Optional[str]:
        """
        Place order on Angel One.
        Returns Angel One order ID or None on failure.
        """
        if not self.is_connected():
            logger.error("Angel One not connected")
            return None

        exchange, token = self.EXCHANGE_MAP.get(symbol, ("NSE", ""))
        if not token:
            token = self._get_token(symbol, exchange)

        if not token:
            logger.error(f"Token not found for {symbol}")
            return None

        try:
            order_params = {
                "variety":          "NORMAL",
                "tradingsymbol":    symbol,
                "symboltoken":      token,
                "transactiontype":  side,
                "exchange":         exchange,
                "ordertype":        order_type,
                "producttype":      product,
                "duration":         "DAY",
                "price":            str(price),
                "squareoff":        str(target),
                "stoploss":         str(stop_loss),
                "quantity":         str(quantity),
            }
            if order_type == "SL":
                order_params["triggerprice"] = str(trigger_price)

            resp = self._api.placeOrder(order_params)

            if resp.get("status"):
                angel_id = resp["data"]["orderid"]
                logger.info(
                    f"Order placed: {side} {quantity}× {symbol} @ "
                    f"{'MARKET' if price == 0 else f'₹{price}'} | ID: {angel_id}"
                )

                # Store locally
                order = LiveOrder(
                    order_id       = angel_id,
                    symbol         = symbol,
                    exchange       = exchange,
                    side           = side,
                    quantity       = quantity,
                    order_type     = order_type,
                    price          = price,
                    trigger_price  = trigger_price,
                    status         = "OPEN",
                    placed_at      = datetime.now().isoformat(),
                    angel_order_id = angel_id,
                    product        = product,
                )
                self._orders[angel_id] = order
                return angel_id
            else:
                logger.error(f"Order failed: {resp.get('message','')}")
                return None

        except Exception as e:
            logger.error(f"Place order error: {e}")
            return None

    def cancel_order(self, order_id: str, variety: str = "NORMAL") -> bool:
        """Cancel an open order."""
        try:
            resp = self._api.cancelOrder(order_id, variety)
            if resp.get("status"):
                if order_id in self._orders:
                    self._orders[order_id].status = "CANCELLED"
                logger.info(f"Order cancelled: {order_id}")
                return True
            logger.warning(f"Cancel failed: {resp.get('message','')}")
            return False
        except Exception as e:
            logger.error(f"Cancel order error: {e}")
            return False

    def modify_sl(self, order_id: str, new_sl_price: float) -> bool:
        """Modify stop-loss order price (trailing stop)."""
        try:
            resp = self._api.modifyOrder({
                "variety":    "STOPLOSS",
                "orderid":    order_id,
                "ordertype":  "SL",
                "price":      str(new_sl_price),
                "triggerprice": str(new_sl_price),
            })
            if resp.get("status"):
                logger.info(f"SL modified: {order_id} → ₹{new_sl_price}")
                return True
            return False
        except Exception as e:
            logger.error(f"Modify SL error: {e}")
            return False

    # ── Position management ───────────────────────────────────────────────────

    def get_positions(self) -> list[LivePosition]:
        """Fetch all open positions from Angel One."""
        try:
            resp = self._api.position()
            if not resp.get("status") or not resp.get("data"):
                return []

            positions = []
            for p in resp["data"]:
                qty = int(p.get("netqty", 0))
                if qty == 0:
                    continue

                sym    = p.get("tradingsymbol","")
                exch   = p.get("exchange","NSE")
                side   = "BUY" if qty > 0 else "SELL"
                entry  = float(p.get("averageprice", 0))
                curr   = float(p.get("ltp", entry))
                pnl    = float(p.get("pnl", 0))

                pos = LivePosition(
                    symbol        = sym,
                    exchange      = exch,
                    quantity      = abs(qty),
                    entry_price   = entry,
                    current_price = curr,
                    side          = side,
                    product       = p.get("producttype","INTRADAY"),
                    pnl           = pnl,
                    pnl_pct       = (curr - entry) / entry if entry > 0 else 0,
                    angel_token   = p.get("symboltoken",""),
                )
                positions.append(pos)
                self._positions[sym] = pos

            return positions

        except Exception as e:
            logger.error(f"Get positions error: {e}")
            return []

    def get_order_book(self) -> list[dict]:
        """Fetch today's order book."""
        try:
            resp = self._api.orderBook()
            if resp.get("status") and resp.get("data"):
                return resp["data"]
            return []
        except Exception as e:
            logger.error(f"Order book error: {e}")
            return []

    def get_portfolio_summary(self) -> dict:
        """Get live portfolio summary."""
        try:
            positions = self.get_positions()
            funds     = self._api.rmsLimit()

            cash      = 0.0
            total_pnl = 0.0

            if funds.get("status") and funds.get("data"):
                d    = funds["data"]
                cash = float(d.get("availablecash", 0))

            for pos in positions:
                total_pnl += pos.pnl

            return {
                "cash":         cash,
                "total_pnl":    total_pnl,
                "n_positions":  len(positions),
                "positions":    [
                    {
                        "symbol":       p.symbol,
                        "quantity":     p.quantity,
                        "entry":        p.entry_price,
                        "current":      p.current_price,
                        "pnl":          p.pnl,
                        "pnl_pct":      p.pnl_pct,
                        "side":         p.side,
                    }
                    for p in positions
                ],
            }
        except Exception as e:
            logger.error(f"Portfolio summary error: {e}")
            return {"cash": 0, "total_pnl": 0, "n_positions": 0, "positions": []}

    # ── Execute interface (matches PaperBroker) ───────────────────────────────

    def execute(self, order, current_price: float = 0):
        """
        Drop-in replacement for PaperBroker.execute().
        Accepts TradeOrder dataclass from src/risk/models.py.
        """
        try:
            symbol   = order.symbol
            side     = order.side.value if hasattr(order.side, "value") else str(order.side)
            side_str = "BUY" if "BUY" in side.upper() else "SELL"
            qty      = order.quantity
            sl       = getattr(order, "stop_loss", 0)
            tgt      = getattr(order, "target_price", 0)

            angel_id = self.place_order(
                symbol     = symbol,
                side       = side_str,
                quantity   = qty,
                price      = 0,          # always MARKET for now
                stop_loss  = sl,
                target     = tgt,
                order_type = "MARKET",
                product    = "INTRADAY",
            )

            if angel_id:
                # Record in event bus
                try:
                    from src.streaming.event_bus import get_bus, Event, EventType
                    get_bus().emit_now(EventType.TRADE_EXECUTED, {
                        "symbol":         symbol,
                        "side":           side_str,
                        "quantity":       qty,
                        "price":          current_price,
                        "angel_order_id": angel_id,
                    }, source="angel_one")
                except Exception:
                    pass

            return type("OrderRecord", (), {
                "order_id":       angel_id or "FAILED",
                "angel_order_id": angel_id or "FAILED",
                "fill_price":     current_price,
                "total_charges":  20.0,   # Angel One flat ₹20
                "status":         "COMPLETE" if angel_id else "FAILED",
            })()

        except Exception as e:
            logger.error(f"Execute error: {e}")
            return None

    # ── WebSocket tick feed ───────────────────────────────────────────────────

    def start_tick_feed(self, symbols: list[str]):
        """
        Start WebSocket feed for real-time tick data.
        Prices fed directly to price_store.
        """
        if not self.is_connected():
            logger.warning("Not connected — cannot start tick feed")
            return

        try:
            from smartapi.smartWebSocketV2 import SmartWebSocketV2

            tokens = []
            for sym in symbols:
                exch, token = self.EXCHANGE_MAP.get(sym, ("NSE",""))
                if token:
                    tokens.append({"exchangeType": 1, "tokens": [token]})

            if not tokens:
                logger.warning("No valid tokens for tick feed")
                return

            def on_open(ws):
                logger.info("Angel One WebSocket connected")
                ws.subscribe("abc123", 3, tokens)  # mode 3 = full quote

            def on_message(ws, message):
                try:
                    data = json.loads(message) if isinstance(message, str) else message
                    token  = str(data.get("token",""))
                    ltp    = float(data.get("last_traded_price", 0)) / 100   # paise → rupees
                    symbol = self._token_to_symbol(token)

                    if symbol and ltp > 0:
                        # Update price store
                        try:
                            from src.streaming.price_store import price_store
                            price_store.set(symbol, ltp)
                        except Exception:
                            pass

                        # Notify callbacks
                        for cb in self._tick_callbacks:
                            try:
                                cb(symbol, ltp)
                            except Exception:
                                pass

                        # Emit to event bus
                        try:
                            from src.streaming.event_bus import get_bus, Event, EventType
                            get_bus().emit_now(EventType.TICK_RECEIVED, {
                                "symbol": symbol, "price": ltp,
                            }, source="angel_ws")
                        except Exception:
                            pass

                except Exception as e:
                    logger.debug(f"Tick parse error: {e}")

            def on_error(ws, error):
                logger.warning(f"Angel One WS error: {error}")

            def on_close(ws):
                logger.warning("Angel One WS closed — will reconnect")
                self._ws_running = False

            self._ws = SmartWebSocketV2(
                auth_token     = self._session_token,
                api_key        = self._api_key,
                client_code    = self._client_id,
                feed_token     = self._api.getfeedToken(),
                on_open        = on_open,
                on_message     = on_message,
                on_error       = on_error,
                on_close       = on_close,
            )

            ws_thread = threading.Thread(
                target=self._ws.connect,
                daemon=True, name="AngelWS",
            )
            ws_thread.start()
            self._ws_running = True
            logger.info(f"Tick feed started for {len(symbols)} symbols")

        except ImportError:
            logger.error("smartapi-python WebSocket not available")
        except Exception as e:
            logger.error(f"Tick feed error: {e}")

    def add_tick_callback(self, callback: Callable):
        """Register callback: callback(symbol: str, price: float)."""
        self._tick_callbacks.append(callback)

    def stop_tick_feed(self):
        if self._ws:
            try:
                self._ws.close_connection()
            except Exception:
                pass
        self._ws_running = False

    # ── Auto SL monitoring ────────────────────────────────────────────────────

    def start_sl_monitor(self, check_interval: int = 5):
        """
        Background thread monitoring SL/target for all positions.
        Executes exit orders when SL or target is hit.
        """
        def _monitor():
            logger.info("SL monitor started")
            while self.is_connected():
                try:
                    self._check_sl_targets()
                except Exception as e:
                    logger.debug(f"SL monitor error: {e}")
                time.sleep(check_interval)

        thread = threading.Thread(target=_monitor, daemon=True, name="SLMonitor")
        thread.start()
        return thread

    def _check_sl_targets(self):
        """Check all positions for SL/target hits."""
        for sym, pos in list(self._positions.items()):
            try:
                from src.streaming.price_store import price_store
                current = price_store.get(sym)
                if not current:
                    continue

                pos.update_pnl(current)

                # SL hit
                if pos.stop_loss > 0:
                    if pos.side == "BUY"  and current <= pos.stop_loss:
                        logger.warning(f"SL hit: {sym} @ ₹{current:.2f} (SL: ₹{pos.stop_loss:.2f})")
                        self._exit_position(sym, pos, "SL_HIT")
                        continue
                    if pos.side == "SELL" and current >= pos.stop_loss:
                        logger.warning(f"SL hit: {sym} @ ₹{current:.2f} (SL: ₹{pos.stop_loss:.2f})")
                        self._exit_position(sym, pos, "SL_HIT")
                        continue

                # Target hit
                if pos.target > 0:
                    if pos.side == "BUY"  and current >= pos.target:
                        logger.info(f"Target hit: {sym} @ ₹{current:.2f}")
                        self._exit_position(sym, pos, "TARGET_HIT")
                        continue
                    if pos.side == "SELL" and current <= pos.target:
                        logger.info(f"Target hit: {sym} @ ₹{current:.2f}")
                        self._exit_position(sym, pos, "TARGET_HIT")
                        continue

            except Exception as e:
                logger.debug(f"SL check error for {sym}: {e}")

    def _exit_position(self, symbol: str, pos: LivePosition, reason: str):
        """Place exit order for a position."""
        exit_side = "SELL" if pos.side == "BUY" else "BUY"
        order_id  = self.place_order(
            symbol    = symbol,
            side      = exit_side,
            quantity  = pos.quantity,
            order_type= "MARKET",
        )
        if order_id:
            del self._positions[symbol]
            # Emit event
            try:
                from src.streaming.event_bus import get_bus, EventType
                event_type = (EventType.SL_HIT if reason == "SL_HIT"
                              else EventType.TARGET_HIT)
                get_bus().emit_now(event_type, {
                    "symbol": symbol, "reason": reason,
                    "pnl":    pos.pnl, "pnl_pct": pos.pnl_pct,
                }, source="sl_monitor")
            except Exception:
                pass

    # ── Token management ──────────────────────────────────────────────────────

    def _get_token(self, symbol: str, exchange: str = "NSE") -> str:
        """Look up instrument token from cached file."""
        try:
            if TOKEN_CACHE.exists():
                tokens = json.loads(TOKEN_CACHE.read_text())
                key    = f"{exchange}:{symbol}"
                return tokens.get(key, "")
        except Exception:
            pass
        return ""

    def _token_to_symbol(self, token: str) -> Optional[str]:
        """Reverse lookup: token → symbol."""
        for sym, (exch, tok) in self.EXCHANGE_MAP.items():
            if tok == token:
                return sym
        return None

    def download_instrument_master(self) -> int:
        """
        Download full instrument master from Angel One.
        Saves token→symbol mapping to cache.
        Returns number of instruments cached.
        """
        if not self.is_connected():
            return 0
        try:
            instruments = self._api.getAllInstruments()
            token_map   = {}
            for inst in instruments or []:
                sym   = inst.get("tradingsymbol","")
                exch  = inst.get("exch_seg","")
                token = str(inst.get("token",""))
                if sym and token:
                    token_map[f"{exch}:{sym}"] = token
                    token_map[token] = sym   # reverse lookup

            TOKEN_CACHE.write_text(json.dumps(token_map, indent=2))
            logger.info(f"Instrument master: {len(instruments)} instruments cached")
            return len(instruments)
        except Exception as e:
            logger.error(f"Instrument master download failed: {e}")
            return 0

    # ── Session persistence ───────────────────────────────────────────────────

    def _save_session(self):
        try:
            SESSION_CACHE.write_text(json.dumps({
                "session_token": self._session_token,
                "refresh_token": self._refresh_token,
                "saved_at":      datetime.now().isoformat(),
            }, indent=2))
        except Exception:
            pass

    def _load_session(self) -> bool:
        try:
            if SESSION_CACHE.exists():
                d = json.loads(SESSION_CACHE.read_text())
                saved_at = datetime.fromisoformat(d["saved_at"])
                age_h    = (datetime.now() - saved_at).total_seconds() / 3600
                if age_h < 20:   # valid for ~24h
                    self._session_token = d["session_token"]
                    self._refresh_token = d["refresh_token"]
                    return True
        except Exception:
            pass
        return False


# ── Dashboard page ────────────────────────────────────────────────────────────

def render_angel_one_page():
    """
    Add to app.py:
        elif page == "🔴 Angel One Live":
            from src.brokers.angel_one_live import render_angel_one_page
            render_angel_one_page()
    """
    import streamlit as st
    import pandas as pd

    st.header("🔴 Angel One Live Trading")
    st.caption(
        "Full live trading integration. ₹20 flat brokerage. "
        "WebSocket tick feed + auto SL monitoring."
    )

    broker = AngelOneLive()

    # Connection status
    col_status, col_btn = st.columns([3, 1])
    with col_status:
        if broker.is_connected():
            st.success(f"✅ Connected — {broker._client_id}")
        else:
            st.error("❌ Not connected")
    with col_btn:
        if st.button("🔌 Connect", type="primary"):
            with st.spinner("Connecting to Angel One..."):
                if broker.connect():
                    st.success("Connected!")
                    st.rerun()
                else:
                    st.error("Connection failed. Check credentials in .env")

    if not broker.is_connected():
        st.info(
            "Add credentials to `.env`:\n"
            "```\nANGEL_API_KEY=...\n"
            "ANGEL_CLIENT_ID=...\n"
            "ANGEL_PASSWORD=...\n"
            "ANGEL_TOTP_SECRET=...\n```"
        )
        return

    st.divider()

    # Portfolio summary
    with st.spinner("Loading live portfolio..."):
        summary = broker.get_portfolio_summary()

    k1, k2, k3 = st.columns(3)
    k1.metric("Cash Available", f"₹{summary['cash']:,.0f}")
    k2.metric("Total P&L",      f"₹{summary['total_pnl']:+,.0f}")
    k3.metric("Open Positions", summary["n_positions"])

    # Positions
    st.subheader("Open Positions")
    if summary["positions"]:
        rows = []
        for p in summary["positions"]:
            rows.append({
                "Symbol":  p["symbol"],
                "Side":    p["side"],
                "Qty":     p["quantity"],
                "Entry":   f"₹{p['entry']:,.2f}",
                "Current": f"₹{p['current']:,.2f}",
                "P&L":     f"₹{p['pnl']:+,.0f}",
                "P&L %":   f"{p['pnl_pct']*100:+.2f}%",
            })
        df_pos = pd.DataFrame(rows)
        def color_pnl(val):
            if "+" in str(val): return "color:#00cc66"
            if "-" in str(val): return "color:#ff4444"
            return ""
        st.dataframe(
            df_pos.style.map(color_pnl, subset=["P&L","P&L %"]),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No open positions.")

    st.divider()

    # Order book
    st.subheader("Today's Orders")
    with st.spinner("Loading order book..."):
        orders = broker.get_order_book()
    if orders:
        rows = []
        for o in orders[:20]:
            rows.append({
                "Time":    o.get("updatetime","")[:16],
                "Symbol":  o.get("tradingsymbol",""),
                "Side":    o.get("transactiontype",""),
                "Qty":     o.get("filledshares","0"),
                "Price":   o.get("averageprice","MKT"),
                "Status":  o.get("status",""),
                "ID":      o.get("orderid",""),
            })
        def color_side(val):
            if val == "BUY":  return "color:#00cc66"
            if val == "SELL": return "color:#ff4444"
            return ""
        df_orders = pd.DataFrame(rows)
        st.dataframe(
            df_orders.style.map(color_side, subset=["Side"]),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No orders today.")

    st.divider()

    # Quick order
    st.subheader("Quick Order")
    st.warning(
        "⚠️ **LIVE TRADING** — Orders placed here use REAL MONEY. "
        "Use the main 📟 Trade page for full risk management."
    )
    qo1, qo2, qo3, qo4 = st.columns(4)
    with qo1:
        q_sym  = st.selectbox("Symbol", ["NIFTY50","BANKNIFTY","RELIANCE","TCS","HDFCBANK"], key="ao_sym")
    with qo2:
        q_side = st.radio("Side", ["BUY","SELL"], horizontal=True, key="ao_side")
    with qo3:
        q_qty  = st.number_input("Qty", 1, 100, 1, key="ao_qty")
    with qo4:
        st.markdown("<br>", unsafe_allow_html=True)
        confirm = st.checkbox("Confirm LIVE order", key="ao_confirm")

    if st.button("Place LIVE Order", type="primary", disabled=not confirm):
        order_id = broker.place_order(q_sym, q_side, q_qty)
        if order_id:
            st.success(f"✅ Order placed: {q_side} {q_qty}× {q_sym} | ID: {order_id}")
        else:
            st.error("Order failed. Check logs.")

    # Controls
    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📡 Start Tick Feed"):
            broker.start_tick_feed(["NIFTY50","BANKNIFTY","RELIANCE","TCS","GOLD"])
            st.success("Tick feed started — prices now live in dashboard")
    with c2:
        if st.button("🛡️ Start SL Monitor"):
            broker.start_sl_monitor(check_interval=5)
            st.success("SL monitor running (checks every 5s)")
    with c3:
        if st.button("📥 Download Instruments"):
            with st.spinner("Downloading..."):
                n = broker.download_instrument_master()
                st.success(f"Downloaded {n:,} instruments")