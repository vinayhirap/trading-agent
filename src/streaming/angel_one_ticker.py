# trading-agent/src/streaming/angel_one_ticker.py
"""
Angel One WebSocket Tick Streamer

Auto-resolves MCX + CDS futures tokens on startup via scrip master.
No manual token updates needed on expiry rollover.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import pyotp
from loguru import logger

from config.settings import settings
from src.streaming.price_store import price_store, ACTIVE_EXPIRY


# ── Static NSE tokens (indices + equities — no expiry) ───────────────────────

NSE_TOKENS: dict[str, tuple[str, str]] = {
    "NIFTY50":    ("NSE", "99926000"),
    "BANKNIFTY":  ("NSE", "99926009"),
    "NIFTYIT":    ("NSE", "99926037"),
    "SENSEX":     ("BSE", "99919000"),
    "FINNIFTY":   ("NSE", "99926025"),
    "RELIANCE":   ("NSE", "2885"),
    "TCS":        ("NSE", "11536"),
    "HDFCBANK":   ("NSE", "1333"),
    "INFY":       ("NSE", "1594"),
    "ICICIBANK":  ("NSE", "4963"),
    "SBIN":       ("NSE", "3045"),
    "WIPRO":      ("NSE", "3787"),
    "AXISBANK":   ("NSE", "5900"),
    "KOTAKBANK":  ("NSE", "1922"),
    "LT":         ("NSE", "11483"),
    "BAJFINANCE": ("NSE", "317"),
    "MARUTI":     ("NSE", "10999"),
    "SUNPHARMA":  ("NSE", "3351"),
    "BHARTIARTL": ("NSE", "10604"),
    "TATASTEEL":  ("NSE", "3499"),
    "TATAMOTORS": ("NSE", "3432"),
    "JSWSTEEL":   ("NSE", "11723"),
    "HINDALCO":   ("NSE", "1363"),
    "ONGC":       ("NSE", "2475"),
    "NTPC":       ("NSE", "11630"),
    "POWERGRID":  ("NSE", "14977"),
    "COALINDIA":  ("NSE", "20374"),
    "ULTRACEMCO": ("NSE", "11532"),
    "DRREDDY":    ("NSE", "881"),
    "BAJAJFINSV": ("NSE", "16675"),
    "ASIANPAINT": ("NSE", "236"),
    "M&M":        ("NSE", "2031"),
}

EXCHANGE_TYPE_MAP = {"NSE": 1, "BSE": 3, "MCX": 5, "CDS": 13}
MCX_TOKENS: dict[str, tuple[str, str]] = {}   # { "GOLD": ("MCX", "495213") }
TOKEN_TO_SYMBOL: dict[str, str] = {}           # { "495213": "GOLD" }

class AngelOneTicker:
    """Real-time WebSocket tick streamer with auto-expiry token resolution."""

    RECONNECT_DELAY  = 5
    MAX_RECONNECTS   = 10
    CHUNK_SIZE       = 50
    LTP_POLL_SECONDS = 3

    def __init__(self):
        self._smart_api       = None
        self._ws_client       = None
        self._auth_token      = None
        self._feed_token      = None
        self._running         = False
        self._connected       = False
        self._reconnect_count = 0
        self._thread: Optional[threading.Thread] = None
        self._lock            = threading.Lock()
        self._tick_count      = 0
        self._last_tick_at    = 0.0
        # Built at runtime from token_resolver
        self._token_to_symbol: dict[str, str] = {}
        self._all_exchange_tokens: dict[str, list[str]] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> bool:
        if self._running:
            return True
        if not self._credentials_available():
            logger.warning(
                "AngelOneTicker: credentials not set — "
                "add ANGEL_API_KEY, ANGEL_CLIENT_ID, ANGEL_PASSWORD, "
                "ANGEL_TOTP_SECRET to .env"
            )
            return False

        # Resolve MCX + CDS tokens from scrip master before connecting
        self._build_token_maps()

        self._running = True
        self._thread  = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="AngelOneTicker",
        )
        self._thread.start()
        logger.info("AngelOneTicker: started")
        return True

    def stop(self):
        self._running = False
        self._disconnect_ws()

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def tick_count(self) -> int:
        return self._tick_count

    def get_status(self) -> dict:
        return {
            "connected":       self._connected,
            "tick_count":      self._tick_count,
            "last_tick_ago":   round(time.time() - self._last_tick_at, 1) if self._last_tick_at else None,
            "reconnect_count": self._reconnect_count,
            "symbols_covered": len(self._token_to_symbol),
        }

    # ── Token resolution ──────────────────────────────────────────────────────
    def _build_token_maps(self):
        """Use mcx_token_manager as single source — no token_resolver."""
        from src.streaming.mcx_token_manager import mcx_token_manager
        mcx_token_manager.start()          # no-op if already started

        tokens = mcx_token_manager.get_tokens()
        mcx_tokens = {sym: (info["exchange"], info["token"]) for sym, info in tokens.items() if info["exchange"] == "MCX"}
        cds_tokens = {sym: (info["exchange"], info["token"]) for sym, info in tokens.items() if info["exchange"] == "CDS"}

        # Update ACTIVE_EXPIRY for display
        from src.streaming.price_store import ACTIVE_EXPIRY
        for sym, info in tokens.items():
            ACTIVE_EXPIRY[sym] = info["expiry"].strftime("%b %Y")

        all_tokens = {**NSE_TOKENS, **mcx_tokens, **cds_tokens}
        self._token_to_symbol = {tok: sym for sym, (_, tok) in all_tokens.items()}

        self._all_exchange_tokens = {
            "NSE": [tok for _, (exch, tok) in NSE_TOKENS.items() if exch == "NSE"],
            "BSE": [tok for _, (exch, tok) in NSE_TOKENS.items() if exch == "BSE"],
            "MCX": [tok for _, tok in mcx_tokens.values()],
            "CDS": [tok for _, tok in cds_tokens.values()],
        }

        # Expose module-level dicts so mcx_token_manager can patch live
        global MCX_TOKENS, TOKEN_TO_SYMBOL
        MCX_TOKENS.update({sym: v for sym, v in mcx_tokens.items()})
        MCX_TOKENS.update({sym: v for sym, v in cds_tokens.items()})
        TOKEN_TO_SYMBOL.update(self._token_to_symbol)

        logger.info(f"AngelOneTicker: {len(mcx_tokens)} MCX + {len(cds_tokens)} CDS tokens loaded")

        # Build reverse map token → symbol
        all_tokens = {**NSE_TOKENS, **mcx_tokens, **cds_tokens}
        self._token_to_symbol = {tok: sym for sym, (_, tok) in all_tokens.items()}

        # Group tokens by exchange for subscription
        self._all_exchange_tokens = {
            "NSE": [tok for _, (exch, tok) in NSE_TOKENS.items() if exch == "NSE"],
            "BSE": [tok for _, (exch, tok) in NSE_TOKENS.items() if exch == "BSE"],
            "MCX": [tok for _, tok in mcx_tokens.values()],
            "CDS": [tok for _, tok in cds_tokens.values()],
        }

    # ── Authentication ────────────────────────────────────────────────────────

    def _credentials_available(self) -> bool:
        return bool(
            settings.ANGEL_API_KEY and
            settings.ANGEL_CLIENT_ID and
            settings.ANGEL_PASSWORD
        )

    def _authenticate(self) -> bool:
        try:
            from SmartApi import SmartConnect
            self._smart_api = SmartConnect(api_key=settings.ANGEL_API_KEY)

            totp_secret = settings.ANGEL_TOTP_SECRET
            totp        = pyotp.TOTP(totp_secret).now() if totp_secret else ""

            data = self._smart_api.generateSession(
                settings.ANGEL_CLIENT_ID,
                settings.ANGEL_PASSWORD,
                totp,
            )

            if not data.get("status"):
                logger.error(f"AngelOneTicker: auth failed — {data.get('message', 'unknown')}")
                return False

            self._auth_token = data["data"]["jwtToken"]
            self._feed_token = data["data"]["feedToken"]
            logger.info(f"AngelOneTicker: authenticated | client={settings.ANGEL_CLIENT_ID}")
            return True

        except ImportError:
            logger.error("AngelOneTicker: smartapi-python not installed")
            return False
        except Exception as e:
            logger.error(f"AngelOneTicker: auth error — {e}")
            return False

    # ── WebSocket streaming ───────────────────────────────────────────────────

    def _run_loop(self):
        while self._running:
            try:
                if not self._authenticate():
                    time.sleep(self.RECONNECT_DELAY)
                    continue
                self._start_websocket()
            except Exception as e:
                logger.warning(f"AngelOneTicker: connection error — {e}")
                self._connected = False
                self._reconnect_count += 1

                if self._reconnect_count >= self.MAX_RECONNECTS:
                    logger.error("AngelOneTicker: max reconnects reached — stopping")
                    self._running = False
                    return

                delay = min(self.RECONNECT_DELAY * self._reconnect_count, 60)
                logger.info(f"AngelOneTicker: reconnecting in {delay}s")
                time.sleep(delay)

    def _start_websocket(self):
        try:
            from SmartApi.smartWebSocketV2 import SmartWebSocketV2
        except ImportError:
            logger.warning("AngelOneTicker: smartWebSocketV2 not available — REST fallback")
            self._rest_poll_fallback()
            return

        self._ws_client = SmartWebSocketV2(
            auth_token  = self._auth_token,
            api_key     = settings.ANGEL_API_KEY,
            client_code = settings.ANGEL_CLIENT_ID,
            feed_token  = self._feed_token,
        )
        self._ws_client.on_open  = self._on_open
        self._ws_client.on_data  = self._on_tick
        self._ws_client.on_error = self._on_error
        self._ws_client.on_close = self._on_close

        logger.info("AngelOneTicker: connecting WebSocket...")
        self._ws_client.connect()

    def _on_open(self, ws):
        self._connected       = True
        self._reconnect_count = 0
        logger.info("AngelOneTicker: connected — subscribing tokens")
        self._subscribe_all_tokens()

    def _on_tick(self, ws, tick: dict):
        try:
            token  = str(tick.get("token", ""))
            symbol = self._token_to_symbol.get(token)
            if not symbol:
                return

            ltp_raw = tick.get("last_traded_price", 0)

            # Angel One sends ALL ticks in paise (NSE, BSE, MCX, CDS)
            ltp = ltp_raw / 100.0

            if ltp > 0:
                price_store.update(symbol, ltp)
                self._tick_count  += 1
                self._last_tick_at = time.time()

        except Exception as e:
            logger.debug(f"AngelOneTicker tick error: {e}")

    def _on_error(self, ws, error):
        logger.warning(f"AngelOneTicker WebSocket error: {error}")
        self._connected = False

    def _on_close(self, ws):
        logger.info("AngelOneTicker: WebSocket closed")
        self._connected = False

    def _subscribe_all_tokens(self):
        for exchange, tokens in self._all_exchange_tokens.items():
            if not tokens:
                continue
            exch_type = EXCHANGE_TYPE_MAP.get(exchange, 1)
            for i in range(0, len(tokens), self.CHUNK_SIZE):
                chunk = tokens[i:i + self.CHUNK_SIZE]
                try:
                    self._ws_client.subscribe(
                        correlation_id=f"tick_{exchange}_{i}",
                        mode=1,
                        token_list=[{"exchangeType": exch_type, "tokens": chunk}],
                    )
                    time.sleep(0.1)
                except Exception as e:
                    logger.warning(f"AngelOneTicker: subscribe error {exchange}: {e}")

        logger.info(f"AngelOneTicker: subscribed {len(self._token_to_symbol)} symbols")

    def _disconnect_ws(self):
        try:
            if self._ws_client:
                self._ws_client.close_connection()
        except Exception:
            pass
        self._connected = False

    # ── REST polling fallback ─────────────────────────────────────────────────

    def _rest_poll_fallback(self):
        logger.info("AngelOneTicker: REST LTP polling fallback")
        all_tokens = {**NSE_TOKENS, **MCX_TOKENS}   # MCX_TOKENS already populated

        while self._running:
            for sym, (exch, token) in all_tokens.items():
                if not self._running:
                    break
                try:
                    resp = self._smart_api.ltpData(exch, sym, token)
                    if resp and resp.get("status") and resp.get("data"):
                        ltp = float(resp["data"].get("ltp", 0))
                        if ltp > 0:
                            price_store.update(sym, ltp)
                            self._tick_count  += 1
                            self._last_tick_at = time.time()
                except Exception as e:
                    logger.debug(f"AngelOneTicker LTP {sym}: {e}")
            time.sleep(self.LTP_POLL_SECONDS)


# Module singleton
angel_one_ticker = AngelOneTicker()