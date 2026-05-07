# trading-agent/src/streaming/tick_engine.py
"""
Tick Engine — Angel One SmartAPI WebSocket streaming.

Changes from original:
  - Added CDS (exchangeType=13) for USDINR/EURINR/GBPINR/JPYINR
  - MCX + CDS tokens auto-resolved from scrip master (no manual rollover)
  - All ticks divided by 100 (Angel One sends paise for ALL exchanges)
"""
import threading
import time
from datetime import datetime, timezone
from loguru import logger

from src.streaming.price_store import price_store
from config.settings import settings

MODE_LTP   = 1
MODE_QUOTE = 2
MODE_SNAP  = 3

# Static NSE symbols (no expiry)
NSE_STREAM_SYMBOLS: dict[str, tuple[str, str]] = {
    "NIFTY50":    ("NSE", "99926000"),
    "BANKNIFTY":  ("NSE", "99926009"),
    "SENSEX":     ("BSE", "1"),
    "NIFTYMID":   ("NSE", "99926012"),
    "NIFTYIT":    ("NSE", "99926019"),
    "FINNIFTY":   ("NSE", "99926037"),
    "NIFTYPHARMA":("NSE", "99926026"),
    "NIFTYAUTO":  ("NSE", "99926014"),
    "NIFTYFMCG":  ("NSE", "99926023"),
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
    "JSWSTEEL":   ("NSE", "11723"),
    "HINDALCO":   ("NSE", "1363"),
    "ONGC":       ("NSE", "2475"),
    "NTPC":       ("NSE", "11630"),
    "POWERGRID":  ("NSE", "14977"),
    "COALINDIA":  ("NSE", "20374"),
    "ASIANPAINT": ("NSE", "236"),
    "ULTRACEMCO": ("NSE", "11532"),
    "DRREDDY":    ("NSE", "881"),
    "BAJAJFINSV": ("NSE", "16675"),
}

EXCHANGE_TYPE_MAP = {"NSE": 1, "BSE": 3, "MCX": 5, "CDS": 13}


class TickEngine:
    RECONNECT_BASE   = 5
    RECONNECT_MAX    = 120
    RECONNECT_FACTOR = 2

    def __init__(self):
        self._sws           = None
        self._thread        = None
        self._running       = False
        self._enabled       = False
        self._connected     = False
        self._reconnects    = 0
        self._dns_failures  = 0
        self._last_error    = None
        self._status        = "idle"
        self._last_tick     = None
        self._lock          = threading.Lock()
        # Built at runtime
        self._stream_symbols: dict[str, tuple[str, str]] = {}
        self._token_to_symbol: dict[str, str] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        if not getattr(settings, "ENABLE_ANGEL_STREAMING", False):
            self._enabled    = False
            self._status     = "disabled"
            self._last_error = "Angel streaming disabled in settings"
            logger.info("TickEngine: disabled — yfinance fallback")
            return
        if not self._has_credentials():
            self._enabled    = False
            self._status     = "disabled"
            self._last_error = "Angel One credentials not set"
            return

        # Resolve dynamic tokens
        self._build_stream_symbols()

        self._enabled = True
        self._running = True
        self._status  = "connecting"
        self._thread  = threading.Thread(
            target=self._run_loop, daemon=True, name="TickEngine"
        )
        self._thread.start()
        logger.info("TickEngine started")

    def stop(self):
        self._running = False
        self._disconnect()
        if self._thread:
            self._thread.join(timeout=5)

    def is_connected(self) -> bool:
        return self._connected

    def get_price(self, symbol: str) -> float | None:
        return price_store.get(symbol)

    def get_all_prices(self) -> dict:
        return price_store.get_all()

    def stats(self) -> dict:
        return {
            "enabled":       self._enabled,
            "running":       self._running,
            "connected":     self._connected,
            "status":        self._status,
            "reconnects":    self._reconnects,
            "dns_failures":  self._dns_failures,
            "last_error":    self._last_error,
            "last_tick":     self._last_tick.isoformat() if self._last_tick else None,
            "symbols":       len(self._stream_symbols),
            "prices_cached": len(price_store.get_all()),
        }

    # ── Token resolution ──────────────────────────────────────────────────────

    def _build_stream_symbols(self):
        try:
            from src.streaming.token_resolver import token_resolver
            mcx = token_resolver.get_mcx_tokens()
            cds = token_resolver.get_cds_tokens()
        except Exception as e:
            logger.warning(f"TickEngine: token resolver error ({e}), using fallback")
            mcx = {
                "GOLD":       ("MCX", "495213"),
                "SILVER":     ("MCX", "495214"),
                "CRUDEOIL":   ("MCX", "488291"),
                "COPPER":     ("MCX", "488791"),
                "NATURALGAS": ("MCX", "488505"),
                "ZINC":       ("MCX", "510478"),
                "ALUMINIUM":  ("MCX", "488790"),
            }
            cds = {
                "USDINR": ("CDS", "11416"),
                "EURINR": ("CDS", "1497"),
                "GBPINR": ("CDS", "1498"),
                "JPYINR": ("CDS", "1510"),
            }

        self._stream_symbols = {**NSE_STREAM_SYMBOLS, **mcx, **cds}
        self._token_to_symbol = {
            tok: sym for sym, (_, tok) in self._stream_symbols.items()
        }
        logger.info(
            f"TickEngine: {len(NSE_STREAM_SYMBOLS)} NSE + "
            f"{len(mcx)} MCX + {len(cds)} CDS symbols"
        )

    # ── Connection loop ───────────────────────────────────────────────────────

    def _run_loop(self):
        delay = self.RECONNECT_BASE
        while self._running:
            try:
                self._status = "connecting"
                self._connect_and_stream()
            except Exception as e:
                error_msg = str(e)
                self._last_error = error_msg
                if "getaddrinfo failed" in error_msg or "Name resolution" in error_msg:
                    self._dns_failures += 1
                    self._status = "dns_error"
                    if self._dns_failures >= 4:
                        logger.error("TickEngine: repeated DNS failures — disabling")
                        self._status  = "disabled"
                        self._running = False
                        break
                else:
                    self._dns_failures = 0
                    self._status = "error"
                    logger.error(f"TickEngine error: {e}")

            if not self._running:
                break

            self._connected   = False
            self._reconnects += 1
            self._status = "reconnecting"
            time.sleep(delay)
            delay = min(delay * self.RECONNECT_FACTOR, self.RECONNECT_MAX)

    def _connect_and_stream(self):
        try:
            from SmartApi.smartWebSocketV2 import SmartWebSocketV2
        except ImportError:
            logger.error("smartapi-python not installed")
            self._running = False
            return

        auth_token, feed_token, client_code = self._get_auth()
        if not auth_token:
            self._status = "auth_failed"
            return

        token_list = self._build_token_list()
        if not token_list:
            logger.error("TickEngine: no tokens to subscribe")
            return

        try:
            self._sws = SmartWebSocketV2(
                auth_token=auth_token,
                api_key=settings.ANGEL_API_KEY,
                client_code=client_code,
                feed_token=feed_token,
                max_retry_attempt=1,
                retry_strategy=0,
                retry_delay=5,
            )
        except Exception as e:
            logger.error(f"TickEngine: init error: {e}")
            self._running = False
            return

        def on_open(wsapp):
            self._connected    = True
            self._status       = "connected"
            self._reconnects   = 0
            self._dns_failures = 0
            self._last_error   = None
            self._sws.subscribe(
                correlation_id="tick_stream",
                mode=MODE_LTP,
                token_list=token_list,
            )
            logger.info(f"TickEngine: subscribed {len(self._stream_symbols)} symbols")

        def on_data(wsapp, message):
            self._on_tick(message)

        def on_error(wsapp, error):
            self._last_error = str(error)
            self._status     = "error"
            self._connected  = False

        def on_close(wsapp, *args):
            self._connected = False
            if self._running:
                self._status = "reconnecting"

        self._sws.on_open  = on_open
        self._sws.on_data  = on_data
        self._sws.on_error = on_error
        self._sws.on_close = on_close
        self._sws.connect()

    def _on_tick(self, message: dict):
        try:
            token = str(message.get("token", ""))
            ltp   = message.get("last_traded_price", 0)
            if not token or not ltp:
                return

            symbol = self._token_to_symbol.get(token)
            if not symbol:
                return

            # Angel One sends paise for ALL exchanges (NSE, BSE, MCX, CDS)
            price = ltp / 100.0

            if price > 0:
                price_store.update(symbol, price)
                self._last_tick = datetime.now(timezone.utc)

        except Exception as e:
            logger.warning(f"TickEngine tick error: {e}")

    def _build_token_list(self) -> list[dict]:
        groups: dict[int, list[str]] = {}
        for sym, (exch, token) in self._stream_symbols.items():
            exch_code = EXCHANGE_TYPE_MAP.get(exch, 1)
            groups.setdefault(exch_code, []).append(token)
        return [
            {"exchangeType": ec, "tokens": toks}
            for ec, toks in groups.items()
        ]

    def _disconnect(self):
        try:
            if self._sws:
                self._sws.close_connection()
        except Exception:
            pass
        self._connected = False
        self._sws = None

    def _get_auth(self) -> tuple[str, str, str]:
        try:
            import pyotp
            from SmartApi import SmartConnect
            smart = SmartConnect(api_key=settings.ANGEL_API_KEY)
            totp  = pyotp.TOTP(settings.ANGEL_TOTP_SECRET).now()
            data  = smart.generateSession(
                settings.ANGEL_CLIENT_ID, settings.ANGEL_PASSWORD, totp
            )
            if not data.get("status"):
                return None, None, None
            return data["data"]["jwtToken"], smart.getfeedToken(), settings.ANGEL_CLIENT_ID
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"TickEngine auth error: {e}")
            return None, None, None

    def _has_credentials(self) -> bool:
        return all([
            getattr(settings, "ANGEL_API_KEY",    None),
            getattr(settings, "ANGEL_CLIENT_ID",  None),
            getattr(settings, "ANGEL_PASSWORD",    None),
            getattr(settings, "ANGEL_TOTP_SECRET", None),
        ])


# Module singleton
tick_engine = TickEngine()