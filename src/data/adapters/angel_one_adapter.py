# trading-agent/src/data/adapters/angel_one_adapter.py
"""
Angel One SmartAPI adapter — real-time data + order execution.

Provides:
- Real-time LTP for all NSE/BSE/MCX instruments
- Historical OHLCV (1min to 1month)
- Order placement (paper mode uses PaperBroker, live uses this)
- WebSocket tick streaming via angel_one_ticker.py

Setup (one-time):
1. Create account at angelone.in
2. Get API key: https://smartapi.angelbroking.com/
3. pip install smartapi-python pyotp
4. Add to .env:
   ANGEL_API_KEY=your_key
   ANGEL_CLIENT_ID=your_client_id
   ANGEL_PASSWORD=your_password
   ANGEL_TOTP_SECRET=your_totp_secret
"""
import pyotp
from datetime import datetime, timedelta
from loguru import logger
import pandas as pd

from src.data.base_adapter import BaseDataAdapter
from src.data.models import Interval
from config.settings import settings

ANGEL_INTERVAL_MAP = {
    Interval.M1:  "ONE_MINUTE",
    Interval.M5:  "FIVE_MINUTE",
    Interval.M15: "FIFTEEN_MINUTE",
    Interval.M30: "THIRTY_MINUTE",
    Interval.H1:  "ONE_HOUR",
    Interval.D1:  "ONE_DAY",
    Interval.W1:  "ONE_WEEK",
}

# Complete NSE + MCX token registry
# Indices use special tokens (99926xxx for NSE indices)
ANGEL_TOKENS: dict[str, tuple[str, str]] = {
    # (exchange, token)
    # ── NSE Indices ──────────────────────────────────────────────────────────
    "NIFTY50":    ("NSE", "99926000"),
    "BANKNIFTY":  ("NSE", "99926009"),
    "NIFTYIT":    ("NSE", "99926037"),
    "FINNIFTY":   ("NSE", "99926025"),
    "SENSEX":     ("BSE", "99919000"),
    # ── NSE Equities ─────────────────────────────────────────────────────────
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
    # Forex (CDS exchange — rolls monthly, auto-managed by mcx_token_manager)
    "USDINR":  ("CDS", "1518"),   # USDINR26508FUT expiry 08May2026
    "EURINR":  ("CDS", "1497"),   # EURINR26508FUT expiry 08May2026
    "GBPINR":  ("CDS", "1498"),   # GBPINR26508FUT expiry 08May2026
    "JPYINR":  ("CDS", "1510"),   # JPYINR26508FUT expiry 08May2026
    # ── MCX Commodities (near-month, update monthly) ──────────────────────────
    "GOLD":       ("MCX", "495213"),
    "SILVER":     ("MCX", "495214"),
    "CRUDEOIL":   ("MCX", "488291"),
    "COPPER":     ("MCX", "488791"),
    "NATURALGAS": ("MCX", "488505"),
    "ZINC":       ("MCX", "510478"),
    "ALUMINIUM":  ("MCX", "488790"),
}

# Fix duplicate tuple for FINNIFTY
ANGEL_TOKENS["FINNIFTY"] = ("NSE", "99926025")


class AngelOneAdapter(BaseDataAdapter):
    """
    Angel One SmartAPI data adapter.
    Falls back to yfinance automatically if credentials not set.
    """

    def __init__(self):
        super().__init__("angel_one")
        self._smart_api  = None
        self._auth_token = None

    def connect(self) -> bool:
        if not settings.ANGEL_API_KEY or not settings.ANGEL_CLIENT_ID:
            logger.warning(
                "Angel One credentials not set — "
                "add ANGEL_API_KEY, ANGEL_CLIENT_ID, ANGEL_PASSWORD, "
                "ANGEL_TOTP_SECRET to .env"
            )
            return False

        try:
            from SmartApi import SmartConnect
            self._smart_api = SmartConnect(api_key=settings.ANGEL_API_KEY)

            totp_secret = settings.ANGEL_TOTP_SECRET
            totp = pyotp.TOTP(totp_secret).now() if totp_secret else ""

            data = self._smart_api.generateSession(
                settings.ANGEL_CLIENT_ID,
                settings.ANGEL_PASSWORD,
                totp,
            )

            if data["status"]:
                self._auth_token = data["data"]["jwtToken"]
                self._connected  = True
                logger.info(f"Angel One connected | client={settings.ANGEL_CLIENT_ID}")
                return True
            else:
                logger.error(f"Angel One auth failed: {data['message']}")
                return False

        except ImportError:
            logger.error("smartapi-python not installed: pip install smartapi-python pyotp")
            return False
        except Exception as e:
            logger.error(f"Angel One connection error: {e}")
            return False

    def fetch_ohlcv(
        self,
        symbol:   str,
        interval: Interval,
        start:    datetime,
        end:      datetime,
    ) -> pd.DataFrame:
        if not self._connected or self._smart_api is None:
            return self._yf_fallback_ohlcv(symbol, interval, start, end)

        token_info = ANGEL_TOKENS.get(symbol)
        if not token_info:
            return self._yf_fallback_ohlcv(symbol, interval, start, end)

        exchange, token = token_info

        try:
            params = {
                "exchange":    exchange,
                "symboltoken": token,
                "interval":    ANGEL_INTERVAL_MAP.get(interval, "ONE_DAY"),
                "fromdate":    start.strftime("%Y-%m-%d %H:%M"),
                "todate":      end.strftime("%Y-%m-%d %H:%M"),
            }
            resp = self._smart_api.getCandleData(params)
            if not resp["status"]:
                raise ValueError(resp["message"])

            candles = resp["data"]
            df = pd.DataFrame(
                candles,
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
            df.index = df.index.tz_localize("Asia/Kolkata").tz_convert("UTC")
            df = df.astype(float)

            logger.debug(f"Angel One OHLCV: {symbol} {len(df)} bars")
            return self.validate_dataframe(df, symbol)

        except Exception as e:
            logger.warning(f"Angel One candle fetch error {symbol}: {e} — using yfinance")
            return self._yf_fallback_ohlcv(symbol, interval, start, end)

    def fetch_latest_price(self, symbol: str) -> float:
        if not self._connected or self._smart_api is None:
            return self._yf_fallback_price(symbol)

        token_info = ANGEL_TOKENS.get(symbol)
        if not token_info:
            return self._yf_fallback_price(symbol)

        exchange, token = token_info

        try:
            resp = self._smart_api.ltpData(exchange, symbol, token)
            if resp and resp.get("status") and resp.get("data"):
                ltp = float(resp["data"].get("ltp", 0))
                if ltp > 0:
                    logger.debug(f"Angel One LTP: {symbol} = {ltp}")
                    return ltp
        except Exception as e:
            logger.debug(f"Angel One LTP failed {symbol}: {e}")

        return self._yf_fallback_price(symbol)

    def fetch_ltp_batch(self, symbols: list[str]) -> dict[str, float]:
        """Fetch LTP for multiple symbols in one call."""
        result = {}
        if not self._connected or self._smart_api is None:
            return result

        for sym in symbols:
            price = self.fetch_latest_price(sym)
            if price > 0:
                result[sym] = price

        return result

    def place_order(
        self,
        symbol:     str,
        side:       str,
        quantity:   int,
        order_type: str   = "MARKET",
        price:      float = 0,
        product:    str   = "DELIVERY",
    ) -> dict:
        """Place a LIVE order. Only works when ENV=live."""
        if settings.ENV != "live":
            raise RuntimeError(
                f"Cannot place live order in ENV={settings.ENV}. "
                "Set ENV=live in .env to enable live trading."
            )
        if not self._connected:
            raise RuntimeError("Angel One not connected")

        token_info = ANGEL_TOKENS.get(symbol, ("NSE", ""))
        exchange, token = token_info

        order_params = {
            "variety":         "NORMAL",
            "tradingsymbol":   symbol,
            "symboltoken":     token,
            "transactiontype": side,
            "exchange":        exchange,
            "ordertype":       order_type,
            "producttype":     product,
            "duration":        "DAY",
            "price":           str(price) if price else "0",
            "squareoff":       "0",
            "stoploss":        "0",
            "quantity":        str(quantity),
        }

        try:
            resp = self._smart_api.placeOrder(order_params)
            logger.info(f"Angel One order placed: {symbol} {side} {quantity} → {resp}")
            return resp
        except Exception as e:
            logger.error(f"Order placement failed {symbol}: {e}")
            raise

    def get_positions(self) -> list[dict]:
        """Get current open positions from Angel One."""
        if not self._connected:
            return []
        try:
            resp = self._smart_api.position()
            if resp and resp.get("status"):
                return resp.get("data", []) or []
        except Exception as e:
            logger.warning(f"Angel One positions error: {e}")
        return []

    def get_order_book(self) -> list[dict]:
        """Get today's order book."""
        if not self._connected:
            return []
        try:
            resp = self._smart_api.orderBook()
            if resp and resp.get("status"):
                return resp.get("data", []) or []
        except Exception as e:
            logger.warning(f"Angel One order book error: {e}")
        return []

    # ── Fallbacks ─────────────────────────────────────────────────────────────

    def _yf_fallback_ohlcv(self, symbol, interval, start, end) -> pd.DataFrame:
        from src.data.adapters.yfinance_adapter import YFinanceAdapter
        return YFinanceAdapter().fetch_ohlcv(symbol, interval, start, end)

    def _yf_fallback_price(self, symbol) -> float:
        from src.data.adapters.yfinance_adapter import YFinanceAdapter
        return YFinanceAdapter().fetch_latest_price(symbol)