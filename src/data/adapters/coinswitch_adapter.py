# trading-agent/src/data/adapters/coinswitch_adapter.py
"""
CoinSwitch Pro API adapter for crypto data and trading.

Setup:
1. Create CoinSwitch Pro account: https://pro.coinswitch.co/
2. Generate API key from Settings → API
3. Add to .env:
   COINSWITCH_API_KEY=your_key
   COINSWITCH_API_SECRET=your_secret

Free tier: public market data (no auth needed)
Trading tier: requires API key
"""
import hmac
import hashlib
import time
import requests
from datetime import datetime
from loguru import logger
import pandas as pd

from src.data.base_adapter import BaseDataAdapter
from src.data.models import Interval
from config.settings import settings

BASE_URL = "https://coinswitch.co/trade/api/v2"

# CoinSwitch symbol format
CS_SYMBOLS = {
    "BTC":   "BTC/INR",
    "ETH":   "ETH/INR",
    "SOL":   "SOL/INR",
    "MATIC": "MATIC/INR",
    "BNB":   "BNB/INR",
    "XRP":   "XRP/INR",
    "ADA":   "ADA/INR",
    "DOGE":  "DOGE/INR",
    "AVAX":  "AVAX/INR",
    "DOT":   "DOT/INR",
}

CS_INTERVAL_MAP = {
    Interval.M1:  "1m",
    Interval.M5:  "5m",
    Interval.M15: "15m",
    Interval.H1:  "1h",
    Interval.D1:  "1d",
}


class CoinSwitchAdapter(BaseDataAdapter):
    """
    CoinSwitch Pro API adapter.
    Falls back to yfinance (USD prices) if API key not set.
    """

    def __init__(self):
        super().__init__("coinswitch")
        self._session = requests.Session()
        self._has_key = bool(
            getattr(settings, "COINSWITCH_API_KEY", None) and
            getattr(settings, "COINSWITCH_API_SECRET", None)
        )

    def connect(self) -> bool:
        if not self._has_key:
            logger.warning(
                "CoinSwitch API key not set — using yfinance for crypto data. "
                "Add COINSWITCH_API_KEY and COINSWITCH_API_SECRET to .env "
                "to get INR prices and trading."
            )
            self._connected = True    # still works via yfinance fallback
            return True

        try:
            # Test connectivity with a public endpoint
            resp = self._session.get(
                f"{BASE_URL}/exchanges/", timeout=10
            )
            resp.raise_for_status()
            self._connected = True
            logger.info("CoinSwitch Pro connected")
            return True
        except Exception as e:
            logger.warning(f"CoinSwitch connection check failed: {e} — using yfinance fallback")
            self._connected = True
            return True

    def fetch_ohlcv(
        self,
        symbol:   str,
        interval: Interval,
        start:    datetime,
        end:      datetime,
    ) -> pd.DataFrame:
        # Always fall back to yfinance for historical data (USD)
        # CoinSwitch API is better for live INR prices
        from src.data.adapters.yfinance_adapter import YFinanceAdapter
        from src.data.models import CRYPTO_SYMBOLS
        adapter = YFinanceAdapter()
        adapter.connect()
        return adapter.fetch_ohlcv(symbol, interval, start, end)

    def fetch_latest_price(self, symbol: str) -> float:
        """Get latest crypto price in INR from CoinSwitch."""
        cs_sym = CS_SYMBOLS.get(symbol)

        if self._has_key and cs_sym:
            try:
                price = self._get_cs_price(cs_sym)
                if price > 0:
                    logger.debug(f"CoinSwitch {symbol}: ₹{price:,.2f}")
                    return price
            except Exception as e:
                logger.warning(f"CoinSwitch price failed: {e}")

        # Fallback: yfinance USD price
        from src.data.adapters.yfinance_adapter import YFinanceAdapter
        usd_price = YFinanceAdapter().fetch_latest_price(symbol)

        # Convert USD → INR using live rate
        usdinr = self._get_usdinr_rate()
        inr_price = usd_price * usdinr
        logger.debug(
            f"{symbol}: ${usd_price:.2f} × ₹{usdinr:.2f}/$ = ₹{inr_price:,.2f}"
        )
        return inr_price

    def _get_cs_price(self, cs_symbol: str) -> float:
        """Fetch live INR price from CoinSwitch public API."""
        try:
            resp = self._session.get(
                f"{BASE_URL}/trades/",
                params={"symbol": cs_symbol, "limit": 1},
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()
            if data and isinstance(data, list):
                return float(data[0].get("price", 0))
        except Exception as e:
            logger.debug(f"CoinSwitch price API: {e}")
        return 0.0

    def _get_usdinr_rate(self) -> float:
        """Get live USD/INR exchange rate."""
        try:
            from src.data.adapters.yfinance_adapter import YFinanceAdapter
            import yfinance as yf
            ticker = yf.Ticker("USDINR=X")
            rate = ticker.fast_info.last_price
            return float(rate) if rate and rate > 0 else 84.0
        except Exception:
            return 84.0   # fallback rate

    def place_order(
        self,
        symbol:   str,
        side:     str,
        quantity: float,
        price:    float = 0,
    ) -> dict:
        """Place a crypto order on CoinSwitch Pro."""
        if not self._has_key:
            raise RuntimeError(
                "CoinSwitch API key not set. "
                "Add COINSWITCH_API_KEY and COINSWITCH_API_SECRET to .env"
            )
        if settings.ENV != "live":
            raise RuntimeError("Live trading requires ENV=live in .env")

        cs_sym = CS_SYMBOLS.get(symbol, symbol)
        payload = {
            "symbol":   cs_sym,
            "side":     side.lower(),
            "type":     "market" if price == 0 else "limit",
            "quantity": str(quantity),
            "price":    str(price),
        }

        try:
            headers = self._auth_headers("POST", "/orders/", payload)
            resp = self._session.post(
                f"{BASE_URL}/orders/",
                json=payload, headers=headers, timeout=10,
            )
            resp.raise_for_status()
            result = resp.json()
            logger.info(f"CoinSwitch order: {result}")
            return result
        except Exception as e:
            logger.error(f"CoinSwitch order failed: {e}")
            raise

    def _auth_headers(self, method: str, path: str, body: dict) -> dict:
        """Generate authenticated request headers for CoinSwitch Pro."""
        ts = str(int(time.time() * 1000))
        import json
        payload_str = json.dumps(body, separators=(",", ":")) if body else ""
        sign_str    = ts + method + path + payload_str
        signature   = hmac.new(
            settings.COINSWITCH_API_SECRET.encode(),
            sign_str.encode(), hashlib.sha256,
        ).hexdigest()
        return {
            "X-AUTH-APIKEY":    settings.COINSWITCH_API_KEY,
            "X-AUTH-SIGNATURE": signature,
            "X-AUTH-EPOCH":     ts,
            "Content-Type":     "application/json",
        }
