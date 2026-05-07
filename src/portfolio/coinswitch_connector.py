"""
CoinSwitch Pro Portfolio Connector.

Fetches live portfolio holdings, P&L, and prices from CoinSwitch Pro API.
Falls back to manual entry if API keys not configured.

API docs: https://coinswitch.co/pro/trade/api
Free tier: portfolio data available with API key.
"""
import hmac
import hashlib
import time
import json
import requests
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger
from cryptography.hazmat.primitives.asymmetric import ed25519

BASE_URL = "https://coinswitch.co"

SYMBOL_MAP = {
    "BTC":   "BTC/INR", "ETH":  "ETH/INR", "SOL":  "SOL/INR",
    "MATIC": "MATIC/INR","BNB":  "BNB/INR", "XRP":  "XRP/INR",
    "ADA":   "ADA/INR",  "DOGE": "DOGE/INR","AVAX": "AVAX/INR",
    "DOT":   "DOT/INR",  "LINK": "LINK/INR",
}


@dataclass
class CryptoHolding:
    symbol:        str
    quantity:      float
    avg_buy_price: float
    current_price: float
    current_value: float
    invested_value: float
    pnl:           float
    pnl_pct:       float
    last_updated:  datetime


@dataclass
class CoinSwitchPortfolio:
    holdings:       list
    total_invested: float
    total_value:    float
    total_pnl:      float
    total_pnl_pct:  float
    fetched_at:     datetime
    is_live:        bool          # True = from API, False = fallback/demo
    error:          Optional[str] = None


class CoinSwitchPortfolioConnector:
    """
    Connects to CoinSwitch Pro and fetches real portfolio data.

    Setup:
    1. Get API key from coinswitch.co/pro → Settings → API
    2. Add to .env:
       COINSWITCH_API_KEY=your_key
       COINSWITCH_API_SECRET=your_secret

    Without keys: returns demo data so dashboard still works.
    """

    CACHE_TTL = 60   # cache portfolio for 60 seconds

    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.has_creds  = bool(api_key and api_secret)
        self._session   = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        self._cache     = None
        self._cache_ts  = 0
        logger.info(f"CoinSwitch connector | live={'yes' if self.has_creds else 'no (add keys to .env)'}")

    def get_portfolio(self) -> CoinSwitchPortfolio:
        """Fetch full portfolio — cached for 60 seconds."""
        now = time.time()
        if self._cache and (now - self._cache_ts) < self.CACHE_TTL:
            return self._cache

        if self.has_creds:
            port = self._fetch_live_portfolio()
        else:
            port = self._demo_portfolio()

        self._cache    = port
        self._cache_ts = now
        return port

    def get_prices(self, symbols: list) -> dict:
        """Get current prices for given symbols in INR."""
        prices = {}
        if self.has_creds:
            try:
                prices = self._fetch_prices_api(symbols)
            except Exception as e:
                logger.warning(f"CoinSwitch prices API: {e}")
                prices = self._fetch_prices_yfinance(symbols)
        else:
            prices = self._fetch_prices_yfinance(symbols)
        return prices

    def validate_keys(self) -> bool:
        """Validate API keys with CoinSwitch."""
        try:
            resp = self._signed_request("GET", "/trade/api/v2/validate/keys")
            return resp.get("message") == "Valid Access"
        except Exception as e:
            logger.error(f"Key validation failed: {e}")
            return False

    def _fetch_live_portfolio(self) -> CoinSwitchPortfolio:
        """Fetch real portfolio from CoinSwitch Pro API."""
        try:
            # Get portfolio/balances
            resp = self._signed_request("GET", "/trade/api/v2/user/portfolio")
            if not resp:
                raise ValueError(f"Empty response: {resp}")

            data     = resp.get("data", [])
            holdings = []
            total_inv = 0.0
            total_val = 0.0

            coins = data if isinstance(data, list) else []
            for coin in coins:
                sym = coin.get("currency", coin.get("symbol", "")).upper()
                if sym in ("INR", "USDT"):
                    continue
                qty      = float(coin.get("main_balance", 0))
                avg_buy  = float(coin.get("buy_average_price", 0))
                if qty <= 0:
                    continue

                # Use API's current_value and invested_value
                cur_val = float(coin.get("current_value", 0))
                inv_val = float(coin.get("invested_value", 0))
                cur_px = cur_val / qty if qty > 0 else 0
                pnl = cur_val - inv_val
                pnl_pct = (pnl / inv_val * 100) if inv_val > 0 else 0

                total_inv += inv_val
                total_val += cur_val

                holdings.append(CryptoHolding(
                    symbol=sym, quantity=qty,
                    avg_buy_price=round(avg_buy, 2),
                    current_price=round(cur_px, 2),
                    current_value=round(cur_val, 2),
                    invested_value=round(inv_val, 2),
                    pnl=round(pnl, 2),
                    pnl_pct=round(pnl_pct, 2),
                    last_updated=datetime.now(timezone.utc),
                ))

            total_pnl = total_val - total_inv
            total_pnl_pct = (total_pnl / total_inv * 100) if total_inv > 0 else 0

            return CoinSwitchPortfolio(
                holdings=holdings,
                total_invested=round(total_inv, 2),
                total_value=round(total_val, 2),
                total_pnl=round(total_pnl, 2),
                total_pnl_pct=round(total_pnl_pct, 2),
                fetched_at=datetime.now(timezone.utc),
                is_live=True,
            )

        except requests.HTTPError as e:
            resp_text = ""
            if e.response is not None:
                try:
                    resp_text = e.response.text
                except Exception:
                    resp_text = "<unreadable response>"
            err_msg = f"HTTP {e.response.status_code} from CoinSwitch: {resp_text or str(e)}"
            logger.error(f"CoinSwitch live portfolio: {err_msg}")
            port = self._demo_portfolio()
            port.error = err_msg
            port.is_live = False
            return port
        except Exception as e:
            logger.error(f"CoinSwitch live portfolio: {e}")
            port = self._demo_portfolio()
            port.error = str(e)
            port.is_live = False
            return port

    def _get_single_price(self, cs_symbol: str) -> float:
        """Get price for one symbol from CoinSwitch."""
        try:
            # Assume coinswitchx for INR pairs, c2c1 for USDT
            exchange = "c2c1" if "/USDT" in cs_symbol else "coinswitchx"
            resp = self._signed_request("GET", f"/trade/api/v2/trades?exchange={exchange}&symbol={cs_symbol}")
            if resp and isinstance(resp, dict) and "data" in resp:
                data = resp["data"]
                if data and isinstance(data, list) and len(data) > 0:
                    return float(data[0].get("p", 0))  # price is "p"
        except Exception:
            pass
        return 0.0

    def _fetch_prices_api(self, symbols: list) -> dict:
        """Batch fetch prices from CoinSwitch."""
        prices = {}
        for sym in symbols:
            cs_sym = SYMBOL_MAP.get(sym)
            if cs_sym:
                px = self._get_single_price(cs_sym)
                if px > 0:
                    prices[sym] = px
        return prices

    def _fetch_prices_yfinance(self, symbols: list) -> dict:
        """Fallback: fetch USD prices from yfinance and convert to INR."""
        import yfinance as yf
        prices = {}
        try:
            usdinr = float(yf.Ticker("USDINR=X").fast_info.last_price or 84)
        except Exception:
            usdinr = 84.0

        from src.data.models import CRYPTO_SYMBOLS
        for sym in symbols:
            info = CRYPTO_SYMBOLS.get(sym)
            yf_sym = info.symbol if info else f"{sym}-USD"
            try:
                px = yf.Ticker(yf_sym).fast_info.last_price
                if px and px > 0:
                    prices[sym] = round(float(px) * usdinr, 2)
            except Exception:
                pass
        return prices

    def _signed_request(self, method: str, path: str, body: dict = None) -> dict:
        """Make signed request to CoinSwitch API."""
        ts = str(int(time.time() * 1000))
        payload = ""
        if body:
            payload = json.dumps(body, separators=(",", ":"), sort_keys=True)
        msg = method + path + payload + ts
        request_string = bytes(msg, 'utf-8')
        secret_key_bytes = bytes.fromhex(self.api_secret)
        secret_key = ed25519.Ed25519PrivateKey.from_private_bytes(secret_key_bytes)
        signature_bytes = secret_key.sign(request_string)
        sig = signature_bytes.hex()

        headers = {
            "X-AUTH-APIKEY":    self.api_key,
            "X-AUTH-SIGNATURE": sig,
            "X-AUTH-EPOCH":     ts,
            "Content-Type":     "application/json",
        }
        url = BASE_URL + path
        resp = self._session.request(
            method, url, headers=headers,
            data=payload if body else None,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def _demo_portfolio(self) -> CoinSwitchPortfolio:
        """Return empty portfolio when no API keys — shows connection instructions."""
        return CoinSwitchPortfolio(
            holdings=[],
            total_invested=0, total_value=0, total_pnl=0, total_pnl_pct=0,
            fetched_at=datetime.now(timezone.utc),
            is_live=False,
            error="API keys not configured. Add COINSWITCH_API_KEY and COINSWITCH_API_SECRET to .env",
        )
