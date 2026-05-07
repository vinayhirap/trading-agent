# trading-agent/src/brokers/token_manager.py
"""
Instrument Token Manager — Angel One SmartAPI

Angel One requires a numeric token for every API call.
The full instrument list (~80,000 symbols) is published daily at:
  https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json

This module:
  1. Downloads the master file (once per day, cached locally)
  2. Builds fast symbol → token lookup
  3. Resolves our internal symbol names (NIFTY50, RELIANCE, etc.)
     to Angel One's tradingsymbol + token + exchange

Cache: data/angel_tokens.json  (refreshed daily at startup)
"""
import json
import requests
from datetime import datetime, date
from pathlib import Path
from loguru import logger

TOKEN_CACHE  = Path("data/angel_tokens.json")
MASTER_URL   = (
    "https://margincalculator.angelbroking.com"
    "/OpenAPI_File/files/OpenAPIScripMaster.json"
)

# Our internal symbol → Angel One tradingsymbol mapping
# Angel One uses exchange-specific symbols (e.g. NIFTY50 → NIFTY50, RELIANCE → RELIANCE-EQ)
SYMBOL_TO_ANGEL = {
    # Indices (exchange: NSE, segment: NSE)
    "NIFTY50":   {"symbol": "NIFTY50",       "exchange": "NSE"},
    "BANKNIFTY": {"symbol": "BANKNIFTY",      "exchange": "NSE"},
    "SENSEX":    {"symbol": "SENSEX",         "exchange": "BSE"},
    "NIFTYMID":  {"symbol": "NIFTYMIDCAP100", "exchange": "NSE"},
    "NIFTYIT":   {"symbol": "NIFTYIT",        "exchange": "NSE"},
    # Equities (exchange: NSE, segment: NSE)
    "RELIANCE":  {"symbol": "RELIANCE-EQ",    "exchange": "NSE"},
    "TCS":       {"symbol": "TCS-EQ",         "exchange": "NSE"},
    "HDFCBANK":  {"symbol": "HDFCBANK-EQ",    "exchange": "NSE"},
    "INFY":      {"symbol": "INFY-EQ",        "exchange": "NSE"},
    "ICICIBANK": {"symbol": "ICICIBANK-EQ",   "exchange": "NSE"},
    "SBIN":      {"symbol": "SBIN-EQ",        "exchange": "NSE"},
    "WIPRO":     {"symbol": "WIPRO-EQ",       "exchange": "NSE"},
    "AXISBANK":  {"symbol": "AXISBANK-EQ",    "exchange": "NSE"},
    "KOTAKBANK": {"symbol": "KOTAKBANK-EQ",   "exchange": "NSE"},
    "LT":        {"symbol": "LT-EQ",          "exchange": "NSE"},
    "BAJFINANCE":{"symbol": "BAJFINANCE-EQ",  "exchange": "NSE"},
    "MARUTI":    {"symbol": "MARUTI-EQ",      "exchange": "NSE"},
    "SUNPHARMA": {"symbol": "SUNPHARMA-EQ",   "exchange": "NSE"},
    "BHARTIARTL":{"symbol": "BHARTIARTL-EQ",  "exchange": "NSE"},
    "TATAMOTORS":{"symbol": "TATAMOTORS-EQ",  "exchange": "NSE"},
    # Commodities (MCX)
    "GOLD":      {"symbol": "GOLD",           "exchange": "MCX"},
    "SILVER":    {"symbol": "SILVER",         "exchange": "MCX"},
    "CRUDEOIL":  {"symbol": "CRUDEOIL",       "exchange": "MCX"},
    "COPPER":    {"symbol": "COPPER",         "exchange": "MCX"},
    "NATURALGAS":{"symbol": "NATURALGAS",     "exchange": "MCX"},
}

# Hardcoded fallback tokens (from adapter — always works offline)
HARDCODED_TOKENS = {
    "NIFTY50":   {"token": "99926000", "exchange": "NSE", "symbol": "NIFTY50"},
    "BANKNIFTY": {"token": "99926009", "exchange": "NSE", "symbol": "BANKNIFTY"},
    "SENSEX":    {"token": "1",        "exchange": "BSE", "symbol": "SENSEX"},
    "RELIANCE":  {"token": "2885",     "exchange": "NSE", "symbol": "RELIANCE-EQ"},
    "TCS":       {"token": "11536",    "exchange": "NSE", "symbol": "TCS-EQ"},
    "HDFCBANK":  {"token": "1333",     "exchange": "NSE", "symbol": "HDFCBANK-EQ"},
    "INFY":      {"token": "1594",     "exchange": "NSE", "symbol": "INFY-EQ"},
    "ICICIBANK": {"token": "4963",     "exchange": "NSE", "symbol": "ICICIBANK-EQ"},
    "SBIN":      {"token": "3045",     "exchange": "NSE", "symbol": "SBIN-EQ"},
    "WIPRO":     {"token": "3787",     "exchange": "NSE", "symbol": "WIPRO-EQ"},
    "AXISBANK":  {"token": "5900",     "exchange": "NSE", "symbol": "AXISBANK-EQ"},
    "KOTAKBANK": {"token": "1922",     "exchange": "NSE", "symbol": "KOTAKBANK-EQ"},
    "LT":        {"token": "11483",    "exchange": "NSE", "symbol": "LT-EQ"},
    "TATAMOTORS":{"token": "3432",     "exchange": "NSE", "symbol": "TATAMOTORS-EQ"},
    "BAJFINANCE":{"token": "317",      "exchange": "NSE", "symbol": "BAJFINANCE-EQ"},
    "MARUTI":    {"token": "10999",    "exchange": "NSE", "symbol": "MARUTI-EQ"},
    "SUNPHARMA": {"token": "3351",     "exchange": "NSE", "symbol": "SUNPHARMA-EQ"},
    "BHARTIARTL":{"token": "10604",    "exchange": "NSE", "symbol": "BHARTIARTL-EQ"},
}


class TokenManager:
    """
    Resolves trading symbols to Angel One instrument tokens.

    Priority:
      1. In-memory cache (fastest)
      2. Local JSON cache (fast, survives restarts)
      3. Download fresh master file (slow, once per day)
      4. Hardcoded fallback (always works for the 19 main symbols)
    """

    def __init__(self):
        self._cache: dict = {}        # token → full instrument record
        self._sym_index: dict = {}    # tradingsymbol → token
        self._loaded_date: date = None
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_token_info(self, our_symbol: str) -> dict | None:
        """
        Returns {'token': str, 'exchange': str, 'symbol': str} or None.
        our_symbol: e.g. 'NIFTY50', 'RELIANCE', 'GOLD'
        """
        self._refresh_if_stale()

        # Try hardcoded first (most reliable for our core symbols)
        if our_symbol in HARDCODED_TOKENS:
            return HARDCODED_TOKENS[our_symbol]

        # Try Angel One symbol mapping
        angel_info = SYMBOL_TO_ANGEL.get(our_symbol)
        if angel_info:
            angel_sym = angel_info["symbol"]
            token = self._sym_index.get(angel_sym)
            if token:
                return {
                    "token":    token,
                    "exchange": angel_info["exchange"],
                    "symbol":   angel_sym,
                }

        # Try direct lookup (sometimes our symbol IS the Angel symbol)
        token = self._sym_index.get(our_symbol)
        if token:
            rec = self._cache.get(token, {})
            return {
                "token":    token,
                "exchange": rec.get("exch_seg", "NSE"),
                "symbol":   our_symbol,
            }

        logger.warning(f"TokenManager: no token found for {our_symbol}")
        return None

    def get_token(self, our_symbol: str) -> str | None:
        info = self.get_token_info(our_symbol)
        return info["token"] if info else None

    def search(self, query: str, exchange: str = "NSE", limit: int = 10) -> list[dict]:
        """Search instruments by partial name. Useful for F&O symbol lookup."""
        self._refresh_if_stale()
        query = query.upper()
        results = []
        for token, rec in self._cache.items():
            if (query in rec.get("symbol", "").upper() or
                    query in rec.get("name", "").upper()):
                if not exchange or rec.get("exch_seg") == exchange:
                    results.append({
                        "token":    token,
                        "symbol":   rec.get("symbol"),
                        "name":     rec.get("name"),
                        "exchange": rec.get("exch_seg"),
                        "lotsize":  rec.get("lotsize", 1),
                    })
                    if len(results) >= limit:
                        break
        return results

    def all_loaded(self) -> bool:
        return len(self._cache) > 100

    def stats(self) -> dict:
        return {
            "total_instruments": len(self._cache),
            "loaded_date":       str(self._loaded_date),
            "source":            "master_file" if self.all_loaded() else "hardcoded_only",
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _refresh_if_stale(self):
        today = date.today()
        if self._loaded_date == today and self.all_loaded():
            return
        if TOKEN_CACHE.exists():
            cache_date = date.fromtimestamp(TOKEN_CACHE.stat().st_mtime)
            if cache_date == today:
                self._load()
                return
        self._download_master()

    def _download_master(self):
        logger.info("Downloading Angel One instrument master (~5 MB)...")
        try:
            resp = requests.get(MASTER_URL, timeout=30)
            resp.raise_for_status()
            instruments = resp.json()

            # Build lookup dicts
            cache     = {}
            sym_index = {}
            for inst in instruments:
                token  = str(inst.get("token", ""))
                symbol = str(inst.get("symbol", ""))
                if token and symbol:
                    cache[token]      = inst
                    sym_index[symbol] = token

            TOKEN_CACHE.parent.mkdir(parents=True, exist_ok=True)
            with open(TOKEN_CACHE, "w") as f:
                json.dump({"cache": cache, "sym_index": sym_index}, f)

            self._cache      = cache
            self._sym_index  = sym_index
            self._loaded_date = date.today()
            logger.info(f"Instrument master loaded: {len(cache):,} instruments")

        except Exception as e:
            logger.warning(f"Master download failed: {e} — using hardcoded tokens only")
            self._loaded_date = date.today()   # don't retry today

    def _load(self):
        if TOKEN_CACHE.exists():
            try:
                with open(TOKEN_CACHE) as f:
                    data = json.load(f)
                self._cache      = data.get("cache", {})
                self._sym_index  = data.get("sym_index", {})
                self._loaded_date = date.fromtimestamp(TOKEN_CACHE.stat().st_mtime)
                logger.info(f"TokenManager: loaded {len(self._cache):,} instruments from cache")
                return
            except Exception as e:
                logger.warning(f"Token cache load failed: {e}")
        # Start with empty — will download on first get_token call
        self._cache = {}
        self._sym_index = {}


# Module-level singleton
token_manager = TokenManager()