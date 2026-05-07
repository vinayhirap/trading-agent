# trading-agent/src/streaming/price_store.py
"""
PriceStore — single source of truth for all market prices.

DESIGN PRINCIPLE:
  Store RAW prices as yfinance returns them.
  Convert ONLY at display time using convert_price().
  Both Python (format_price_with_unit) and JS (live_ticker) 
  use the SAME conversion logic — this eliminates ticker≠tile discrepancy.

MCX Note:
  GC=F  → USD per troy oz  (COMEX Gold)
  SI=F  → USD per troy oz  (COMEX Silver)  
  CL=F  → USD per barrel   (WTI Crude)
  These are NOT the same as MCX contract prices but are the best
  freely available proxy. MCX premium is typically 2-5% above COMEX.
"""
import threading
from datetime import datetime, timezone
from collections import deque
from loguru import logger


# ── MCX Active Contract Expiry (update monthly on rollover) ───────────────────
ACTIVE_EXPIRY: dict[str, str] = {
    "GOLD":       "Jun 2026",
    "SILVER":     "May 2026",
    "CRUDEOIL":   "May 2026",
    "COPPER":     "Apr 2026",
    "NATURALGAS": "Apr 2026",
    "ZINC":       "Apr 2026",
    "ALUMINIUM":  "Apr 2026",
}


# ── Canonical conversion table ────────────────────────────────────────────────
# "convert" values:
#   none       → already in INR, display as-is
#   usd_to_inr → multiply by USDINR
#   gold       → USD/troy oz → INR/10g  (price * usdinr * 10 / 31.1035)
#   silver     → USD/troy oz → INR/kg   (price * usdinr * 1000 / 31.1035)
#   copper     → USD/lb → INR/kg        (price * usdinr / 0.453592)

# NSE Equities — already in INR, no conversion
PRICE_META: dict[str, dict] = {
    # NSE Indices
    "NIFTY50":    {"convert": "none",      "label": " pts"},
    "BANKNIFTY":  {"convert": "none",      "label": " pts"},
    "NIFTYIT":    {"convert": "none",      "label": " pts"},
    "SENSEX":     {"convert": "none",      "label": " pts"},
    "NIFTYMID":   {"convert": "none",      "label": " pts"},
    "FINNIFTY":   {"convert": "none",      "label": " pts"},
    "NIFTYPHARMA":{"convert": "none",      "label": " pts"},
    "NIFTYAUTO":  {"convert": "none",      "label": " pts"},
    "NIFTYFMCG":  {"convert": "none",      "label": " pts"},
    "NIFTYINFRA": {"convert": "none",      "label": " pts"},
    # MCX Commodities
    "GOLD":       {"convert": "none", "label": "/10g"},
    "SILVER":     {"convert": "none", "label": "/kg"},
    "CRUDEOIL":   {"convert": "none", "label": "/bbl"},
    "COPPER":     {"convert": "none", "label": "/kg"},
    "NATURALGAS": {"convert": "none", "label": "/mmBtu"},
    "ZINC":       {"convert": "none", "label": "/kg"},
    "ALUMINIUM":  {"convert": "none", "label": ""},
    # Crypto
    "BTC":        {"convert": "usd_to_inr","label": ""},
    "ETH":        {"convert": "usd_to_inr","label": ""},
    "SOL":        {"convert": "usd_to_inr","label": ""},
    "BNB":        {"convert": "usd_to_inr","label": ""},
    "XRP":        {"convert": "usd_to_inr","label": ""},
    "ADA":        {"convert": "usd_to_inr","label": ""},
    "DOGE":       {"convert": "usd_to_inr","label": ""},
    "AVAX":       {"convert": "usd_to_inr","label": ""},
    "DOT":        {"convert": "usd_to_inr","label": ""},
    "MATIC":      {"convert": "usd_to_inr","label": ""},
    # Forex
    "USDINR":     {"convert": "none",      "label": ""},
    "EURUSD":     {"convert": "none",      "label": ""},
    "GBPUSD":     {"convert": "none",      "label": ""},
    "EURINR":     {"convert": "none",      "label": ""},
    "GBPINR":     {"convert": "none",      "label": ""},
    "JPYINR":     {"convert": "none",      "label": ""},
    # NSE Equities
    "RELIANCE":   {"convert": "none",      "label": ""},
    "TCS":        {"convert": "none",      "label": ""},
    "HDFCBANK":   {"convert": "none",      "label": ""},
    "INFY":       {"convert": "none",      "label": ""},
    "ICICIBANK":  {"convert": "none",      "label": ""},
    "SBIN":       {"convert": "none",      "label": ""},
    "WIPRO":      {"convert": "none",      "label": ""},
    "AXISBANK":   {"convert": "none",      "label": ""},
    "KOTAKBANK":  {"convert": "none",      "label": ""},
    "LT":         {"convert": "none",      "label": ""},
    "BAJFINANCE": {"convert": "none",      "label": ""},
    "MARUTI":     {"convert": "none",      "label": ""},
    "SUNPHARMA":  {"convert": "none",      "label": ""},
    "BHARTIARTL": {"convert": "none",      "label": ""},
    "TATASTEEL":  {"convert": "none",      "label": ""},
    "TATAMOTORS": {"convert": "none",      "label": ""},
    "JSWSTEEL":   {"convert": "none",      "label": ""},
    "HINDALCO":   {"convert": "none",      "label": ""},
    "ONGC":       {"convert": "none",      "label": ""},
    "NTPC":       {"convert": "none",      "label": ""},
    "POWERGRID":  {"convert": "none",      "label": ""},
    "COALINDIA":  {"convert": "none",      "label": ""},
    "ULTRACEMCO": {"convert": "none",      "label": ""},
    "DRREDDY":    {"convert": "none",      "label": ""},
    "BAJAJFINSV": {"convert": "none",      "label": ""},
    "ASIANPAINT": {"convert": "none",      "label": ""},
    "M&M":        {"convert": "none",      "label": ""},
}

# JS-compatible conversion string (used by live_ticker.py to inject into JS)
# Maps each symbol to a JS expression where `p` = raw price, `u` = usdinr
JS_CONVERSION: dict[str, str] = {
    "GOLD":    "p * u * 10 / 31.1035",
    "SILVER":  "p * u * 1000 / 31.1035",
    "CRUDEOIL":"p * u",
    "COPPER":  "p * u / 0.453592",
    "ZINC":    "p * u / 0.453592",
    "ALUMINIUM":"p * u",
    "NATURALGAS":"p * u",
    "BTC":     "p * u",
    "ETH":     "p * u",
    "SOL":     "p * u",
    "BNB":     "p * u",
    "XRP":     "p * u",
    "ADA":     "p * u",
    "DOGE":    "p * u",
    "AVAX":    "p * u",
    "DOT":     "p * u",
    "MATIC":   "p * u",
}


def convert_price(symbol: str, raw_price: float, usdinr: float) -> float:
    """
    THE canonical Python price converter.
    Takes raw yfinance price → returns display INR value.
    Must stay in sync with JS_CONVERSION above.
    """
    if not raw_price or raw_price <= 0:
        return 0.0
    sym  = symbol.upper()
    meta = PRICE_META.get(sym, {})
    conv = meta.get("convert", "usd_to_inr")

    if conv == "none":
        return float(raw_price)
    elif conv == "usd_to_inr":
        return raw_price * usdinr
    elif conv == "gold":
        return raw_price * usdinr * 10 / 31.1035
    elif conv == "silver":
        return raw_price * usdinr * 1000 / 31.1035
    elif conv == "copper":
        return raw_price * usdinr / 0.453592
    else:
        return float(raw_price)


def format_price_display(symbol: str, raw_price: float, usdinr: float) -> str:
    """
    Returns a fully formatted price string for display.
    e.g. "₹98,660 /10g" or "24,196.75 pts"
    This replaces format_price_with_unit() in app.py.
    """
    if not raw_price or raw_price <= 0:
        return "—"
    sym   = symbol.upper()
    meta  = PRICE_META.get(sym, {})
    label = meta.get("label", "")
    conv  = meta.get("convert", "usd_to_inr")
    price = convert_price(sym, raw_price, usdinr)

    if conv == "none" and "pts" in label:
        return f"{price:,.2f}{label}"
    if conv == "none" and sym in ("USDINR", "EURINR", "GBPINR", "JPYINR"):
        return f"₹{price:.4f}"
    if conv == "none" and sym in ("EURUSD", "GBPUSD"):
        return f"${price:.4f}"
    # INR values — always full number, never lakh shorthand
    if price >= 1_000:
        return f"₹{price:,.0f}{label}"
    else:
        return f"₹{price:,.2f}{label}"


class PriceStore:
    MAX_HISTORY              = 500
    STALE_SECONDS            = 300
    FALLBACK_REFRESH_SECONDS = 300

    def __init__(self):
        self._prices:     dict[str, float]    = {}
        self._timestamps: dict[str, datetime] = {}
        self._history:    dict[str, deque]    = {}
        self._lock        = threading.Lock()
        self._tick_count  = 0
        self._bg_started  = False

    # ── Write ─────────────────────────────────────────────────────────────────

    def update(self, symbol: str, price: float):
        """Store RAW price — no conversion."""
        sym = symbol.upper()
        now = datetime.now(timezone.utc)
        with self._lock:
            self._prices[sym]     = price
            self._timestamps[sym] = now
            if sym not in self._history:
                self._history[sym] = deque(maxlen=self.MAX_HISTORY)
            self._history[sym].append((now, price))
            self._tick_count += 1
        # Fire tick callbacks (used for instant WS push on Angel One ticks)
        if hasattr(self, "_tick_callbacks"):
            for _cb in self._tick_callbacks:
                try:
                    _cb(sym, price)
                except Exception:
                    pass

    def set(self, symbol: str, price: float):
        """Alias for update(). Used by MCX bootstrap and Angel One ticks."""
        self.update(symbol, price)

    def update_batch(self, prices: dict[str, float]):
        now = datetime.now(timezone.utc)
        with self._lock:
            for sym, price in prices.items():
                if price and price > 0:
                    key = sym.upper()
                    self._prices[key]     = price
                    self._timestamps[key] = now
                    if key not in self._history:
                        self._history[key] = deque(maxlen=self.MAX_HISTORY)
                    self._history[key].append((now, price))

    # ── Read ──────────────────────────────────────────────────────────────────

    def get(self, symbol: str, fallback: bool = True) -> float | None:
        """Returns RAW price. Use convert_price() or get_display() for INR."""
        sym = symbol.upper()
        with self._lock:
            price = self._prices.get(sym)
            ts    = self._timestamps.get(sym)

        if price and price > 0:
            if fallback and ts:
                age = (datetime.now(timezone.utc) - ts).total_seconds()
                if age > self.FALLBACK_REFRESH_SECONDS:
                    fresh = self._yfinance_fallback(sym)
                    if fresh:
                        self.update(sym, fresh)
                        return fresh
            return price

        if fallback:
            price = self._yfinance_fallback(sym)
            if price:
                self.update(sym, price)
            return price
        return None

    def get_display(self, symbol: str, usdinr: float = 92.99) -> float:
        """Returns price converted to INR for display."""
        raw = self.get(symbol)
        if raw is None:
            return 0.0
        return convert_price(symbol, raw, usdinr)

    def get_formatted(self, symbol: str, usdinr: float = 92.99) -> str:
        """Returns fully formatted price string."""
        raw = self.get(symbol)
        if raw is None:
            return "—"
        return format_price_display(symbol, raw, usdinr)

    def get_all(self) -> dict[str, float]:
        """Return all raw prices."""
        with self._lock:
            return {k: v for k, v in self._prices.items() if v and v > 0}

    def get_history(self, symbol: str, n: int = 100) -> list[tuple[datetime, float]]:
        sym = symbol.upper()
        with self._lock:
            h = self._history.get(sym, deque())
            return list(h)[-n:]

    def get_history_display(
        self, symbol: str, usdinr: float, n: int = 100
    ) -> list[tuple[datetime, float]]:
        """History with prices converted to INR — for sparklines."""
        raw = self.get_history(symbol, n)
        return [(ts, convert_price(symbol, px, usdinr)) for ts, px in raw]

    def is_stale(self, symbol: str) -> bool:
        sym = symbol.upper()
        with self._lock:
            ts = self._timestamps.get(sym)
        if not ts:
            return True
        return (datetime.now(timezone.utc) - ts).total_seconds() > self.STALE_SECONDS

    def age_seconds(self, symbol: str) -> float | None:
        sym = symbol.upper()
        with self._lock:
            ts = self._timestamps.get(sym)
        if not ts:
            return None
        return (datetime.now(timezone.utc) - ts).total_seconds()

    def stats(self) -> dict:
        with self._lock:
            n_symbols  = len(self._prices)
            tick_count = self._tick_count
            prices     = dict(self._prices)
            timestamps = dict(self._timestamps)
        stale = sum(
            1 for sym in prices
            if (datetime.now(timezone.utc) - timestamps.get(
                sym, datetime.min.replace(tzinfo=timezone.utc)
            )).total_seconds() > self.STALE_SECONDS
        )
        return {
            "symbols":     n_symbols,
            "total_ticks": tick_count,
            "stale":       stale,
            "fresh":       n_symbols - stale,
        }

    # ── Background refresh ────────────────────────────────────────────────────

    def start_background_refresh(self, symbols: list[str], interval_seconds: int = 3):
        if self._bg_started:
            return
        self._bg_started = True

        def _poll_loop():
            import time
            import yfinance as yf
            from src.data.models import ALL_SYMBOLS

            yf_map: dict[str, str] = {}
            for sym in symbols:
                info   = ALL_SYMBOLS.get(sym)
                yf_sym = info.symbol if info else f"{sym}.NS"
                yf_map[sym.upper()] = yf_sym

            all_yf = list(dict.fromkeys(yf_map.values()))

            while True:
                try:
                    raw = yf.download(
                        all_yf,
                        period      = "1d",
                        interval    = "1m",
                        progress    = False,
                        auto_adjust = True,
                        timeout     = 15,
                    )
                    if not raw.empty:
                        close = raw.get("Close", raw.get("close"))
                        if close is not None:
                            for sym, yf_sym in yf_map.items():
                                try:
                                    if hasattr(close, "columns"):
                                        col = close[yf_sym].dropna() if yf_sym in close.columns else None
                                    else:
                                        col = close.dropna()
                                    if col is not None and not col.empty:
                                        px = float(col.iloc[-1])
                                        if px > 0:
                                            self.update(sym, px)
                                except Exception:
                                    pass
                except Exception:
                    pass
                time.sleep(interval_seconds)

        t = threading.Thread(target=_poll_loop, daemon=True, name="PricePoller")
        t.start()
        logger.info(
            f"PriceStore: background refresh started "
            f"({interval_seconds}s, {len(symbols)} symbols)"
        )

    # ── yfinance fallback ─────────────────────────────────────────────────────

    MCX_SYMBOLS_NO_FALLBACK = {
        "GOLD", "SILVER", "CRUDEOIL", "COPPER",
        "NATURALGAS", "ZINC", "ALUMINIUM"
    }

    def _yfinance_fallback(self, symbol: str) -> float | None:
        if symbol.upper() in self.MCX_SYMBOLS_NO_FALLBACK:
            return None
        try:
            import yfinance as yf
            from src.data.models import ALL_SYMBOLS
            info   = ALL_SYMBOLS.get(symbol)
            yf_sym = info.symbol if info else f"{symbol}.NS"
            px     = yf.Ticker(yf_sym).fast_info.last_price
            return float(px) if px and px > 0 else None
        except Exception:
            return None
        
    def warm_up(self, symbols: list[str]):
        logger.info(f"PriceStore: warming up {len(symbols)} symbols...")
        fetched = 0
        for sym in symbols:
            px = self._yfinance_fallback(sym)
            if px:
                self.update(sym, px)
                fetched += 1
        logger.info(f"PriceStore: warmed up {fetched}/{len(symbols)} symbols")


    def register_tick_callback(self, cb):
        """Register a callable(symbol, price) fired on every update."""
        if not hasattr(self, "_tick_callbacks"):
            self._tick_callbacks = []
        self._tick_callbacks.append(cb)

# Module-level singleton
price_store = PriceStore()