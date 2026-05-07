# trading-agent/src/data/manager.py
from datetime import datetime, timedelta, timezone
from typing import Optional
import pandas as pd
from loguru import logger

from src.data.adapters.yfinance_adapter import YFinanceAdapter
from src.data.store import LocalDataStore
from src.data.models import Interval
from config.settings import settings


class DataManager:
    """
    The single source of truth for all market data.

    Usage:
        dm = DataManager()
        df = dm.get_ohlcv("RELIANCE", Interval.D1, days_back=365)
        price = dm.get_latest_price("NIFTY50")

    Strategy:
        1. Check local cache first (instant)
        2. If stale or missing → fetch from API
        3. Save to cache → return
    """

    def __init__(self):
        self.store   = LocalDataStore()
        self.yf      = YFinanceAdapter()
        self.yf.connect()
        logger.info("DataManager ready")

    def get_ohlcv(
        self,
        symbol: str,
        interval: Interval = Interval.D1,
        days_back: int = 365,
        force_refresh: bool = False,
        asset_type: str = "equity",
    ) -> pd.DataFrame:
        """
        Main method to get historical OHLCV data.

        Parameters
        ----------
        symbol      : short name ("RELIANCE") or yfinance ticker ("RELIANCE.NS")
        interval    : Interval enum (D1, H1, M15, etc.)
        days_back   : how many calendar days of history to fetch
        force_refresh: bypass cache and re-fetch from API
        asset_type  : "equity", "crypto", "index"
        """
        from datetime import timezone
        end = datetime.now(timezone.utc).replace(tzinfo=None)
        start = end - timedelta(days=days_back)

        cache_key = symbol.replace(".", "_").replace("^", "IDX_")

        # --- Try cache first ---
        if not force_refresh:
            cached = self.store.load(cache_key, interval.value, asset_type, start, end)
            if not cached.empty:
                # Cache hit — check if it's recent enough
                latest_cached = cached.index.max()
                staleness_hours = (pd.Timestamp.now("UTC") - latest_cached).total_seconds() / 3600

                # For daily data, 24h stale is fine; for intraday, 1h is the limit
                threshold = 24 if interval in (Interval.D1, Interval.W1) else 1
                if staleness_hours < threshold:
                    logger.info(f"Cache hit: {symbol} ({len(cached)} bars, {staleness_hours:.1f}h old)")
                    return cached

        # --- Fetch from API ---
        logger.info(f"Fetching from API: {symbol} | {interval.value} | {days_back}d")
        df = self.yf.fetch_ohlcv(symbol, interval, start, end)

        if df.empty:
            logger.error(f"No data received for {symbol}")
            return df

        # --- Save to cache ---
        self.store.save(df, cache_key, interval.value, asset_type)

        return df

    def get_multiple(
        self,
        symbols: list[str],
        interval: Interval = Interval.D1,
        days_back: int = 365,
        asset_type: str = "equity",
    ) -> dict[str, pd.DataFrame]:
        """Fetch multiple symbols efficiently."""
        results = {}
        for sym in symbols:
            results[sym] = self.get_ohlcv(sym, interval, days_back, asset_type=asset_type)
        return results

    def get_latest_price(self, symbol: str) -> float:
        """Get current market price."""
        return self.yf.fetch_latest_price(symbol)

    def get_nifty50_constituents_data(
        self,
        interval: Interval = Interval.D1,
        days_back: int = 252,
    ) -> dict[str, pd.DataFrame]:
        """Convenience: fetch all watchlist stocks at once."""
        from src.data.models import NSE_SYMBOLS
        symbols = list(NSE_SYMBOLS.keys())
        logger.info(f"Fetching data for {len(symbols)} NSE symbols")
        return self.yf.fetch_multiple(symbols, interval,
                                    datetime.now(timezone.utc) - timedelta(days=days_back),
                                    datetime.now(timezone.utc))

    def show_cache_summary(self) -> None:
        """Print a table of what's stored locally."""
        files = self.store.list_available()
        if not files:
            logger.info("Cache is empty")
            return
        logger.info(f"{'File':<35} {'Bars':>6} {'From':<12} {'To':<12} {'KB':>6}")
        logger.info("-" * 75)
        for f in files:
            logger.info(f"{f['file']:<35} {f['bars']:>6} {f['from']:<12} {f['to']:<12} {f['size_kb']:>6}")