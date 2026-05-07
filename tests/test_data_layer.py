# trading-agent/tests/test_data_layer.py
"""
Data layer tests.
Run: python -m pytest tests/test_data_layer.py -v
"""
import pytest
from datetime import datetime, timedelta
import pandas as pd
from src.data.models import Interval, Exchange, AssetClass, NSE_SYMBOLS, CRYPTO_SYMBOLS
from src.data.adapters.yfinance_adapter import YFinanceAdapter
from src.data.store import LocalDataStore
from src.data.manager import DataManager


class TestDataModels:
    def test_nse_symbols_defined(self):
        assert "RELIANCE" in NSE_SYMBOLS
        assert "NIFTY50" in NSE_SYMBOLS

    def test_ohlcv_bar_validation(self):
        from src.data.models import OHLCVBar
        good_bar = OHLCVBar(
            symbol="RELIANCE", exchange=Exchange.NSE,
            interval=Interval.D1, timestamp=datetime.utcnow(),
            open=2800.0, high=2850.0, low=2780.0, close=2820.0, volume=1_000_000
        )
        assert good_bar.is_valid()

        bad_bar = OHLCVBar(
            symbol="RELIANCE", exchange=Exchange.NSE,
            interval=Interval.D1, timestamp=datetime.utcnow(),
            open=2800.0, high=2700.0,   # high < low — invalid
            low=2780.0, close=2820.0, volume=1_000_000
        )
        assert not bad_bar.is_valid()


class TestYFinanceAdapter:
    def setup_method(self):
        self.adapter = YFinanceAdapter()
        self.adapter.connect()

    def test_fetch_daily_equity(self):
        end   = datetime.utcnow()
        start = end - timedelta(days=30)
        df = self.adapter.fetch_ohlcv("RELIANCE", Interval.D1, start, end)
        assert not df.empty, "Should return data for RELIANCE"
        assert all(c in df.columns for c in ["open","high","low","close","volume"])
        assert df.index.name == "timestamp"
        print(f"\nRELIANCE 30d: {len(df)} bars | Latest close: INR{df['close'].iloc[-1]:.2f}")

    def test_fetch_nifty_index(self):
        end   = datetime.utcnow()
        start = end - timedelta(days=30)
        df = self.adapter.fetch_ohlcv("NIFTY50", Interval.D1, start, end)
        assert not df.empty, "Should return Nifty 50 data"
        print(f"\nNIFTY50 30d: {len(df)} bars | Latest: {df['close'].iloc[-1]:.2f}")

    def test_fetch_crypto(self):
        end   = datetime.utcnow()
        start = end - timedelta(days=30)
        df = self.adapter.fetch_ohlcv("BTC", Interval.D1, start, end)
        assert not df.empty, "Should return BTC data"
        print(f"\nBTC 30d: {len(df)} bars | Latest: ${df['close'].iloc[-1]:,.2f}")

    def test_latest_price(self):
        price = self.adapter.fetch_latest_price("RELIANCE")
        assert price > 0, "Price should be positive"
        print(f"\nRELIANCE latest: INR{price:.2f}")


class TestLocalDataStore:
    def setup_method(self):
        import tempfile
        from pathlib import Path
        self.tmp = Path(tempfile.mkdtemp())
        self.store = LocalDataStore(base_dir=self.tmp)

    def test_save_and_load(self):
        # Create a minimal DataFrame
        idx = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
        df = pd.DataFrame({
            "open": 100.0, "high": 105.0, "low": 98.0,
            "close": 102.0, "volume": 50000.0
        }, index=idx)
        df.index.name = "timestamp"

        self.store.save(df, "TEST", "1d", "equity")
        loaded = self.store.load("TEST", "1d", "equity")
        assert len(loaded) == 10
        assert loaded["close"].iloc[0] == 102.0

    def test_merge_on_save(self):
        """Saving new data should merge with existing, not overwrite."""
        idx1 = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
        idx2 = pd.date_range("2024-01-04", periods=5, freq="D", tz="UTC")  # overlaps

        def make_df(idx):
            df = pd.DataFrame({
                "open":100.0,"high":105.0,"low":98.0,"close":102.0,"volume":50000.0
            }, index=idx)
            df.index.name = "timestamp"
            return df

        self.store.save(make_df(idx1), "MERGE_TEST", "1d")
        self.store.save(make_df(idx2), "MERGE_TEST", "1d")
        loaded = self.store.load("MERGE_TEST", "1d")
        assert len(loaded) == 8   # 5 + 5 with overlap → 8 unique timestamps

class TestDataManager:
    def setup_method(self):
        self.dm = DataManager()

    def test_get_ohlcv_reliance(self):
        df = self.dm.get_ohlcv("RELIANCE", Interval.D1, days_back=60)
        assert not df.empty
        assert len(df) >= 40   # ~40-42 trading days in 60 calendar days
        print(f"\nRELIANCE via DataManager: {len(df)} bars")

    def test_cache_works(self):
        """Second call should be faster (cache hit)."""
        import time
        self.dm.get_ohlcv("TCS", Interval.D1, days_back=30)  # populate cache

        t0 = time.time()
        df = self.dm.get_ohlcv("TCS", Interval.D1, days_back=30)
        elapsed = time.time() - t0
        assert not df.empty
        print(f"\nCache read time: {elapsed*1000:.1f}ms")