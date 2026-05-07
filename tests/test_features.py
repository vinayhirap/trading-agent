# trading-agent/tests/test_features.py
"""
Run: python -m pytest tests/test_features.py -v -s
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from src.utils.market_hours import MarketHours, MarketSession, IST
from src.features.indicators import ema, rsi, macd, bollinger_bands, atr, vwap
from src.features.feature_engine import FeatureEngine


# ── Market Hours Tests ───────────────────────────────────────────────────────

class TestMarketHours:
    def setup_method(self):
        self.mh = MarketHours()

    def _make_ist(self, hour, minute, weekday_offset=0):
        """Helper: create an IST datetime for a known trading day."""
        from datetime import date
        from zoneinfo import ZoneInfo
        # Use a known Monday (2026-03-30) as base
        base = datetime(2026, 3, 30, hour, minute, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
        return base + timedelta(days=weekday_offset)

    def test_market_open_hours(self):
        dt = self._make_ist(10, 30)   # 10:30 IST Monday
        assert self.mh.is_tradeable(dt), "10:30 on a weekday should be tradeable"

    def test_pre_open_not_tradeable(self):
        dt = self._make_ist(9, 5)    # 9:05 IST — pre-open
        assert not self.mh.is_tradeable(dt)

    def test_after_close_not_tradeable(self):
        dt = self._make_ist(16, 0)   # 4 PM IST
        assert not self.mh.is_tradeable(dt)

    def test_saturday_not_tradeable(self):
        dt = self._make_ist(10, 30, weekday_offset=5)   # Saturday
        assert not self.mh.is_tradeable(dt)

    def test_seconds_to_close_during_session(self):
        dt = self._make_ist(14, 30)   # 2:30 PM = 60 min to close
        secs = self.mh.seconds_to_close(dt)
        assert 3500 <= secs <= 3601, f"Expected ~3600s, got {secs}"

    def test_format_status_runs(self):
        status = self.mh.format_status()
        assert isinstance(status, str)
        assert "NSE" in status
        print(f"\nMarket status: {status}")

    def test_get_status_dict(self):
        s = self.mh.get_status()
        assert "session" in s
        assert "tradeable" in s
        assert "ist_time" in s
        print(f"\nFull status: {s}")


# ── Indicator Tests ──────────────────────────────────────────────────────────

def make_ohlcv(n: int = 300) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    close = 1000 + np.cumsum(np.random.randn(n) * 10)
    close = np.maximum(close, 100)   # no negative prices
    high  = close + np.abs(np.random.randn(n) * 5)
    low   = close - np.abs(np.random.randn(n) * 5)
    open_ = close + np.random.randn(n) * 3
    open_ = np.clip(open_, low, high)
    vol   = np.abs(np.random.randn(n) * 1e6 + 2e6)

    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    df  = pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": vol,
    }, index=idx)
    df.index.name = "timestamp"
    return df


class TestIndicators:
    def setup_method(self):
        self.df = make_ohlcv(300)

    def test_ema_values(self):
        e = ema(self.df["close"], 21)
        assert len(e) == len(self.df)
        assert not e.iloc[21:].isna().any(), "EMA should have no NaN after warmup"

    def test_rsi_bounds(self):
        r = rsi(self.df["close"], 14)
        valid = r.dropna()
        assert (valid >= 0).all() and (valid <= 100).all(), "RSI must be 0-100"
        print(f"\nRSI range: {valid.min():.1f} – {valid.max():.1f}")

    def test_macd_columns(self):
        m = macd(self.df["close"])
        assert all(c in m.columns for c in ["macd", "signal", "histogram"])

    def test_bollinger_bands(self):
        bb = bollinger_bands(self.df["close"])
        valid = bb.dropna()
        assert (valid["bb_upper"] >= valid["bb_lower"]).all()
        within = (
            (self.df["close"] >= bb["bb_lower"]) &
            (self.df["close"] <= bb["bb_upper"])
        ).dropna().mean()
        assert within > 0.80, f"Price inside bands: {within:.1%}"
        print(f"\nPrice inside Bollinger Bands: {within:.0%}")

    def test_atr_positive(self):
        a = atr(self.df)
        assert (a.dropna() > 0).all(), "ATR must always be positive"


class TestFeatureEngine:
    def setup_method(self):
        self.df = make_ohlcv(300)
        self.engine = FeatureEngine(interval="1d")

    def test_build_returns_more_columns(self):
        featured = self.engine.build(self.df)
        assert len(featured.columns) > len(self.df.columns)
        ohlcv = {"open","high","low","close","volume"}
        feature_cols = [c for c in featured.columns if c not in ohlcv]
        print(f"\nFeature count: {len(feature_cols)}")
        print(f"Features: {feature_cols[:10]}...")   # first 10

    def test_no_lookahead(self):
        """Features at bar t must not use data from bars t+1, t+2, ..."""
        featured = self.engine.build(self.df, drop_na=False)
        # If we shuffle the last 10 rows and rebuild, early features must be identical
        df_shuffled = self.df.copy()
        df_shuffled.iloc[-10:] = df_shuffled.iloc[-10:].sample(frac=1, random_state=99).values
        featured_shuffled = self.engine.build(df_shuffled, drop_na=False)
        # First 280 rows should be identical
        assert (featured.iloc[:280]["rsi_14"].dropna() ==
                featured_shuffled.iloc[:280]["rsi_14"].dropna()).all()

    def test_no_inf_values(self):
        featured = self.engine.build(self.df)
        inf_cols = [c for c in featured.columns if np.isinf(featured[c]).any()]
        assert len(inf_cols) == 0, f"Inf values in: {inf_cols}"

    def test_latest_features(self):
        latest = self.engine.latest_features(self.df)
        assert isinstance(latest, pd.Series)
        assert "rsi_14" in latest.index
        print(f"\nLatest RSI-14: {latest['rsi_14']:.1f}")
        print(f"Latest ATR%: {latest['atr_pct']:.3f}")
        print(f"Latest BB %B: {latest['bb_pct_b']:.2f}")