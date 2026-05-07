# trading-agent/tests/test_prediction.py
"""
Run: python -m pytest tests/test_prediction.py -v -s
"""
import pytest
import numpy as np
import pandas as pd
from datetime import timezone
from src.prediction.labels import make_labels, Signal
from src.prediction.model import TradingModel, PredictionResult
from src.features.feature_engine import FeatureEngine


def make_ohlcv(n: int = 500) -> pd.DataFrame:
    np.random.seed(42)
    close = 1000 + np.cumsum(np.random.randn(n) * 10)
    close = np.maximum(close, 100)
    high  = close + np.abs(np.random.randn(n) * 5)
    low   = close - np.abs(np.random.randn(n) * 5)
    open_ = np.clip(close + np.random.randn(n) * 3, low, high)
    vol   = np.abs(np.random.randn(n) * 1e6 + 2e6)
    idx   = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    df    = pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": vol,
    }, index=idx)
    df.index.name = "timestamp"
    return df


class TestLabels:
    def test_label_distribution(self):
        df = make_ohlcv(500)
        labels = make_labels(df["close"], horizon=5)
        valid  = labels.dropna()
        buy_pct  = (valid == Signal.BUY).mean()
        sell_pct = (valid == Signal.SELL).mean()
        hold_pct = (valid == Signal.HOLD).mean()
        print(f"\nLabel distribution: BUY={buy_pct:.1%} HOLD={hold_pct:.1%} SELL={sell_pct:.1%}")
        assert abs(buy_pct  - 0.40) < 0.05, "BUY should be ~40%"
        assert abs(sell_pct - 0.40) < 0.05, "SELL should be ~40%"

    def test_no_future_leak(self):
        df = make_ohlcv(100)
        labels = make_labels(df["close"], horizon=5)
        # Last 5 bars must be NaN
        assert labels.iloc[-5:].isna().all(), "Last horizon bars must be NaN"
        # First bars must have labels
        assert not labels.iloc[:-5].isna().any(), "All other bars must have labels"

    def test_signal_values(self):
        df = make_ohlcv(200)
        labels = make_labels(df["close"])
        valid  = labels.dropna()
        assert set(valid.unique()).issubset({-1, 0, 1})


class TestTradingModel:
    def setup_method(self):
        df = make_ohlcv(600)
        engine = FeatureEngine()
        featured = engine.build(df, drop_na=False)
        ohlcv = {"open","high","low","close","volume"}
        feature_cols = [c for c in featured.columns if c not in ohlcv]
        labels = make_labels(featured["close"], horizon=5)
        valid  = ~(featured[feature_cols].isna().any(axis=1) | labels.isna())
        self.X = featured[feature_cols][valid]
        self.y = labels[valid]
        self.model = TradingModel(min_confidence=0.40)

    def test_fit_predict(self):
        split = int(len(self.X) * 0.8)
        X_tr, X_te = self.X.iloc[:split], self.X.iloc[split:]
        y_tr, y_te = self.y.iloc[:split], self.y.iloc[split:]
        self.model.fit(X_tr, y_tr, X_te, y_te)
        preds = self.model.predict(X_te)
        assert len(preds) == len(X_te)
        assert set(preds.unique()).issubset({-1, 0, 1})

    def test_predict_proba_sums_to_one(self):
        split = int(len(self.X) * 0.8)
        self.model.fit(self.X.iloc[:split], self.y.iloc[:split])
        probs = self.model.predict_proba(self.X.iloc[split:])
        row_sums = probs.sum(axis=1)
        assert (row_sums.between(0.999, 1.001)).all(), "Probabilities must sum to 1"

    def test_predict_latest(self):
        split = int(len(self.X) * 0.8)
        self.model.fit(self.X.iloc[:split], self.y.iloc[:split])
        result = self.model.predict_latest(self.X.iloc[-1])
        assert isinstance(result, PredictionResult)
        assert result.signal in (Signal.BUY, Signal.HOLD, Signal.SELL)
        assert 0 <= result.confidence <= 1
        print(f"\nLatest prediction: {result}")

    def test_walk_forward(self):
        result = self.model.walk_forward_validate(
            self.X, self.y,
            train_months=6,
            test_months=1,
            min_train_bars=100,
        )
        print(f"\nWalk-forward accuracy: {result.accuracy:.1%}")
        print(f"Folds completed: {len(result.fold_results)}")
        print(f"\n{result.report}")
        assert result.accuracy > 0.30, "Accuracy should beat random (33%)"
        assert len(result.fold_results) >= 2

    def test_feature_importance(self):
        split = int(len(self.X) * 0.8)
        self.model.fit(self.X.iloc[:split], self.y.iloc[:split])
        imp = self.model.feature_importance(top_n=5)
        assert len(imp) <= 5
        assert "feature" in imp.columns
        print(f"\nTop 5 features:\n{imp.to_string(index=False)}")

    def test_save_and_load(self, tmp_path):
        split = int(len(self.X) * 0.8)
        self.model.fit(self.X.iloc[:split], self.y.iloc[:split])
        path = tmp_path / "test_model.pkl"
        self.model.save(path)
        loaded = TradingModel().load(path)
        preds_orig   = self.model.predict(self.X.iloc[split:])
        preds_loaded = loaded.predict(self.X.iloc[split:])
        assert (preds_orig == preds_loaded).all(), "Loaded model must give identical predictions"