# trading-agent/src/prediction/pipeline.py
"""
End-to-end prediction pipeline.
This is what you call to get a trading signal for any symbol.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

from src.data.manager import DataManager
from src.data.models import Interval
from src.features.feature_engine import FeatureEngine
from src.prediction.labels import make_labels, Signal
from src.prediction.model import TradingModel, PredictionResult, WalkForwardResult
from config.settings import settings


OHLCV_COLS = {"open", "high", "low", "close", "volume"}


class PredictionPipeline:
    """
    One object to rule them all for predictions.
    """

    MODEL_DIR = Path("models")

    def __init__(
        self,
        symbol: str,
        interval: Interval = Interval.D1,
        horizon: int = 5,
        min_confidence: float = 0.55,
    ):
        self.symbol         = symbol
        self.interval       = interval
        self.horizon        = horizon
        self.min_confidence = min_confidence

        self.dm      = DataManager()
        self.engine  = FeatureEngine(interval=interval.value)
        self.model   = TradingModel(min_confidence=min_confidence)

        self.MODEL_DIR.mkdir(exist_ok=True)
        self._model_path = self.MODEL_DIR / f"{symbol}_{interval.value}_h{horizon}.pkl"

    def _prepare_data(
            self,
            days_back: int = 730,
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
            """
            UPGRADED: Now uses advanced features for better accuracy.
            Fetch → clean → base features → advanced features → labels
            """
            from src.features.advanced_features import build_advanced_features

            raw = self.dm.get_ohlcv(
                self.symbol, self.interval,
                days_back=days_back, asset_type="equity"
            )

            if raw.empty or len(raw) < 200:
                raise ValueError(f"Insufficient data for {self.symbol}: {len(raw)} bars")

            # Step 1: Base features (existing)
            featured = self.engine.build(raw, drop_na=True)

            # Step 2: Advanced features (NEW — regime, momentum, gaps, patterns, structure)
            featured = build_advanced_features(featured)

            # Step 3: Labels
            labels = make_labels(
                featured["close"],
                horizon=self.horizon,
            )

            feature_cols = [c for c in featured.columns if c not in OHLCV_COLS]
            X = featured[feature_cols].copy()
            y = labels.copy()

            # Clean
            X = X.replace([np.inf, -np.inf], np.nan)
            valid_mask = ~(X.isna().any(axis=1) | y.isna())
            X = X.loc[valid_mask]
            y = y.loc[valid_mask]

            logger.info(
                f"{self.symbol}: {len(raw)} raw → {len(X)} samples | "
                f"{len(feature_cols)} features | "
                f"BUY:{(y==1).sum()} HOLD:{(y==0).sum()} SELL:{(y==-1).sum()}"
            )
            return raw, X, y

    def train_and_validate(
        self,
        days_back: int = 730,
        train_months: int = 12,
        test_months: int = 1,
    ) -> WalkForwardResult:
        """
        Full training + walk-forward validation pipeline.
        """
        logger.info(f"Training pipeline: {self.symbol} | {days_back}d history")

        _, X, y = self._prepare_data(days_back)

        result = self.model.walk_forward_validate(
            X, y,
            train_months=train_months,
            test_months=test_months,
        )

        # Train final model on all data
        feature_cols = list(X.columns)

        val_size = max(20, len(X) // 5)

        self.model.fit(
            X.iloc[:-val_size], y.iloc[:-val_size],
            X.iloc[-val_size:], y.iloc[-val_size:],
        )

        self.model.save(self._model_path)

        return result

    def get_signal(self, days_back: int = 300) -> PredictionResult:
        """
        Generate a live signal for the latest bar.
        """
        if self.model.model is None:
            if self._model_path.exists():
                self.model.load(self._model_path)
            else:
                raise RuntimeError(
                    f"No trained model found for {self.symbol}. "
                    f"Run train_and_validate() first."
                )

        raw = self.dm.get_ohlcv(self.symbol, self.interval, days_back=days_back)

        # Latest features (DO NOT drop NaN here)
        latest = self.engine.latest_features(raw)

        # Keep only trained columns
        feature_series = latest[self.model.feature_cols]

        # Safety: remove inf just in case
        feature_series = feature_series.replace([np.inf, -np.inf], np.nan).fillna(0)

        return self.model.predict_latest(feature_series)

    def load_model(self) -> None:
        self.model.load(self._model_path)

    def print_feature_importance(self, top_n: int = 10) -> None:
        imp = self.model.feature_importance(top_n)

        print(f"\nTop {top_n} features for {self.symbol}:")
        print(f"{'Feature':<20} {'Importance':>12}")
        print("-" * 35)

        for _, row in imp.iterrows():
            print(f"{row['feature']:<20} {row['importance_pct']:>11.1%}")

    def train_ensemble(
        self,
        days_back: int = 1095,   # 3 years for better accuracy
        train_months: int = 12,
        test_months: int  = 1,
    ) -> dict:
        """
        Train the ensemble model (XGBoost + LightGBM) for better accuracy.
        Use this instead of train_and_validate for production.
        """
        from src.prediction.ensemble_model import EnsembleModel

        logger.info(f"Training ensemble for {self.symbol} | {days_back}d history")
        _, X, y = self._prepare_data(days_back)

        self._ensemble = EnsembleModel(min_confidence=self.min_confidence)
        result = self._ensemble.walk_forward_validate(X, y, train_months, test_months)

        # Train final ensemble on all data
        val_size = max(20, len(X) // 5)
        self._ensemble.fit(
            X.iloc[:-val_size], y.iloc[:-val_size],
            X.iloc[-val_size:], y.iloc[-val_size:],
        )

        ensemble_path = self.MODEL_DIR / f"{self.symbol}_{self.interval.value}_ensemble.pkl"
        self._ensemble.save(ensemble_path)
        logger.info(f"Ensemble accuracy: {result['accuracy']:.1%} over {result['n_folds']} folds")
        return result

    def get_ensemble_signal(self, days_back: int = 300):
        """Get signal from ensemble model."""
        from src.prediction.ensemble_model import EnsembleModel

        ensemble_path = self.MODEL_DIR / f"{self.symbol}_{self.interval.value}_ensemble.pkl"

        if not hasattr(self, "_ensemble") or self._ensemble is None:
            self._ensemble = EnsembleModel(min_confidence=self.min_confidence)
            if ensemble_path.exists():
                self._ensemble.load(ensemble_path)
            else:
                raise RuntimeError(f"No ensemble model for {self.symbol}. Run train_ensemble() first.")

        raw = self.dm.get_ohlcv(self.symbol, self.interval, days_back=days_back)

        from src.features.advanced_features import build_advanced_features
        featured = self.engine.build(raw, drop_na=False)
        featured = build_advanced_features(featured)

        ohlcv = {"open","high","low","close","volume"}
        feature_cols = [c for c in featured.columns if c not in ohlcv]
        latest = featured[feature_cols].iloc[-1]
        latest = latest.replace([float("inf"), float("-inf")], 0).fillna(0)

        # Detect regime for ensemble weighting
        atr_ratio = float(featured.get("vol_regime", pd.Series([1.0])).iloc[-1])
        adx_val   = float(featured.get("adx", pd.Series([20.0])).iloc[-1])
        if adx_val > 25 and atr_ratio < 1.5:
            regime = "trending"
        elif atr_ratio > 1.5:
            regime = "high_vol"
        else:
            regime = "normal"

        return self._ensemble.predict_latest(latest, regime=regime)