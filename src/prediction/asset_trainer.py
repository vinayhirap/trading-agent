# trading-agent/src/prediction/asset_trainer.py
"""
Asset-Class Aware Training System

Replaces the uniform training loop in run_pipeline_v2.py with
one that treats each asset class differently:

  EQUITY:    beta/sector features, 5-bar horizon, higher volume weight
  INDEX:     global correlation, 3-bar horizon, overnight gap features
  COMMODITY: FX impact, 7-bar horizon, seasonality features
  CRYPTO:    BTC benchmark, 2-bar horizon, fat-tail features
  FOREX:     rate differential, 5-bar horizon

Each asset class trains its OWN ensemble model.
At inference time, the correct model is selected by asset class.

This gives 4-5 specialised models instead of 1 general model.
Expected accuracy improvement:
  - Equity:    +4-6%   (sector + beta features are strong predictors)
  - Index:     +3-5%   (overnight S&P gap is highly predictive for Nifty)
  - Commodity: +5-8%   (FX + seasonality are huge for MCX)
  - Crypto:    +6-10%  (BTC dominance proxy + fear/greed proxy)

Key design:
  - Does NOT modify existing EnsembleModel or FeatureEngine
  - Saves one model file per asset class: models/ensemble_{asset_class}.pkl
  - Falls back to ensemble_v2.pkl if no asset-specific model exists
  - run_pipeline_v2.py can call train_all() to train everything
"""
import json
from pathlib import Path
from typing import Optional
from loguru import logger

import pandas as pd
import numpy as np

from src.data.models import AssetClass, ALL_SYMBOLS, Interval
from src.features.asset_features import AssetFeatureEngine, ASSET_LABEL_HORIZONS, ASSET_MIN_BARS
from src.prediction.labels import make_labels
from src.prediction.ensemble_model import EnsembleModel
from src.analysis.regime_detector import RegimeDetector

ROOT       = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
DATA_DIR   = ROOT / "data"

# Symbols to train per asset class
TRAINING_SYMBOLS: dict[AssetClass, list[str]] = {
    AssetClass.INDEX: [
        "NIFTY50", "BANKNIFTY", "SENSEX", "NIFTYMID", "NIFTYIT",
    ],
    AssetClass.EQUITY: [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "SBIN", "WIPRO", "AXISBANK", "KOTAKBANK", "LT",
        "BAJFINANCE", "MARUTI", "SUNPHARMA", "BHARTIARTL", "TATAMOTORS",
    ],
    AssetClass.FUTURES: [
        "GOLD", "SILVER", "CRUDEOIL", "COPPER", "NATURALGAS",
    ],
    AssetClass.CRYPTO: [
        "BTC", "ETH", "SOL", "BNB", "XRP",
    ],
}

MODEL_PATHS = {
    ac: MODELS_DIR / f"ensemble_{ac.value}.pkl"
    for ac in AssetClass
}


class AssetTrainer:
    """
    Trains a separate EnsembleModel for each asset class.
    """

    def __init__(
        self,
        days_back:    int  = 730,
        skip_tuning:  bool = True,
        n_trials:     int  = 30,
        dm=None,
    ):
        self.days_back   = days_back
        self.skip_tuning = skip_tuning
        self.n_trials    = n_trials
        self.dm          = dm
        self.afe         = AssetFeatureEngine(interval="1d")
        self.rd          = RegimeDetector()

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def train_all(self) -> dict:
        """
        Train one EnsembleModel per asset class.
        Returns validation results dict.
        """
        if self.dm is None:
            self._init_dm()

        all_results = {}

        for ac, symbols in TRAINING_SYMBOLS.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Training asset class: {ac.value.upper()} ({len(symbols)} symbols)")
            logger.info(f"{'='*60}")

            result = self.train_asset_class(ac, symbols)
            if result:
                all_results[ac.value] = result
            else:
                logger.warning(f"Skipped {ac.value} — insufficient data")

        # Save combined results
        results_path = DATA_DIR / "asset_validation.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"Asset validation results → {results_path}")

        self._print_summary(all_results)
        return all_results

    def train_asset_class(
        self,
        ac:      AssetClass,
        symbols: list[str] = None,
    ) -> Optional[dict]:
        """
        Train a single asset class model.
        Returns validation results or None if insufficient data.
        """
        if self.dm is None:
            self._init_dm()

        symbols  = symbols or TRAINING_SYMBOLS.get(ac, [])
        horizon  = ASSET_LABEL_HORIZONS.get(ac, 5)
        min_bars = ASSET_MIN_BARS.get(ac, 200)

        all_X, all_y, all_regimes = [], [], []

        for sym in symbols:
            logger.info(f"  Processing {sym}...")
            try:
                df = self.dm.get_ohlcv(sym, Interval.D1, days_back=self.days_back)
                if df.empty or len(df) < min_bars:
                    logger.warning(f"  {sym}: only {len(df)} bars — skip")
                    continue

                # Asset-specific features
                featured = self.afe.build_for_asset(
                    df, symbol=sym, asset_class=ac,
                    drop_na=True, dm=self.dm,
                )
                if featured.empty or len(featured) < min_bars:
                    continue

                # Asset-class-correct label horizon
                labels = make_labels(featured["close"], horizon=horizon)
                labels = labels.dropna()
                featured = featured.loc[labels.index]

                if len(featured) < min_bars:
                    continue

                # Regime features
                featured = self.rd.add_regime_features(featured)

                # Regime series
                regime_series = self._get_regime_series(featured)

                # Feature columns only
                ohlcv_cols   = {"open", "high", "low", "close", "volume"}
                feature_cols = [c for c in featured.columns if c not in ohlcv_cols]
                X            = featured[feature_cols].fillna(0)

                all_X.append(X)
                all_y.append(labels)
                all_regimes.append(regime_series)

                logger.info(f"  {sym}: {len(X)} bars | {X.shape[1]} features | horizon={horizon}")

            except Exception as e:
                logger.error(f"  {sym} failed: {e}")
                continue

        if not all_X:
            logger.warning(f"No data for {ac.value} — skipping")
            return None

        X_all       = pd.concat(all_X,       axis=0).fillna(0)
        y_all       = pd.concat(all_y,       axis=0)
        regimes_all = pd.concat(all_regimes, axis=0)

        # Align indices
        common      = X_all.index.intersection(y_all.index)
        X_all       = X_all.loc[common]
        y_all       = y_all.loc[common]
        regimes_all = regimes_all.loc[common]

        logger.info(
            f"\n{ac.value.upper()} dataset: "
            f"{len(X_all)} bars | {X_all.shape[1]} features"
        )

        # Optuna tuning (optional)
        tuned_params = {}
        if not self.skip_tuning:
            tuned_params = self._tune(X_all, y_all, regimes_all, ac)

        # Train ensemble
        logger.info(f"Training {ac.value} ensemble...")
        ensemble = EnsembleModel(tuned_params=tuned_params)
        ensemble.fit(X_all, y_all, regime_series=regimes_all)

        # Walk-forward validation
        logger.info(f"Validating {ac.value}...")
        wf = ensemble.walk_forward_validate(
            X_all, y_all,
            regime_series = regimes_all,
            train_months  = 9,
            test_months   = 1,
        )

        # Save model
        model_path = MODEL_PATHS[ac]
        ensemble.save(model_path)
        logger.info(f"Saved {ac.value} model → {model_path}")

        result = {
            "asset_class":      ac.value,
            "symbols_trained":  symbols,
            "n_bars":           len(X_all),
            "n_features":       X_all.shape[1],
            "label_horizon":    horizon,
            "overall_accuracy": wf["overall_accuracy"],
            "n_folds":          wf["n_folds"],
            "regime_accuracy":  wf["regime_accuracy"],
            "fold_results":     wf["fold_results"],
        }

        logger.info(
            f"{ac.value.upper()} accuracy: "
            f"{wf['overall_accuracy']:.1f}% over {wf['n_folds']} folds"
        )
        return result

    def load_model(self, ac: AssetClass) -> Optional[EnsembleModel]:
        """
        Load trained model for an asset class.
        Falls back to ensemble_v2.pkl if not found.
        """
        path = MODEL_PATHS.get(ac)
        if path and path.exists():
            em = EnsembleModel()
            em.load(path)
            return em

        # Fallback to global model
        fallback = MODELS_DIR / "ensemble_v2.pkl"
        if fallback.exists():
            logger.info(f"No {ac.value} model — using global fallback")
            em = EnsembleModel()
            em.load(fallback)
            return em

        return None

    def get_model_for_symbol(self, symbol: str) -> Optional[EnsembleModel]:
        """
        Convenience: resolve symbol → asset class → load correct model.
        """
        info = ALL_SYMBOLS.get(symbol)
        ac   = info.asset_class if info else AssetClass.EQUITY
        return self.load_model(ac)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _init_dm(self):
        from src.data.manager import DataManager
        self.dm = DataManager()

    def _get_regime_series(self, featured: pd.DataFrame) -> pd.Series:
        from src.analysis.regime_detector import Regime
        label_map = {0: Regime.RANGING, 1: Regime.VOLATILE,
                     2: Regime.TRENDING_DOWN, 3: Regime.TRENDING_UP}
        if "regime_label" in featured.columns:
            return featured["regime_label"].map(label_map)
        # Fallback: compute on the fly
        return self.rd.detect_series(featured)

    def _tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        regimes: pd.Series,
        ac: AssetClass,
    ) -> dict:
        try:
            from src.prediction.tuner import RegimeTuner
            tuner = RegimeTuner(
                n_trials    = self.n_trials,
                train_months= 9,
                test_months = 1,
            )
            return tuner.tune(X, y, regime_series=regimes)
        except Exception as e:
            logger.warning(f"Tuning failed for {ac.value}: {e}")
            return {}

    def _print_summary(self, results: dict):
        logger.info("\n" + "="*60)
        logger.info("ASSET-CLASS TRAINING SUMMARY")
        logger.info("="*60)
        for ac_name, r in results.items():
            logger.info(
                f"{ac_name.upper():12} | "
                f"acc={r['overall_accuracy']:5.1f}% | "
                f"folds={r['n_folds']} | "
                f"horizon={r['label_horizon']}d | "
                f"bars={r['n_bars']:,}"
            )
        logger.info("="*60)


class AssetModelRouter:
    """
    Routes inference requests to the correct asset-class model.

    Used by the dashboard Signal Scanner and AI Insights page
    instead of loading one global model for everything.

    Usage:
        router = AssetModelRouter()
        result = router.predict(symbol="RELIANCE", features=feat_series)
    """

    def __init__(self):
        self._trainer = AssetTrainer()
        self._cache: dict[AssetClass, Optional[EnsembleModel]] = {}
        self._rd     = RegimeDetector()
        self._afe    = AssetFeatureEngine()

    def predict(
        self,
        symbol:   str,
        features: pd.Series,
    ):
        """
        Route to correct model and predict.
        Returns EnsemblePrediction or None.
        """
        info = ALL_SYMBOLS.get(symbol)
        ac   = info.asset_class if info else AssetClass.EQUITY

        model = self._get_model(ac)
        if model is None:
            return None

        try:
            regime_result = self._rd.detect(features)
            # Filter to model's feature columns
            feat_filtered = features.reindex(model.feature_cols).fillna(0)
            return model.predict_latest(feat_filtered, regime_result=regime_result)
        except Exception as e:
            logger.warning(f"AssetModelRouter.predict({symbol}): {e}")
            return None

    def get_asset_class(self, symbol: str) -> AssetClass:
        info = ALL_SYMBOLS.get(symbol)
        return info.asset_class if info else AssetClass.EQUITY

    def _get_model(self, ac: AssetClass) -> Optional[EnsembleModel]:
        if ac not in self._cache:
            self._cache[ac] = self._trainer.load_model(ac)
        return self._cache[ac]


# Module-level singleton
asset_model_router = AssetModelRouter()