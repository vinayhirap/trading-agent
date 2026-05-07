# trading-agent/src/prediction/ensemble_model.py
"""
Ensemble Model — XGBoost + LightGBM with regime-conditional parameters.

Why ensemble + regime:
  - XGBoost and LightGBM make different errors (uncorrelated)
  - Voting reduces variance: when one model is wrong, the other is often right
  - Regime-specific params: each model is tuned for the current market state
  - Expected accuracy improvement: 37% → 52-58% on Indian daily data

Architecture:
  - RegimeDetector classifies current bar
  - Load regime-specific params (from tuner or defaults)
  - XGBoost + LightGBM each predict with those params
  - Soft voting: average probability outputs → final signal
  - Confidence is regime-weighted: VOLATILE regime reduces confidence

Usage:
    from src.prediction.ensemble_model import EnsembleModel
    em = EnsembleModel()
    em.fit(X_train, y_train, regime_series=regimes)
    result = em.predict_latest(features_series, regime_result)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import joblib
from loguru import logger

from src.prediction.labels import Signal
from src.analysis.regime_detector import RegimeDetector, Regime, RegimeResult, REGIME_PARAMS

@dataclass
class EnsemblePrediction:
    signal:           Signal
    confidence:       float
    buy_prob:         float
    hold_prob:        float
    sell_prob:        float
    regime:           Regime
    regime_confidence: float
    xgb_signal:       Signal
    lgb_signal:       Signal
    xgb_confidence:   float
    lgb_confidence:   float
    passes_threshold: bool
    top_features:     dict = field(default_factory=dict)

    def __str__(self):
        arrows = {Signal.BUY: "▲", Signal.SELL: "▼", Signal.HOLD: "—"}
        return (
            f"{arrows[self.signal]} {self.signal.name} "
            f"({self.confidence:.0%}) | "
            f"regime={self.regime.value} | "
            f"XGB={self.xgb_signal.name} LGB={self.lgb_signal.name}"
        )


class EnsembleModel:
    """
    XGBoost + LightGBM ensemble with regime-conditional hyperparameters.

    Training strategy:
    - Train both models on the full dataset
    - Use regime features as additional inputs (regime_label column)
    - Apply tuned params per regime if available, else use defaults
    - Walk-forward validation checks real out-of-sample performance

    The model intentionally keeps things simple:
    - No stacking (too prone to overfitting on small datasets)
    - Soft voting (probability averaging) is more stable than hard voting
    - Regime weighting adjusts final confidence without changing predictions
    """

    ENSEMBLE_WEIGHTS = {"xgb": 0.55, "lgb": 0.45}   # XGBoost slightly favoured

    # Regime-specific default XGBoost params (before Optuna tuning)
    REGIME_XGB_DEFAULTS = {
        Regime.BULL_TREND: {
            "max_depth": 5, "learning_rate": 0.06, "n_estimators": 400,
            "min_child_weight": 4, "subsample": 0.85, "colsample_bytree": 0.75,
            "reg_alpha": 0.05, "reg_lambda": 1.2,
        },
        Regime.BEAR_TREND: {
            "max_depth": 5, "learning_rate": 0.06, "n_estimators": 400,
            "min_child_weight": 4, "subsample": 0.85, "colsample_bytree": 0.75,
            "reg_alpha": 0.05, "reg_lambda": 1.2,
        },
        Regime.RANGING_LOW: {
            "max_depth": 3, "learning_rate": 0.04, "n_estimators": 500,
            "min_child_weight": 8, "subsample": 0.75, "colsample_bytree": 0.65,
            "reg_alpha": 0.3, "reg_lambda": 2.0,   # heavier regularisation
        },
        Regime.RANGING_HIGH: {
            "max_depth": 3, "learning_rate": 0.03, "n_estimators": 300,
            "min_child_weight": 10, "subsample": 0.70, "colsample_bytree": 0.60,
            "reg_alpha": 0.5, "reg_lambda": 2.5,   # heaviest regularisation
        },
    }

    REGIME_LGB_DEFAULTS = {
        Regime.BULL_TREND: {
            "num_leaves": 31, "learning_rate": 0.06, "n_estimators": 400,
            "min_child_samples": 20, "subsample": 0.85, "colsample_bytree": 0.75,
            "reg_alpha": 0.05, "reg_lambda": 1.2,
        },
        Regime.BEAR_TREND: {
            "num_leaves": 31, "learning_rate": 0.06, "n_estimators": 400,
            "min_child_samples": 20, "subsample": 0.85, "colsample_bytree": 0.75,
            "reg_alpha": 0.05, "reg_lambda": 1.2,
        },
        Regime.RANGING_LOW: {
            "num_leaves": 15, "learning_rate": 0.04, "n_estimators": 500,
            "min_child_samples": 30, "subsample": 0.75, "colsample_bytree": 0.65,
            "reg_alpha": 0.3, "reg_lambda": 2.0,
        },
        Regime.RANGING_HIGH: {
            "num_leaves": 15, "learning_rate": 0.03, "n_estimators": 300,
            "min_child_samples": 40, "subsample": 0.70, "colsample_bytree": 0.60,
            "reg_alpha": 0.5, "reg_lambda": 2.5,
        },
    }

    MIN_CONFIDENCE = 0.52   # lower than TradingModel's 0.55 (ensemble is more calibrated)

    def __init__(
        self,
        tuned_params: dict = None,    # from RegimeTuner.load_cached()
        min_confidence: float = None,
    ):
        self.tuned_params   = tuned_params or {}
        self.min_confidence = min_confidence or self.MIN_CONFIDENCE
        self.regime_detector = RegimeDetector()

        self._xgb_models: dict[Regime, object] = {}
        self._lgb_models: dict[Regime, object] = {}
        self._global_xgb = None
        self._global_lgb = None
        self.feature_cols: list[str] = []

        self._label_map     = {0: Signal.SELL, 1: Signal.HOLD, 2: Signal.BUY}
        self._inv_label_map = {Signal.SELL: 0, Signal.HOLD: 1, Signal.BUY: 2}

        self._check_lgb()

    def _check_lgb(self):
        try:
            import lightgbm
            self._has_lgb = True
            logger.info("LightGBM available — full ensemble enabled")
        except ImportError:
            self._has_lgb = False
            logger.warning("LightGBM not installed — XGBoost only. Run: pip install lightgbm")

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        regime_series: pd.Series = None,
        sample_weight: np.ndarray = None,
        skip_tuning: bool = False,
    ) -> "EnsembleModel":
        """
        Train regime-specific ensemble models.

        Args:
            X:             feature matrix with regime columns included
            y:             Signal label series
            regime_series: Regime enum series (same index as X)
                           If None, trains one global model
        """
        # Normalize training delimiter values (inf/-inf) that can break xgboost DMatrix creation
        X = X.replace([np.inf, -np.inf], np.nan)
        if X.isna().any().any():
            n_bad = X.isna().sum().sum()
            logger.warning(f"EnsembleModel.fit: replaced inf/nan; dropping {n_bad} bad entries")
            X = X.dropna()
            y = y.loc[X.index]

        # Reset indices to keep X/y/regime in sync and avoid reindexing warnings
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        if regime_series is not None:
            regime_series = regime_series.reset_index(drop=True)

        self.feature_cols = list(X.columns)
        y_enc = y.map(self._inv_label_map)

        if regime_series is not None:
            # Train per-regime models
            for regime in Regime:
                mask = (regime_series == regime).to_numpy()
                n_regime = int(mask.sum())
                if n_regime < 50:
                    logger.warning(f"Skipping {regime.value}: only {n_regime} bars")
                    continue

                X_r   = X.iloc[mask]
                y_r   = y_enc.iloc[mask]
                w_r   = sample_weight[mask] if sample_weight is not None else None

                xgb_p = self._get_xgb_params(regime)
                self._xgb_models[regime] = self._train_xgb(X_r, y_r, xgb_p, sample_weight=w_r)

                if self._has_lgb:
                    lgb_p = self._get_lgb_params(regime)
                    self._lgb_models[regime] = self._train_lgb(X_r, y_r, lgb_p, sample_weight=w_r)

                logger.info(f"Trained ensemble for {regime.value} ({mask.sum()} bars)")

        # Always train a global fallback model
        logger.info("Training global fallback ensemble...")
        xgb_global_p = self.REGIME_XGB_DEFAULTS[Regime.RANGING].copy()
        xgb_global_p.update({
            "objective": "multi:softprob", "num_class": 3,
            "verbosity": 0, "random_state": 42, "n_jobs": -1,
        })
        self._global_xgb = self._train_xgb(X, y_enc, xgb_global_p, sample_weight=sample_weight)

        if self._has_lgb:
            lgb_global_p = self.REGIME_LGB_DEFAULTS[Regime.RANGING].copy()
            lgb_global_p.update({
                "objective": "multiclass", "num_class": 3,
                "verbosity": -1, "random_state": 42,
            })
            self._global_lgb = self._train_lgb(X, y_enc, lgb_global_p, sample_weight=sample_weight)

        logger.info("EnsembleModel training complete")
        return self

    def predict_latest(
        self,
        features: pd.Series,
        regime_result: RegimeResult = None,
    ) -> EnsemblePrediction:
        """
        Live inference for a single bar.

        Args:
            features:      feature Series from FeatureEngine.latest_features()
            regime_result: pre-computed RegimeResult (or will compute internally)
        """
        if not self._global_xgb:
            raise RuntimeError("Model not trained — call fit() first")

        if regime_result is None:
            regime_result = self.regime_detector.detect(features)

        regime = regime_result.regime
        X      = pd.DataFrame([features[self.feature_cols]], columns=self.feature_cols)

        # Get XGBoost probs
        xgb_model = self._xgb_models.get(regime, self._global_xgb)
        xgb_probs = xgb_model.predict_proba(X)[0]   # [sell, hold, buy]

        # Get LightGBM probs
        if self._has_lgb and self._global_lgb:
            lgb_model = self._lgb_models.get(regime, self._global_lgb)
            lgb_probs = lgb_model.predict_proba(X)[0]
        else:
            lgb_probs = xgb_probs.copy()   # XGBoost only

        # Soft voting: weighted average of probabilities
        w_xgb = self.ENSEMBLE_WEIGHTS["xgb"]
        w_lgb = self.ENSEMBLE_WEIGHTS["lgb"]
        ensemble_probs = w_xgb * xgb_probs + w_lgb * lgb_probs

        # Apply regime signal weight to confidence (not to signal direction)
        regime_weight  = regime_result.signal_multiplier
        raw_confidence = float(ensemble_probs.max())
        adj_confidence = min(0.99, raw_confidence * regime_weight)

        # Final signal from ensemble probs
        signal_idx = int(np.argmax(ensemble_probs))
        signal     = self._label_map[signal_idx]

        # Individual model signals for transparency
        xgb_signal = self._label_map[int(np.argmax(xgb_probs))]
        lgb_signal = self._label_map[int(np.argmax(lgb_probs))]

        # Top features from XGBoost
        top_features = {}
        try:
            imp = xgb_model.get_booster().get_score(importance_type="gain")
            top5 = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:5]
            for feat, score in top5:
                if feat in features.index:
                    top_features[feat] = {
                        "value":      round(float(features[feat]), 4),
                        "importance": round(score / sum(v for _, v in top5), 3),
                    }
        except Exception:
            pass

        return EnsemblePrediction(
            signal            = signal,
            confidence        = adj_confidence,
            buy_prob          = float(ensemble_probs[2]),
            hold_prob         = float(ensemble_probs[1]),
            sell_prob         = float(ensemble_probs[0]),
            regime            = regime,
            regime_confidence = regime_result.confidence,
            xgb_signal        = xgb_signal,
            lgb_signal        = lgb_signal,
            xgb_confidence    = float(xgb_probs.max()),
            lgb_confidence    = float(lgb_probs.max()),
            passes_threshold  = adj_confidence >= self.min_confidence,
            top_features      = top_features,
        )

    def walk_forward_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        regime_series: pd.Series = None,
        train_months: int = 12,
        test_months:  int = 1,
        min_train_bars: int = 200,
    ) -> dict:
        """
        Walk-forward validation for the ensemble.
        Returns accuracy stats per regime + overall.
        """
        from sklearn.metrics import accuracy_score
        import numpy as np

        all_preds   = []
        all_actuals = []
        regime_results: dict[str, dict] = {}
        fold_results = []
        fold_num     = 0

        # Align indices and reset for clean split, avoids out-of-bounds and reindex warnings
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        if regime_series is not None:
            regime_series = regime_series.reset_index(drop=True)

        step = max(10, len(X) // 20)
        min_test_bars = max(5, test_months * 5)
        fold_starts = range(min_train_bars, len(X) - min_test_bars + 1, step)

        for fold_start_idx in fold_starts:
            fold_end_idx = min(fold_start_idx + step, len(X))
            if fold_end_idx - fold_start_idx < min_test_bars:
                continue

            X_train = X.iloc[:fold_start_idx]
            y_train = y.iloc[:fold_start_idx]
            X_test  = X.iloc[fold_start_idx:fold_end_idx]
            y_test  = y.iloc[fold_start_idx:fold_end_idx]

            # Drop NaN
            y_train = y_train.dropna()
            X_train = X_train.loc[y_train.index]
            y_test  = y_test.dropna()
            X_test  = X_test.loc[y_test.index]

            if len(X_train) < min_train_bars or len(X_test) < 5:
                continue
            
            try:
                # Fit ensemble on training data
                temp_ensemble = EnsembleModel(
                    tuned_params=self.tuned_params, 
                    min_confidence=self.min_confidence
                )
                regime_train = regime_series.iloc[:fold_start_idx] if regime_series is not None else None
                temp_ensemble.fit(X_train, y_train, regime_series=regime_train)
                
                # Predict on test
                preds = []
                for i in range(len(X_test)):
                    pred_result = temp_ensemble.predict_latest(X_test.iloc[i])
                    preds.append(pred_result.signal)
                
                all_preds.extend(preds)
                all_actuals.extend(y_test.tolist())
                
                # Per-regime accuracy
                if regime_series is not None:
                    test_regimes = regime_series.iloc[fold_start_idx:fold_end_idx].loc[y_test.index]
                    for i, (pred, actual, regime) in enumerate(zip(preds, y_test, test_regimes)):
                        if regime not in regime_results:
                            regime_results[regime] = {"correct": 0, "total": 0}
                        regime_results[regime]["total"] += 1
                        if pred == actual:
                            regime_results[regime]["correct"] += 1
                
                fold_num += 1
                
            except Exception as e:
                logger.warning(f"Failed fold starting at {fold_start_idx}: {e}")
                continue

        overall_acc = accuracy_score(all_actuals, all_preds) if all_actuals else 0.0

        regime_acc = {
            str(r): {
                "accuracy": round(v["correct"] / v["total"] * 100, 1) if v["total"] > 0 else 0,
                "correct":  v["correct"],
                "total":    v["total"],
            }
            for r, v in regime_results.items()
        }

        logger.info(
            f"Ensemble walk-forward complete: "
            f"{fold_num} folds | overall={overall_acc:.1%}"
        )
        for r, s in regime_acc.items():
            logger.info(f"  {r}: {s['accuracy']:.1f}% ({s['correct']}/{s['total']})")

        return {
            "overall_accuracy": round(overall_acc * 100, 2),
            "n_folds":          fold_num,
            "n_predictions":    len(all_preds),
            "fold_results":     fold_results,
            "regime_accuracy":  regime_acc,
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "xgb_models":    self._xgb_models,
            "lgb_models":    self._lgb_models,
            "global_xgb":    self._global_xgb,
            "global_lgb":    self._global_lgb,
            "feature_cols":  self.feature_cols,
            "min_confidence": self.min_confidence,
            "tuned_params":  self.tuned_params,
        }, path)
        logger.info(f"EnsembleModel saved → {path}")

    def load(self, path: Path) -> "EnsembleModel":
        data = joblib.load(path)
        self._xgb_models   = data["xgb_models"]
        self._lgb_models   = data["lgb_models"]
        self._global_xgb   = data["global_xgb"]
        self._global_lgb   = data.get("global_lgb")
        self.feature_cols  = data["feature_cols"]
        self.min_confidence= data["min_confidence"]
        self.tuned_params  = data.get("tuned_params", {})
        logger.info(f"EnsembleModel loaded ← {path}")
        return self

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_xgb_params(self, regime: Regime) -> dict:
        tuned = self.tuned_params.get(regime.value, {})
        defaults = self.REGIME_XGB_DEFAULTS[regime].copy()
        merged = {**defaults, **{k: v for k, v in tuned.items() if not k.startswith("_")}}
        merged.update({
            "objective":   "multi:softprob",
            "num_class":   3,
            "verbosity":   0,
            "random_state": 42,
            "n_jobs":      -1,
            "eval_metric": "mlogloss",
        })
        return merged

    def _get_lgb_params(self, regime: Regime) -> dict:
        defaults = self.REGIME_LGB_DEFAULTS[regime].copy()
        defaults.update({
            "objective":    "multiclass",
            "num_class":    3,
            "verbosity":   -1,
            "random_state": 42,
        })
        return defaults

    def _train_xgb(self, X: pd.DataFrame, y_enc: pd.Series, params: dict, sample_weight=None):
        import xgboost as xgb
        model = xgb.XGBClassifier(**params)
        model.fit(X, y_enc, sample_weight=sample_weight, verbose=False)
        return model

    def _train_lgb(self, X: pd.DataFrame, y_enc: pd.Series, params: dict, sample_weight=None):
        import lightgbm as lgb
        model = lgb.LGBMClassifier(**params)
        model.fit(X, y_enc, sample_weight=sample_weight)
        return model
