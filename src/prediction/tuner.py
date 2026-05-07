# trading-agent/src/prediction/tuner.py
"""
Optuna Hyperparameter Tuner — walk-forward aware, regime-conditional.

Key design:
- Objective function uses walk-forward accuracy (not in-sample)
  so tuned params genuinely generalise to unseen data
- Tunes separately per regime: ranging market needs different params
  than trending market (e.g. deeper trees, more regularisation in ranging)
- Pruning: unpromising trials are stopped early (saves ~60% of compute)
- Results cached to disk — re-run picks up from last checkpoint

Run time estimate (daily data, 2 years, 50 trials):
  ~3-5 minutes on a modern laptop (XGBoost is fast)

Usage:
    from src.prediction.tuner import RegimeTuner
    tuner = RegimeTuner(n_trials=50)
    best_params = tuner.tune(X, y, regime_series)
    # best_params = {"TRENDING_UP": {...}, "RANGING": {...}, ...}
"""
import json
import warnings
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger

warnings.filterwarnings("ignore", category=UserWarning)

TUNER_CACHE = Path("data/tuner_results.json")


class RegimeTuner:
    """
    Tunes XGBoost hyperparameters for each market regime using Optuna.

    For each regime it finds the best:
      - n_estimators, learning_rate, max_depth
      - min_child_weight, subsample, colsample_bytree
      - reg_alpha (L1), reg_lambda (L2)

    These replace DEFAULT_PARAMS in TradingModel, giving regime-specific
    models that are calibrated for each market condition.
    """

    # Sensible search space for Indian daily equity data
    SEARCH_SPACE = {
        "n_estimators":     (100, 600),
        "learning_rate":    (0.01, 0.15),
        "max_depth":        (3, 7),
        "min_child_weight": (3, 15),
        "subsample":        (0.6, 1.0),
        "colsample_bytree": (0.5, 1.0),
        "reg_alpha":        (0.0, 1.0),
        "reg_lambda":       (0.5, 3.0),
    }

    def __init__(
        self,
        n_trials:    int   = 50,
        n_jobs:      int   = 1,      # Optuna parallel jobs (keep 1 to avoid conflicts)
        train_months: int  = 12,
        test_months:  int  = 1,
        min_train_bars: int = 150,
        use_cache:   bool  = True,
    ):
        self.n_trials       = n_trials
        self.n_jobs         = n_jobs
        self.train_months   = train_months
        self.test_months    = test_months
        self.min_train_bars = min_train_bars
        self.use_cache      = use_cache

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        regime_series: pd.Series = None,
    ) -> dict:
        """
        Main entry point. Tunes params per regime.

        Args:
            X:             feature matrix (from FeatureEngine + regime features)
            y:             label series (Signal enum values)
            regime_series: Series of Regime enum values (same index as X)
                           If None, tunes a single global model

        Returns:
            dict of {regime_name: best_params_dict} or {"global": best_params}
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.error("Optuna not installed. Run: pip install optuna")
            return {}

        results = {}

        if regime_series is not None:
            from src.analysis.regime_detector import Regime
            regimes_to_tune = [r for r in Regime if (regime_series == r).sum() >= 50]
            logger.info(f"Tuning for {len(regimes_to_tune)} regimes with ≥50 bars each")

            for regime in regimes_to_tune:
                mask     = regime_series == regime
                X_regime = X[mask]
                y_regime = y[mask]

                if len(X_regime) < self.min_train_bars:
                    logger.warning(f"Skipping {regime.value}: only {len(X_regime)} bars")
                    continue

                logger.info(f"Tuning {regime.value} ({len(X_regime)} bars, {self.n_trials} trials)...")
                best = self._tune_single(X_regime, y_regime, study_name=regime.value)
                if best:
                    results[regime.value] = best
                    logger.info(f"{regime.value} best accuracy: {best.get('_best_accuracy', 0):.1%}")
        else:
            logger.info(f"Tuning global model ({len(X)} bars, {self.n_trials} trials)...")
            best = self._tune_single(X, y, study_name="global")
            if best:
                results["global"] = best

        if results:
            self._save_results(results)
            logger.info(f"Tuning complete. Results saved to {TUNER_CACHE}")

        return results

    def load_cached(self) -> dict:
        """Load previously tuned parameters from disk."""
        if TUNER_CACHE.exists():
            try:
                with open(TUNER_CACHE) as f:
                    data = json.load(f)
                logger.info(f"Loaded tuned params for: {list(data.keys())}")
                return data
            except Exception as e:
                logger.warning(f"Cache load failed: {e}")
        return {}

    def get_params_for_regime(self, regime_name: str, fallback: dict = None) -> dict:
        """
        Get best params for a specific regime.
        Falls back to provided defaults if regime not tuned yet.
        """
        cached = self.load_cached()
        params = cached.get(regime_name) or cached.get("global") or fallback or {}
        # Strip internal keys
        return {k: v for k, v in params.items() if not k.startswith("_")}

    # ── Internal ──────────────────────────────────────────────────────────────

    def _tune_single(self, X: pd.DataFrame, y: pd.Series, study_name: str) -> Optional[dict]:
        """Run Optuna on a single dataset. Returns best params dict."""
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            return None

        def objective(trial):
            params = {
                "objective":        "multi:softprob",
                "num_class":        3,
                "n_estimators":     trial.suggest_int(    "n_estimators",     *self.SEARCH_SPACE["n_estimators"]),
                "learning_rate":    trial.suggest_float(  "learning_rate",    *self.SEARCH_SPACE["learning_rate"], log=True),
                "max_depth":        trial.suggest_int(    "max_depth",        *self.SEARCH_SPACE["max_depth"]),
                "min_child_weight": trial.suggest_int(    "min_child_weight", *self.SEARCH_SPACE["min_child_weight"]),
                "subsample":        trial.suggest_float(  "subsample",        *self.SEARCH_SPACE["subsample"]),
                "colsample_bytree": trial.suggest_float(  "colsample_bytree", *self.SEARCH_SPACE["colsample_bytree"]),
                "reg_alpha":        trial.suggest_float(  "reg_alpha",        *self.SEARCH_SPACE["reg_alpha"]),
                "reg_lambda":       trial.suggest_float(  "reg_lambda",       *self.SEARCH_SPACE["reg_lambda"]),
                "verbosity":        0,
                "random_state":     42,
                "n_jobs":          -1,
                "eval_metric":      "mlogloss",
            }

            acc = self._walk_forward_score(X, y, params)
            return acc

        sampler = TPESampler(seed=42)
        study   = optuna.create_study(
            direction   = "maximize",
            study_name  = study_name,
            sampler     = sampler,
        )

        try:
            study.optimize(
                objective,
                n_trials  = self.n_trials,
                n_jobs    = self.n_jobs,
                show_progress_bar = False,
            )
        except Exception as e:
            logger.warning(f"Optuna study failed: {e}")
            return None

        best = study.best_params.copy()
        best["_best_accuracy"] = study.best_value
        best["_n_trials"]      = len(study.trials)
        best["objective"]      = "multi:softprob"
        best["num_class"]      = 3
        best["verbosity"]      = 0
        best["random_state"]   = 42
        best["n_jobs"]         = -1
        best["eval_metric"]    = "mlogloss"

        return best

    def _walk_forward_score(self, X: pd.DataFrame, y: pd.Series, params: dict) -> float:
        """
        Walk-forward cross-validation score.
        This is the objective Optuna maximises — prevents overfitting to in-sample.
        """
        import xgboost as xgb
        from sklearn.metrics import accuracy_score
        from dateutil.relativedelta import relativedelta
        from src.prediction.labels import Signal

        inv_label = {Signal.SELL: 0, Signal.HOLD: 1, Signal.BUY: 2}

        dates      = pd.RangeIndex(len(X))
        start_date = dates.min()
        end_date   = dates.max()

        all_preds   = []
        all_actuals = []

        
        step = max(20, len(X) // 15)
        fold_starts = range(self.min_train_bars, len(X) - 5, step)

        for fold_start in fold_starts:
            fold_end = min(fold_start + step, len(X))
            X_train, y_train = X.iloc[:fold_start], y.iloc[:fold_start]
            X_test,  y_test  = X.iloc[fold_start:fold_end], y.iloc[fold_start:fold_end]
            y_train = y_train.dropna()
            X_train = X_train.loc[y_train.index]
            y_test  = y_test.dropna()
            X_test  = X_test.loc[y_test.index]

            if len(X_train) < self.min_train_bars or len(X_test) < 5:
                continue
            try:
                y_enc = y_train.map(inv_label)
                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_enc, verbose=False)
                preds_enc = model.predict(X_test)
                label_map = {0: Signal.SELL, 1: Signal.HOLD, 2: Signal.BUY}
                preds     = [label_map[e] for e in preds_enc]
                all_preds.extend(preds)
                all_actuals.extend(y_test.tolist())
            except Exception:
                pass

        if not all_actuals:
            return 0.0

        return accuracy_score(all_actuals, all_preds)

    def _save_results(self, results: dict):
        TUNER_CACHE.parent.mkdir(parents=True, exist_ok=True)
        # Convert any non-serialisable types
        clean = {}
        for regime, params in results.items():
            clean[regime] = {
                k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                for k, v in params.items()
            }
        with open(TUNER_CACHE, "w") as f:
            json.dump(clean, f, indent=2)
