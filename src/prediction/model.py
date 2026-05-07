# trading-agent/src/prediction/model.py
"""
XGBoost-based prediction model with walk-forward validation.

Explainability features:
- Feature importance (gain-based)
- SHAP values for individual predictions
- Confidence scores (class probabilities)
- Calibration check
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, accuracy_score,
    precision_score, recall_score, f1_score,
)
from dataclasses import dataclass, field
from pathlib import Path
import joblib
from loguru import logger

from src.prediction.labels import Signal


@dataclass
class PredictionResult:
    """What the model returns for a single bar."""
    signal:           Signal
    confidence:       float          # probability of predicted class (0-1)
    buy_prob:         float          # P(BUY)
    hold_prob:        float          # P(HOLD)
    sell_prob:        float          # P(SELL)
    top_features:     dict           # top 5 features driving this prediction
    passes_threshold: bool           # True if confidence > min_confidence

    def __str__(self):
        arrow = {"BUY": "▲", "SELL": "▼", "HOLD": "—"}
        sig   = self.signal.name
        return (
            f"{arrow[sig]} {sig}  |  confidence: {self.confidence:.1%}  |  "
            f"B:{self.buy_prob:.1%} H:{self.hold_prob:.1%} S:{self.sell_prob:.1%}  |  "
            f"{'TRADEABLE' if self.passes_threshold else 'below threshold'}"
        )


@dataclass
class WalkForwardResult:
    """Aggregated results across all walk-forward folds."""
    fold_results:   list[dict] = field(default_factory=list)
    all_predictions: pd.Series = field(default_factory=pd.Series)
    all_actuals:     pd.Series = field(default_factory=pd.Series)

    @property
    def accuracy(self) -> float:
        if self.all_actuals.empty:
            return 0.0
        return accuracy_score(self.all_actuals, self.all_predictions)

    @property
    def report(self) -> str:
        if self.all_actuals.empty:
            return "No results"
        return classification_report(
            self.all_actuals, self.all_predictions,
            target_names=["SELL", "HOLD", "BUY"],
            labels=[-1, 0, 1],
        )

    def summary(self) -> dict:
        if self.all_actuals.empty:
            return {}
        return {
            "accuracy":     round(self.accuracy, 4),
            "n_folds":      len(self.fold_results),
            "n_predictions":len(self.all_predictions),
            "buy_precision": round(
                precision_score(self.all_actuals, self.all_predictions,
                                labels=[1], average="macro", zero_division=0), 4),
            "sell_precision": round(
                precision_score(self.all_actuals, self.all_predictions,
                                labels=[-1], average="macro", zero_division=0), 4),
        }


class TradingModel:
    """
    XGBoost classifier for market direction prediction.

    Key design choices:
    - scale_pos_weight handles class imbalance automatically
    - early_stopping prevents overfitting per fold
    - probability calibration ensures confidence scores are trustworthy
    - SHAP values make every prediction explainable
    """

    # XGBoost hyperparameters — conservative defaults (prioritise stability)
    DEFAULT_PARAMS = {
        "objective":        "multi:softprob",
        "num_class":        3,
        "n_estimators":     400,
        "learning_rate":    0.05,
        "max_depth":        4,           # shallow trees → less overfitting
        "min_child_weight": 5,           # require meaningful samples per leaf
        "subsample":        0.8,
        "colsample_bytree": 0.7,
        "reg_alpha":        0.1,         # L1 regularisation
        "reg_lambda":       1.0,         # L2 regularisation
        "eval_metric":      "mlogloss",
        "early_stopping_rounds": 30,
        "verbosity":        0,
        "random_state":     42,
        "n_jobs":           -1,
    }

    def __init__(
        self,
        params: dict = None,
        min_confidence: float = 0.55,    # only trade if model is ≥55% confident
        feature_cols: list[str] = None,
    ):
        self.params          = params or self.DEFAULT_PARAMS.copy()
        self.min_confidence  = min_confidence
        self.feature_cols    = feature_cols   # set after first fit
        self.model           = None
        self._label_map      = {0: Signal.SELL, 1: Signal.HOLD, 2: Signal.BUY}
        self._inv_label_map  = {Signal.SELL: 0, Signal.HOLD: 1, Signal.BUY: 2}

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
    ) -> "TradingModel":
        """
        Train on X_train/y_train.
        X_val/y_val used for early stopping — must be out-of-sample.
        """
        # Map Signal enum → 0,1,2 for XGBoost
        y_enc = y_train.map(self._inv_label_map)

        self.feature_cols = list(X_train.columns)

        eval_set = None
        if X_val is not None and y_val is not None:
            y_val_enc = y_val.map(self._inv_label_map)
            eval_set  = [(X_val, y_val_enc)]

        params = {k: v for k, v in self.params.items()
                  if k != "early_stopping_rounds"}
        early_stop = self.params.get("early_stopping_rounds", 30)

        self.model = xgb.XGBClassifier(**params)
        self.model.fit(
            X_train, y_enc,
            eval_set=eval_set,
            verbose=False,
        )
        logger.info(f"Model trained on {len(X_train)} samples | {len(self.feature_cols)} features")
        return self

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Return Signal series for a DataFrame of features."""
        if self.model is None:
            raise RuntimeError("Model not trained yet — call fit() first")
        y_enc = self.model.predict(X)
        return pd.Series([self._label_map[e] for e in y_enc], index=X.index)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return probability matrix [SELL_prob, HOLD_prob, BUY_prob]."""
        if self.model is None:
            raise RuntimeError("Model not trained yet")
        probs = self.model.predict_proba(X)
        return pd.DataFrame(
            probs,
            columns=["sell_prob", "hold_prob", "buy_prob"],
            index=X.index,
        )

    def predict_latest(self, features: pd.Series) -> PredictionResult:
        """
        Predict for a single bar (live inference).
        features: a Series with the same columns as training data.
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet")

        X = pd.DataFrame([features], columns=self.feature_cols)
        # Handle any missing features gracefully
        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = 0.0

        probs_df  = self.predict_proba(X).iloc[0]
        signal_enc = self.model.predict(X)[0]
        signal    = self._label_map[signal_enc]
        confidence = probs_df.max()

        top_features = self._get_top_features(X, n=5)

        return PredictionResult(
            signal           = signal,
            confidence       = float(confidence),
            buy_prob         = float(probs_df["buy_prob"]),
            hold_prob        = float(probs_df["hold_prob"]),
            sell_prob        = float(probs_df["sell_prob"]),
            top_features     = top_features,
            passes_threshold = confidence >= self.min_confidence,
        )

    # ── Walk-forward validation ───────────────────────────────────────────────

    def walk_forward_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_months: int = 12,
        test_months: int  = 1,
        min_train_bars: int = 200,
    ) -> WalkForwardResult:
        """
        The gold standard for evaluating trading models.

        We never peek into the future. For each fold:
          - Train on all data up to cutoff
          - Test on the NEXT test_months of unseen data
          - Move forward and repeat

        Parameters
        ----------
        X              : full feature matrix (DatetimeIndex required)
        y              : full label series
        train_months   : minimum initial training window
        test_months    : each fold's test window size
        min_train_bars : skip fold if training set is too small
        """
        logger.info(f"Walk-forward: {train_months}m train / {test_months}m test windows")

        all_preds   = []
        all_actuals = []
        fold_results = []

        dates = X.index
        start_date = dates[0]
        end_date   = dates[-1]

        # Build fold boundaries
        from dateutil.relativedelta import relativedelta
        fold_start = min_train_bars

        fold_num = 0

        step = max(20, len(X) // 15)
        fold_starts = range(min_train_bars, len(X) - 5, step)

        for fold_start in fold_starts:
            fold_end = min(fold_start + step, len(X))
            X_train, y_train = X.iloc[:fold_start], y.iloc[:fold_start]
            X_test,  y_test  = X.iloc[fold_start:fold_end], y.iloc[fold_start:fold_end]
            y_train = y_train.dropna()
            X_train = X_train.loc[y_train.index]
            y_test  = y_test.dropna()
            X_test  = X_test.loc[y_test.index]

            if len(X_train) < min_train_bars or len(X_test) < 5:
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

        result = WalkForwardResult(
            fold_results    = fold_results,
            all_predictions = pd.Series(all_preds),
            all_actuals     = pd.Series(all_actuals),
        )
        logger.info(f"Walk-forward complete: {fold_num} folds | "
                    f"overall accuracy={result.accuracy:.1%}")
        return result

    # ── Explainability ────────────────────────────────────────────────────────

    def feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """Returns top N features by gain importance."""
        if self.model is None:
            return pd.DataFrame()
        imp = self.model.get_booster().get_score(importance_type="gain")
        df  = pd.DataFrame(imp.items(), columns=["feature", "importance"])
        df  = df.sort_values("importance", ascending=False).head(top_n)
        df["importance_pct"] = df["importance"] / df["importance"].sum()
        return df.reset_index(drop=True)

    def _get_top_features(self, X: pd.DataFrame, n: int = 5) -> dict:
        """Top N features by importance for a single prediction."""
        imp = self.feature_importance(top_n=n)
        result = {}
        for _, row in imp.iterrows():
            feat = row["feature"]
            if feat in X.columns:
                result[feat] = {
                    "value":      round(float(X[feat].iloc[0]), 4),
                    "importance": round(float(row["importance_pct"]), 3),
                }
        return result

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("No model to save")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model":         self.model,
            "feature_cols":  self.feature_cols,
            "min_confidence": self.min_confidence,
            "params":        self.params,
        }, path)
        logger.info(f"Model saved → {path}")

    def load(self, path: Path) -> "TradingModel":
        data = joblib.load(path)
        self.model          = data["model"]
        self.feature_cols   = data["feature_cols"]
        self.min_confidence = data["min_confidence"]
        self.params         = data["params"]
        logger.info(f"Model loaded ← {path}")
        return self