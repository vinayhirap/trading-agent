# trading-agent/run_pipeline_v2.py
"""
Training Pipeline v2 — Regime + Ensemble + Optuna

Replaces run_pipeline.py. Run this to train the improved model.

Steps:
  1. Fetch data for all configured symbols
  2. Build features (FeatureEngine)
  3. Generate labels (forward returns)
  4. Detect regimes (RegimeDetector)
  5. Optuna hyperparameter tuning per regime (skippable)
  6. Train EnsembleModel (XGBoost + LightGBM per regime)
  7. Walk-forward validation — print accuracy per regime
  8. Save model to models/

Usage:
    python run_pipeline_v2.py                    # full run with tuning
    python run_pipeline_v2.py --skip-tuning      # use cached or default params
    python run_pipeline_v2.py --symbol NIFTY50   # single symbol only
    python run_pipeline_v2.py --trials 30        # fewer Optuna trials (faster)
"""
import sys
import argparse
from pathlib import Path
from turtle import pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger
from src.data.manager import DataManager
from src.data.models import Interval, NSE_SYMBOLS, MCX_SYMBOLS, CRYPTO_SYMBOLS
from src.features.feature_engine import FeatureEngine
from src.prediction.labels import make_labels
from src.analysis.regime_detector import RegimeDetector
from src.prediction.ensemble_model import EnsembleModel
from src.prediction.tuner import RegimeTuner


# ── Symbols to train on ───────────────────────────────────────────────────────
TRAIN_SYMBOLS = [
    "NIFTY50", "BANKNIFTY",                    # Indices
    "RELIANCE", "TCS", "HDFCBANK", "INFY",     # Large cap
    "GOLD", "CRUDEOIL",                         # Commodities
    "BTC",                                      # Crypto
]


def run(
    symbols:      list[str] = None,
    skip_tuning:  bool      = False,
    n_trials:     int       = 50,
    days_back:    int       = 730,              # 2 years of daily data
    save_path:    Path      = Path("models/ensemble_v2.pkl"),
):
    symbols = symbols or TRAIN_SYMBOLS
    logger.info(f"Pipeline v2 | symbols={symbols} | trials={n_trials} | skip_tuning={skip_tuning}")

    dm       = DataManager()
    fe       = FeatureEngine()
    rd       = RegimeDetector()

    all_X       = []
    all_y       = []
    all_regimes = []

    # ── Step 1-4: Data → Features → Labels → Regimes ─────────────────────────
    for sym in symbols:
        logger.info(f"Processing {sym}...")
        try:
            df = dm.get_ohlcv(sym, Interval.D1, days_back=days_back)
            if df.empty or len(df) < 200:
                logger.warning(f"{sym}: not enough data ({len(df)} bars), skipping")
                continue

            # Features
            featured = fe.build(df, drop_na=True)
            if featured.empty:
                continue

            # Regime features added as columns
            featured = rd.add_regime_features(featured)

            # Labels
            labels = make_labels(featured["close"], horizon=5)
            labels = labels.dropna()
            featured = featured.loc[labels.index]

            if len(featured) < 150:
                logger.warning(f"{sym}: only {len(featured)} labeled bars, skipping")
                continue

            # Regime series for per-regime training
            from src.analysis.regime_detector import Regime
            regime_series = featured["regime_label"].map({
                0: Regime.RANGING,
                1: Regime.VOLATILE,
                2: Regime.TRENDING_DOWN,
                3: Regime.TRENDING_UP,
            })

            # Feature columns only (exclude OHLCV + raw regime label)
            ohlcv_cols   = {"open", "high", "low", "close", "volume"}
            feature_cols = [c for c in featured.columns if c not in ohlcv_cols]
            X            = featured[feature_cols]

            all_X.append(X)
            all_y.append(labels)
            all_regimes.append(regime_series)

            # Log regime distribution
            stats = rd.get_regime_stats(featured)
            logger.info(
                f"{sym}: {len(X)} bars | "
                + " ".join(f"{r}={v['pct']:.0f}%" for r, v in stats.items())
            )

        except Exception as e:
            logger.error(f"{sym} failed: {e}")
            continue

    if not all_X:
        logger.error("No data processed. Check data layer.")
        return

    import numpy as np
    import pandas as pd

    X_all       = pd.concat(all_X,       axis=0).reset_index(drop=True)
    y_all       = pd.concat(all_y,       axis=0).reset_index(drop=True)
    regimes_all = pd.concat(all_regimes, axis=0).reset_index(drop=True)

    # Align
    common_idx = X_all.index.intersection(y_all.index)
    X_all       = X_all.loc[common_idx]
    y_all       = y_all.loc[common_idx]
    regimes_all = regimes_all.loc[common_idx]

    inf_mask = ~np.isfinite(X_all.to_numpy(dtype=float, copy=False))
    inf_count = int(inf_mask.sum())
    if inf_count:
        logger.warning(f"Replacing {inf_count} non-finite feature values with 0 before training")
    X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0)

    logger.info(f"Combined dataset: {len(X_all)} bars | {X_all.shape[1]} features")

    # Log class balance
    from collections import Counter
    label_counts = Counter(str(v) for v in y_all)
    logger.info(f"Label distribution: {dict(label_counts)}")

    # ── Step 5: Optuna tuning ─────────────────────────────────────────────────
    tuned_params = {}
    if not skip_tuning:
        logger.info(f"Starting Optuna tuning ({n_trials} trials per regime)...")
        tuner       = RegimeTuner(n_trials=n_trials, train_months=12, test_months=1)
        tuned_params = tuner.tune(X_all, y_all, regime_series=regimes_all)
        logger.info(f"Tuning complete. Regimes tuned: {list(tuned_params.keys())}")
        for regime, params in tuned_params.items():
            acc = params.get("_best_accuracy", 0)
            logger.info(f"  {regime}: best walk-forward accuracy = {acc:.1%}")
    else:
        logger.info("Skipping tuning — using cached/default params")
        tuner = RegimeTuner()
        tuned_params = tuner.load_cached()
        if tuned_params:
            logger.info(f"Loaded cached params for: {list(tuned_params.keys())}")
        else:
            logger.info("No cached params found — using hardcoded regime defaults")

    # ── Step 6: Train ensemble ────────────────────────────────────────────────
    logger.info("Training EnsembleModel...")
    ensemble = EnsembleModel(tuned_params=tuned_params)
    ensemble.fit(X_all, y_all, regime_series=regimes_all)

    # ── Step 7: Walk-forward validation ──────────────────────────────────────
    logger.info("Running walk-forward validation on combined dataset...")
    wf_results = ensemble.walk_forward_validate(
        X_all, y_all,
        regime_series = regimes_all,
        train_months  = 12,
        test_months   = 1,
    )

    logger.info("=" * 60)
    logger.info(f"OVERALL ACCURACY: {wf_results['overall_accuracy']:.1f}%")
    logger.info(f"Folds: {wf_results['n_folds']} | Predictions: {wf_results['n_predictions']}")
    logger.info("Per-regime accuracy:")
    for regime, stats in wf_results["regime_accuracy"].items():
        logger.info(
            f"  {regime:20}: {stats['accuracy']:5.1f}% "
            f"({stats['correct']}/{stats['total']})"
        )
    logger.info("=" * 60)

    # ── Step 8: Save ─────────────────────────────────────────────────────────
    ensemble.save(save_path)
    logger.info(f"Model saved → {save_path}")

    # Also save validation results for the dashboard
    import json
    results_path = Path("data/ensemble_validation.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "overall_accuracy": wf_results["overall_accuracy"],
            "n_folds":          wf_results["n_folds"],
            "n_predictions":    wf_results["n_predictions"],
            "regime_accuracy":  wf_results["regime_accuracy"],
            "fold_results":     wf_results["fold_results"],
            "symbols_trained":  symbols,
            "n_features":       X_all.shape[1],
        }, f, indent=2, default=str)
    logger.info(f"Validation results saved → {results_path}")

    return wf_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ensemble model v2")
    parser.add_argument("--skip-tuning", action="store_true",
                        help="Skip Optuna tuning, use cached/default params")
    parser.add_argument("--symbol",  type=str, default=None,
                        help="Train on a single symbol only")
    parser.add_argument("--trials",  type=int, default=50,
                        help="Number of Optuna trials per regime")
    parser.add_argument("--days",    type=int, default=730,
                        help="Days of historical data to fetch")
    args = parser.parse_args()

    symbols = [args.symbol.upper()] if args.symbol else None

    run(
        symbols      = symbols,
        skip_tuning  = args.skip_tuning,
        n_trials     = args.trials,
        days_back    = args.days,
    )
