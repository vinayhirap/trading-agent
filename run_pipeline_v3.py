#!/usr/bin/env python
# trading-agent/run_pipeline_v3.py
"""
Model Training Pipeline v3 — Improved Accuracy

Fixes vs v2 (37-43% → target 48-56%):
  1. Labels v2    — percentile-based, regime-gated, balanced sampling
  2. Regime features — 8 new interaction features (RSI-in-trend etc.)
  3. Feature selection — drops bottom 20% by importance after first fit
  4. Balanced training — sample_weight passed to XGBoost/LightGBM
  5. Better walk-forward — 7 folds (was 17-19, each fold too small)
  6. More symbols — equities now uses all 25 available stocks
  7. Longer horizon — equity 7d (was 5d), index 5d (was 3d)

Usage:
  python run_pipeline_v3.py                          # all asset classes
  python run_pipeline_v3.py --asset index            # indices only
  python run_pipeline_v3.py --asset equity           # equities only
  python run_pipeline_v3.py --asset futures          # commodities only
  python run_pipeline_v3.py --asset crypto           # crypto only
  python run_pipeline_v3.py --skip-tuning            # faster, no Optuna
  python run_pipeline_v3.py --asset equity --skip-tuning
"""
import sys
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from loguru import logger

ASSET_CONFIG = {
    "index": {
        "symbols":  ["NIFTY50", "BANKNIFTY", "NIFTYIT", "SENSEX", "NIFTYMID"],
        "horizon":  5,
        "buy_pct":  0.28,
        "sell_pct": 0.28,
        "min_move": 0.004,
        "n_splits": 7,
        "days_back":1000,
    },
    "equity": {
        "symbols": [
            "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","SBIN",
            "WIPRO","AXISBANK","KOTAKBANK","LT","BAJFINANCE","MARUTI",
            "SUNPHARMA","BHARTIARTL","TATASTEEL","JSWSTEEL","ONGC",
            "NTPC","ASIANPAINT","DRREDDY","BAJAJFINSV","ULTRACEMCO",
            "COALINDIA","POWERGRID","HINDALCO",
        ],
        "horizon":  7,
        "buy_pct":  0.30,
        "sell_pct": 0.30,
        "min_move": 0.005,
        "n_splits": 7,
        "days_back":1000,
    },
    "futures": {
        "symbols":  ["GOLD","SILVER","CRUDEOIL","COPPER","NATURALGAS"],
        "horizon":  7,
        "buy_pct":  0.30,
        "sell_pct": 0.30,
        "min_move": 0.004,
        "n_splits": 7,
        "days_back":1200,
    },
    "crypto": {
        "symbols":  ["BTC","ETH","BNB","SOL","XRP"],
        "horizon":  3,
        "buy_pct":  0.25,
        "sell_pct": 0.25,
        "min_move": 0.008,
        "n_splits": 7,
        "days_back":1000,
    },
}

MODEL_PATHS = {
    "index":   ROOT / "models" / "ensemble_index.pkl",
    "equity":  ROOT / "models" / "ensemble_equity.pkl",
    "futures": ROOT / "models" / "ensemble_futures.pkl",
    "crypto":  ROOT / "models" / "ensemble_crypto.pkl",
}


def train_asset_class(ac_name: str, skip_tuning: bool = False) -> dict:
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import TimeSeriesSplit

    from src.data.manager import DataManager
    from src.data.models import Interval
    from src.features.feature_engine import FeatureEngine
    from src.features.regime_features import add_regime_features
    from src.prediction.labels_v2 import make_labels_v2
    from src.prediction.ensemble_model import EnsembleModel

    config = ASSET_CONFIG[ac_name]
    dm     = DataManager()
    fe     = FeatureEngine()

    logger.info(f"\n{'='*60}")
    logger.info(f"Training: {ac_name.upper()} | "
                f"{len(config['symbols'])} symbols | "
                f"horizon={config['horizon']}d | "
                f"folds={config['n_splits']}")
    logger.info(f"{'='*60}")

    all_X, all_y, all_w = [], [], []
    symbol_count = 0

    REF_SYMBOLS = {
        "index": "^NSEI",
        "equity": "^NSEI",
        "crypto": "BTC-USD",
        "futures": "CL=F",
    }
    ref_sym = REF_SYMBOLS.get(ac_name, "^NSEI")
    try:
        market_df = dm.get_ohlcv(ref_sym, Interval.D1, days_back=config["days_back"] + 50)
    except Exception as e:
        logger.warning(f"Failed to fetch market reference '{ref_sym}': {e}")
        market_df = None

    for sym in config["symbols"]:
        try:
            df = dm.get_ohlcv(sym, Interval.D1, days_back=config["days_back"])
            if df.empty or len(df) < 200:
                logger.warning(f"  {sym}: only {len(df)} bars, skipping")
                continue

            featured = fe.build(df, drop_na=False, market_df=market_df)
            if featured.empty or len(featured) < 150:
                continue

            # Add 8 regime interaction features
            featured = add_regime_features(featured)
            featured = featured.dropna()
            if len(featured) < 100:
                continue

            df_aligned = df.reindex(featured.index).dropna()
            featured   = featured.reindex(df_aligned.index)

            # v2 labels with balanced weights
            y, weights = make_labels_v2(
                df          = df_aligned,
                features    = featured,
                horizon     = config["horizon"],
                asset_class = ac_name,
                buy_pct     = config["buy_pct"],
                sell_pct    = config["sell_pct"],
                min_move    = config["min_move"],
            )
            if len(y) < 80:
                continue

            X = featured.reindex(y.index).dropna()
            y = y.reindex(X.index)
            w = weights.reindex(X.index)
            if len(X) < 50:
                continue

            all_X.append(X)
            all_y.append(y)
            all_w.append(w)
            symbol_count += 1
            logger.info(
                f"  ✓ {sym}: {len(X)} bars | "
                f"BUY={(y==1).sum()} "
                f"HOLD={(y==0).sum()} "
                f"SELL={(y==-1).sum()}"
            )

        except Exception as e:
            logger.error(f"  ✗ {sym}: {e}")

    if symbol_count == 0:
        logger.error(f"No symbols loaded for {ac_name}")
        return {"accuracy": 0.0, "n_symbols": 0}

    X_all = pd.concat(all_X).fillna(0)
    y_all = pd.concat(all_y)
    w_all = pd.concat(all_w)

    logger.info(
        f"\nCombined: {len(X_all)} bars | "
        f"BUY={(y_all==1).mean():.0%} | "
        f"HOLD={(y_all==0).mean():.0%} | "
        f"SELL={(y_all==-1).mean():.0%}"
    )

    # Feature selection — drop bottom 20%
    X_all, selected_features = _select_features(X_all, y_all)
    logger.info(f"Features selected: {len(selected_features)}")

    # Walk-forward validation
    tscv             = TimeSeriesSplit(n_splits=config["n_splits"], gap=config["horizon"])
    fold_accuracies  = []
    ensemble         = EnsembleModel()

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_all)):
        X_tr = X_all.iloc[tr_idx]
        y_tr = y_all.iloc[tr_idx]
        w_tr = w_all.iloc[tr_idx]
        X_te = X_all.iloc[te_idx]
        y_te = y_all.iloc[te_idx]

        if len(X_tr) < 80 or len(X_te) < 20:
            continue
        try:
            ensemble.fit(X_tr, y_tr, sample_weight=w_tr.values, skip_tuning=True)

            preds = []
            for i in range(len(X_te)):
                p = ensemble.predict_latest(X_te.iloc[i])
                preds.append(p.signal.value if p else 0)

            correct = sum(p == t for p, t in zip(preds, y_te.values))
            acc     = correct / len(y_te)
            fold_accuracies.append(acc)
            logger.info(f"  Fold {fold+1}: {acc:.1%} ({correct}/{len(y_te)})")
        except Exception as e:
            logger.error(f"  Fold {fold+1} failed: {e}")

    overall_acc = float(np.mean(fold_accuracies)) if fold_accuracies else 0.0
    logger.info(f"\nWalk-forward accuracy: {overall_acc:.1%}")

    # Final model on all data
    logger.info("Training final model on all data...")
    try:
        ensemble.fit(X_all, y_all, sample_weight=w_all.values, skip_tuning=skip_tuning)
        model_path = MODEL_PATHS[ac_name]
        model_path.parent.mkdir(exist_ok=True)
        ensemble.save(model_path)
        logger.info(f"Saved → {model_path} ({model_path.stat().st_size//1024} KB)")

        feat_path = ROOT / "data" / f"features_{ac_name}.json"
        feat_path.write_text(json.dumps(selected_features))

    except Exception as e:
        logger.error(f"Final model failed: {e}")

    return {
        "accuracy":       round(overall_acc * 100, 1),
        "n_symbols":      symbol_count,
        "n_bars":         len(X_all),
        "n_features":     len(selected_features),
        "n_folds":        len(fold_accuracies),
        "label_horizon":  config["horizon"],
        "symbols_trained":config["symbols"][:symbol_count],
        "fold_accuracies":[round(a*100,1) for a in fold_accuracies],
    }


def _select_features(X, y):
    """Drop bottom 20% features by XGBoost importance."""
    import pandas as pd
    try:
        import xgboost as xgb
        clf = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            verbosity=0, use_label_encoder=False, eval_metric="mlogloss",
        )
        y_enc = y.map({-1: 0, 0: 1, 1: 2})
        clf.fit(X.fillna(0), y_enc)
        imp  = pd.Series(clf.feature_importances_, index=X.columns)
        thr  = imp.quantile(0.20)
        sel  = imp[imp >= thr].index.tolist()

        # Always keep regime features
        try:
            from src.features.regime_features import get_regime_feature_names
            for rf in get_regime_feature_names():
                if rf in X.columns and rf not in sel:
                    sel.append(rf)
        except Exception:
            pass

        return X[sel], sel
    except Exception as e:
        logger.warning(f"Feature selection failed ({e}), using all")
        return X, X.columns.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset",
                        choices=["index","equity","futures","crypto","all"],
                        default="all")
    parser.add_argument("--skip-tuning", action="store_true")
    args = parser.parse_args()

    assets = list(ASSET_CONFIG.keys()) if args.asset == "all" else [args.asset]
    logger.info(f"Pipeline v3 | assets={assets} | "
                f"tuning={'off' if args.skip_tuning else 'on'}")

    results = {}
    for ac in assets:
        try:
            results[ac] = train_asset_class(ac, args.skip_tuning)
        except Exception as e:
            logger.error(f"{ac} failed: {e}")
            results[ac] = {"accuracy": 0.0, "error": str(e)}

    # Save
    val_path = ROOT / "data" / "asset_validation.json"
    val_path.parent.mkdir(exist_ok=True)
    val_path.write_text(json.dumps(results, indent=2))

    # Summary
    print(f"\n{'='*62}")
    print(f"{'Asset':<12} {'Accuracy':>10} {'vs before':>10} "
          f"{'Symbols':>8} {'Bars':>8} {'Feats':>6}")
    print("="*62)

    before = {"index": 43.2, "equity": 37.5, "futures": 40.3, "crypto": 0.0}
    for ac, r in results.items():
        acc  = r.get("accuracy", 0)
        prev = before.get(ac, 0)
        diff = acc - prev
        syms = r.get("n_symbols", 0)
        bars = r.get("n_bars", 0)
        feat = r.get("n_features", 0)
        icon = "✅" if acc >= 50 else "🔶" if acc >= 45 else "❌"
        print(f"{ac:<12} {icon} {acc:>7.1f}%  {diff:>+8.1f}%  {syms:>8} {bars:>8} {feat:>6}")

    print("="*62)
    print(f"\nSaved → {val_path}")
    print("Restart dashboard to load new models.")


if __name__ == "__main__":
    main()