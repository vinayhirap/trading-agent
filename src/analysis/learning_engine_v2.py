# trading-agent/src/analysis/learning_engine_v2.py
"""
Learning Engine V2 — Self-Improving Prediction System

Replaces the basic LearningEngine with a full feedback loop:

  1. Track every prediction → actual outcome
  2. Analyse misclassifications (false BUY, false SELL patterns)
  3. Adjust per-asset-class model trust scores
  4. Adjust hybrid engine component weights based on which source
     was right most often
  5. Feed feature importance back into signal scoring

Storage: lightweight JSON (no database dependency)

Key metrics tracked:
  - Accuracy by signal type (BUY/SELL/HOLD)
  - Accuracy by regime (TRENDING_UP/DOWN/RANGING/VOLATILE)
  - Accuracy by asset class
  - Which signal source (technical/news/events/behavior) was correct
  - Confidence calibration (is 70% confidence actually right 70%?)

Integration:
  - ActionEngine reads trust scores before issuing final verdict
  - HybridEngine reads component accuracy to adjust weights
  - PredictionEngine reads feature importance to prioritise signals
"""
import json
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
from loguru import logger

ROOT       = Path(__file__).resolve().parents[3]
STORE_PATH = ROOT / "data" / "learning_v2.json"

# ── Default component weights (adjusted by learning) ─────────────────────────
DEFAULT_COMPONENT_WEIGHTS = {
    "technical":  0.40,
    "news":       0.20,
    "events":     0.20,
    "behavior":   0.20,
}

# ── Default asset class trust (1.0 = full trust) ─────────────────────────────
DEFAULT_ASSET_TRUST = {
    "equity":   1.0,
    "index":    1.0,
    "futures":  1.0,   # commodities
    "crypto":   1.0,
    "options":  0.8,
}

MIN_SAMPLES_TO_ADJUST = 10   # need this many resolved predictions before adjusting
MAX_WEIGHT             = 0.70
MIN_WEIGHT             = 0.05


class LearningEngineV2:
    """
    Self-learning feedback system.

    Usage:
        le = LearningEngineV2()

        # Record a prediction
        pid = le.record(
            symbol       = "NIFTY50",
            signal       = "BUY",
            confidence   = 0.67,
            asset_class  = "index",
            regime       = "TRENDING_DOWN",
            source_scores= {"technical": 0.45, "news": -0.1, "events": 0.0, "behavior": 0.05},
            horizon_bars = 5,
        )

        # Later, resolve it
        le.resolve(pid, actual_return=0.023)   # +2.3% = BUY was correct

        # Get adjusted weights for hybrid engine
        weights = le.get_component_weights()

        # Get asset trust score
        trust = le.get_asset_trust("index")   # e.g. 0.85
    """

    def __init__(self):
        self._db = self._load()
        logger.info(
            f"LearningEngineV2 | "
            f"{len(self._db['predictions'])} predictions | "
            f"weights: {self._db['component_weights']}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def record(
        self,
        symbol:        str,
        signal:        str,      # BUY / SELL / HOLD
        confidence:    float,
        asset_class:   str       = "equity",
        regime:        str       = "RANGING",
        source_scores: dict      = None,   # {technical: 0.3, news: 0.1, ...}
        horizon_bars:  int       = 5,
        entry_price:   float     = 0,
        extra:         dict      = None,
    ) -> str:
        """Record a new prediction. Returns prediction ID."""
        pid = str(uuid.uuid4())[:12]
        self._db["predictions"].append({
            "id":           pid,
            "symbol":       symbol,
            "signal":       signal,
            "confidence":   round(confidence, 4),
            "asset_class":  asset_class,
            "regime":       regime,
            "source_scores":source_scores or {},
            "horizon_bars": horizon_bars,
            "entry_price":  entry_price,
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "resolved":     False,
            "outcome":      None,      # CORRECT / WRONG
            "actual_return":None,
            "resolved_at":  None,
            "extra":        extra or {},
        })
        self._save()
        return pid

    def resolve(self, pid: str, actual_return: float) -> bool:
        """
        Resolve a prediction with the actual outcome.
        actual_return: price change as fraction (0.02 = +2%)
        """
        for p in self._db["predictions"]:
            if p["id"] == pid and not p["resolved"]:
                signal = p["signal"]
                # BUY correct if price went up, SELL correct if price went down
                if signal in ("BUY", "STRONG BUY"):
                    correct = actual_return > 0.002    # > 0.2% counts as correct
                elif signal in ("SELL", "STRONG SELL"):
                    correct = actual_return < -0.002
                else:  # HOLD
                    correct = abs(actual_return) < 0.01

                p["resolved"]     = True
                p["outcome"]      = "CORRECT" if correct else "WRONG"
                p["actual_return"]= round(actual_return, 6)
                p["resolved_at"]  = datetime.now(timezone.utc).isoformat()

                self._save()
                self._maybe_update_weights()
                return True
        return False

    def resolve_by_symbol_date(
        self,
        symbol:        str,
        signal:        str,
        actual_return: float,
        max_age_hours: int = 48,
    ) -> int:
        """Resolve pending predictions for a symbol by matching signal type."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        count  = 0
        for p in self._db["predictions"]:
            if (p["symbol"] == symbol and p["signal"] == signal
                    and not p["resolved"]):
                ts = datetime.fromisoformat(p["timestamp"])
                if ts >= cutoff:
                    self.resolve(p["id"], actual_return)
                    count += 1
        return count

    def check_and_resolve_all(self, dm=None) -> int:
        """
        Auto-resolve pending predictions using current prices.
        Compares entry_price to current price to compute return.
        """
        if dm is None:
            try:
                from src.data.manager import DataManager
                dm = DataManager()
            except Exception:
                return 0

        resolved = 0
        pending  = [
            p for p in self._db["predictions"]
            if not p["resolved"] and p.get("entry_price", 0) > 0
        ]

        for p in pending:
            try:
                age_bars = (
                    datetime.now(timezone.utc) -
                    datetime.fromisoformat(p["timestamp"])
                ).days
                if age_bars < p["horizon_bars"]:
                    continue   # not yet time to resolve

                current_price = dm.get_latest_price(p["symbol"])
                if not current_price:
                    continue

                ret = (current_price - p["entry_price"]) / p["entry_price"]
                self.resolve(p["id"], ret)
                resolved += 1
            except Exception:
                continue

        if resolved:
            logger.info(f"LearningEngineV2: auto-resolved {resolved} predictions")
        return resolved

    def get_component_weights(self) -> dict:
        """Get current component weights for hybrid engine."""
        return dict(self._db["component_weights"])

    def get_asset_trust(self, asset_class: str) -> float:
        """Get trust score for an asset class (0-1)."""
        return self._db["asset_trust"].get(asset_class, 1.0)

    def get_accuracy_stats(self, symbol: str = None, asset_class: str = None) -> dict:
        """Compute accuracy statistics, optionally filtered."""
        preds = [
            p for p in self._db["predictions"] if p["resolved"]
        ]
        if symbol:
            preds = [p for p in preds if p["symbol"] == symbol]
        if asset_class:
            preds = [p for p in preds if p["asset_class"] == asset_class]

        if not preds:
            return {"total": 0, "accuracy": 0.0, "by_signal": {}, "by_regime": {}}

        n_total   = len(preds)
        n_correct = sum(1 for p in preds if p["outcome"] == "CORRECT")
        accuracy  = n_correct / n_total if n_total else 0

        # By signal type
        by_signal = {}
        for sig in ("BUY", "SELL", "HOLD", "STRONG BUY", "STRONG SELL"):
            sp = [p for p in preds if p["signal"] == sig]
            if sp:
                by_signal[sig] = {
                    "total":    len(sp),
                    "correct":  sum(1 for p in sp if p["outcome"] == "CORRECT"),
                    "accuracy": sum(1 for p in sp if p["outcome"] == "CORRECT") / len(sp),
                }

        # By regime
        by_regime = {}
        for p in preds:
            r = p.get("regime", "UNKNOWN")
            if r not in by_regime:
                by_regime[r] = {"total": 0, "correct": 0}
            by_regime[r]["total"] += 1
            if p["outcome"] == "CORRECT":
                by_regime[r]["correct"] += 1
        for r in by_regime:
            d = by_regime[r]
            d["accuracy"] = d["correct"] / d["total"] if d["total"] else 0

        # Confidence calibration
        calibration = self._compute_calibration(preds)

        # Misclassification patterns
        errors = self._analyse_errors(preds)

        return {
            "total":        n_total,
            "correct":      n_correct,
            "accuracy":     round(accuracy, 3),
            "by_signal":    by_signal,
            "by_regime":    by_regime,
            "calibration":  calibration,
            "errors":       errors,
            "weights":      self.get_component_weights(),
            "asset_trust":  dict(self._db["asset_trust"]),
        }

    def get_misclassification_report(self) -> list[dict]:
        """
        Identify the most common error patterns.
        Returns list of patterns with frequency and suggested fix.
        """
        errors = [
            p for p in self._db["predictions"]
            if p["resolved"] and p["outcome"] == "WRONG"
        ]
        if len(errors) < 5:
            return []

        patterns = []

        # Pattern 1: BUY in TRENDING_DOWN regime
        bt_down = [
            e for e in errors
            if "BUY" in e["signal"] and e.get("regime") == "TRENDING_DOWN"
        ]
        if len(bt_down) >= 3:
            patterns.append({
                "pattern":     "BUY in TRENDING_DOWN",
                "count":       len(bt_down),
                "severity":    "HIGH",
                "description": f"Buying against trend — {len(bt_down)} losses. "
                               f"Avoid BUY signals when regime is TRENDING_DOWN.",
                "fix":         "Block BUY when regime=TRENDING_DOWN unless confidence > 0.75",
            })

        # Pattern 2: SELL in TRENDING_UP regime
        st_up = [
            e for e in errors
            if "SELL" in e["signal"] and e.get("regime") == "TRENDING_UP"
        ]
        if len(st_up) >= 3:
            patterns.append({
                "pattern":     "SELL in TRENDING_UP",
                "count":       len(st_up),
                "severity":    "HIGH",
                "description": f"Selling against trend — {len(st_up)} losses.",
                "fix":         "Block SELL when regime=TRENDING_UP unless confidence > 0.75",
            })

        # Pattern 3: High confidence wrong predictions
        hc_wrong = [e for e in errors if e.get("confidence", 0) > 0.70]
        if len(hc_wrong) >= 3:
            patterns.append({
                "pattern":     "Overconfident wrong signals",
                "count":       len(hc_wrong),
                "severity":    "MEDIUM",
                "description": f"{len(hc_wrong)} high-confidence (>70%) predictions were wrong. "
                               f"Model is overconfident.",
                "fix":         "Apply confidence penalty: multiply all conf > 0.7 by 0.85",
            })

        # Pattern 4: News-driven signals wrong
        if "news" in str(errors[:10]):
            news_driven = [
                e for e in errors
                if abs(e.get("source_scores", {}).get("news", 0)) > 0.3
            ]
            if len(news_driven) >= 3:
                patterns.append({
                    "pattern":     "News-driven signals underperform",
                    "count":       len(news_driven),
                    "severity":    "MEDIUM",
                    "description": f"News sentiment was primary driver but signal was wrong "
                                   f"{len(news_driven)} times.",
                    "fix":         "Reduce news weight in hybrid engine",
                })

        return sorted(patterns, key=lambda p: p["count"], reverse=True)

    # ── Weight adjustment logic ───────────────────────────────────────────────

    def _maybe_update_weights(self):
        """Update weights if we have enough resolved predictions."""
        resolved = [p for p in self._db["predictions"] if p["resolved"]]
        if len(resolved) < MIN_SAMPLES_TO_ADJUST:
            return

        # Only use last 100 predictions for recency bias
        recent = resolved[-100:]

        # ── Component weight adjustment ───────────────────────────────────
        # For each prediction, figure out which component was most right
        component_correct = {k: 0 for k in DEFAULT_COMPONENT_WEIGHTS}
        component_total   = {k: 0 for k in DEFAULT_COMPONENT_WEIGHTS}

        for p in recent:
            scores = p.get("source_scores", {})
            outcome_dir = (
                1 if p["outcome"] == "CORRECT" and "BUY"  in p["signal"]
                else -1 if p["outcome"] == "CORRECT" and "SELL" in p["signal"]
                else 0
            )
            if outcome_dir == 0:
                continue

            for comp, score in scores.items():
                if comp not in component_total:
                    continue
                component_total[comp] += 1
                # Component was "right" if its score direction matches outcome
                if (score > 0 and outcome_dir > 0) or (score < 0 and outcome_dir < 0):
                    component_correct[comp] += 1

        # Compute new weights proportional to accuracy
        new_weights = {}
        for comp in DEFAULT_COMPONENT_WEIGHTS:
            total = component_total.get(comp, 0)
            if total >= 5:
                acc = component_correct[comp] / total
                # Weight proportional to accuracy, bounded
                new_weights[comp] = max(MIN_WEIGHT, min(MAX_WEIGHT, acc))
            else:
                new_weights[comp] = self._db["component_weights"].get(
                    comp, DEFAULT_COMPONENT_WEIGHTS[comp]
                )

        # Normalise to sum = 1.0
        total_w = sum(new_weights.values())
        if total_w > 0:
            new_weights = {k: round(v / total_w, 3) for k, v in new_weights.items()}
            self._db["component_weights"] = new_weights
            logger.info(f"LearningV2: component weights updated → {new_weights}")

        # ── Asset trust adjustment ────────────────────────────────────────
        for ac in DEFAULT_ASSET_TRUST:
            ac_preds = [p for p in recent if p.get("asset_class") == ac]
            if len(ac_preds) >= MIN_SAMPLES_TO_ADJUST:
                ac_acc = sum(1 for p in ac_preds if p["outcome"] == "CORRECT") / len(ac_preds)
                # Trust = accuracy / 0.5 (normalised around 50% baseline)
                trust = min(1.2, max(0.4, ac_acc / 0.5))
                self._db["asset_trust"][ac] = round(trust, 3)

        self._save()

    # ── Analysis helpers ──────────────────────────────────────────────────────

    def _compute_calibration(self, preds: list) -> dict:
        """
        Confidence calibration: is 70% conf actually 70% accurate?
        Groups predictions by confidence bucket and measures accuracy.
        """
        buckets = {
            "50-60%": (0.50, 0.60),
            "60-70%": (0.60, 0.70),
            "70-80%": (0.70, 0.80),
            "80%+":   (0.80, 1.00),
        }
        calibration = {}
        for label, (lo, hi) in buckets.items():
            bucket_preds = [
                p for p in preds
                if lo <= p.get("confidence", 0) < hi
            ]
            if bucket_preds:
                acc = sum(1 for p in bucket_preds if p["outcome"] == "CORRECT") / len(bucket_preds)
                calibration[label] = {
                    "n":        len(bucket_preds),
                    "accuracy": round(acc, 3),
                    "expected": (lo + hi) / 2,
                    "overconfident": acc < lo,
                }
        return calibration

    def _analyse_errors(self, preds: list) -> dict:
        """Identify top error patterns."""
        errors = [p for p in preds if p["outcome"] == "WRONG"]
        if not errors:
            return {}

        by_signal = {}
        for p in errors:
            sig = p["signal"]
            by_signal[sig] = by_signal.get(sig, 0) + 1

        by_regime = {}
        for p in errors:
            r = p.get("regime", "UNKNOWN")
            by_regime[r] = by_regime.get(r, 0) + 1

        return {
            "total_errors":   len(errors),
            "error_rate":     round(len(errors) / len(preds), 3),
            "by_signal":      by_signal,
            "by_regime":      by_regime,
            "worst_regime":   max(by_regime, key=by_regime.get) if by_regime else None,
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> dict:
        STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        if STORE_PATH.exists():
            try:
                with open(STORE_PATH) as f:
                    data = json.load(f)
                # Ensure all keys exist
                data.setdefault("component_weights", dict(DEFAULT_COMPONENT_WEIGHTS))
                data.setdefault("asset_trust",       dict(DEFAULT_ASSET_TRUST))
                data.setdefault("predictions",       [])
                return data
            except Exception as e:
                logger.warning(f"LearningV2: corrupt store, resetting — {e}")

        return {
            "component_weights": dict(DEFAULT_COMPONENT_WEIGHTS),
            "asset_trust":       dict(DEFAULT_ASSET_TRUST),
            "predictions":       [],
            "version":           2,
        }

    def _save(self):
        try:
            with open(STORE_PATH, "w") as f:
                json.dump(self._db, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"LearningV2 save failed: {e}")


# Module-level singleton
learning_v2 = LearningEngineV2()