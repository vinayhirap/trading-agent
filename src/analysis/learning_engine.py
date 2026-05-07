# trading-agent/src/analysis/learning_engine.py

"""
Self-Learning Engine — tracks predictions vs actual outcomes.

Records every prediction, checks outcome after N bars,
computes accuracy, detects error patterns, adjusts weights.
Storage: JSON file — no database needed.
"""
import json
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from loguru import logger

LEARNING_DB = Path("data/learning_engine.json")


@dataclass
class PredictionRecord:
    id: str
    symbol: str
    timestamp: str
    signal: str
    confidence: float
    entry_price: float
    horizon_bars: int
    check_time: str
    source: str
    actual_price: Optional[float] = None
    actual_return: Optional[float] = None
    outcome: Optional[str] = None
    checked: bool = False


class LearningEngine:
    """
    Tracks all predictions and computes accuracy over time.
    Identifies what works and what doesn't.
    Provides weight adjustments to improve future signals.
    """

    def __init__(self):
        LEARNING_DB.parent.mkdir(parents=True, exist_ok=True)
        self._db = self._load()
        logger.info(f"LearningEngine | {len(self._db['predictions'])} records")

    def record_prediction(
        self,
        symbol: str,
        signal: str,
        confidence: float,
        entry_price: float,
        horizon_bars: int = 5,
        source: str = "rule",
    ) -> str:
        pred_id = str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc)
        rec = PredictionRecord(
            id=pred_id, symbol=symbol,
            timestamp=now.isoformat(),
            signal=signal,
            confidence=round(confidence, 3),
            entry_price=entry_price,
            horizon_bars=horizon_bars,
            check_time=(now + timedelta(days=horizon_bars)).isoformat(),
            source=source,
        )
        self._db["predictions"].append(asdict(rec))
        self._save()
        return pred_id

    def check_outcomes(self) -> int:
        now = datetime.now(timezone.utc)
        updated = 0
        for rec in self._db["predictions"]:
            if rec.get("checked"):
                continue
            ct = datetime.fromisoformat(rec["check_time"])
            if ct.tzinfo is None:
                ct = ct.replace(tzinfo=timezone.utc)
            if now < ct:
                continue
            px = self._fetch_price(rec["symbol"])
            if px and px > 0:
                entry = rec["entry_price"]
                ret = (px - entry) / entry
                sig = rec["signal"]
                if sig in ("BUY", "GAP_UP"):
                    outcome = "CORRECT" if ret > 0.003 else "WRONG"
                elif sig in ("SELL", "GAP_DOWN"):
                    outcome = "CORRECT" if ret < -0.003 else "WRONG"
                else:
                    outcome = "CORRECT" if abs(ret) < 0.005 else "WRONG"
                rec.update({
                    "actual_price": round(px, 2),
                    "actual_return": round(ret, 5),
                    "outcome": outcome,
                    "checked": True,
                })
                updated += 1
                logger.info(f"Checked {rec['symbol']} {sig} → {outcome} ({ret:+.2%})")
        if updated:
            self._save()
        return updated

    def get_accuracy_stats(self, symbol: str = None) -> dict:
        preds = self._db["predictions"]
        if symbol:
            preds = [p for p in preds if p["symbol"] == symbol]
        checked = [p for p in preds if p.get("checked")]
        pending = [p for p in preds if not p.get("checked")]
        correct = [p for p in checked if p.get("outcome") == "CORRECT"]
        wrong   = [p for p in checked if p.get("outcome") == "WRONG"]
        accuracy = len(correct) / len(checked) if checked else 0.0
        avg_conf = sum(p["confidence"] for p in preds) / len(preds) if preds else 0.0

        by_signal = {}
        for sig in ("BUY", "SELL", "HOLD", "GAP_UP", "GAP_DOWN", "FLAT"):
            sp = [p for p in checked if p["signal"] == sig]
            sk = [p for p in sp if p.get("outcome") == "CORRECT"]
            if sp:
                by_signal[sig] = {
                    "correct": len(sk),
                    "total": len(sp),
                    "accuracy": round(len(sk) / len(sp) * 100, 1),
                }

        best = max(by_signal, key=lambda k: by_signal[k]["accuracy"], default="N/A") if by_signal else "N/A"

        return {
            "symbol": symbol or "ALL",
            "total": len(preds),
            "correct": len(correct),
            "wrong": len(wrong),
            "pending": len(pending),
            "accuracy_pct": round(accuracy * 100, 1),
            "avg_confidence": round(avg_conf * 100, 1),
            "best_signal": best,
            "by_signal": by_signal,
        }

    def get_recent(self, n: int = 20, symbol: str = None) -> list:
        preds = self._db["predictions"]
        if symbol:
            preds = [p for p in preds if p["symbol"] == symbol]
        return sorted(preds, key=lambda p: p["timestamp"], reverse=True)[:n]

    def get_weight_adjustments(self) -> dict:
        checked = [p for p in self._db["predictions"] if p.get("checked")]
        if len(checked) < 10:
            return {"rule": 1.0, "model": 1.0, "ensemble": 1.0, "news": 1.0}
        by_source = {}
        for p in checked:
            src = p.get("source", "rule")
            by_source.setdefault(src, {"correct": 0, "total": 0})
            by_source[src]["total"] += 1
            if p.get("outcome") == "CORRECT":
                by_source[src]["correct"] += 1
        return {
            src: round(min(1.5, max(0.5, 0.6 + s["correct"]/s["total"] * 1.0)), 2)
            for src, s in by_source.items()
        }

    def get_error_patterns(self) -> list:
        wrong = [p for p in self._db["predictions"] if p.get("outcome") == "WRONG"]
        patterns = []
        hc = [p for p in wrong if p["confidence"] > 0.65]
        if hc:
            patterns.append({
                "pattern": "High confidence but wrong",
                "count": len(hc),
                "description": "Model was confident but failed. Raise confidence threshold.",
                "severity": "HIGH",
            })
        for sig in ("BUY", "SELL"):
            sw = [p for p in wrong if p["signal"] == sig]
            st = [p for p in self._db["predictions"] if p.get("checked") and p["signal"] == sig]
            if st and len(sw)/len(st) > 0.6:
                patterns.append({
                    "pattern": f"{sig} signals underperforming",
                    "count": len(sw),
                    "description": f"{sig} accuracy only {(1-len(sw)/len(st))*100:.0f}%. Review logic.",
                    "severity": "MEDIUM",
                })
        return patterns

    def _fetch_price(self, symbol: str) -> Optional[float]:
        try:
            import yfinance as yf
            from src.data.models import ALL_SYMBOLS
            info = ALL_SYMBOLS.get(symbol)
            yf_sym = info.symbol if info else f"{symbol}.NS"
            px = yf.Ticker(yf_sym).fast_info.last_price
            return float(px) if px and px > 0 else None
        except Exception:
            return None

    def _load(self) -> dict:
        if LEARNING_DB.exists():
            try:
                with open(LEARNING_DB) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"predictions": [], "version": 1}

    def _save(self):
        with open(LEARNING_DB, "w") as f:
            json.dump(self._db, f, indent=2)
