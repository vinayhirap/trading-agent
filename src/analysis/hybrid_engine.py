# trading-agent/src/analysis/hybrid_engine.py
"""
Hybrid Prediction Engine — Step 6

Fuses four signal sources into one unified prediction:
  1. Technical (XGBoost+LightGBM ensemble or rule-based)
  2. News sentiment (symbol-linked, parallel-fetched)
  3. Global events (war, pandemic, monetary shock detection)
  4. Trader behavior (session quality, time-of-day patterns)

Dynamic weight allocation:
  Base weights: Technical=40%, News=20%, Events=20%, Behavior=20%

  Weights shift based on context:
    - Market closed      → Behavior→0%, Technical→60%, News→25%, Events→15%
    - Active war event   → Events→35%, Technical→35%, News→20%, Behavior→10%
    - Fresh news < 30min → News→30%, Technical→35%, Events→20%, Behavior→15%
    - Expiry day chaos   → Behavior→35%, Technical→35%, News→20%, Events→10%
    - Low volatility     → Technical→50%, News→20%, Events→15%, Behavior→15%

Output: HybridSignal with:
  - final_bias: BUY / SELL / HOLD / NEUTRAL / STRONG BUY / STRONG SELL
  - final_confidence: 0-1 (pass threshold = 0.52)
  - component_scores: breakdown of each source's contribution
  - reasoning: human-readable explanation of the fusion
  - trade_verdict: PROCEED / CAUTION / AVOID
"""
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo
from loguru import logger

IST = ZoneInfo("Asia/Kolkata")

# ── Base weights ───────────────────────────────────────────────────────────────
BASE_WEIGHTS = {
    "technical":  0.40,
    "news":       0.20,
    "events":     0.20,
    "behavior":   0.20,
}

# Minimum confidence to generate a tradeable signal
MIN_CONFIDENCE = 0.52


@dataclass
class ComponentScore:
    """Score from a single signal source."""
    name:        str
    raw_score:   float    # -1 to +1 (negative = bearish, positive = bullish)
    confidence:  float    # 0-1
    weight:      float    # allocated weight in fusion
    weighted:    float    # raw_score * confidence * weight
    available:   bool     # was this source available?
    notes:       list[str] = field(default_factory=list)


@dataclass
class HybridSignal:
    """Complete hybrid signal output."""
    symbol:             str
    final_bias:         str       # BUY / SELL / HOLD / NEUTRAL / STRONG BUY / STRONG SELL
    final_confidence:   float     # 0-1
    final_score:        float     # -1 to +1 (raw fusion score)
    trade_verdict:      str       # PROCEED / CAUTION / AVOID

    # Component breakdown
    technical:          ComponentScore
    news:               ComponentScore
    events:             ComponentScore
    behavior:           ComponentScore

    # Effective weights (after dynamic adjustment)
    weights_used:       dict      # {"technical": 0.45, ...}

    # Reasoning
    reasoning:          list[str]
    warnings:           list[str]
    opportunities:      list[str]

    # Context
    session_label:      str
    day_of_week:        str
    is_expiry_day:      bool
    active_event_count: int
    news_article_count: int
    computed_at:        str

    def __str__(self):
        arrow = "▲" if "BUY" in self.final_bias else "▼" if "SELL" in self.final_bias else "—"
        return (
            f"{arrow} {self.final_bias} ({self.final_confidence:.0%}) | "
            f"T:{self.technical.weighted:+.2f} "
            f"N:{self.news.weighted:+.2f} "
            f"E:{self.events.weighted:+.2f} "
            f"B:{self.behavior.weighted:+.2f}"
        )

    def to_dict(self) -> dict:
        return {
            "symbol":           self.symbol,
            "final_bias":       self.final_bias,
            "final_confidence": self.final_confidence,
            "final_score":      self.final_score,
            "trade_verdict":    self.trade_verdict,
            "weights":          self.weights_used,
            "reasoning":        self.reasoning,
            "warnings":         self.warnings,
            "session":          self.session_label,
        }


class HybridEngine:
    """
    Fuses technical, news, event, and behavioral signals.

    Usage:
        engine = HybridEngine()
        signal = engine.predict(
            symbol        = "NIFTY50",
            features      = feat_series,        # from FeatureEngine
            df            = ohlcv_df,
            dm            = data_manager,
        )
        print(signal.final_bias, signal.final_confidence)
    """

    def __init__(self, min_confidence: float = MIN_CONFIDENCE):
        self.min_confidence = min_confidence

    # ── Main entry point ──────────────────────────────────────────────────────

    def predict(
        self,
        symbol:    str,
        features=None,        # pd.Series from FeatureEngine
        df=None,              # OHLCV DataFrame
        dm=None,              # DataManager instance
        # Pre-computed inputs (pass if already computed to avoid re-fetching)
        technical_bias:   str   = None,
        technical_conf:   float = None,
        news_sentiment:   dict  = None,   # from NewsIntelligence
        event_summary:    dict  = None,   # from EventEngine
        behavior_reading=None,            # from BehaviorModel
        dt:               datetime = None,
    ) -> HybridSignal:
        """
        Compute hybrid signal for a symbol.

        All inputs are optional — the engine fetches what's missing.
        Pass pre-computed values to avoid redundant API calls.
        """
        ist_now = (dt or datetime.now(IST)).astimezone(IST)

        # ── Step 1: Get technical signal ──────────────────────────────────────
        tech_score = self._get_technical_score(
            symbol, features, df, dm, technical_bias, technical_conf
        )

        # ── Step 2: Get news sentiment ────────────────────────────────────────
        news_score = self._get_news_score(symbol, news_sentiment)

        # ── Step 3: Get event impact ──────────────────────────────────────────
        event_score = self._get_event_score(symbol, event_summary, features, df)

        # ── Step 4: Get behavioral quality ───────────────────────────────────
        behav_score = self._get_behavior_score(symbol, behavior_reading, ist_now)

        # ── Step 5: Compute dynamic weights ──────────────────────────────────
        weights = self._compute_weights(
            tech_score, news_score, event_score, behav_score, ist_now
        )

        # ── Step 6: Apply weights ─────────────────────────────────────────────
        tech_score.weight  = weights["technical"]
        news_score.weight  = weights["news"]
        event_score.weight = weights["events"]
        behav_score.weight = weights["behavior"]

        tech_score.weighted  = tech_score.raw_score  * tech_score.confidence  * tech_score.weight
        news_score.weighted  = news_score.raw_score  * news_score.confidence  * news_score.weight
        event_score.weighted = event_score.raw_score * event_score.confidence * event_score.weight
        behav_score.weighted = behav_score.raw_score * behav_score.confidence * behav_score.weight

        # ── Step 7: Fuse into final score ─────────────────────────────────────
        fusion_score = (
            tech_score.weighted +
            news_score.weighted +
            event_score.weighted +
            behav_score.weighted
        )
        fusion_score = max(-1.0, min(1.0, fusion_score))   # clip to [-1, 1]

        # ── Step 8: Convert score to bias + confidence ────────────────────────
        final_bias, final_confidence = self._score_to_signal(
            fusion_score, tech_score, behav_score
        )

        # ── Step 9: Build reasoning ───────────────────────────────────────────
        reasoning, warnings, opportunities = self._build_reasoning(
            symbol, tech_score, news_score, event_score, behav_score,
            fusion_score, weights
        )

        # ── Step 10: Verdict ──────────────────────────────────────────────────
        verdict = self._get_verdict(
            final_confidence, behav_score, event_score, warnings
        )

        # Context info
        session_label = "Unknown"
        day_of_week   = "Unknown"
        is_expiry     = False
        if behavior_reading:
            session_label = getattr(behavior_reading, "session", session_label)
            if hasattr(session_label, "value"):
                session_label = session_label.value
            day_of_week = getattr(behavior_reading, "day_of_week", day_of_week)
            is_expiry   = getattr(behavior_reading, "is_expiry_day", False)

        event_count = 0
        if event_summary:
            event_count = event_summary.get("event_count", 0)

        news_count = 0
        if news_sentiment:
            news_count = news_sentiment.get("n", 0)

        return HybridSignal(
            symbol             = symbol,
            final_bias         = final_bias,
            final_confidence   = round(final_confidence, 3),
            final_score        = round(fusion_score, 3),
            trade_verdict      = verdict,
            technical          = tech_score,
            news               = news_score,
            events             = event_score,
            behavior           = behav_score,
            weights_used       = weights,
            reasoning          = reasoning,
            warnings           = warnings,
            opportunities      = opportunities,
            session_label      = session_label,
            day_of_week        = day_of_week,
            is_expiry_day      = is_expiry,
            active_event_count = event_count,
            news_article_count = news_count,
            computed_at        = ist_now.strftime("%Y-%m-%d %H:%M IST"),
        )

    # ── Component scorers ─────────────────────────────────────────────────────

    def _get_technical_score(
        self, symbol, features, df, dm,
        technical_bias, technical_conf
    ) -> ComponentScore:
        """Convert technical signal to -1..+1 score."""
        notes = []

        # Use pre-computed if available
        if technical_bias and technical_conf is not None:
            raw  = self._bias_to_score(technical_bias)
            conf = technical_conf
            notes.append(f"Pre-computed: {technical_bias} ({conf:.0%})")
            return ComponentScore("technical", raw, conf, 0.0, 0.0, True, notes)

        # Try asset router → global ensemble → rule-based
        try:
            from src.prediction.asset_trainer import asset_model_router
            if features is not None and asset_model_router:
                pred = asset_model_router.predict(symbol, features)
                if pred:
                    raw  = self._bias_to_score(pred.signal.name)
                    conf = pred.confidence
                    notes.append(f"Asset model: {pred.signal.name} ({conf:.0%})")
                    notes.append(f"Regime: {pred.regime.value}")
                    return ComponentScore("technical", raw, conf, 0.0, 0.0, True, notes)
        except Exception:
            pass

        # Rule-based fallback
        try:
            if features is not None:
                from src.dashboard.app import get_overall_bias
                lat  = features.to_dict() if hasattr(features, "to_dict") else dict(features)
                bias, conf_str, reasons = get_overall_bias(lat)
                raw  = self._bias_to_score(bias)
                # Rule-based confidence: map label to number
                conf_map = {"High conviction": 0.70, "Moderate": 0.55, "Low confidence": 0.40}
                conf = conf_map.get(conf_str, 0.50)
                notes.append(f"Rule-based: {bias} ({conf:.0%})")
                notes.extend(reasons[:2])
                return ComponentScore("technical", raw, conf, 0.0, 0.0, True, notes)
        except Exception:
            pass

        notes.append("Technical signal unavailable")
        return ComponentScore("technical", 0.0, 0.0, 0.0, 0.0, False, notes)

    def _get_news_score(self, symbol, news_sentiment) -> ComponentScore:
        """Convert news sentiment to -1..+1 score."""
        notes = []

        if news_sentiment and isinstance(news_sentiment, dict):
            score  = news_sentiment.get("score", 0.0)
            n      = news_sentiment.get("n", 0)
            label  = news_sentiment.get("label", "NEUTRAL")

            if n == 0:
                notes.append("No news articles found for this symbol")
                return ComponentScore("news", 0.0, 0.1, 0.0, 0.0, False, notes)

            # Confidence based on number of articles
            conf = min(0.90, 0.30 + n * 0.08)
            notes.append(f"{n} articles | sentiment: {label} ({score:+.2f})")

            # Pull headlines if available
            articles = news_sentiment.get("articles", [])
            for a in articles[:2]:
                notes.append(f"  • {a.get('title','')[:60]}")

            return ComponentScore("news", score, conf, 0.0, 0.0, True, notes)

        # Try fetching from news intelligence
        try:
            from src.news.news_intelligence import news_intelligence
            result = news_intelligence.get_symbol_news(symbol, max_age=120, top_n=5)
            sent   = result.get("sentiment", {})
            score  = sent.get("score", 0.0)
            n      = sent.get("n", 0)
            label  = sent.get("label", "NEUTRAL")
            conf   = min(0.85, 0.25 + n * 0.08) if n > 0 else 0.1
            notes.append(f"{n} articles | {label} ({score:+.2f})")
            return ComponentScore("news", score, conf, 0.0, 0.0, n > 0, notes)
        except Exception as e:
            notes.append(f"News fetch failed: {e}")

        return ComponentScore("news", 0.0, 0.0, 0.0, 0.0, False, notes)

    def _get_event_score(self, symbol, event_summary, features, df) -> ComponentScore:
        """Convert event impact to -1..+1 score."""
        notes = []

        # Get event impact from pre-computed summary
        if event_summary and isinstance(event_summary, dict):
            events_list = event_summary.get("events", [])
            if not events_list:
                notes.append("No significant global events")
                return ComponentScore("events", 0.0, 0.5, 0.0, 0.0, True, notes)

            # Find impact for this specific symbol
            try:
                from src.analysis.event_detector import event_detector
                # Reconstruct event objects from summary
                from src.analysis.event_detector import (
                    DetectedEvent, EventType, EventPhase, EVENT_PATTERNS
                )
                mock_events = []
                for ev in events_list[:3]:
                    et = EventType(ev["type"])
                    pattern = EVENT_PATTERNS.get(et, {})
                    mock_events.append(DetectedEvent(
                        event_type       = et,
                        phase            = EventPhase(ev.get("phase", "ACTIVE")),
                        severity         = ev["severity"],
                        confidence       = ev["confidence"],
                        title            = ev["title"],
                        description      = ev["title"],
                        affected_assets  = ev.get("affected", []),
                        signal_dampening = ev.get("dampening", 0.7),
                        historical_analog= ev.get("analog", "Unknown"),
                    ))

                impact = event_detector.get_asset_impact(symbol, mock_events)
                direction = impact["direction"]   # +1, -1, or 0
                dampening = impact["dampening"]   # 0-1

                # Score: direction × dampening (dampening reduces magnitude)
                score = direction * (1 - dampening) * 0.8
                conf  = min(0.85, max(e["severity"] * e["confidence"] / 10
                                      for e in events_list) if events_list else 0.3)

                notes.append(
                    f"{len(events_list)} events | "
                    f"impact: {'bullish' if direction > 0 else 'bearish' if direction < 0 else 'neutral'} | "
                    f"dampening: {dampening:.0%}"
                )
                for r in impact.get("reasoning", [])[:2]:
                    notes.append(f"  • {r[:70]}")

                return ComponentScore("events", score, conf, 0.0, 0.0, True, notes)
            except Exception as e:
                notes.append(f"Event impact calc failed: {e}")

        # Try live detection
        try:
            from src.analysis.event_engine import event_engine
            from src.news.news_intelligence import news_intelligence
            articles = news_intelligence.get_all_linked(max_age=120)
            events   = event_engine.detect_all(articles=articles)
            if events:
                impact    = event_engine._detector.get_asset_impact(symbol, events)
                direction = impact["direction"]
                score     = direction * 0.4
                conf      = min(0.75, max(e.severity * e.confidence / 10 for e in events))
                notes.append(f"{len(events)} live events detected")
                return ComponentScore("events", score, conf, 0.0, 0.0, True, notes)
        except Exception:
            pass

        notes.append("No global events detected")
        return ComponentScore("events", 0.0, 0.3, 0.0, 0.0, True, notes)

    def _get_behavior_score(self, symbol, behavior_reading, ist_now) -> ComponentScore:
        """
        Convert behavioral session quality to a score modifier.

        Behavior doesn't have direction — it amplifies or dampens the signal.
        Score = session_quality * day_bias_modifier (range: -0.2 to +0.2)
        Confidence = session quality itself.
        """
        notes = []

        reading = behavior_reading
        if reading is None:
            try:
                from src.analysis.behavior_model import behavior_model
                reading = behavior_model.get_reading(ist_now)
            except Exception:
                notes.append("Behavior model unavailable")
                return ComponentScore("behavior", 0.0, 0.5, 0.0, 0.0, False, notes)

        q       = reading.characteristics.signal_quality
        action  = reading.recommended_action
        dow_bias = reading.day_bias

        # Day-of-week directional bias (small effect)
        dow_direction = (
            +0.15 if dow_bias == "BULLISH"
            else -0.10 if dow_bias == "BEARISH"
            else 0.0
        )

        # Session-based score modifier (positive = good time to trade)
        if action == "TRADE":
            score = dow_direction + 0.05    # slight upward for good sessions
        elif action in ("AVOID", "EXIT_ONLY"):
            score = -0.20                   # negative modifier in bad sessions
        else:
            score = dow_direction           # neutral modifier

        # Expiry day penalty
        if reading.is_expiry_day:
            score *= 0.7
            notes.append("⚡ Expiry day — extra caution near close")

        notes.append(
            f"Session: {reading.session.value.replace('_',' ')} | "
            f"Quality: {q:.0%} | Action: {action}"
        )
        notes.append(f"Day: {reading.day_of_week} ({dow_bias} bias)")

        return ComponentScore("behavior", score, q, 0.0, 0.0, True, notes)

    # ── Dynamic weight computation ────────────────────────────────────────────

    def _compute_weights(
        self,
        tech, news, event, behav,
        ist_now: datetime,
    ) -> dict:
        """
        Compute dynamic weights based on signal availability and context.
        Weights always sum to 1.0.
        """
        w = dict(BASE_WEIGHTS)

        try:
            from src.analysis.learning_engine_v2 import learning_v2
            base = learning_v2.get_component_weights()
            if base:
                w.update({k: v for k, v in base.items() if k in ("technical", "news", "events", "behavior", "cross_asset")})
        except Exception:
            pass

        # PredictionEngine v2 may learn a cross-asset slot. HybridEngine does not
        # score that component directly, so reserve it if missing and fold it back
        # into technical for backward-compatible four-source fusion.
        if "cross_asset" not in w:
            w["cross_asset"] = 0.05
            w["technical"] = max(0.20, w.get("technical", 0.40) - 0.05)

        # Adjust for unavailable sources
        if not tech.available:
            w["news"]     += w["technical"] * 0.5
            w["events"]   += w["technical"] * 0.3
            w["behavior"] += w["technical"] * 0.2
            w["technical"] = 0.0

        if not news.available:
            w["technical"] += w["news"] * 0.6
            w["events"]    += w["news"] * 0.4
            w["news"]       = 0.0

        # Context-based shifts
        t = ist_now.time()
        from datetime import time as dtime

        # Market closed: behavior → 0, technical dominates
        if t < dtime(9, 15) or t > dtime(15, 30):
            shift = w["behavior"] * 0.8
            w["behavior"]  -= shift
            w["technical"] += shift * 0.6
            w["news"]      += shift * 0.4

        # Active events: event weight rises
        if event.available and event.confidence > 0.6:
            shift = 0.08
            w["events"]   += shift
            w["technical"] -= shift * 0.6
            w["news"]      -= shift * 0.4

        # Fresh news: news weight rises
        if news.available and news.confidence > 0.7:
            shift = 0.07
            w["news"]      += shift
            w["technical"] -= shift

        # Opening chaos: behavior weight rises (session timing matters most)
        if dtime(9, 15) <= t < dtime(9, 30):
            shift = 0.12
            w["behavior"]  += shift
            w["technical"] -= shift * 0.7
            w["news"]      -= shift * 0.3

        if "cross_asset" in w:
            w["technical"] += w.pop("cross_asset")

        # Normalise to sum = 1.0
        total = sum(w.values())
        if total > 0:
            w = {k: round(v / total, 3) for k, v in w.items()}

        return w

    # ── Signal conversion helpers ─────────────────────────────────────────────

    def _bias_to_score(self, bias: str) -> float:
        """Convert text bias to -1..+1 score."""
        bias_u = (bias or "").upper()
        if "STRONG BUY"  in bias_u: return +0.90
        if "BUY"         in bias_u: return +0.60
        if "STRONG SELL" in bias_u: return -0.90
        if "SELL"        in bias_u: return -0.60
        return 0.0

    def _score_to_signal(
        self, score: float,
        tech: ComponentScore,
        behav: ComponentScore,
    ) -> tuple[str, float]:
        """Convert fusion score to bias label + confidence."""
        abs_score = abs(score)

        # Confidence from score magnitude + component confidences
        base_conf = abs_score * 0.7 + tech.confidence * 0.3
        base_conf = min(0.95, base_conf)

        # Behavior acts as a gate on confidence
        behavior_gate = behav.confidence
        final_conf    = base_conf * (0.5 + 0.5 * behavior_gate)
        final_conf    = min(0.95, max(0.05, final_conf))

        if score >= 0.55:    bias = "STRONG BUY"
        elif score >= 0.25:  bias = "BUY"
        elif score <= -0.55: bias = "STRONG SELL"
        elif score <= -0.25: bias = "SELL"
        else:                bias = "HOLD"

        return bias, round(final_conf, 3)

    def _get_verdict(
        self,
        confidence: float,
        behav:  ComponentScore,
        event:  ComponentScore,
        warnings: list[str],
    ) -> str:
        if behav.confidence < 0.30:          return "AVOID"   # bad session
        if confidence < self.min_confidence: return "AVOID"
        if event.raw_score < -0.5:           return "CAUTION" # strong negative event
        if warnings:                          return "CAUTION"
        return "PROCEED"

    # ── Reasoning builder ─────────────────────────────────────────────────────

    def _build_reasoning(
        self, symbol,
        tech, news, event, behav,
        fusion_score, weights,
    ) -> tuple[list, list, list]:
        reasoning    = []
        warnings     = []
        opportunities = []

        # Technical contribution
        direction_t = "bullish" if tech.raw_score > 0 else "bearish" if tech.raw_score < 0 else "neutral"
        reasoning.append(
            f"📊 Technical ({weights['technical']:.0%} weight): "
            f"{direction_t} with {tech.confidence:.0%} confidence"
        )
        for note in tech.notes[:2]:
            reasoning.append(f"   • {note}")

        # News contribution
        if news.available:
            direction_n = "bullish" if news.raw_score > 0.05 else "bearish" if news.raw_score < -0.05 else "neutral"
            reasoning.append(
                f"📰 News ({weights['news']:.0%} weight): "
                f"{news.notes[0] if news.notes else direction_n}"
            )
        else:
            reasoning.append(f"📰 News ({weights['news']:.0%} weight): No articles found")

        # Event contribution
        if event.available and event.confidence > 0.2:
            reasoning.append(
                f"🌍 Events ({weights['events']:.0%} weight): "
                f"{event.notes[0] if event.notes else 'No events'}"
            )
            if event.raw_score < -0.3:
                warnings.append(
                    f"⚠️ Global event creating negative headwind for {symbol}"
                )
            elif event.raw_score > 0.3:
                opportunities.append(
                    f"📈 Global event creating tailwind for {symbol}"
                )

        # Behavior contribution
        reasoning.append(
            f"⏰ Behavior ({weights['behavior']:.0%} weight): "
            f"{behav.notes[0] if behav.notes else 'Unknown session'}"
        )
        if behav.confidence < 0.35:
            warnings.append(
                "⚠️ Poor session quality — technical signals unreliable right now"
            )

        # Overall fusion note
        reasoning.append(
            f"→ Fusion score: {fusion_score:+.2f} | "
            f"Weights: T={weights['technical']:.0%} "
            f"N={weights['news']:.0%} "
            f"E={weights['events']:.0%} "
            f"B={weights['behavior']:.0%}"
        )

        return reasoning, warnings, opportunities


# ── Module-level singleton ────────────────────────────────────────────────────
hybrid_engine = HybridEngine()
