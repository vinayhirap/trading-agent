# trading-agent/src/analysis/prediction_engine.py
"""
Prediction Engine — Unified Multi-Factor Inference Layer

Orchestrates all signal sources into one coherent prediction:
  1. Technical (ensemble model or rule-based)
  2. Cross-asset signals (Index↔Equity, USDINR↔Gold, BTC dominance)
  3. News + sentiment (symbol-linked, recency-weighted)
  4. Global events (dampening + direction)
  5. Behavior (session quality, expiry effects)
  6. Learning adjustments (trust scores, weight feedback)

Key improvements over raw HybridEngine:
  - Cross-asset features wired in (S&P500 overnight, USDINR, DXY)
  - Recency weighting for news (< 30 min = 2× weight)
  - Regime-gated signals (blocks BUY in TRENDING_DOWN)
  - Learning engine trust scores applied before final output
  - All reasoning captured per-component for ActionEngine

Output: PredictionResult with full per-component breakdown
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from loguru import logger


@dataclass
class ComponentSignal:
    """Signal contribution from one source."""
    name:       str
    score:      float     # -1 to +1 (negative = bearish)
    confidence: float     # 0-1
    weight:     float     # allocated weight
    weighted:   float     # score × confidence × weight
    available:  bool
    reasoning:  str       # human-readable explanation


@dataclass
class PredictionResult:
    """Full prediction output from PredictionEngine."""
    symbol:        str
    asset_class:   str

    # Final signal
    signal:        str     # BUY / SELL / HOLD / STRONG BUY / STRONG SELL
    confidence:    float   # 0-1 (adjusted for trust and calibration)
    raw_score:     float   # -1 to +1 fusion score before label conversion

    # Component breakdown
    technical:     ComponentSignal
    cross_asset:   ComponentSignal
    news:          ComponentSignal
    events:        ComponentSignal
    behavior:      ComponentSignal

    # Context
    regime:        str
    session:       str
    is_expiry:     bool
    weights_used:  dict

    # Learning adjustments applied
    asset_trust:      float
    conf_multiplier:  float   # applied from learning calibration

    # Regime gate
    regime_blocked:   bool    # True if signal was blocked by regime filter
    regime_block_note:str

    computed_at:   str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def reasoning_dict(self) -> dict:
        """Returns reasoning in the strict output format."""
        return {
            "technical":  self.technical.reasoning,
            "cross_asset":self.cross_asset.reasoning,
            "news":       self.news.reasoning,
            "event":      self.events.reasoning,
            "behavior":   self.behavior.reasoning,
            "regime":     (
                f"Regime: {self.regime} | "
                f"Session: {self.session} | "
                f"Expiry: {'yes' if self.is_expiry else 'no'}"
            ),
        }


class PredictionEngine:
    """
    Unified prediction engine combining all signal sources.

    Usage:
        engine = PredictionEngine()
        result = engine.predict(
            symbol   = "NIFTY50",
            features = feat_series,   # from FeatureEngine
            df       = ohlcv_df,
            dm       = data_manager,
        )
        print(result.signal, result.confidence)
    """

    def __init__(self):
        # Lazy load all dependencies to avoid circular imports
        self._learning  = None
        self._hybrid    = None
        self._behavior  = None
        self._events    = None
        self._news      = None

    def predict(
        self,
        symbol:    str,
        features=None,      # pd.Series from FeatureEngine
        df=None,            # OHLCV DataFrame
        dm=None,
        dt:        datetime = None,
        # Pre-computed (pass to avoid re-fetching)
        technical_bias:   str   = None,
        technical_conf:   float = None,
        news_result:      dict  = None,
        event_summary:    dict  = None,
        behavior_reading=None,
        regime_str:       str   = None,
    ) -> PredictionResult:
        """
        Compute full multi-factor prediction for a symbol.
        """
        from src.data.models import ALL_SYMBOLS
        from zoneinfo import ZoneInfo
        import pandas as pd

        ist_now    = (dt or datetime.now(ZoneInfo("Asia/Kolkata")))
        info       = ALL_SYMBOLS.get(symbol)
        asset_class = info.asset_class.value if info else "equity"

        # ── Load learning engine ─────────────────────────────────────────
        le = self._get_learning()

        # ── Step 1: Technical signal ──────────────────────────────────────
        tech = self._get_technical(
            symbol, features, df, dm, technical_bias, technical_conf
        )

        # ── Step 2: Cross-asset signal ────────────────────────────────────
        cross = self._get_cross_asset(symbol, asset_class, features, df, dm)

        # ── Step 3: News sentiment ────────────────────────────────────────
        news = self._get_news(symbol, news_result)

        # ── Step 4: Event impact ──────────────────────────────────────────
        events = self._get_events(symbol, event_summary, features, df)

        # ── Step 5: Behavior ─────────────────────────────────────────────
        behav, session_label, is_expiry = self._get_behavior(
            symbol, behavior_reading, ist_now
        )

        # ── Step 6: Regime ────────────────────────────────────────────────
        regime = regime_str or self._detect_regime(features)

        # ── Step 7: Dynamic weights (learning-adjusted) ───────────────────
        weights = self._compute_weights(
            le, tech, cross, news, events, behav, ist_now
        )

        # Apply weights
        for comp, key in [
            (tech,   "technical"),
            (cross,  "cross_asset"),
            (news,   "news"),
            (events, "events"),
            (behav,  "behavior"),
        ]:
            comp.weight   = weights.get(key, 0.20)
            comp.weighted = comp.score * comp.confidence * comp.weight

        # ── Step 8: Fuse ──────────────────────────────────────────────────
        fusion = sum([
            tech.weighted, cross.weighted,
            news.weighted, events.weighted, behav.weighted
        ])
        fusion = max(-1.0, min(1.0, fusion))

        # ── Step 9: Convert to signal ─────────────────────────────────────
        signal, raw_conf = self._score_to_signal(fusion)

        # ── Step 10: Apply asset trust from learning ──────────────────────
        asset_trust    = le.get_asset_trust(asset_class) if le else 1.0
        conf_multiplier = asset_trust
        confidence      = min(0.95, raw_conf * conf_multiplier)

        # ── Step 11: Regime gate ──────────────────────────────────────────
        blocked, block_note = self._regime_gate(signal, confidence, regime, le)
        if blocked:
            signal     = "HOLD"
            confidence = min(confidence, 0.45)

        return PredictionResult(
            symbol          = symbol,
            asset_class     = asset_class,
            signal          = signal,
            confidence      = round(confidence, 3),
            raw_score       = round(fusion, 3),
            technical       = tech,
            cross_asset     = cross,
            news            = news,
            events          = events,
            behavior        = behav,
            regime          = regime,
            session         = session_label,
            is_expiry       = is_expiry,
            weights_used    = weights,
            asset_trust     = asset_trust,
            conf_multiplier = round(conf_multiplier, 3),
            regime_blocked  = blocked,
            regime_block_note=block_note,
        )

    # ── Component builders ────────────────────────────────────────────────────

    def _get_technical(
        self, symbol, features, df, dm, bias, conf
    ) -> ComponentSignal:
        """Get technical signal from ensemble or rule-based."""
        if bias and conf is not None:
            score  = self._bias_to_score(bias)
            reason = f"Pre-computed: {bias} ({conf:.0%})"
            return ComponentSignal("technical", score, conf, 0, 0, True, reason)

        # Try asset model router
        try:
            from src.prediction.asset_trainer import asset_model_router
            if features is not None:
                pred = asset_model_router.predict(symbol, features)
                if pred:
                    score  = self._bias_to_score(pred.signal.name)
                    reason = (
                        f"Asset model: {pred.signal.name} ({pred.confidence:.0%}) | "
                        f"XGB={pred.xgb_signal.name} LGB={pred.lgb_signal.name} | "
                        f"B:{pred.buy_prob:.0%} H:{pred.hold_prob:.0%} S:{pred.sell_prob:.0%}"
                    )
                    return ComponentSignal("technical", score, pred.confidence, 0, 0, True, reason)
        except Exception:
            pass

        # Rule-based fallback
        if features is not None:
            score, conf_rb, reason = self._rule_based(features)
            return ComponentSignal("technical", score, conf_rb, 0, 0, True, reason)

        return ComponentSignal("technical", 0.0, 0.0, 0, 0, False, "Technical unavailable")

    def _get_cross_asset(
        self, symbol, asset_class, features, df, dm
    ) -> ComponentSignal:
        """Cross-asset correlation signals."""
        try:
            import numpy as np
            score    = 0.0
            reasons  = []
            n_signals = 0

            feat_dict = {}
            if features is not None:
                feat_dict = features.to_dict() if hasattr(features, "to_dict") else dict(features)

            # ── Index / Equity: S&P500 overnight ──────────────────────────
            if asset_class in ("index", "equity"):
                sp500_ret = float(feat_dict.get("sp500_ret_1d", 0))
                if abs(sp500_ret) > 0.001:
                    direction = 1 if sp500_ret > 0 else -1
                    magnitude = min(abs(sp500_ret) / 0.02, 1.0)
                    score    += direction * magnitude * 0.6
                    reasons.append(
                        f"S&P500 overnight: {sp500_ret:+.1%} → "
                        f"{'bullish' if direction > 0 else 'bearish'} for {symbol}"
                    )
                    n_signals += 1

                # Nifty vs Bank Nifty divergence
                beta = float(feat_dict.get("beta_nifty", 1.0))
                if beta > 1.2:
                    reasons.append(f"High beta ({beta:.1f}×) — amplifies index move")

            # ── Commodity: USDINR impact ──────────────────────────────────
            if asset_class == "futures":
                usdinr_ret = float(feat_dict.get("usdinr_ret_1d", 0))
                fx_impact  = float(feat_dict.get("fx_impact",     0))
                if abs(usdinr_ret) > 0.002:
                    # INR weakening → MCX commodity prices rise in INR terms
                    direction = 1 if usdinr_ret > 0 else -1
                    score    += direction * 0.4
                    reasons.append(
                        f"USDINR: {usdinr_ret:+.1%} → "
                        f"MCX {'gains' if direction > 0 else 'falls'} in INR terms"
                    )
                    n_signals += 1

                # Gold-specific: DXY inverse
                if symbol == "GOLD":
                    safe_haven = float(feat_dict.get("safe_haven_flag", 0))
                    if safe_haven:
                        score    += 0.3
                        reasons.append("Gold safe-haven mode active")
                        n_signals += 1

            # ── Crypto: BTC dominance proxy ───────────────────────────────
            if asset_class == "crypto":
                btc_ret  = float(feat_dict.get("btc_ret_1d",    0))
                btc_corr = float(feat_dict.get("btc_correlation",1.0))
                btc_alpha= float(feat_dict.get("btc_alpha",     0))
                if symbol != "BTC" and abs(btc_ret) > 0.01:
                    # Alt coin follows BTC with correlation weight
                    direction = 1 if btc_ret > 0 else -1
                    score    += direction * btc_corr * 0.5
                    reasons.append(
                        f"BTC: {btc_ret:+.1%} | correlation: {btc_corr:.2f} | "
                        f"alpha: {btc_alpha:+.2%}"
                    )
                    n_signals += 1

            # ── Volatility regime cross-signal ────────────────────────────
            vol_regime = float(feat_dict.get("vol_regime", 1.0))
            if vol_regime > 1.8:
                score    *= 0.7   # reduce signal in high-vol environment
                reasons.append(f"High volatility regime ({vol_regime:.1f}×) — signal dampened")
            elif vol_regime < 0.6:
                reasons.append(f"Low volatility — trend signals more reliable")

            score = max(-1.0, min(1.0, score))
            conf  = min(0.80, 0.30 + n_signals * 0.15)
            reason_str = " | ".join(reasons) if reasons else "No cross-asset signals"

            return ComponentSignal("cross_asset", score, conf, 0, 0, n_signals > 0, reason_str)

        except Exception as e:
            return ComponentSignal("cross_asset", 0.0, 0.1, 0, 0, False, f"Cross-asset failed: {e}")

    def _get_news(self, symbol, news_result) -> ComponentSignal:
        """News sentiment with recency weighting."""
        try:
            # Use pre-computed if available
            if news_result and isinstance(news_result, dict):
                score  = news_result.get("score", 0.0)
                n      = news_result.get("n", 0)
                label  = news_result.get("label", "NEUTRAL")
                articles = news_result.get("articles", [])

                if n == 0:
                    return ComponentSignal("news", 0.0, 0.1, 0, 0, False, "No news found")

                # Recency weighting: articles < 30 min get 2× weight
                weighted_score = 0.0
                total_weight   = 0.0
                for a in articles[:8]:
                    age  = a.get("age_m", 60)
                    w    = 2.0 if age < 30 else 1.5 if age < 60 else 1.0
                    sent = a.get("sentiment_score", score)
                    weighted_score += sent * w
                    total_weight   += w

                final_score = (weighted_score / total_weight) if total_weight > 0 else score
                conf        = min(0.85, 0.30 + n * 0.07)

                # Sentiment strength modifier
                if abs(final_score) > 0.5:
                    reason = (
                        f"STRONG {'BULLISH' if final_score > 0 else 'BEARISH'} news | "
                        f"{n} articles | score: {final_score:+.2f}"
                    )
                else:
                    reason = f"{label} news | {n} articles | score: {final_score:+.2f}"

                return ComponentSignal("news", final_score, conf, 0, 0, True, reason)

            # Live fetch
            ni = self._get_news_intel()
            if ni:
                result     = ni.get_symbol_news(symbol, max_age=120, top_n=8)
                sent       = result.get("sentiment", {})
                score      = sent.get("score", 0.0)
                n          = sent.get("n", 0)
                articles   = sent.get("articles", [])
                conf       = min(0.80, 0.25 + n * 0.07)

                # Recency weighting
                if articles:
                    ws, wt = 0.0, 0.0
                    for a in articles:
                        age = a.get("age_m", 60)
                        w   = 2.0 if age < 30 else 1.0
                        ws += a.get("sentiment_score", score) * w
                        wt += w
                    score = ws / wt if wt > 0 else score

                reason = f"{n} articles | score: {score:+.2f}"
                return ComponentSignal("news", score, conf, 0, 0, n > 0, reason)

        except Exception as e:
            pass

        return ComponentSignal("news", 0.0, 0.0, 0, 0, False, "News unavailable")

    def _get_events(self, symbol, event_summary, features, df) -> ComponentSignal:
        """Global event impact."""
        try:
            if event_summary and isinstance(event_summary, dict):
                events_list = event_summary.get("events", [])
                dampening   = event_summary.get("global_dampening", 1.0)

                if not events_list:
                    return ComponentSignal(
                        "events", 0.0, 0.3, 0, 0, True, "No significant events"
                    )

                # Get symbol-specific direction from event impact
                from src.analysis.event_detector import event_detector, DetectedEvent, EventType, EventPhase, EVENT_PATTERNS

                mock_events = []
                for ev in events_list[:3]:
                    try:
                        mock_events.append(DetectedEvent(
                            event_type        = EventType(ev["type"]),
                            phase             = EventPhase(ev.get("phase", "ACTIVE")),
                            severity          = ev["severity"],
                            confidence        = ev["confidence"],
                            title             = ev["title"],
                            description       = ev["title"],
                            affected_assets   = ev.get("affected", []),
                            signal_dampening  = ev.get("dampening", 0.7),
                            historical_analog = ev.get("analog", "Unknown"),
                        ))
                    except Exception:
                        continue

                impact    = event_detector.get_asset_impact(symbol, mock_events)
                direction = impact["direction"]
                score     = direction * (1 - dampening) * 0.8
                score     = max(-1.0, min(1.0, score))
                conf      = min(0.80, max(ev["severity"] * ev["confidence"] / 10
                                          for ev in events_list))

                top_event = events_list[0]["title"]
                reason    = (
                    f"{top_event} | "
                    f"Impact: {'bullish' if direction > 0 else 'bearish' if direction < 0 else 'neutral'} | "
                    f"Dampening: {dampening:.0%}"
                )
                if impact.get("reasoning"):
                    reason += f" | {impact['reasoning'][0][:60]}"

                return ComponentSignal("events", score, conf, 0, 0, True, reason)

        except Exception:
            pass

        return ComponentSignal("events", 0.0, 0.2, 0, 0, False, "No events detected")

    def _get_behavior(self, symbol, behavior_reading, ist_now):
        """Session quality and behavioral bias."""
        try:
            reading = behavior_reading
            if reading is None:
                bm = self._get_behavior_model()
                if bm:
                    reading = bm.get_reading(ist_now)

            if reading:
                q      = reading.characteristics.signal_quality
                action = reading.recommended_action
                dow    = reading.day_bias
                expiry = reading.is_expiry_day

                dow_score = (
                    +0.12 if dow == "BULLISH"
                    else -0.08 if dow == "BEARISH"
                    else 0.0
                )

                session_penalty = (
                    -0.25 if action in ("AVOID", "EXIT_ONLY")
                    else -0.10 if action == "WAIT"
                    else 0.0
                )

                score  = dow_score + session_penalty
                if expiry:
                    score *= 0.7

                session_label = reading.characteristics.label
                reason = (
                    f"Session: {session_label} | Quality: {q:.0%} | "
                    f"Day: {reading.day_of_week} ({dow} bias) | "
                    f"Action: {action}"
                )
                if expiry:
                    reason += " | ⚡ EXPIRY DAY"

                return (
                    ComponentSignal("behavior", score, q, 0, 0, True, reason),
                    session_label,
                    expiry,
                )
        except Exception:
            pass

        return (
            ComponentSignal("behavior", 0.0, 0.5, 0, 0, False, "Behavior unavailable"),
            "UNKNOWN",
            False,
        )

    # ── Weight computation ────────────────────────────────────────────────────

    def _compute_weights(self, le, tech, cross, news, events, behav, ist_now) -> dict:
        """Compute weights from learning engine + context."""
        from datetime import time as dtime
        from zoneinfo import ZoneInfo

        # Start with learning-adjusted weights
        if le:
            base = le.get_component_weights()
        else:
            base = {
                "technical":   0.35,
                "cross_asset": 0.15,
                "news":        0.20,
                "events":      0.15,
                "behavior":    0.15,
            }

        # Add cross_asset if not present
        base.setdefault("cross_asset", 0.10)

        # Context adjustments
        t = ist_now.time() if hasattr(ist_now, "time") else ist_now

        # Market closed → behavior → 0
        if t < dtime(9, 15) or t > dtime(15, 30):
            base["behavior"]   = max(0.02, base.get("behavior", 0.15) * 0.1)

        # Strong event → event weight up
        if events.available and events.confidence > 0.65:
            base["events"]     = min(0.35, base.get("events", 0.15) + 0.10)
            base["technical"]  = max(0.20, base.get("technical", 0.35) - 0.05)

        # Fresh strong news → news weight up
        if news.available and news.confidence > 0.65:
            base["news"]       = min(0.30, base.get("news", 0.20) + 0.08)

        # Unavailable sources → redistribute
        for comp_name, comp in [
            ("technical", tech), ("cross_asset", cross),
            ("news", news), ("events", events), ("behavior", behav)
        ]:
            if not comp.available:
                w = base.pop(comp_name, 0)
                base["technical"] = base.get("technical", 0) + w * 0.6
                base["news"]      = base.get("news", 0)      + w * 0.4

        # Normalise
        total = sum(base.values())
        if total > 0:
            base = {k: round(v / total, 3) for k, v in base.items()}

        return base

    # ── Regime gate ───────────────────────────────────────────────────────────

    def _regime_gate(
        self, signal: str, confidence: float, regime: str, le
    ) -> tuple[bool, str]:
        """
        Block signals that go against the regime.
        Uses misclassification data from learning engine.
        """
        block_threshold = 0.72   # only block if confidence < this

        if confidence > block_threshold:
            return False, ""   # high confidence signals pass through

        if "BUY" in signal and regime == "TRENDING_DOWN":
            # Check if learning engine has seen this pattern fail
            if le:
                stats = le.get_accuracy_stats()
                errors = stats.get("errors", {})
                by_regime = errors.get("by_regime", {})
                if by_regime.get("TRENDING_DOWN", 0) >= 3:
                    return True, f"Blocked: BUY in {regime} historically wrong (learning data)"
            return True, f"Blocked: BUY against {regime} trend (confidence {confidence:.0%} < {block_threshold:.0%})"

        if "SELL" in signal and regime == "TRENDING_UP":
            return True, f"Blocked: SELL against {regime} trend (confidence {confidence:.0%} < {block_threshold:.0%})"

        return False, ""

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _score_to_signal(self, score: float) -> tuple[str, float]:
        abs_s = abs(score)
        conf  = min(0.92, 0.35 + abs_s * 0.60)
        if   score >= 0.55: return "STRONG BUY",  conf
        elif score >= 0.25: return "BUY",          conf
        elif score <= -0.55:return "STRONG SELL",  conf
        elif score <= -0.25:return "SELL",          conf
        return "HOLD", min(conf, 0.50)

    def _bias_to_score(self, bias: str) -> float:
        b = (bias or "").upper()
        if "STRONG BUY"  in b: return +0.85
        if "BUY"         in b: return +0.55
        if "STRONG SELL" in b: return -0.85
        if "SELL"        in b: return -0.55
        return 0.0

    def _detect_regime(self, features) -> str:
        try:
            if features is None:
                return "RANGING"
            adx    = float(features.get("adx", 20))
            e9     = float(features.get("ema9_pct", 0))
            di_diff= float(features.get("di_diff", 0))
            if adx > 25 and di_diff > 0:   return "TRENDING_UP"
            if adx > 25 and di_diff < 0:   return "TRENDING_DOWN"
            atr_r  = float(features.get("atr_ratio", 1.0))
            if atr_r > 1.8:                return "VOLATILE"
            return "RANGING"
        except Exception:
            return "RANGING"

    def _rule_based(self, features) -> tuple[float, float, str]:
        """Simple rule-based signal as fallback."""
        try:
            rsi    = float(features.get("rsi_14", 50))
            macd_h = float(features.get("macd_hist", 0))
            e9     = float(features.get("ema9_pct", 0))
            e50    = float(features.get("ema50_pct", 0))
            adx    = float(features.get("adx", 20))

            score = 0
            if rsi < 35:   score += 2
            elif rsi > 65: score -= 2
            if macd_h > 0: score += 1
            elif macd_h < 0: score -= 1
            if e9 > 0 and e50 > 0:   score += 2
            elif e9 < 0 and e50 < 0: score -= 2

            norm  = max(-1.0, min(1.0, score / 5))
            conf  = 0.50 + abs(norm) * 0.15
            reason = f"Rule: RSI={rsi:.0f} MACD={'pos' if macd_h > 0 else 'neg'} EMA={'bull' if e9 > 0 else 'bear'}"
            return norm, conf, reason
        except Exception:
            return 0.0, 0.40, "Rule-based unavailable"

    # ── Lazy singleton getters ────────────────────────────────────────────────

    def _get_learning(self):
        if self._learning is None:
            try:
                from src.analysis.learning_engine_v2 import learning_v2
                self._learning = learning_v2
            except Exception:
                pass
        return self._learning

    def _get_news_intel(self):
        if self._news is None:
            try:
                from src.news.news_intelligence import news_intelligence
                self._news = news_intelligence
            except Exception:
                pass
        return self._news

    def _get_behavior_model(self):
        if self._behavior is None:
            try:
                from src.analysis.behavior_model import behavior_model
                self._behavior = behavior_model
            except Exception:
                pass
        return self._behavior


# Module-level singleton
prediction_engine = PredictionEngine()