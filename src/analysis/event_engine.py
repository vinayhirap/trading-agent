# trading-agent/src/analysis/event_engine.py
"""
Event Engine — applies global event scenarios to trading signals.

This is the decision layer on top of EventDetector.
It takes detected events + technical signals and outputs:
  1. Adjusted signal confidence (dampened during crises)
  2. Event-specific trade warnings ("Avoid buying during war event")
  3. Scenario-based recommendations per asset
  4. Historical analog context ("This looks like Ukraine 2022")

Key principle: Events don't override signals — they adjust them.
A STRONG BUY during a war event becomes BUY with lower confidence.
An extreme event (pandemic) drops confidence to below threshold = no trade.

Integration points:
  - get_signal_with_ensemble() in app.py reads event adjustments
  - Signal Scanner shows event warnings per symbol
  - AI Insights gets event context for its reasoning
"""
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from loguru import logger

from src.analysis.event_detector import (
    EventDetector, DetectedEvent, EventType, EventPhase,
    EVENT_ASSET_IMPACT, event_detector,
)

# ── Historical scenarios with market behavior templates ───────────────────────
HISTORICAL_SCENARIOS = {
    "Russia-Ukraine-2022": {
        "summary": "Russia invaded Ukraine Feb 24, 2022. Markets initially crashed then partially recovered.",
        "duration_days": 180,
        "phases": {
            "week_1":  "Panic sell. Nifty -5%, Gold +3%, Crude +10%",
            "month_1": "Partial recovery. Commodity surge continues.",
            "month_3": "Markets stabilise. Crude stays elevated.",
            "month_6": "New normal. Energy companies outperform.",
        },
        "key_trades": [
            "Buy GOLD on initial dip (safe haven demand)",
            "Avoid buying equities in week 1",
            "Energy (ONGC, CRUDEOIL) outperform long term",
            "INR weakens — USDINR long profitable",
        ],
        "mistakes_to_avoid": [
            "Buying the first bounce — usually a dead cat",
            "Shorting gold early — it kept rising",
            "Ignoring oil impact on India CAD",
        ],
    },

    "COVID-2020": {
        "summary": "COVID-19 pandemic declared March 2020. Markets crashed 40%, then recovered sharply.",
        "duration_days": 365,
        "phases": {
            "week_1":  "Initial uncertainty. Market down -10%",
            "month_1": "Full panic. Nifty -40% from peak. Crude -60%.",
            "month_3": "Stimulus announced. Markets begin recovery.",
            "month_6": "V-shape recovery begins. IT stocks lead.",
            "year_1":  "Markets at new highs despite ongoing pandemic.",
        },
        "key_trades": [
            "Gold initially fell with everything, then surged",
            "IT sector (WFH beneficiary) recovered first and strongest",
            "Avoid banks and aviation in early phase",
            "BTC crashed then became biggest winner",
        ],
        "mistakes_to_avoid": [
            "Buying crude on first dip — it went to negative prices",
            "Selling everything at the bottom",
            "Missing the IT sector's outperformance",
        ],
    },

    "Fed-Rate-Shock-2022": {
        "summary": "Fed raised rates 425bps in 2022. Most aggressive tightening since 1980.",
        "duration_days": 365,
        "phases": {
            "month_1": "Bond yields spike. Growth stocks crash.",
            "month_3": "Full bear market. Nifty -15%, Nasdaq -35%",
            "month_6": "Gold struggles. INR under pressure.",
            "year_1":  "Markets find bottom. Value stocks outperform.",
        },
        "key_trades": [
            "Short duration bonds",
            "Avoid high-PE growth stocks",
            "Banks outperform initially (higher spreads)",
            "Avoid crypto (liquidity tightening hits hardest)",
        ],
        "mistakes_to_avoid": [
            "Buying gold expecting safe haven — rates hurt gold",
            "Holding long-duration fixed income",
        ],
    },

    "SVB-2023": {
        "summary": "Silicon Valley Bank collapsed March 2023. Banking contagion fears.",
        "duration_days": 30,
        "phases": {
            "day_1_3": "Panic. Regional banks crash. Gold surges.",
            "week_2":  "Fed backstops deposits. Contagion fears ease.",
            "month_1": "Recovery. Markets move on.",
        },
        "key_trades": [
            "Buy gold on the announcement day",
            "Avoid banks in first week",
            "BTC rallied (anti-bank narrative)",
        ],
        "mistakes_to_avoid": [
            "Selling everything — contained quickly",
            "Shorting large banks (only small ones affected)",
        ],
    },

    "OPEC-Cut-2023": {
        "summary": "Saudi Arabia announced surprise 1mbpd production cut, April 2023.",
        "duration_days": 60,
        "phases": {
            "day_1":   "Crude +6%, energy stocks surge",
            "week_1":  "Crude settles +3-4% above pre-cut",
            "month_2": "Impact fades as demand concerns return",
        },
        "key_trades": [
            "Buy crude immediately on announcement",
            "ONGC benefits from higher crude",
            "Aviation stocks hurt (fuel costs)",
            "INR weakens slightly (oil import bill)",
        ],
        "mistakes_to_avoid": [
            "Chasing crude after 2-3 day surge — often mean-reverts",
        ],
    },

    "China-Crypto-Ban-2021": {
        "summary": "China banned crypto mining and trading, May-September 2021.",
        "duration_days": 90,
        "phases": {
            "day_1":   "BTC -30% in weeks following announcement",
            "month_2": "Hash rate migrates out of China",
            "month_3": "Recovery as mining decentralises",
        },
        "key_trades": [
            "Short BTC on announcement",
            "Wait for month-2 entry as hash stabilises",
        ],
        "mistakes_to_avoid": [
            "Buying the initial dip — kept falling",
            "Assuming all crypto equally affected",
        ],
    },
}


@dataclass
class EventAdjustedSignal:
    """
    A trading signal adjusted for global event context.
    """
    symbol:              str
    original_bias:       str        # BUY / SELL / HOLD / NEUTRAL
    original_confidence: float
    adjusted_bias:       str        # may be unchanged or downgraded
    adjusted_confidence: float      # dampened by event severity
    event_dampening:     float      # what we multiplied by
    active_events:       list[DetectedEvent]
    event_warnings:      list[str]  # human-readable warnings
    event_opportunities: list[str]  # event-specific opportunities
    historical_context:  str        # what history says
    trade_verdict:       str        # PROCEED / CAUTION / AVOID / EVENT_OPPORTUNITY


@dataclass
class ScenarioReport:
    """Full scenario analysis for a detected event."""
    event:              DetectedEvent
    analog:             dict          # historical_scenarios entry
    asset_impacts:      dict          # {symbol: {direction, magnitude, reasoning}}
    recommended_trades: list[str]
    avoid_trades:       list[str]
    time_horizon:       str
    confidence_note:    str


class EventEngine:
    """
    Applies event context to trading signals.

    Usage:
        engine = EventEngine()

        # Detect events
        events = engine.detect_all(articles, price_data)

        # Adjust a signal
        adjusted = engine.adjust_signal("NIFTY50", "BUY", 0.72, events)
        # adjusted.adjusted_confidence may be 0.72 * 0.6 = 0.43 (below threshold)

        # Get full scenario report
        report = engine.get_scenario_report(events[0])

        # Get dashboard summary
        summary = engine.get_event_summary(events)
    """

    MIN_CONFIDENCE_TO_TRADE = 0.52

    def __init__(self):
        self._detector = event_detector

    def detect_all(
        self,
        articles: list = None,
        price_data: dict = None,
        max_age: float = 120,
    ) -> list[DetectedEvent]:
        """
        Run both news + price detection, merge and deduplicate.
        """
        news_events  = []
        price_events = []

        if articles:
            try:
                news_events = self._detector.detect_from_news(articles, max_age)
            except Exception as e:
                logger.warning(f"News event detection failed: {e}")

        if price_data:
            try:
                price_events = self._detector.detect_from_price_action(price_data)
            except Exception as e:
                logger.warning(f"Price event detection failed: {e}")

        # Merge: if same event type detected by both, boost confidence
        merged = {}
        for event in news_events + price_events:
            key = event.event_type
            if key in merged:
                existing = merged[key]
                # Boost confidence when confirmed by both signals
                existing.confidence = min(0.95, existing.confidence + 0.15)
                existing.severity   = max(existing.severity, event.severity)
                if event.price_signals:
                    existing.price_signals.update(event.price_signals)
            else:
                merged[key] = event

        result = sorted(merged.values(),
                        key=lambda e: e.severity * e.confidence, reverse=True)
        logger.info(
            f"EventEngine: {len(result)} events detected "
            f"({len(news_events)} news, {len(price_events)} price)"
        )
        return result

    def adjust_signal(
        self,
        symbol:     str,
        bias:       str,
        confidence: float,
        events:     list[DetectedEvent],
    ) -> EventAdjustedSignal:
        """
        Adjust a technical signal based on active global events.

        The adjustment:
          1. Gets asset-specific impact for this symbol
          2. Applies confidence dampening based on event severity
          3. Checks if event creates a contrarian opportunity
          4. Returns verdict: PROCEED / CAUTION / AVOID / EVENT_OPPORTUNITY
        """
        if not events:
            return EventAdjustedSignal(
                symbol              = symbol,
                original_bias       = bias,
                original_confidence = confidence,
                adjusted_bias       = bias,
                adjusted_confidence = confidence,
                event_dampening     = 1.0,
                active_events       = [],
                event_warnings      = [],
                event_opportunities = [],
                historical_context  = "No significant global events detected.",
                trade_verdict       = "PROCEED",
            )

        impact = self._detector.get_asset_impact(symbol, events)

        warnings     = []
        opportunities = []
        dampening    = impact["dampening"]

        # Build warnings
        for event in events:
            if event.severity >= 5 and symbol in event.affected_assets:
                warnings.append(
                    f"⚠️ {event.event_type.value}: {event.title} "
                    f"(severity {event.severity:.0f}/10, {event.phase.value})"
                )

        # Check for event-specific opportunities
        for event in events:
            imp = EVENT_ASSET_IMPACT.get(event.event_type, {}).get(symbol)
            if imp:
                direction, magnitude, reason = imp
                if direction == 1 and magnitude >= 2:
                    opportunities.append(
                        f"📈 {event.event_type.value} historically bullish for {symbol}: {reason}"
                    )
                elif direction == -1 and magnitude >= 2:
                    opportunities.append(
                        f"📉 {event.event_type.value} historically bearish for {symbol}: {reason}"
                    )

        # Adjusted confidence
        adj_conf = round(confidence * dampening, 3)

        # Check signal alignment with event impact
        event_direction = impact["direction"]
        bias_direction  = (
            1 if "BUY" in bias
            else -1 if "SELL" in bias
            else 0
        )

        # If event and signal agree → slight boost
        if event_direction != 0 and event_direction == bias_direction:
            adj_conf = min(0.95, adj_conf * 1.1)
            opportunities.append(
                f"✅ Technical signal aligns with event impact direction"
            )

        # Adjust bias label based on confidence
        adjusted_bias = bias
        if adj_conf < 0.40:
            adjusted_bias = "HOLD"    # confidence too low to act
        elif "STRONG" in bias and adj_conf < 0.55:
            adjusted_bias = bias.replace("STRONG ", "")  # downgrade strong signal

        # Historical context
        most_severe = max(events, key=lambda e: e.severity) if events else None
        hist_context = ""
        if most_severe:
            analog = HISTORICAL_SCENARIOS.get(most_severe.historical_analog, {})
            hist_context = (
                f"Historical analog: {most_severe.historical_analog}. "
                f"{analog.get('summary', '')} "
                f"Week 1 behavior: {analog.get('phases', {}).get('week_1', 'Unknown')}"
            )

        # Verdict
        if not events or all(e.severity < 3 for e in events):
            verdict = "PROCEED"
        elif adj_conf >= self.MIN_CONFIDENCE_TO_TRADE and not warnings:
            verdict = "PROCEED"
        elif opportunities and event_direction == bias_direction:
            verdict = "EVENT_OPPORTUNITY"
        elif adj_conf >= self.MIN_CONFIDENCE_TO_TRADE:
            verdict = "CAUTION"
        else:
            verdict = "AVOID"

        return EventAdjustedSignal(
            symbol              = symbol,
            original_bias       = bias,
            original_confidence = confidence,
            adjusted_bias       = adjusted_bias,
            adjusted_confidence = adj_conf,
            event_dampening     = dampening,
            active_events       = events,
            event_warnings      = warnings,
            event_opportunities = opportunities,
            historical_context  = hist_context,
            trade_verdict       = verdict,
        )

    def get_scenario_report(self, event: DetectedEvent) -> ScenarioReport:
        """Full scenario analysis for a detected event."""
        analog = HISTORICAL_SCENARIOS.get(event.historical_analog, {})

        asset_impacts = {}
        for symbol in event.affected_assets:
            asset_impacts[symbol] = self._detector.get_asset_impact(symbol, [event])

        return ScenarioReport(
            event             = event,
            analog            = analog,
            asset_impacts     = asset_impacts,
            recommended_trades= analog.get("key_trades", []),
            avoid_trades      = analog.get("mistakes_to_avoid", []),
            time_horizon      = f"~{analog.get('duration_days', 30)} days",
            confidence_note   = (
                f"Based on {event.historical_analog} analog. "
                f"Markets are never identical — use as context, not prediction."
            ),
        )

    def get_event_summary(self, events: list[DetectedEvent]) -> dict:
        """
        Dashboard-ready summary of all active events.
        """
        if not events:
            return {
                "has_events":      False,
                "event_count":     0,
                "max_severity":    0,
                "global_dampening":1.0,
                "events":          [],
                "market_regime":   None,
            }

        significant = [e for e in events if e.is_significant]
        max_sev     = max((e.severity for e in events), default=0)
        min_damp    = min((e.signal_dampening for e in events), default=1.0)

        # Check for regime override
        regime_override = None
        for e in events:
            override = e.market_regime_override
            if override:
                regime_override = override
                break

        return {
            "has_events":      len(significant) > 0,
            "event_count":     len(events),
            "significant":     len(significant),
            "max_severity":    max_sev,
            "global_dampening":round(min_damp, 2),
            "market_regime":   regime_override,
            "events": [
                {
                    "type":        e.event_type.value,
                    "title":       e.title,
                    "severity":    e.severity,
                    "confidence":  e.confidence,
                    "phase":       e.phase.value,
                    "analog":      e.historical_analog,
                    "dampening":   e.signal_dampening,
                    "affected":    e.affected_assets[:5],
                    "keywords":    e.keywords_matched[:4],
                }
                for e in events[:5]
            ],
        }


# Module-level singleton
event_engine = EventEngine()