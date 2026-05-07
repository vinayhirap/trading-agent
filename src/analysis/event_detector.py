# trading-agent/src/analysis/event_detector.py
"""
Global Event Detector

Detects and classifies macro events from:
  1. News headlines (keyword pattern matching)
  2. Price action signatures (sudden moves, vol spikes)
  3. Cross-asset correlations breaking down

Event taxonomy:
  GEOPOLITICAL   — war, conflict, sanctions, coup
  PANDEMIC       — disease outbreak, lockdown, health emergency
  MONETARY       — central bank decision, rate shock, currency crisis
  FINANCIAL      — banking crisis, market crash, liquidity crunch
  ENERGY         — oil shock, supply disruption, pipeline attack
  NATURAL        — earthquake, hurricane, flood (regional impact)
  ELECTION       — election outcome, political transition
  TECH_REGULATION — crypto ban, AI regulation, data law

Each detected event has:
  - type: EventType enum
  - severity: 1-10 (10 = market-moving like COVID crash)
  - confidence: 0-1 (how sure we are it's real)
  - affected_assets: which symbols are most impacted
  - phase: EMERGING / ACTIVE / FADING
  - historical_analog: closest historical precedent
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from loguru import logger


class EventType(str, Enum):
    GEOPOLITICAL    = "GEOPOLITICAL"
    PANDEMIC        = "PANDEMIC"
    MONETARY        = "MONETARY"
    FINANCIAL       = "FINANCIAL"
    ENERGY          = "ENERGY"
    NATURAL         = "NATURAL"
    ELECTION        = "ELECTION"
    TECH_REGULATION = "TECH_REGULATION"
    NONE            = "NONE"


class EventPhase(str, Enum):
    EMERGING = "EMERGING"    # just detected, uncertain
    ACTIVE   = "ACTIVE"      # confirmed, markets reacting
    FADING   = "FADING"      # markets absorbing, normalising


@dataclass
class DetectedEvent:
    event_type:        EventType
    phase:             EventPhase
    severity:          float          # 1-10
    confidence:        float          # 0-1
    title:             str            # human-readable description
    description:       str
    affected_assets:   list[str]      # symbols most affected
    signal_dampening:  float          # multiply signal confidence by this (0-1)
    historical_analog: str            # "COVID-2020", "Ukraine-2022", etc.
    detected_at:       datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    keywords_matched:  list[str] = field(default_factory=list)
    price_signals:     dict = field(default_factory=dict)

    @property
    def is_significant(self) -> bool:
        return self.severity >= 5 and self.confidence >= 0.5

    @property
    def market_regime_override(self) -> Optional[str]:
        """If severe enough, override regime detection with event regime."""
        if self.severity >= 8:
            return "VOLATILE"
        if self.severity >= 6:
            return "VOLATILE"
        return None

    def __str__(self):
        icons = {
            EventType.GEOPOLITICAL:   "⚔️",
            EventType.PANDEMIC:       "🦠",
            EventType.MONETARY:       "🏦",
            EventType.FINANCIAL:      "💥",
            EventType.ENERGY:         "🛢️",
            EventType.NATURAL:        "🌊",
            EventType.ELECTION:       "🗳️",
            EventType.TECH_REGULATION:"📋",
            EventType.NONE:           "✅",
        }
        icon = icons.get(self.event_type, "⚠️")
        return (
            f"{icon} {self.event_type.value} | "
            f"Severity: {self.severity:.0f}/10 | "
            f"Confidence: {self.confidence:.0%} | "
            f"{self.phase.value}"
        )


# ── Keyword patterns per event type ──────────────────────────────────────────
# Each entry: (keywords_required, keywords_boosting, severity_base)
EVENT_PATTERNS = {
    EventType.GEOPOLITICAL: {
        "required":  ["war", "attack", "invasion", "military", "conflict",
                      "missile", "bomb", "troops", "strike", "airstrike",
                      "sanction", "blockade", "coup", "nuclear"],
        "boosting":  ["iran", "russia", "china", "ukraine", "israel",
                      "pakistan", "north korea", "nato", "un security"],
        "severity":  6.0,
        "analog":    "Russia-Ukraine-2022",
        "dampening": 0.6,
    },
    EventType.PANDEMIC: {
        "required":  ["pandemic", "outbreak", "virus", "disease", "epidemic",
                      "lockdown", "quarantine", "health emergency",
                      "who declares", "novel virus", "contagion"],
        "boosting":  ["global", "spread", "mortality", "cases rising",
                      "travel ban", "border close"],
        "severity":  8.0,
        "analog":    "COVID-2020",
        "dampening": 0.4,
    },
    EventType.MONETARY: {
        "required":  ["rate hike", "rate cut", "fed decision", "rbi decision",
                      "ecb decision", "emergency rate", "quantitative",
                      "taper", "hawkish surprise", "dovish surprise",
                      "interest rate shock"],
        "boosting":  ["50 basis", "75 basis", "100 basis", "emergency",
                      "surprise", "unexpected"],
        "severity":  4.0,
        "analog":    "Fed-Rate-Shock-2022",
        "dampening": 0.75,
    },
    EventType.FINANCIAL: {
        "required":  ["bank collapse", "bank failure", "banking crisis",
                      "credit crunch", "liquidity crisis", "default",
                      "market crash", "circuit breaker", "black monday",
                      "lehman", "systemic risk"],
        "boosting":  ["contagion", "bailout", "fdic", "rbi intervention",
                      "sebi halt"],
        "severity":  7.5,
        "analog":    "SVB-2023",
        "dampening": 0.5,
    },
    EventType.ENERGY: {
        "required":  ["oil shock", "pipeline attack", "refinery attack",
                      "opec emergency", "energy crisis", "gas shortage",
                      "supply disruption", "oil embargo"],
        "boosting":  ["saudi", "iran", "russia gas", "strategic reserve",
                      "eia emergency"],
        "severity":  5.5,
        "analog":    "OPEC-Cut-2023",
        "dampening": 0.7,
    },
    EventType.NATURAL: {
        "required":  ["earthquake", "tsunami", "hurricane", "cyclone",
                      "flood", "wildfire", "volcano", "natural disaster"],
        "boosting":  ["major city", "port disruption", "supply chain",
                      "oil facility", "refinery damage"],
        "severity":  4.0,
        "analog":    "Japan-Earthquake-2011",
        "dampening": 0.8,
    },
    EventType.ELECTION: {
        "required":  ["election result", "election outcome", "won election",
                      "political crisis", "government collapse",
                      "hung parliament", "snap election"],
        "boosting":  ["india election", "us election", "surprise win",
                      "market reaction", "policy change"],
        "severity":  3.5,
        "analog":    "India-Election-2024",
        "dampening": 0.8,
    },
    EventType.TECH_REGULATION: {
        "required":  ["crypto ban", "bitcoin ban", "crypto regulation",
                      "sec charges", "exchange shutdown", "ai regulation",
                      "antitrust", "big tech break"],
        "boosting":  ["binance", "coinbase", "sec", "china ban",
                      "india crypto", "cbdc replace"],
        "severity":  5.0,
        "analog":    "China-Crypto-Ban-2021",
        "dampening": 0.65,
    },
}

# ── Asset impact table per event type ─────────────────────────────────────────
# Direction: +1 = typically rises, -1 = typically falls, 0 = neutral/unclear
# Magnitude: 1=small, 2=medium, 3=large move expected
EVENT_ASSET_IMPACT = {
    EventType.GEOPOLITICAL: {
        "GOLD":      (+1, 3, "safe haven demand surges"),
        "SILVER":    (+1, 2, "safe haven, industrial demand falls"),
        "CRUDEOIL":  (+1, 3, "supply disruption fear"),
        "NIFTY50":   (-1, 2, "risk-off, FII selling"),
        "BANKNIFTY": (-1, 2, "credit risk rises"),
        "USDINR":    (+1, 2, "INR weakens (risk-off)"),
        "BTC":       (-1, 1, "initial sell-off, then sometimes recovery"),
    },
    EventType.PANDEMIC: {
        "GOLD":      (+1, 2, "safe haven"),
        "CRUDEOIL":  (-1, 3, "demand collapse"),
        "NIFTY50":   (-1, 3, "broad market crash"),
        "BANKNIFTY": (-1, 3, "NPA fears, credit stress"),
        "USDINR":    (+1, 3, "INR crashes"),
        "BTC":       (-1, 2, "initial crash, then recovery if stimulus"),
        "NATURALGAS":(-1, 2, "industrial demand drops"),
    },
    EventType.MONETARY: {
        "GOLD":      (-1, 2, "higher rates = lower gold (opportunity cost)"),
        "SILVER":    (-1, 2, "follows gold"),
        "NIFTY50":   (-1, 2, "P/E compression on rate hike"),
        "BANKNIFTY": (+1, 1, "banks benefit from higher spreads (short term)"),
        "USDINR":    (-1, 2, "INR strengthens if RBI hikes"),
        "BTC":       (-1, 2, "liquidity tightening hits risk assets"),
        "CRUDEOIL":  (-1, 1, "demand reduction from slowdown"),
    },
    EventType.FINANCIAL: {
        "GOLD":      (+1, 3, "flight to safety"),
        "NIFTY50":   (-1, 3, "systemic fear"),
        "BANKNIFTY": (-1, 3, "direct impact on banks"),
        "USDINR":    (+1, 2, "INR weakens"),
        "BTC":       (-1, 2, "deleveraging hits all risk assets"),
        "CRUDEOIL":  (-1, 2, "demand destruction fears"),
    },
    EventType.ENERGY: {
        "CRUDEOIL":  (+1, 3, "direct supply shock"),
        "NATURALGAS":(+1, 2, "energy substitute demand"),
        "GOLD":      (+1, 1, "inflation hedge"),
        "NIFTY50":   (-1, 2, "India imports oil, CAD widens"),
        "USDINR":    (+1, 2, "INR weakens on higher oil import bill"),
        "SILVER":    (+1, 1, "industrial metal, energy input costs"),
    },
    EventType.NATURAL: {
        "CRUDEOIL":  (+1, 1, "if oil infrastructure affected"),
        "GOLD":      (+1, 1, "mild safe haven"),
        "NIFTY50":   (-1, 1, "mild risk-off"),
    },
    EventType.ELECTION: {
        "NIFTY50":   (0, 2, "direction depends on winner"),
        "BANKNIFTY": (0, 2, "policy uncertainty"),
        "USDINR":    (0, 1, "policy uncertainty"),
    },
    EventType.TECH_REGULATION: {
        "BTC":       (-1, 3, "direct regulatory risk"),
        "ETH":       (-1, 2, "follows BTC"),
        "NIFTY50":   (-1, 1, "sentiment impact"),
    },
}


def compute_dampening(severity: float, event_type_str: str = "GEOPOLITICAL") -> float:
    """
    Dynamic signal dampening by severity + event type.
    Replaces hardcoded 0.6 used everywhere.

    Returns fraction to KEEP (0.4 = 40% of normal signal strength).
    1.0 = no dampening, 0.05 = 95% dampened.
    """
    if severity <= 3:
        base = 0.80 - (severity - 1) * 0.10    # 0.80, 0.70, 0.60  (mild)
    elif severity <= 6:
        base = 0.55 - (severity - 4) * 0.08    # 0.55, 0.47, 0.39  (moderate)
    elif severity <= 9:
        base = 0.30 - (severity - 7) * 0.08    # 0.30, 0.22, 0.14  (major)
    else:
        base = 0.05                              # catastrophic

    # Event type modifier
    type_mod = {
        "PANDEMIC":       0.90,   # pandemics damp the most
        "GEOPOLITICAL":   0.95,
        "FINANCIAL":      0.90,
        "MONETARY":       1.10,   # central bank → less damping
        "ENERGY":         1.05,
        "NATURAL":        1.10,
        "ELECTION":       1.15,
        "TECH_REGULATION":1.20,
    }
    mod  = type_mod.get(event_type_str, 1.0)
    return round(min(0.95, max(0.05, base * mod)), 2)


def compute_dampening(severity: float, event_type_str: str = "GEOPOLITICAL") -> float:
    """
    Dynamic signal dampening by severity + event type.
    Replaces hardcoded 0.6 used everywhere.
 
    Returns fraction to KEEP (0.4 = 40% of normal signal strength).
    1.0 = no dampening, 0.05 = 95% dampened.
    """
    if severity <= 3:
        base = 0.80 - (severity - 1) * 0.10    # 0.80, 0.70, 0.60  (mild)
    elif severity <= 6:
        base = 0.55 - (severity - 4) * 0.08    # 0.55, 0.47, 0.39  (moderate)
    elif severity <= 9:
        base = 0.30 - (severity - 7) * 0.08    # 0.30, 0.22, 0.14  (major)
    else:
        base = 0.05                              # catastrophic
 
    # Event type modifier
    type_mod = {
        "PANDEMIC":       0.90,   # pandemics damp the most
        "GEOPOLITICAL":   0.95,
        "FINANCIAL":      0.90,
        "MONETARY":       1.10,   # central bank → less damping
        "ENERGY":         1.05,
        "NATURAL":        1.10,
        "ELECTION":       1.15,
        "TECH_REGULATION":1.20,
    }
    mod  = type_mod.get(event_type_str, 1.0)
    return round(min(0.95, max(0.05, base * mod)), 2)


class EventDetector:
    """
    Detects global macro events from news + price action.

    Two detection modes:
      1. News-driven: scan headlines for event keywords
      2. Price-driven: detect anomalous price moves across assets
      3. Combined: both signals reinforce each other
    """

    def __init__(self):
        self._active_events: list[DetectedEvent] = []
        self._last_scan: datetime = None

    def detect_from_news(
        self,
        articles: list,       # list of LinkedArticle or ScoredNewsItem
        max_age_minutes: float = 120,
    ) -> list[DetectedEvent]:
        """
        Scan news articles for global event patterns.
        Returns list of DetectedEvent (may be empty).
        """
        # Collect all text from recent articles
        recent = [
            a for a in articles
            if getattr(a, "age_minutes", 0) <= max_age_minutes
        ]

        if not recent:
            return []

        all_text = " ".join(
            f"{getattr(a, 'title', '')} {getattr(a, 'summary', '')}"
            for a in recent
        ).lower()

        events = []
        for event_type, pattern in EVENT_PATTERNS.items():
            result = self._check_pattern(all_text, event_type, pattern, recent)
            if result:
                events.append(result)

        # Sort by severity × confidence
        events.sort(key=lambda e: e.severity * e.confidence, reverse=True)
        self._active_events = events
        return events

    def detect_from_price_action(
        self,
        price_data: dict,   # {symbol: {"ret_1d": float, "atr_ratio": float, ...}}
    ) -> list[DetectedEvent]:
        """
        Detect events from unusual price movements.

        Signatures:
          - Gold +3%+ AND Nifty -2%+ same day → geopolitical/financial stress
          - Crude +5%+ → energy shock
          - Nifty -5%+ with high volume → financial/panic
          - BTC -15%+ → crypto-specific event
          - USDINR +1%+ in one day → currency/macro stress
        """
        events = []

        gold_ret   = price_data.get("GOLD",    {}).get("ret_1d", 0)
        crude_ret  = price_data.get("CRUDEOIL",{}).get("ret_1d", 0)
        nifty_ret  = price_data.get("NIFTY50", {}).get("ret_1d", 0)
        btc_ret    = price_data.get("BTC",      {}).get("ret_1d", 0)
        usdinr_ret = price_data.get("USDINR",   {}).get("ret_1d", 0)
        nifty_atr  = price_data.get("NIFTY50",  {}).get("atr_ratio", 1.0)

        # Pattern: gold up + nifty down = stress event
        if gold_ret > 0.02 and nifty_ret < -0.015:
            events.append(self._make_price_event(
                EventType.GEOPOLITICAL,
                severity    = min(9, 5 + abs(nifty_ret) * 100),
                confidence  = 0.55,
                title       = "Risk-off: Gold rising, Nifty falling",
                description = (
                    f"Gold +{gold_ret:.1%} while Nifty {nifty_ret:.1%} suggests "
                    f"flight to safety. Possible geopolitical or financial stress."
                ),
                price_signals = {"GOLD": gold_ret, "NIFTY50": nifty_ret},
            ))

        # Pattern: crude spike > 4%
        if crude_ret > 0.04:
            events.append(self._make_price_event(
                EventType.ENERGY,
                severity    = min(8, 4 + crude_ret * 60),
                confidence  = 0.65,
                title       = f"Crude oil shock: +{crude_ret:.1%}",
                description = (
                    f"Crude up {crude_ret:.1%} in one session. "
                    f"Supply disruption or geopolitical event likely."
                ),
                price_signals = {"CRUDEOIL": crude_ret},
            ))

        # Pattern: nifty -4%+ (circuit breaker territory)
        if nifty_ret < -0.04:
            events.append(self._make_price_event(
                EventType.FINANCIAL,
                severity    = min(10, 5 + abs(nifty_ret) * 80),
                confidence  = 0.60,
                title       = f"Market crash signal: Nifty {nifty_ret:.1%}",
                description = (
                    f"Nifty down {nifty_ret:.1%} — approaching circuit breaker range. "
                    f"Systemic or macro shock likely."
                ),
                price_signals = {"NIFTY50": nifty_ret},
            ))

        # Pattern: INR weakness > 0.8% in one day
        if usdinr_ret > 0.008:
            events.append(self._make_price_event(
                EventType.MONETARY,
                severity    = min(7, 3 + usdinr_ret * 200),
                confidence  = 0.55,
                title       = f"INR under pressure: +{usdinr_ret:.1%}",
                description = (
                    f"USDINR rose {usdinr_ret:.1%} — significant INR weakness. "
                    f"FII outflows, oil shock, or global risk-off."
                ),
                price_signals = {"USDINR": usdinr_ret},
            ))

        # Pattern: BTC crash > 8%
        if btc_ret < -0.08:
            events.append(self._make_price_event(
                EventType.TECH_REGULATION,
                severity    = min(8, 4 + abs(btc_ret) * 30),
                confidence  = 0.50,
                title       = f"Crypto crash: BTC {btc_ret:.1%}",
                description = (
                    f"Bitcoin down {btc_ret:.1%} — regulatory action, "
                    f"exchange issue, or macro deleveraging."
                ),
                price_signals = {"BTC": btc_ret},
            ))

        return events

    def get_active_events(self) -> list[DetectedEvent]:
        """Return currently active events."""
        return self._active_events

    def get_most_severe(self) -> Optional[DetectedEvent]:
        """Return the single most severe active event."""
        if not self._active_events:
            return None
        return max(self._active_events, key=lambda e: e.severity * e.confidence)

    def get_asset_impact(
        self,
        symbol: str,
        events: list[DetectedEvent] = None,
    ) -> dict:
        """
        Get expected impact on a specific asset from all active events.

        Returns:
          {
            "direction":  +1 | -1 | 0,
            "magnitude":  1-3,
            "reasoning":  [str, ...],
            "dampening":  float,  # multiply signal confidence by this
            "events":     [str, ...],
          }
        """
        events = events or self._active_events
        if not events:
            return {
                "direction": 0, "magnitude": 0,
                "reasoning": [], "dampening": 1.0, "events": [],
            }

        net_direction = 0.0
        max_magnitude = 0
        all_reasoning = []
        min_dampening = 1.0
        event_names   = []

        for event in events:
            impact = EVENT_ASSET_IMPACT.get(event.event_type, {}).get(symbol)
            if impact:
                direction, magnitude, reason = impact
                weight        = event.severity * event.confidence / 10
                net_direction += direction * weight * magnitude
                max_magnitude  = max(max_magnitude, magnitude)
                all_reasoning.append(
                    f"{event.event_type.value}: {reason} "
                    f"(severity={event.severity:.0f})"
                )
                min_dampening  = min(min_dampening, event.signal_dampening)
                event_names.append(event.title)

        return {
            "direction": (
                1 if net_direction > 0.5
                else -1 if net_direction < -0.5
                else 0
            ),
            "magnitude":  max_magnitude,
            "reasoning":  all_reasoning,
            "dampening":  round(min_dampening, 2),
            "events":     event_names,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _check_pattern(
        self,
        text: str,
        event_type: EventType,
        pattern: dict,
        articles: list,
    ) -> Optional[DetectedEvent]:
        required = pattern["required"]
        boosting = pattern["boosting"]

        # Must match at least 1 required keyword
        matched_required = [kw for kw in required if kw in text]
        if not matched_required:
            return None

        matched_boosting = [kw for kw in boosting if kw in text]

        # Confidence: more matches = higher confidence
        base_conf    = min(0.4 + len(matched_required) * 0.15, 0.85)
        boost_conf   = min(len(matched_boosting) * 0.05, 0.15)
        confidence   = min(0.95, base_conf + boost_conf)

        # Severity: base + boost from multiple matches
        severity = min(
            10,
            pattern["severity"] +
            (len(matched_required) - 1) * 0.3 +
            len(matched_boosting) * 0.2
        )

        # Phase: more keywords = more active
        if len(matched_required) >= 3 or len(matched_boosting) >= 2:
            phase = EventPhase.ACTIVE
        else:
            phase = EventPhase.EMERGING

        # Affected assets from impact table
        impact_table    = EVENT_ASSET_IMPACT.get(event_type, {})
        affected_assets = list(impact_table.keys())

        return DetectedEvent(
            event_type       = event_type,
            phase            = phase,
            severity         = round(severity, 1),
            confidence       = round(confidence, 2),
            title            = f"{event_type.value.replace('_', ' ').title()} Event Detected",
            description      = (
                f"Matched keywords: {', '.join(matched_required[:3])}. "
                f"Boosters: {', '.join(matched_boosting[:3]) if matched_boosting else 'none'}."
            ),
            affected_assets  = affected_assets,
            signal_dampening = compute_dampening(severity, event_type.value),
            historical_analog= pattern["analog"],
            keywords_matched = matched_required + matched_boosting,
        )

    def _make_price_event(
        self,
        event_type: EventType,
        severity: float,
        confidence: float,
        title: str,
        description: str,
        price_signals: dict,
    ) -> DetectedEvent:
        pattern = EVENT_PATTERNS.get(event_type, {})
        return DetectedEvent(
            event_type       = event_type,
            phase            = EventPhase.ACTIVE,
            severity         = round(severity, 1),
            confidence       = round(confidence, 2),
            title            = title,
            description      = description,
            affected_assets  = list(EVENT_ASSET_IMPACT.get(event_type, {}).keys()),
            signal_dampening = compute_dampening(severity, event_type.value),
            historical_analog= pattern.get("analog", "Unknown"),
            price_signals    = price_signals,
        )


# Module-level singleton
event_detector = EventDetector()