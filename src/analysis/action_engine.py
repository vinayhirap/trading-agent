# trading-agent/src/analysis/action_engine.py
"""
Action Engine — Converts predictions into actionable trading decisions.

Takes a PredictionResult and produces the strict output format:
  {
    "symbol":     "NIFTY50",
    "signal":     "BUY",
    "confidence": 0.67,
    "action":     "ENTER",
    "reasoning":  {
      "technical":  "...",
      "news":       "...",
      "event":      "...",
      "behavior":   "..."
    }
  }

Action rules (in priority order):
  1. AVOID  — regime blocked / bad session / extreme event / confidence < 0.45
  2. WAIT   — confidence 0.45-0.52 / transitional regime / mixed signals
  3. ENTER  — confidence >= 0.52 / regime aligned / session good

Also computes:
  - Stop loss recommendation (ATR-based)
  - Position size recommendation
  - Hold type (INTRADAY / SWING / POSITIONAL)
  - Risk:Reward ratio

Records every decision in LearningEngineV2 for feedback loop.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from loguru import logger

# Thresholds
ENTER_THRESHOLD  = 0.52
WAIT_THRESHOLD   = 0.45
STRONG_THRESHOLD = 0.68


@dataclass
class ActionDecision:
    """
    Full actionable trading decision in strict output format.
    Compatible with Telegram alerts, dashboard display, and API output.
    """
    # Core fields (strict output format)
    symbol:     str
    signal:     str       # BUY / SELL / HOLD / STRONG BUY / STRONG SELL
    confidence: float     # 0-1
    action:     str       # ENTER / WAIT / AVOID

    reasoning: dict = field(default_factory=dict)
    # reasoning keys: technical, cross_asset, news, event, behavior, regime

    # Extended fields
    raw_score:      float  = 0.0
    asset_class:    str    = "equity"
    regime:         str    = "RANGING"
    session:        str    = "UNKNOWN"

    # Risk management
    atr_pct:        float  = 0.015
    sl_pct:         float  = 0.0    # suggested SL %
    target_pct:     float  = 0.0    # suggested target %
    risk_reward:    float  = 0.0
    position_size_pct: float = 0.0  # % of capital to allocate
    hold_type:      str    = "INTRADAY"   # INTRADAY / SWING / POSITIONAL

    # Warnings and opportunities
    warnings:       list[str] = field(default_factory=list)
    opportunities:  list[str] = field(default_factory=list)

    # Meta
    weights_used:   dict = field(default_factory=dict)
    asset_trust:    float = 1.0
    learning_pid:   str  = ""    # prediction ID for tracking
    computed_at:    str  = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        """Strict output format for API / Telegram / dashboard."""
        return {
            "symbol":     self.symbol,
            "signal":     self.signal,
            "confidence": round(self.confidence, 3),
            "action":     self.action,
            "reasoning":  self.reasoning,
        }

    def to_full_dict(self) -> dict:
        """Extended dict with risk management and meta."""
        d = self.to_dict()
        d.update({
            "raw_score":   self.raw_score,
            "asset_class": self.asset_class,
            "regime":      self.regime,
            "session":     self.session,
            "sl_pct":      self.sl_pct,
            "target_pct":  self.target_pct,
            "risk_reward": self.risk_reward,
            "position_pct":self.position_size_pct,
            "hold_type":   self.hold_type,
            "warnings":    self.warnings,
            "opportunities":self.opportunities,
            "weights":     self.weights_used,
            "trust":       self.asset_trust,
        })
        return d

    def __str__(self):
        return (
            f"{self.action} | {self.signal} {self.symbol} "
            f"({self.confidence:.0%}) | regime:{self.regime}"
        )


class ActionEngine:
    """
    Converts PredictionResult → ActionDecision.

    Also:
      - Records the decision in LearningEngineV2 for tracking
      - Computes ATR-based SL/target
      - Determines position size from risk parameters

    Usage:
        ae     = ActionEngine()
        result = prediction_engine.predict("NIFTY50", features, df)
        action = ae.decide(result, current_price=22713, atr_pct=0.023)

        print(action.to_dict())
        # {"symbol": "NIFTY50", "signal": "SELL", "confidence": 0.71,
        #  "action": "ENTER", "reasoning": {...}}
    """

    def __init__(
        self,
        risk_per_trade:  float = 0.01,   # 1% of capital per trade
        atr_sl_mult:     float = 1.5,
        target_rr:       float = 2.0,
        capital:         float = 10_000,
    ):
        self.risk_per_trade = risk_per_trade
        self.atr_sl_mult    = atr_sl_mult
        self.target_rr      = target_rr
        self.capital        = capital

    def decide(
        self,
        prediction,        # PredictionResult from PredictionEngine
        current_price: float = 0,
        atr_pct:       float = None,
        record:        bool  = True,   # record in learning engine
    ) -> ActionDecision:
        """
        Convert PredictionResult to ActionDecision.
        """
        from src.analysis.prediction_engine import PredictionResult

        p = prediction

        # ── Determine action ──────────────────────────────────────────────
        action, warnings = self._determine_action(p)

        # ── Risk management ───────────────────────────────────────────────
        atr_pct_used = atr_pct or float(0.015)
        sl_pct, target_pct, rr, pos_pct = self._compute_risk(
            p.signal, atr_pct_used, current_price
        )

        # ── Hold type ─────────────────────────────────────────────────────
        hold_type = self._hold_type(p.regime, p.session, atr_pct_used)

        # ── Build reasoning dict ──────────────────────────────────────────
        reasoning = self._build_reasoning(p)

        # ── Opportunities ─────────────────────────────────────────────────
        opportunities = []
        if p.events.score > 0.3 and "BUY" in p.signal:
            opportunities.append(f"Event tailwind: {p.events.reasoning[:60]}")
        if p.news.score > 0.2 and "BUY" in p.signal:
            opportunities.append(f"Positive news: {p.news.reasoning[:60]}")
        if p.cross_asset.score > 0.2:
            opportunities.append(f"Cross-asset support: {p.cross_asset.reasoning[:60]}")
        if p.regime_blocked:
            warnings.append(f"Regime gate: {p.regime_block_note}")

        # ── Record in learning engine ─────────────────────────────────────
        pid = ""
        if record and current_price > 0:
            pid = self._record_decision(p, action, current_price)

        return ActionDecision(
            symbol      = p.symbol,
            signal      = p.signal,
            confidence  = p.confidence,
            action      = action,
            reasoning   = reasoning,
            raw_score   = p.raw_score,
            asset_class = p.asset_class,
            regime      = p.regime,
            session     = p.session,
            atr_pct     = atr_pct_used,
            sl_pct      = sl_pct,
            target_pct  = target_pct,
            risk_reward = rr,
            position_size_pct = pos_pct,
            hold_type   = hold_type,
            warnings    = warnings,
            opportunities=opportunities,
            weights_used = p.weights_used,
            asset_trust  = p.asset_trust,
            learning_pid = pid,
        )

    def decide_batch(
        self,
        predictions: list,
        prices:      dict = None,   # {symbol: price}
        atr_pcts:    dict = None,   # {symbol: atr_pct}
    ) -> list[ActionDecision]:
        """Decide for multiple predictions at once."""
        results = []
        for pred in predictions:
            try:
                price   = (prices   or {}).get(pred.symbol, 0)
                atr_pct = (atr_pcts or {}).get(pred.symbol, 0.015)
                decision = self.decide(pred, current_price=price, atr_pct=atr_pct)
                results.append(decision)
            except Exception as e:
                logger.warning(f"ActionEngine.decide failed for {pred.symbol}: {e}")
        return results

    def format_telegram(self, decision: ActionDecision) -> str:
        """Format decision as Telegram HTML message."""
        icons = {"ENTER": "✅", "WAIT": "⏳", "AVOID": "🚫"}
        sig_icons = {
            "STRONG BUY":  "🚀", "BUY":  "🟢",
            "STRONG SELL": "💥", "SELL": "🔴",
            "HOLD": "⚪",
        }
        icon    = icons.get(decision.action, "⚠️")
        sig_icon= sig_icons.get(decision.signal, "⚠️")

        msg = (
            f"{icon} <b>{decision.action}</b> — {sig_icon} {decision.signal}\n"
            f"<b>{decision.symbol}</b> | Confidence: <code>{decision.confidence:.0%}</code>\n"
            f"Regime: {decision.regime} | {decision.session}\n\n"
        )

        r = decision.reasoning
        if r.get("technical"):
            msg += f"📊 <i>{r['technical'][:80]}</i>\n"
        if r.get("news"):
            msg += f"📰 <i>{r['news'][:80]}</i>\n"
        if r.get("event"):
            msg += f"🌍 <i>{r['event'][:80]}</i>\n"
        if r.get("behavior"):
            msg += f"⏰ <i>{r['behavior'][:80]}</i>\n"

        if decision.action == "ENTER":
            msg += (
                f"\nSL: <code>{decision.sl_pct:.1f}%</code> | "
                f"Target: <code>{decision.target_pct:.1f}%</code> | "
                f"R:R: <code>1:{decision.risk_reward:.1f}</code>"
            )

        if decision.warnings:
            msg += f"\n⚠️ {decision.warnings[0]}"

        return msg

    # ── Internal ──────────────────────────────────────────────────────────────

    def _determine_action(
        self, p
    ) -> tuple[str, list[str]]:
        """Determine ENTER / WAIT / AVOID based on prediction quality."""
        warnings = []

        # ── Hard AVOID conditions ─────────────────────────────────────────
        if p.behavior.confidence < 0.25:
            warnings.append(f"Bad session: {p.session}")
            return "AVOID", warnings

        if p.regime_blocked:
            return "AVOID", warnings

        if p.signal == "HOLD":
            return "WAIT", warnings

        if p.confidence < WAIT_THRESHOLD:
            warnings.append(f"Low confidence: {p.confidence:.0%}")
            return "AVOID", warnings

        # ── Event dampening ───────────────────────────────────────────────
        if p.events.available and p.events.confidence > 0.7:
            # Severe event → reduce confidence gate to 65%
            if p.events.score < -0.4 and "BUY" in p.signal:
                warnings.append("Active negative event — elevated risk")
                if p.confidence < 0.65:
                    return "AVOID", warnings

        # ── WAIT conditions ───────────────────────────────────────────────
        if WAIT_THRESHOLD <= p.confidence < ENTER_THRESHOLD:
            return "WAIT", warnings

        if p.asset_trust < 0.6:
            warnings.append(f"Low asset trust: {p.asset_trust:.0%}")
            return "WAIT", warnings

        # ── ENTER ─────────────────────────────────────────────────────────
        return "ENTER", warnings

    def _build_reasoning(self, p) -> dict:
        """Build the strict reasoning dict."""
        # Technical reasoning
        tech_text = p.technical.reasoning
        if p.regime_blocked:
            tech_text += f" | ⚠️ {p.regime_block_note}"

        # Confidence modifier note
        if p.conf_multiplier < 0.90:
            tech_text += (
                f" | Trust adjustment: ×{p.conf_multiplier:.2f} "
                f"(asset class performance-based)"
            )

        return {
            "technical":   tech_text,
            "cross_asset": p.cross_asset.reasoning,
            "news":        p.news.reasoning,
            "event":       p.events.reasoning,
            "behavior":    p.behavior.reasoning,
            "regime":      (
                f"{p.regime} | Session: {p.session} | "
                f"Expiry: {'YES ⚡' if p.is_expiry else 'no'} | "
                f"Weights: T={p.weights_used.get('technical',0):.0%} "
                f"N={p.weights_used.get('news',0):.0%} "
                f"E={p.weights_used.get('events',0):.0%} "
                f"B={p.weights_used.get('behavior',0):.0%}"
            ),
        }

    def _compute_risk(
        self, signal: str, atr_pct: float, price: float
    ) -> tuple[float, float, float, float]:
        """Compute SL%, target%, R:R, position size%."""
        sl_pct     = atr_pct * self.atr_sl_mult * 100
        target_pct = sl_pct  * self.target_rr
        rr         = self.target_rr

        # Position size: risk_per_trade / sl_pct
        if sl_pct > 0:
            pos_pct = min(50.0, self.risk_per_trade / (sl_pct / 100) * 100)
        else:
            pos_pct = 5.0

        return round(sl_pct, 2), round(target_pct, 2), rr, round(pos_pct, 1)

    def _hold_type(self, regime: str, session: str, atr_pct: float) -> str:
        if atr_pct > 0.025 or "VOLATILE" in regime:
            return "INTRADAY"
        if "TREND_WINDOW" in session or "INSTITUTIONAL" in session:
            return "SWING"
        if session == "UNKNOWN" or "CLOSED" in session:
            return "POSITIONAL"
        return "INTRADAY"

    def _record_decision(self, p, action: str, price: float) -> str:
        """Record in LearningEngineV2 for future feedback."""
        try:
            from src.analysis.learning_engine_v2 import learning_v2
            source_scores = {
                "technical":   p.technical.weighted,
                "cross_asset": p.cross_asset.weighted,
                "news":        p.news.weighted,
                "events":      p.events.weighted,
                "behavior":    p.behavior.weighted,
            }
            pid = learning_v2.record(
                symbol        = p.symbol,
                signal        = p.signal,
                confidence    = p.confidence,
                asset_class   = p.asset_class,
                regime        = p.regime,
                source_scores = source_scores,
                horizon_bars  = 5,
                entry_price   = price,
                extra         = {"action": action, "session": p.session},
            )
            return pid
        except Exception:
            return ""


# Module-level singleton
action_engine = ActionEngine()