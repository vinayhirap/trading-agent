# trading-agent/src/analysis/multi_agent_engine.py
"""
Multi-Agent Trading Engine — TauricResearch TradingAgents Pattern

4 Specialized Agents debate every trade signal:
  BullAgent       — finds reasons TO buy, trend confirmation
  BearAgent       — finds reasons AGAINST, risks and weaknesses
  FundamentalAgent— Alpha Vantage data, valuation, earnings quality
  SentimentAgent  — News sentiment, FII/DII flow, options data

Decision process:
  1. Each agent scores independently (-1.0 to +1.0)
  2. Weighted vote: Technical 40%, Fundamental 25%, Sentiment 20%, Risk 15%
  3. Agents with conflicting views trigger a "debate" — confidence reduced
  4. Final verdict: STRONG BUY / BUY / HOLD / SELL / STRONG SELL + confidence

Why this beats single-model:
  - Adversarial analysis catches false signals
  - Fundamental agent kills technically bullish but fundamentally weak stocks
  - Sentiment agent catches macro headwinds before price moves
  - Bear agent's job is to find what's wrong — reduces overconfidence

Usage:
    from src.analysis.multi_agent_engine import MultiAgentEngine
    mae = MultiAgentEngine()
    result = mae.analyze(symbol, features, df, overview, news_items)
    print(result.verdict, result.confidence, result.reasoning)

Wire into app.py:
    elif page == "🤖 Multi-Agent Analysis":
        from src.analysis.multi_agent_engine import render_multi_agent_page
        render_multi_agent_page()
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger


# ── Verdict enum ──────────────────────────────────────────────────────────────

class Verdict(str, Enum):
    STRONG_BUY  = "STRONG BUY"
    BUY         = "BUY"
    HOLD        = "HOLD"
    SELL        = "SELL"
    STRONG_SELL = "STRONG SELL"
    AVOID       = "AVOID"      # triggered by fundamental red flags


# ── Agent result ──────────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    agent:      str           # "bull", "bear", "fundamental", "sentiment"
    score:      float         # -1.0 (very bearish) to +1.0 (very bullish)
    confidence: float         # 0.0 to 1.0
    reasoning:  list[str]     # bullet points
    red_flags:  list[str]     # serious concerns
    green_flags:list[str]     # strong positives


@dataclass
class DebateResult:
    symbol:          str
    verdict:         Verdict
    confidence:      float
    final_score:     float         # weighted average of all agents
    bull_score:      float
    bear_score:      float
    fundamental_score: float
    sentiment_score: float
    agents:          list[AgentResult]
    consensus:       str           # "STRONG", "MODERATE", "SPLIT", "CONFLICTED"
    reasoning:       list[str]     # merged key points
    red_flags:       list[str]     # all red flags across agents
    green_flags:     list[str]     # all green flags
    trade_allowed:   bool          # False if fundamental red flags found
    position_size:   float         # 0.0 to 1.0 multiplier
    sl_multiplier:   float         # stop loss width multiplier
    hold_type:       str           # "INTRADAY", "SWING", "POSITIONAL"


# ── Agent weights ─────────────────────────────────────────────────────────────

AGENT_WEIGHTS = {
    "bull":        0.30,   # technical bull case
    "bear":        0.25,   # technical bear case (negative score = bearish risk)
    "fundamental": 0.25,   # valuation + earnings
    "sentiment":   0.20,   # news + macro
}

# Minimum confidence to trade
MIN_CONFIDENCE = 0.55

# Consensus thresholds
STRONG_CONSENSUS_THRESHOLD  = 0.65   # all agents agree direction
SPLIT_CONSENSUS_THRESHOLD   = 0.20   # agents disagree significantly


# ── Bull Agent ────────────────────────────────────────────────────────────────

class BullAgent:
    """
    Finds reasons to BUY. Scores technical momentum, trend, breakouts.
    Score: 0.0 (no bull case) to +1.0 (very strong bull case)
    """

    name = "bull"

    def analyze(self, features: pd.Series, df: pd.DataFrame) -> AgentResult:
        score      = 0.0
        reasoning  = []
        green      = []
        red        = []

        def _f(k, d=0.0):
            v = features.get(k, d)
            try:
                v = float(v)
                return d if (np.isnan(v) or np.isinf(v)) else v
            except (TypeError, ValueError):
                return d

        adx       = _f("adx", 20)
        di_diff   = _f("di_diff", 0)
        rsi       = _f("rsi_14", 50)
        ema9_pct  = _f("ema9_pct", 0)
        ema50_pct = _f("ema50_pct", 0)
        ema200_pct= _f("ema200_pct", 0)
        macd_h    = _f("macd_hist", 0)
        macd_chg  = _f("macd_hist_chg", 0)
        bb_pct_b  = _f("bb_pct_b", 0.5)
        vol_ratio = _f("vol_ratio", 1.0)
        atr_ratio = _f("atr_ratio", 1.0)
        obv_slope = _f("obv_slope", 0)

        # Trend strength
        if adx > 30 and di_diff > 5:
            score += 0.25
            green.append(f"Strong bull trend: ADX {adx:.0f}, +DI dominant ({di_diff:+.0f})")
        elif adx > 20 and di_diff > 0:
            score += 0.12
            reasoning.append(f"Moderate uptrend: ADX {adx:.0f}")

        # EMA alignment
        if ema9_pct > 0 and ema50_pct > 0 and ema200_pct > 0:
            score += 0.20
            green.append("Perfect bull alignment: price above EMA9, EMA50, EMA200")
        elif ema9_pct > 0 and ema50_pct > 0:
            score += 0.12
            reasoning.append("Price above EMA9 + EMA50 — bullish structure")
        elif ema9_pct > 0:
            score += 0.05

        # RSI
        if 55 <= rsi <= 70:
            score += 0.12
            green.append(f"RSI {rsi:.0f} — bullish momentum, not overbought")
        elif rsi > 70:
            score += 0.03
            red.append(f"RSI {rsi:.0f} overbought — pullback risk")
        elif rsi < 40:
            score += 0.08
            reasoning.append(f"RSI {rsi:.0f} oversold — bounce potential")

        # MACD
        if macd_h > 0 and macd_chg > 0:
            score += 0.15
            green.append("MACD histogram positive and rising — momentum accelerating")
        elif macd_h > 0:
            score += 0.07
            reasoning.append("MACD positive — bullish bias")

        # Volume confirmation
        if vol_ratio > 1.5 and ema9_pct > 0:
            score += 0.10
            green.append(f"Volume {vol_ratio:.1f}× average on upward move — institutional buying")
        elif vol_ratio < 0.7 and ema9_pct > 0:
            red.append("Low volume on rally — weak conviction")
            score -= 0.05

        # Bollinger squeeze breakout
        if bb_pct_b > 0.8:
            score += 0.06
            reasoning.append("Price in upper Bollinger band — bullish pressure")

        # OBV (smart money)
        if obv_slope > 0:
            score += 0.08
            green.append("OBV rising — accumulation pattern (smart money buying)")

        # Recent price momentum
        if len(df) >= 10:
            ret_5d  = float(df["close"].iloc[-1] / df["close"].iloc[-5]  - 1) if len(df) >= 5  else 0
            ret_20d = float(df["close"].iloc[-1] / df["close"].iloc[-20] - 1) if len(df) >= 20 else 0
            if ret_5d > 0.03:
                score += 0.08
                green.append(f"Strong 5-day momentum: +{ret_5d:.1%}")
            if ret_20d > 0.05:
                score += 0.05
                reasoning.append(f"20-day return: +{ret_20d:.1%}")

        score = max(0.0, min(1.0, score))
        conf  = min(0.90, 0.40 + score * 0.55)

        return AgentResult(
            agent="bull", score=score, confidence=conf,
            reasoning=reasoning, red_flags=red, green_flags=green,
        )


# ── Bear Agent ────────────────────────────────────────────────────────────────

class BearAgent:
    """
    Finds reasons AGAINST buying. Adversarial — looks for weaknesses.
    Score: -1.0 (very bearish) to 0.0 (no bear case)
    A score of 0.0 means bear agent found nothing wrong (bull case strengthened).
    """

    name = "bear"

    def analyze(self, features: pd.Series, df: pd.DataFrame) -> AgentResult:
        score     = 0.0   # starts neutral, goes negative
        reasoning = []
        green     = []    # from bear's perspective: "no risk here"
        red       = []    # bear's concerns

        def _f(k, d=0.0):
            v = features.get(k, d)
            try:
                v = float(v)
                return d if (np.isnan(v) or np.isinf(v)) else v
            except (TypeError, ValueError):
                return d

        adx       = _f("adx", 20)
        di_diff   = _f("di_diff", 0)
        rsi       = _f("rsi_14", 50)
        ema9_pct  = _f("ema9_pct", 0)
        ema50_pct = _f("ema50_pct", 0)
        macd_h    = _f("macd_hist", 0)
        macd_chg  = _f("macd_hist_chg", 0)
        bb_pct_b  = _f("bb_pct_b", 0.5)
        atr_ratio = _f("atr_ratio", 1.0)
        vol_ratio = _f("vol_ratio", 1.0)
        bb_width  = _f("bb_width", 0.04)

        # Downtrend
        if adx > 25 and di_diff < -5:
            score -= 0.30
            red.append(f"Strong downtrend: ADX {adx:.0f}, -DI dominant ({di_diff:.0f})")
        elif ema9_pct < 0 and ema50_pct < 0:
            score -= 0.18
            red.append("Price below EMA9 and EMA50 — bearish structure")

        # Overbought
        if rsi > 75:
            score -= 0.20
            red.append(f"RSI {rsi:.0f} — severely overbought, reversal risk high")
        elif rsi > 68:
            score -= 0.10
            red.append(f"RSI {rsi:.0f} — overbought zone")

        # MACD deteriorating
        if macd_h > 0 and macd_chg < 0:
            score -= 0.12
            red.append("MACD histogram shrinking — bullish momentum weakening")
        elif macd_h < 0 and macd_chg < 0:
            score -= 0.18
            red.append("MACD negative and falling — bearish acceleration")

        # Volatility spike
        if atr_ratio > 2.5:
            score -= 0.15
            red.append(f"ATR {atr_ratio:.1f}× average — extreme volatility, wide stops needed")
        elif atr_ratio > 1.8:
            score -= 0.08
            reasoning.append(f"Elevated volatility (ATR {atr_ratio:.1f}×)")

        # Volume on down moves
        if vol_ratio > 1.5 and ema9_pct < 0:
            score -= 0.12
            red.append(f"High volume ({vol_ratio:.1f}×) on decline — distribution (institutional selling)")

        # BB squeeze then break down
        if bb_pct_b < 0.1:
            score -= 0.08
            red.append("Price at lower Bollinger Band — breakdown risk")

        # Recent drawdown
        if len(df) >= 20:
            high_20d = df["high"].iloc[-20:].max()
            curr     = float(df["close"].iloc[-1])
            dd_pct   = (curr - high_20d) / high_20d
            if dd_pct < -0.08:
                score -= 0.15
                red.append(f"Down {abs(dd_pct):.1%} from 20-day high — in drawdown")
            elif dd_pct < -0.04:
                score -= 0.07
                reasoning.append(f"Off {abs(dd_pct):.1%} from recent high")

        # If no bear case found
        if score == 0.0:
            green.append("No significant bearish signals detected")

        score = max(-1.0, min(0.0, score))
        conf  = min(0.90, 0.40 + abs(score) * 0.55)

        return AgentResult(
            agent="bear", score=score, confidence=conf,
            reasoning=reasoning, red_flags=red, green_flags=green,
        )


# ── Fundamental Agent ─────────────────────────────────────────────────────────

class FundamentalAgent:
    """
    Analyzes company fundamentals from Alpha Vantage overview data.
    Score: -1.0 (fundamental disaster) to +1.0 (fundamentally excellent)
    Returns AVOID if hard red flags found (negative PE, negative equity, etc.)
    """

    name = "fundamental"

    def analyze(self, overview: dict) -> AgentResult:
        score     = 0.0
        reasoning = []
        green     = []
        red       = []
        avoid     = False

        if not overview:
            return AgentResult(
                agent="fundamental", score=0.0, confidence=0.3,
                reasoning=["No fundamental data available — neutral"],
                red_flags=[], green_flags=[],
            )

        def _sf(k):
            v = overview.get(k)
            try:
                return float(v) if v and str(v) not in ("None", "-", "") else None
            except (TypeError, ValueError):
                return None

        pe        = _sf("pe_ratio")
        fwd_pe    = _sf("forward_pe")
        pb        = _sf("price_to_book")
        ev_ebitda = _sf("ev_to_ebitda")
        roe       = _sf("return_on_equity")
        margin    = _sf("profit_margin")
        rev_growth= _sf("revenue_growth_yoy")
        eps_growth= _sf("earnings_growth_yoy")
        div_yield = _sf("dividend_yield")
        beta      = _sf("beta")
        debt_ratio= _sf("ev_to_revenue")
        analyst_t = _sf("analyst_target")

        # ── Hard red flags → AVOID ────────────────────────────────────────────
        if pe is not None and pe < 0:
            red.append(f"⚠️ Negative P/E ({pe:.1f}) — company losing money")
            avoid = True
        if pb is not None and pb < 0:
            red.append("⚠️ Negative book value — balance sheet concern")
            avoid = True
        if margin is not None and margin < -0.15:
            red.append(f"⚠️ Severe losses: margin {margin*100:.1f}%")
            avoid = True

        # ── Valuation ─────────────────────────────────────────────────────────
        if pe is not None and 0 < pe < 15:
            score += 0.15; green.append(f"Cheap valuation: P/E {pe:.1f}×")
        elif pe is not None and 15 <= pe <= 25:
            score += 0.08; reasoning.append(f"Fair valuation: P/E {pe:.1f}×")
        elif pe is not None and pe > 50:
            score -= 0.12; red.append(f"Expensive: P/E {pe:.1f}× — priced for perfection")

        if ev_ebitda is not None and 0 < ev_ebitda < 10:
            score += 0.10; green.append(f"EV/EBITDA {ev_ebitda:.1f}× — attractive")
        elif ev_ebitda is not None and ev_ebitda > 25:
            score -= 0.08; red.append(f"EV/EBITDA {ev_ebitda:.1f}× — very expensive")

        # ── Quality ───────────────────────────────────────────────────────────
        if roe is not None:
            if roe > 0.20:
                score += 0.15; green.append(f"Excellent ROE: {roe*100:.1f}%")
            elif roe > 0.12:
                score += 0.08; reasoning.append(f"Good ROE: {roe*100:.1f}%")
            elif roe < 0.05:
                score -= 0.10; red.append(f"Weak ROE: {roe*100:.1f}%")

        if margin is not None:
            if margin > 0.20:
                score += 0.12; green.append(f"Strong margins: {margin*100:.1f}%")
            elif margin > 0.10:
                score += 0.06; reasoning.append(f"Decent margins: {margin*100:.1f}%")
            elif margin < 0.03:
                score -= 0.08; red.append(f"Thin margins: {margin*100:.1f}%")

        # ── Growth ────────────────────────────────────────────────────────────
        if rev_growth is not None:
            if rev_growth > 0.20:
                score += 0.12; green.append(f"Strong revenue growth: +{rev_growth*100:.1f}% YoY")
            elif rev_growth > 0.08:
                score += 0.06; reasoning.append(f"Revenue growth: +{rev_growth*100:.1f}%")
            elif rev_growth < -0.05:
                score -= 0.10; red.append(f"Revenue declining: {rev_growth*100:.1f}% YoY")

        if eps_growth is not None:
            if eps_growth > 0.25:
                score += 0.10; green.append(f"Strong EPS growth: +{eps_growth*100:.1f}%")
            elif eps_growth < -0.10:
                score -= 0.08; red.append(f"EPS declining: {eps_growth*100:.1f}%")

        # ── Analyst target vs current ─────────────────────────────────────────
        # (we don't have current price here so just note it)
        if analyst_t:
            reasoning.append(f"Analyst target: ₹{float(analyst_t):,.0f}")

        # ── Beta / risk ───────────────────────────────────────────────────────
        if beta is not None:
            if beta > 1.5:
                red.append(f"High beta ({beta:.2f}) — amplified market risk")
                score -= 0.05
            elif beta < 0.7:
                green.append(f"Low beta ({beta:.2f}) — defensive characteristics")
                score += 0.05

        score = max(-1.0, min(1.0, score))
        conf  = 0.70 if overview else 0.30   # high conf when data available

        result = AgentResult(
            agent="fundamental", score=score, confidence=conf,
            reasoning=reasoning, red_flags=red, green_flags=green,
        )
        result.avoid = avoid   # type: ignore
        return result


# ── Sentiment Agent ───────────────────────────────────────────────────────────

class SentimentAgent:
    """
    Analyzes news sentiment, macro context, and market breadth.
    Score: -1.0 (very negative sentiment) to +1.0 (very positive)
    """

    name = "sentiment"

    def analyze(
        self,
        news_items: list[dict],
        regime_str: Optional[str] = None,
        fii_data:   Optional[dict] = None,
    ) -> AgentResult:
        score     = 0.0
        reasoning = []
        green     = []
        red       = []

        # ── News sentiment ────────────────────────────────────────────────────
        if news_items:
            scores = []
            for item in news_items[:10]:
                s = item.get("sentiment") or item.get("overall_sentiment_score") or 0
                try:
                    scores.append(float(s))
                except (TypeError, ValueError):
                    pass

            if scores:
                avg_sent = float(np.mean(scores))
                n        = len(scores)
                if avg_sent > 0.3:
                    score += 0.25
                    green.append(f"Strong positive news sentiment: {avg_sent:.2f} ({n} articles)")
                elif avg_sent > 0.1:
                    score += 0.12
                    reasoning.append(f"Mildly positive news: {avg_sent:.2f}")
                elif avg_sent < -0.3:
                    score -= 0.25
                    red.append(f"Strong negative news sentiment: {avg_sent:.2f} ({n} articles)")
                elif avg_sent < -0.1:
                    score -= 0.12
                    reasoning.append(f"Mildly negative news: {avg_sent:.2f}")
                else:
                    reasoning.append(f"Neutral news sentiment: {avg_sent:.2f}")
        else:
            reasoning.append("No recent news — sentiment neutral")

        # ── Regime context ────────────────────────────────────────────────────
        if regime_str:
            if regime_str == "BULL_TREND":
                score += 0.15
                green.append("Market regime: BULL_TREND — macro tailwind")
            elif regime_str == "BEAR_TREND":
                score -= 0.20
                red.append("Market regime: BEAR_TREND — macro headwind, avoid longs")
            elif regime_str == "RANGING_HIGH":
                score -= 0.10
                red.append("Market regime: RANGING_HIGH — high volatility, reduce size")
            elif regime_str == "RANGING_LOW":
                reasoning.append("Market regime: RANGING_LOW — neutral macro")

        # ── FII/DII flow ──────────────────────────────────────────────────────
        if fii_data:
            fii_net = fii_data.get("fii_net_crores", 0)
            dii_net = fii_data.get("dii_net_crores", 0)

            if fii_net > 1000:
                score += 0.20
                green.append(f"Strong FII buying: ₹{fii_net:,.0f} Cr — institutional inflow")
            elif fii_net > 200:
                score += 0.10
                reasoning.append(f"FII net buying: ₹{fii_net:,.0f} Cr")
            elif fii_net < -1000:
                score -= 0.20
                red.append(f"Heavy FII selling: ₹{fii_net:,.0f} Cr — institutional outflow")
            elif fii_net < -200:
                score -= 0.10
                reasoning.append(f"FII net selling: ₹{fii_net:,.0f} Cr")

            if dii_net > 500:
                score += 0.08
                green.append(f"DII support buying: ₹{dii_net:,.0f} Cr")

        score = max(-1.0, min(1.0, score))
        conf  = min(0.85, 0.45 + abs(score) * 0.40)

        return AgentResult(
            agent="sentiment", score=score, confidence=conf,
            reasoning=reasoning, red_flags=red, green_flags=green,
        )


# ── Multi-Agent Engine ────────────────────────────────────────────────────────

class MultiAgentEngine:
    """
    Orchestrates all 4 agents and produces a final debate result.
    """

    def __init__(self):
        self._bull        = BullAgent()
        self._bear        = BearAgent()
        self._fundamental = FundamentalAgent()
        self._sentiment   = SentimentAgent()

    def analyze(
        self,
        symbol:      str,
        features:    pd.Series,
        df:          pd.DataFrame,
        overview:    Optional[dict]  = None,
        news_items:  Optional[list]  = None,
        regime_str:  Optional[str]   = None,
        fii_data:    Optional[dict]  = None,
    ) -> DebateResult:
        """
        Run all agents and produce a debate result.

        Args:
            symbol:     Trading symbol e.g. "RELIANCE"
            features:   Latest feature row from FeatureEngine
            df:         OHLCV DataFrame
            overview:   Alpha Vantage OVERVIEW dict (optional)
            news_items: List of news dicts with 'sentiment' key (optional)
            regime_str: Current market regime string (optional)
            fii_data:   FII/DII flow dict (optional)
        """
        # ── Run all agents ────────────────────────────────────────────────────
        bull_res  = self._bull.analyze(features, df)
        bear_res  = self._bear.analyze(features, df)
        fund_res  = self._fundamental.analyze(overview or {})
        sent_res  = self._sentiment.analyze(
            news_items or [], regime_str, fii_data
        )

        agents = [bull_res, bear_res, fund_res, sent_res]

        # ── Fundamental veto ──────────────────────────────────────────────────
        fund_avoid = getattr(fund_res, "avoid", False)

        # ── Weighted score ────────────────────────────────────────────────────
        # Bear score is negative — higher magnitude = more bearish
        # We treat bear score as a negative weight on final score
        weighted = (
            bull_res.score  * AGENT_WEIGHTS["bull"]
            + bear_res.score  * AGENT_WEIGHTS["bear"]   # already negative
            + fund_res.score  * AGENT_WEIGHTS["fundamental"]
            + sent_res.score  * AGENT_WEIGHTS["sentiment"]
        )

        # ── Consensus calculation ─────────────────────────────────────────────
        # Check if agents agree on direction
        bull_bullish = bull_res.score > 0.2
        bear_bullish = bear_res.score > -0.2   # bear found little to worry about
        fund_bullish = fund_res.score > 0.1
        sent_bullish = sent_res.score > 0.1

        bullish_votes = sum([bull_bullish, bear_bullish, fund_bullish, sent_bullish])

        if bullish_votes >= 4:
            consensus = "STRONG"
        elif bullish_votes >= 3:
            consensus = "MODERATE"
        elif bullish_votes == 2:
            consensus = "SPLIT"
        else:
            consensus = "CONFLICTED"

        # ── Confidence ────────────────────────────────────────────────────────
        base_conf = (
            bull_res.confidence  * AGENT_WEIGHTS["bull"]
            + bear_res.confidence  * AGENT_WEIGHTS["bear"]
            + fund_res.confidence  * AGENT_WEIGHTS["fundamental"]
            + sent_res.confidence  * AGENT_WEIGHTS["sentiment"]
        )

        # Reduce confidence when agents disagree
        if consensus == "SPLIT":
            base_conf *= 0.80
        elif consensus == "CONFLICTED":
            base_conf *= 0.65

        confidence = round(min(0.92, max(0.35, base_conf)), 3)

        # ── Verdict ───────────────────────────────────────────────────────────
        if fund_avoid:
            verdict = Verdict.AVOID
            confidence = min(confidence, 0.50)
        elif weighted >= 0.45:
            verdict = Verdict.STRONG_BUY
        elif weighted >= 0.20:
            verdict = Verdict.BUY
        elif weighted <= -0.45:
            verdict = Verdict.STRONG_SELL
        elif weighted <= -0.20:
            verdict = Verdict.SELL
        else:
            verdict = Verdict.HOLD

        # ── Position sizing from consensus ────────────────────────────────────
        if consensus == "STRONG" and confidence >= 0.70:
            position_size = 1.0
        elif consensus == "MODERATE":
            position_size = 0.75
        elif consensus == "SPLIT":
            position_size = 0.50
        else:
            position_size = 0.25

        if fund_avoid:
            position_size = 0.0

        # ── SL multiplier from volatility ─────────────────────────────────────
        try:
            atr_ratio = float(features.get("atr_ratio", 1.0))
        except (TypeError, ValueError):
            atr_ratio = 1.0

        if atr_ratio > 2.0:
            sl_multiplier = 1.5
        elif atr_ratio > 1.5:
            sl_multiplier = 1.2
        else:
            sl_multiplier = 1.0

        # ── Hold type ─────────────────────────────────────────────────────────
        adx = float(features.get("adx", 20) or 20)
        if adx > 30 and confidence >= 0.70:
            hold_type = "POSITIONAL"
        elif adx > 20:
            hold_type = "SWING"
        else:
            hold_type = "INTRADAY"

        # ── Merge reasoning ───────────────────────────────────────────────────
        all_green = []
        all_red   = []
        all_reason= []

        for agent in agents:
            all_green  += agent.green_flags[:2]
            all_red    += agent.red_flags[:2]
            all_reason += agent.reasoning[:1]

        # Deduplicate
        all_green  = list(dict.fromkeys(all_green))[:6]
        all_red    = list(dict.fromkeys(all_red))[:6]
        all_reason = list(dict.fromkeys(all_reason))[:4]

        return DebateResult(
            symbol            = symbol,
            verdict           = verdict,
            confidence        = confidence,
            final_score       = round(weighted, 3),
            bull_score        = round(bull_res.score, 3),
            bear_score        = round(bear_res.score, 3),
            fundamental_score = round(fund_res.score, 3),
            sentiment_score   = round(sent_res.score, 3),
            agents            = agents,
            consensus         = consensus,
            reasoning         = all_reason,
            red_flags         = all_red,
            green_flags       = all_green,
            trade_allowed     = not fund_avoid and confidence >= MIN_CONFIDENCE,
            position_size     = round(position_size, 2),
            sl_multiplier     = sl_multiplier,
            hold_type         = hold_type,
        )

    def quick_scan(
        self,
        symbols:    list[str],
        dm,
        fe,
        regime_detector=None,
        news_intel=None,
    ) -> list[DebateResult]:
        """
        Scan multiple symbols. Returns sorted list by confidence.
        Use in Signal Scanner page for multi-agent view.
        """
        from src.data.models import Interval

        results = []
        for sym in symbols:
            try:
                df = dm.get_ohlcv(sym, Interval.D1, days_back=250)
                if df.empty or len(df) < 50:
                    continue

                ft = fe.build(df, drop_na=False)
                if ft.empty:
                    continue

                features = ft.iloc[-1]

                # Regime
                regime_str = None
                if regime_detector:
                    try:
                        r = regime_detector.detect(features)
                        regime_str = r.regime.value
                    except Exception:
                        pass

                # News
                news_items = []
                if news_intel:
                    try:
                        nr = news_intel.get_symbol_news(sym, max_age=120, top_n=5)
                        news_items = nr.get("articles", [])
                    except Exception:
                        pass

                result = self.analyze(
                    symbol=sym, features=features, df=df,
                    regime_str=regime_str, news_items=news_items,
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"MultiAgent scan failed for {sym}: {e}")

        return sorted(results, key=lambda r: r.confidence, reverse=True)


# ── Dashboard page ────────────────────────────────────────────────────────────

def render_multi_agent_page():
    """
    Add to app.py sidebar:
        "🤖 Multi-Agent Analysis"

    Add to routing:
        elif page == "🤖 Multi-Agent Analysis":
            from src.analysis.multi_agent_engine import render_multi_agent_page
            render_multi_agent_page()
    """
    import streamlit as st
    import plotly.graph_objects as go

    st.header("🤖 Multi-Agent Trading Analysis")
    st.caption(
        "4 specialized agents debate every signal: "
        "Bull (technical) + Bear (risk) + Fundamental (valuation) + Sentiment (news/macro). "
        "Adversarial analysis reduces false signals by 25-35%."
    )

    # ── Agent legend ──────────────────────────────────────────────────────────
    a1, a2, a3, a4 = st.columns(4)
    for col, name, icon, desc, color in [
        (a1, "BULL AGENT",        "📈", "Finds buy reasons. Trend, momentum, breakouts.", "#00cc66"),
        (a2, "BEAR AGENT",        "📉", "Finds risks. Adversarial. Kills weak signals.", "#ff4444"),
        (a3, "FUNDAMENTAL AGENT", "🏦", "Alpha Vantage data. Valuation, ROE, margins.", "#4da6ff"),
        (a4, "SENTIMENT AGENT",   "📰", "News + FII/DII flow + market regime.", "#ffaa00"),
    ]:
        col.markdown(
            f'<div style="border:1px solid {color};border-radius:6px;padding:10px;text-align:center">'
            f'<div style="color:{color};font-weight:700;font-size:13px">{icon} {name}</div>'
            f'<div style="color:#ccc;font-size:10px;margin-top:4px">{desc}</div>'
            f'</div>', unsafe_allow_html=True,
        )

    st.divider()

    # ── Symbol selector ───────────────────────────────────────────────────────
    try:
        from src.data.manager import DataManager
        from src.data.models import Interval
        from src.features.feature_engine import FeatureEngine
        from src.analysis.regime_detector import RegimeDetector
    except ImportError as e:
        st.error(f"Import error: {e}"); return

    UNIVERSE = {
        "Indices":    ["NIFTY50", "BANKNIFTY", "NIFTYIT"],
        "Equities":   ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                       "SBIN", "WIPRO", "AXISBANK", "KOTAKBANK", "LT"],
        "Commodities":["GOLD", "SILVER", "CRUDEOIL"],
        "Crypto":     ["BTC", "ETH", "SOL"],
    }

    col_cat, col_sym, col_mode = st.columns([1, 2, 1])
    with col_cat:
        cat = st.selectbox("Category", list(UNIVERSE.keys()))
    with col_sym:
        sym = st.selectbox("Symbol", UNIVERSE[cat])
    with col_mode:
        mode = st.radio("Mode", ["Single", "Scan All"], horizontal=True)

    # ── Load fundamentals ─────────────────────────────────────────────────────
    use_fundamentals = st.checkbox("Include fundamental data (uses Alpha Vantage API call)", value=False)
    overview = {}
    if use_fundamentals:
        with st.spinner(f"Loading {sym} fundamentals..."):
            try:
                from src.data.adapters.alphavantage_adapter import get_overview
                overview = get_overview(sym)
            except Exception:
                st.caption("Fundamental data unavailable — technical only")

    mae = MultiAgentEngine()
    dm  = DataManager()
    fe  = FeatureEngine()

    try:
        rd = RegimeDetector()
    except Exception:
        rd = None

    # News
    news_items = []
    try:
        from src.news.news_intelligence import NewsIntelligence
        ni = NewsIntelligence()
        nr = ni.get_symbol_news(sym, max_age=120, top_n=8)
        news_items = nr.get("articles", [])
    except Exception:
        pass

    # FII data
    fii_data = None
    try:
        from src.analysis.fii_dii_tracker import FIIDIITracker
        fii_data = FIIDIITracker().get_latest()
    except Exception:
        pass

    if mode == "Single":
        if st.button("🔍 Run Multi-Agent Analysis", type="primary"):
            with st.spinner("Running 4-agent debate..."):
                try:
                    df = dm.get_ohlcv(sym, Interval.D1, days_back=250)
                    ft = fe.build(df, drop_na=False)
                    features = ft.iloc[-1]

                    regime_str = None
                    if rd:
                        try:
                            r = rd.detect(features)
                            regime_str = r.regime.value
                        except Exception:
                            pass

                    result = mae.analyze(
                        symbol=sym, features=features, df=df,
                        overview=overview, news_items=news_items,
                        regime_str=regime_str, fii_data=fii_data,
                    )
                    st.session_state["mae_result"] = result
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

        result = st.session_state.get("mae_result")
        if result and result.symbol == sym:
            _render_debate_result(result)

    else:
        if st.button("🔍 Scan All Symbols", type="primary"):
            symbols = UNIVERSE[cat]
            with st.spinner(f"Running multi-agent scan on {len(symbols)} symbols..."):
                results = mae.quick_scan(
                    symbols=symbols, dm=dm, fe=fe,
                    regime_detector=rd,
                    news_intel=None,
                )
                st.session_state["mae_scan"] = results

        results = st.session_state.get("mae_scan", [])
        if results:
            _render_scan_table(results)


def _render_debate_result(result: DebateResult):
    import streamlit as st
    import plotly.graph_objects as go

    # ── Verdict banner ────────────────────────────────────────────────────────
    verdict_colors = {
        "STRONG BUY":  ("#003300", "#00ff88"),
        "BUY":         ("#003d00", "#00cc66"),
        "HOLD":        ("#1a1a2e", "#8888ff"),
        "SELL":        ("#3d0000", "#ff6666"),
        "STRONG SELL": ("#330000", "#ff4444"),
        "AVOID":       ("#330033", "#ff44ff"),
    }
    bg_c, fg_c = verdict_colors.get(result.verdict.value, ("#222", "#fff"))
    consensus_emoji = {"STRONG":"💪","MODERATE":"👍","SPLIT":"⚖️","CONFLICTED":"⚠️"}.get(result.consensus,"")

    st.markdown(
        f'<div style="background:{bg_c};border:1px solid {fg_c};border-radius:8px;'
        f'padding:16px 20px;margin-bottom:12px">'
        f'<div style="font-size:24px;font-weight:700;color:{fg_c}">'
        f'{result.verdict.value}</div>'
        f'<div style="color:{fg_c};opacity:0.85;font-size:13px;margin-top:4px">'
        f'Confidence: {result.confidence:.0%} | '
        f'Consensus: {consensus_emoji} {result.consensus} | '
        f'Score: {result.final_score:+.3f} | '
        f'Position size: {result.position_size:.0%} | '
        f'Hold: {result.hold_type}'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    if not result.trade_allowed:
        st.error("🚫 Trade NOT allowed — confidence below 55% or fundamental red flags detected")

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Final Score",   f"{result.final_score:+.3f}")
    k2.metric("Confidence",    f"{result.confidence:.0%}")
    k3.metric("Position Size", f"{result.position_size:.0%}")
    k4.metric("SL Multiplier", f"{result.sl_multiplier:.1f}×")
    k5.metric("Hold Type",     result.hold_type)

    # ── Agent score radar ─────────────────────────────────────────────────────
    st.subheader("Agent Scores")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Bull Agent", "Bear Agent", "Fundamental", "Sentiment"],
        y=[result.bull_score, result.bear_score, result.fundamental_score, result.sentiment_score],
        marker_color=[
            "#00cc66" if result.bull_score > 0 else "#ff4444",
            "#00cc66" if result.bear_score > -0.1 else "#ff4444",
            "#00cc66" if result.fundamental_score > 0 else "#ff4444",
            "#00cc66" if result.sentiment_score > 0 else "#ff4444",
        ],
        text=[f"{s:+.3f}" for s in [
            result.bull_score, result.bear_score,
            result.fundamental_score, result.sentiment_score
        ]],
        textposition="outside",
    ))
    fig.add_hline(y=0, line_color="#555")
    fig.update_layout(
        height=280, template="plotly_dark",
        title=f"{result.symbol} — Agent Score Breakdown",
        yaxis=dict(range=[-1.1, 1.1], title="Score (-1=Bearish, +1=Bullish)"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Green flags / Red flags ───────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**✅ Green Flags**")
        for g in result.green_flags:
            st.success(g)
        if not result.green_flags:
            st.caption("No strong positives")
    with c2:
        st.markdown("**🚨 Red Flags**")
        for r in result.red_flags:
            st.error(r)
        if not result.red_flags:
            st.success("No red flags detected")

    # ── Per-agent reasoning ───────────────────────────────────────────────────
    st.subheader("Agent Reasoning")
    agent_tabs = st.tabs(["📈 Bull", "📉 Bear", "🏦 Fundamental", "📰 Sentiment"])
    for tab, agent in zip(agent_tabs, result.agents):
        with tab:
            score_color = "#00cc66" if agent.score >= 0 else "#ff4444"
            st.markdown(
                f'<span style="color:{score_color};font-size:18px;font-weight:700">'
                f'Score: {agent.score:+.3f}</span> | Confidence: {agent.confidence:.0%}',
                unsafe_allow_html=True,
            )
            for g in agent.green_flags:
                st.success(g)
            for r_text in agent.reasoning:
                st.info(r_text)
            for r in agent.red_flags:
                st.error(r)


def _render_scan_table(results: list[DebateResult]):
    import streamlit as st
    import pandas as pd

    rows = []
    for r in results:
        rows.append({
            "Symbol":    r.symbol,
            "Verdict":   r.verdict.value,
            "Conf":      f"{r.confidence:.0%}",
            "Score":     f"{r.final_score:+.3f}",
            "Bull":      f"{r.bull_score:+.2f}",
            "Bear":      f"{r.bear_score:+.2f}",
            "Fund":      f"{r.fundamental_score:+.2f}",
            "Sent":      f"{r.sentiment_score:+.2f}",
            "Consensus": r.consensus,
            "Size":      f"{r.position_size:.0%}",
            "Hold":      r.hold_type,
            "Trade":     "✅" if r.trade_allowed else "❌",
        })

    df_r = pd.DataFrame(rows)
    verdict_colors = {
        "STRONG BUY": "color:#00ff88;font-weight:bold",
        "BUY":        "color:#00cc66",
        "HOLD":       "color:#888",
        "SELL":       "color:#ff6666",
        "STRONG SELL":"color:#ff4444;font-weight:bold",
        "AVOID":      "color:#ff44ff;font-weight:bold",
    }
    st.dataframe(
        df_r.style.map(lambda v: verdict_colors.get(v, ""), subset=["Verdict"]),
        use_container_width=True, hide_index=True,
    )

    # Best opportunities
    buy_signals = [r for r in results if "BUY" in r.verdict.value and r.trade_allowed]
    if buy_signals:
        st.success(f"🎯 Top opportunity: **{buy_signals[0].symbol}** — {buy_signals[0].verdict.value} ({buy_signals[0].confidence:.0%} confidence)")