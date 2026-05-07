"""
AI Trading Assistant — upgraded with Anthropic API + richer context.
Rule-based fallback always works. API enhances when available.
"""
import time
import re
from dataclasses import dataclass
from typing import Optional
from loguru import logger
from src.utils.market_hours import MarketHours


@dataclass
class AIResponse:
    answer: str
    sources: list
    confidence: str
    timestamp: float


class AIEngine:
    CACHE_TTL = 90  # 90 seconds

    def __init__(self, use_api: bool = False, api_key: Optional[str] = None):
        self.use_api = use_api
        self.api_key = api_key
        self._cache: dict = {}
        logger.info(f"AIEngine ready | API={'yes' if use_api and api_key else 'rule-based'}")

    def ask(
        self,
        question: str,
        indicators: dict = None,
        news_sentiment: dict = None,
        crude_pred: dict = None,
        smart_signal=None,
        symbol: str = "CRUDEOIL",
    ) -> AIResponse:
        cache_key = f"{question[:50]}_{symbol}"
        if cache_key in self._cache:
            c = self._cache[cache_key]
            if time.time() - c.timestamp < self.CACHE_TTL:
                return c

        context = self._build_context(indicators, news_sentiment, crude_pred, smart_signal, symbol)
        answer  = self._rule_based(question, context, symbol, indicators, news_sentiment, crude_pred)
        sources = ["technical_indicators", "news_sentiment", "global_markets"]
        conf    = "MEDIUM"

        if self.use_api and self.api_key:
            try:
                enhanced = self._api_answer(question, context, symbol)
                if enhanced:
                    answer  = enhanced
                    sources.append("Anthropic_Claude")
                    conf    = "HIGH"
            except Exception as e:
                logger.debug(f"API fallback: {e}")

        resp = AIResponse(answer=answer, sources=sources, confidence=conf, timestamp=time.time())
        self._cache[cache_key] = resp
        return resp

    def _build_context(self, indicators, news, crude_pred, smart_signal, symbol) -> str:
        parts = [f"Symbol: {symbol}"]
        if indicators:
            parts.append(
                f"RSI={indicators.get('rsi',indicators.get('rsi_14',50)):.0f} "
                f"MACD={indicators.get('macd_hist',0):.3f} "
                f"ADX={indicators.get('adx',20):.0f} "
                f"ATR%={indicators.get('atr_pct',0.015)*100:.1f}% "
                f"Price=${indicators.get('price',0):.2f}"
            )
        if news:
            parts.append(
                f"News={news.get('label','NEUTRAL')} "
                f"score={news.get('score',0):+.2f} "
                f"n={news.get('n_articles',0)} "
                f"bull={news.get('bullish_n',0)} bear={news.get('bearish_n',0)}"
            )
        if crude_pred:
            parts.append(
                f"MCX_pred={crude_pred.get('direction','?')} "
                f"conf={crude_pred.get('confidence',0):.0%} "
                f"WTI={crude_pred.get('wti_change',0):+.2f}% "
                f"Brent={crude_pred.get('brent_change',0):+.2f}%"
            )
        if smart_signal:
            parts.append(
                f"SmartSignal={smart_signal.signal} "
                f"composite={smart_signal.composite_score:+.2f} "
                f"tech={smart_signal.tech_score:+.2f} "
                f"news={smart_signal.news_score:+.2f} "
                f"global={smart_signal.global_score:+.2f}"
            )
        return " | ".join(parts)

    def _rule_based(self, question: str, context: str, symbol: str,
                    indicators: dict, news: dict, crude_pred: dict) -> str:
        q   = question.lower()
        ctx = self._parse(context)
        rsi      = ctx.get("rsi", 50.0)
        macd     = ctx.get("macd_hist", 0.0)
        adx      = ctx.get("adx", 20.0)
        atr_pct  = ctx.get("atr_pct", 0.015)
        news_lb  = ctx.get("news_label", "NEUTRAL")
        news_sc  = ctx.get("news_score", 0.0)
        pred     = ctx.get("mcx_direction", "FLAT")
        pconf    = ctx.get("mcx_confidence", 0.5)
        wti      = ctx.get("wti_change", 0.0)
        brent    = ctx.get("brent_change", 0.0)
        smart_s  = ctx.get("smart_signal", "")
        composite= ctx.get("composite", 0.0)
        price    = ctx.get("price", 0.0)

        # Market open status
        if any(phrase in q for phrase in ["is open today", "open today", "market open", "trading today"]):
            mh = MarketHours()
            now = mh.now_ist()
            mcx_session = mh.get_mcx_session(now)
            nse_session = mh.get_nse_session(now)
            holiday_name = mh.get_holiday_name(now) if mh.is_nse_holiday(now) or mh.is_mcx_full_closure(now) else ""

            if "mcx" in q or symbol == "CRUDEOIL":
                status = "OPEN" if mh.is_mcx_tradeable(now) else "CLOSED"
                session_desc = mcx_session
                if holiday_name:
                    session_desc += f" ({holiday_name})"
                return f"MCX is {status} today. Session: {session_desc}."
            else:
                status = "OPEN" if mh.is_nse_tradeable(now) else "CLOSED"
                session_desc = nse_session
                if holiday_name:
                    session_desc += f" ({holiday_name})"
                return f"NSE is {status} today. Session: {session_desc}."

        # MCX / Opening
        if any(w in q for w in ["opening","gap","tomorrow"]) or ("mcx" in q and "open" in q):
            dir_txt = {
                "GAP_UP":   f"▲ GAP UP ({pconf:.0%} confidence, ~{wti:+.1f}% expected)",
                "GAP_DOWN": f"▼ GAP DOWN ({pconf:.0%} confidence, ~{wti:+.1f}% expected)",
                "FLAT":     f"— FLAT (no strong overnight signal)",
            }.get(pred, "uncertain")

            qual = ""
            if pred == "GAP_UP":
                qual = (
                    "Strategy: Wait for first 15-minute candle to confirm. "
                    "Buy above the opening range high with stop below opening range low. "
                    f"WTI overnight: {wti:+.2f}%. Brent: {brent:+.2f}%."
                )
            elif pred == "GAP_DOWN":
                qual = (
                    "Strategy: Avoid buying on gap down. "
                    "Short below opening range low if confirmed. "
                    f"WTI overnight: {wti:+.2f}%. Brent: {brent:+.2f}%."
                )
            else:
                qual = "Wait for first 30 minutes before taking any position."

            return (
                f"MCX Crude opening prediction: {dir_txt}.\n\n"
                f"Analysis:\n"
                f"• WTI (primary): {wti:+.2f}%\n"
                f"• Brent (confirmation): {brent:+.2f}%\n"
                f"• News sentiment: {news_lb}\n"
                f"• Risk level: {'HIGH — be cautious' if abs(wti)>1.5 else 'NORMAL'}\n\n"
                f"{qual}"
            )

        # Bull / buy questions
        if any(w in q for w in ["up","bull","buy","long","rise","increase","should i buy"]):
            bull, bear = [], []
            if rsi < 35:   bull.append(f"RSI {rsi:.0f} oversold — bounce likely")
            elif rsi > 65: bear.append(f"RSI {rsi:.0f} overbought — pullback risk")
            if macd > 0:   bull.append("MACD histogram positive — upward momentum")
            else:          bear.append("MACD histogram negative — downward momentum")
            if news_lb == "BULLISH": bull.append(f"News bullish (score {news_sc:+.2f})")
            elif news_lb == "BEARISH": bear.append(f"News bearish (score {news_sc:+.2f})")
            if pred == "GAP_UP": bull.append(f"WTI up {wti:+.1f}% overnight — MCX gap up expected")
            elif pred == "GAP_DOWN": bear.append(f"WTI down {wti:+.1f}% overnight")

            bc, rc = len(bull), len(bear)
            trend_txt = (
                f"Strong trend (ADX={adx:.0f}) — trend-following works. "
                if adx > 25 else
                f"Ranging market (ADX={adx:.0f}) — momentum trades risky. "
            )
            if bc >= rc + 1:
                verdict = "LEAN BULLISH"
                action  = f"Consider small BUY position. Risk = 1% of capital (₹100 on ₹10,000). "
            elif rc >= bc + 1:
                verdict = "LEAN BEARISH"
                action  = "Avoid buying. Wait for RSI below 40 or price to reach support. "
            else:
                verdict = "MIXED / NO EDGE"
                action  = "No clear edge. Best action: WAIT. Trade only when 3+ signals agree. "

            return (
                f"{verdict} for {symbol}.\n\n"
                f"Bull signals: {', '.join(bull) if bull else 'none'}\n"
                f"Bear signals: {', '.join(bear) if bear else 'none'}\n\n"
                f"{trend_txt}\n{action}"
                f"Stop loss: 2×ATR = ~{atr_pct*100*2:.1f}% below entry. "
                f"Never risk more than ₹100 on ₹10,000 capital."
            )

        # Stop loss
        if any(w in q for w in ["stop","sl","stop loss","stoploss","exit"]):
            vol_txt = ("HIGH volatility — use 2.5-3×ATR stop to avoid whipsaw."
                       if atr_pct > 0.02 else "NORMAL volatility — 2×ATR stop is appropriate.")
            return (
                f"Stop loss guidance for {symbol}:\n\n"
                f"• ATR (14): ~{atr_pct*100:.1f}% of price per day\n"
                f"• Standard stop: 2×ATR below entry\n"
                f"• Volatile market stop: 2.5-3×ATR\n"
                f"• {vol_txt}\n\n"
                f"On ₹10,000 capital:\n"
                f"• Max loss per trade: ₹100 (1% rule)\n"
                f"• Never widen stop to 'hope' — if stop hits, accept the loss\n"
                f"• Consider no stop for high-conviction swing trades; instead size down to 0.5% risk"
            )

        # News / sentiment
        if any(w in q for w in ["news","sentiment","latest","happening"]):
            return (
                f"Market intelligence for {symbol}:\n\n"
                f"• News sentiment: {news_lb} (score {news_sc:+.2f})\n"
                f"• Bullish articles: {news.get('bullish_n',0) if news else '?'}\n"
                f"• Bearish articles: {news.get('bearish_n',0) if news else '?'}\n\n"
                f"{'Positive news flow supports bullish trades.' if news_lb=='BULLISH' else 'Negative news flow — caution on longs.' if news_lb=='BEARISH' else 'Neutral news — look at technicals for direction.'}\n\n"
                f"For crude oil: always check WTI direction (3:30 AM IST) before MCX open."
            )

        # Generic
        trend = ("BULLISH" if rsi > 55 and macd > 0 else
                 "BEARISH" if rsi < 45 and macd < 0 else "NEUTRAL")
        return (
            f"Current snapshot for {symbol}:\n\n"
            f"• Price: {'$'+str(round(price,2)) if price > 0 else 'see chart'}\n"
            f"• RSI {rsi:.0f} ({'oversold' if rsi<30 else 'overbought' if rsi>70 else 'neutral'})\n"
            f"• MACD: {'positive ↑' if macd>0 else 'negative ↓'}\n"
            f"• News: {news_lb}\n"
            f"• Overall bias: {trend}\n\n"
            f"Try asking: 'MCX opening today?', 'Will {symbol} go up?', "
            f"'What stop loss?', 'Latest news?'"
        )

    def _api_answer(self, question: str, context: str, symbol: str) -> Optional[str]:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            system = (
                "You are a professional quantitative trader specialising in Indian markets "
                "(NSE, BSE, MCX), crude oil, crypto, and macro analysis. "
                "Answer in 4-6 sentences. Be specific, give actual numbers, "
                "mention entry/exit levels when relevant. "
                "Always include one sentence about risk management. "
                "Avoid generic advice. Think like a prop trader."
            )
            prompt = f"Market context: {context}\n\nQuestion: {question}"
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=350,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        except Exception as e:
            logger.debug(f"Anthropic: {e}")
            return None

    def _parse(self, ctx: str) -> dict:
        r = {}
        patterns = {
            "rsi":            r"RSI=(\d+\.?\d*)",
            "macd_hist":      r"MACD=(-?\d+\.?\d*)",
            "adx":            r"ADX=(\d+\.?\d*)",
            "atr_pct":        r"ATR%=(\d+\.?\d*)",
            "price":          r"Price\$?=?(\d+\.?\d*)",
            "news_score":     r"score=([+-]?\d+\.?\d*)",
            "news_label":     r"News=(\w+)",
            "wti_change":     r"WTI=([+-]?\d+\.?\d*)",
            "brent_change":   r"Brent=([+-]?\d+\.?\d*)",
            "mcx_direction":  r"MCX_pred=(\w+)",
            "mcx_confidence": r"conf=(\d+\.?\d*)",
            "smart_signal":   r"SmartSignal=(\w+)",
            "composite":      r"composite=([+-]?\d+\.?\d*)",
        }
        for k, pat in patterns.items():
            m = re.search(pat, ctx)
            if m:
                v = m.group(1)
                try:
                    r[k] = float(v) if k not in ("news_label","mcx_direction","smart_signal") else v
                except ValueError:
                    r[k] = v
        return r
