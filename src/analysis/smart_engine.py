"""
Smart Signal Fusion Engine — combines technical + news + global signals.
Also handles price unit display correctly for MCX/crypto/forex.
"""
from datetime import datetime
from zoneinfo import ZoneInfo
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger

IST = ZoneInfo("Asia/Kolkata")

# Price unit metadata — prevents ₹29 BTC or ₹4500 Gold confusion
PRICE_UNITS = {
    "GOLD":       {"unit": "USD/troy oz",  "mcx_factor": lambda p, r: p * r * 31.1035 / 10, "mcx_unit": "₹/10g"},
    "SILVER":     {"unit": "USD/troy oz",  "mcx_factor": lambda p, r: p * r * 31.1035 / 1000, "mcx_unit": "₹/g"},
    "CRUDEOIL":   {"unit": "USD/barrel",   "mcx_factor": lambda p, r: p * r, "mcx_unit": "₹/bbl"},
    "COPPER":     {"unit": "USD/lb",       "mcx_factor": None, "mcx_unit": ""},
    "NATURALGAS": {"unit": "USD/MMBtu",    "mcx_factor": None, "mcx_unit": ""},
    "NIFTY50":    {"unit": "Index pts",    "mcx_factor": None, "mcx_unit": ""},
    "BANKNIFTY":  {"unit": "Index pts",    "mcx_factor": None, "mcx_unit": ""},
    "SENSEX":     {"unit": "Index pts",    "mcx_factor": None, "mcx_unit": ""},
    "BTC":        {"unit": "USD",          "mcx_factor": lambda p, r: p * r, "mcx_unit": "₹"},
    "ETH":        {"unit": "USD",          "mcx_factor": lambda p, r: p * r, "mcx_unit": "₹"},
    "SOL":        {"unit": "USD",          "mcx_factor": lambda p, r: p * r, "mcx_unit": "₹"},
    "MATIC":      {"unit": "USD",          "mcx_factor": lambda p, r: p * r, "mcx_unit": "₹"},
    "USDINR":     {"unit": "₹ per $",      "mcx_factor": None, "mcx_unit": ""},
}

USDINR_FALLBACK = 84.0


@dataclass
class SmartSignal:
    symbol: str
    signal: str
    confidence: float
    tech_score: float
    news_score: float
    global_score: float
    composite_score: float
    reasoning: list
    price: float
    price_display: str
    price_inr: Optional[str]
    stop_loss: Optional[float]
    target: Optional[float]
    risk_reward: Optional[float]
    timestamp: datetime = field(default_factory=lambda: datetime.now(IST))

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "signal": self.signal,
            "confidence": self.confidence,
            "tech_score": self.tech_score,
            "news_score": self.news_score,
            "global_score": self.global_score,
            "composite": self.composite_score,
            "reasoning": self.reasoning,
            "price_display": self.price_display,
            "price_inr": self.price_inr,
        }


class SmartSignalEngine:

    def __init__(self, tech_w=0.50, news_w=0.25, global_w=0.25):
        self.tw, self.nw, self.gw = tech_w, news_w, global_w
        self._usdinr = None

    def get_usdinr(self) -> float:
        if self._usdinr:
            return self._usdinr
        try:
            import yfinance as yf
            r = yf.Ticker("USDINR=X").fast_info.last_price
            self._usdinr = float(r) if r and r > 0 else USDINR_FALLBACK
        except Exception:
            self._usdinr = USDINR_FALLBACK
        return self._usdinr

    def format_price(self, symbol: str, price: float) -> tuple:
        """Returns (display_str, inr_str_or_None)."""
        info = PRICE_UNITS.get(symbol)
        if not info:
            return f"₹{price:,.2f}", None
        unit = info["unit"]
        if "USD" in unit:
            display = f"${price:,.2f} ({unit})"
            mcx_fn  = info.get("mcx_factor")
            if mcx_fn:
                try:
                    inr = mcx_fn(price, self.get_usdinr())
                    mcx_u = info.get("mcx_unit", "₹")
                    inr_s = f"{mcx_u}{inr:,.2f} (approx INR)"
                except Exception:
                    inr_s = None
            else:
                inr_s = None
            return display, inr_s
        elif "Index" in unit:
            return f"{price:,.2f} pts", None
        else:
            return f"₹{price:,.2f}", None

    def generate(
        self,
        symbol: str,
        features: dict,
        news_sentiment: dict = None,
        learning_weights: dict = None,
    ) -> SmartSignal:
        price = features.get("close", 0.0)
        price_disp, price_inr = self.format_price(symbol, price)

        tech_score, tech_r   = self._tech(features)
        news_score, news_r   = self._news(news_sentiment)
        glob_score, glob_r   = self._global(symbol)

        tw, nw, gw = self.tw, self.nw, self.gw
        if learning_weights:
            rw = learning_weights.get("rule", 1.0)
            nwm = learning_weights.get("news", 1.0)
            tot = tw*rw + nw*nwm + gw
            tw, nw, gw = tw*rw/tot, nw*nwm/tot, gw/tot

        composite = max(-1.0, min(1.0, tw*tech_score + nw*news_score + gw*glob_score))

        if composite >= 0.25:
            signal = "BUY";  conf = min(1.0, 0.5 + composite*0.5)
        elif composite <= -0.25:
            signal = "SELL"; conf = min(1.0, 0.5 + abs(composite)*0.5)
        else:
            signal = "HOLD"; conf = 0.5

        atr = price * features.get("atr_pct", 0.015)
        if signal == "BUY":
            sl = round(price - 2*atr, 2); tgt = round(price + 3*atr, 2); rr = 1.5
        elif signal == "SELL":
            sl = round(price + 2*atr, 2); tgt = round(price - 3*atr, 2); rr = 1.5
        else:
            sl = tgt = rr = None

        return SmartSignal(
            symbol=symbol, signal=signal, confidence=round(conf, 3),
            tech_score=round(tech_score, 3),
            news_score=round(news_score, 3),
            global_score=round(glob_score, 3),
            composite_score=round(composite, 3),
            reasoning=tech_r + news_r + glob_r,
            price=price, price_display=price_disp, price_inr=price_inr,
            stop_loss=sl, target=tgt, risk_reward=rr,
        )

    def _tech(self, f: dict) -> tuple:
        s, r = 0.0, []
        rsi = f.get("rsi_14", 50)
        if rsi < 30:   s += 0.25;  r.append(f"RSI oversold ({rsi:.0f}) → strong bullish")
        elif rsi > 70: s -= 0.25;  r.append(f"RSI overbought ({rsi:.0f}) → strong bearish")
        elif rsi > 55: s += 0.10;  r.append(f"RSI bullish zone ({rsi:.0f})")
        elif rsi < 45: s -= 0.10;  r.append(f"RSI bearish zone ({rsi:.0f})")

        mh = f.get("macd_hist", 0); mc = f.get("macd_hist_chg", 0)
        if mh > 0 and mc > 0:    s += 0.20; r.append("MACD expanding positive → bull momentum")
        elif mh < 0 and mc < 0:  s -= 0.20; r.append("MACD expanding negative → bear momentum")
        elif mh > 0:              s += 0.05; r.append("MACD positive but weakening")
        elif mh < 0:              s -= 0.05; r.append("MACD negative but recovering")

        e9 = f.get("ema9_pct", 0); e50 = f.get("ema50_pct", 0); e200 = f.get("ema200_pct", 0)
        if e9 > 0 and e50 > 0 and e200 > 0:
            s += 0.25; r.append("Above EMA9 + EMA50 + EMA200 → full bull alignment")
        elif e9 > 0 and e50 > 0:
            s += 0.15; r.append("Above EMA9 & EMA50 → bullish")
        elif e9 < 0 and e50 < 0 and e200 < 0:
            s -= 0.25; r.append("Below EMA9 + EMA50 + EMA200 → full bear alignment")
        elif e9 < 0 and e50 < 0:
            s -= 0.15; r.append("Below EMA9 & EMA50 → bearish")

        bb = f.get("bb_pct_b", 0.5)
        if bb < 0.05:  s += 0.15; r.append("At lower Bollinger Band → bounce zone")
        elif bb > 0.95:s -= 0.15; r.append("At upper Bollinger Band → overbought")

        adx = f.get("adx", 20); dd = f.get("di_diff", 0)
        if adx > 30:
            s += 0.15 if dd > 0 else -0.15
            r.append(f"ADX {adx:.0f} → strong {'bull' if dd>0 else 'bear'} trend confirmed")

        return max(-1.0, min(1.0, s)), r

    def _news(self, ns: dict) -> tuple:
        if not ns: return 0.0, ["No news data"]
        sc = ns.get("score", 0.0); lb = ns.get("label","NEUTRAL")
        bn = ns.get("bullish_n", 0); brn = ns.get("bearish_n", 0); n = ns.get("n_articles", 0)
        r = []
        if n > 0:
            r.append(f"News: {lb} | {n} articles ({bn}↑ {brn}↓)")
            if abs(sc) > 0.2:
                r.append(f"{'Positive' if sc>0 else 'Negative'} news flow (score {sc:+.2f}) → market {'tailwind' if sc>0 else 'headwind'}")
        return max(-1.0, min(1.0, sc * 2)), r

    def _global(self, symbol: str) -> tuple:
        s, r = 0.0, []
        try:
            import yfinance as yf
            sp = yf.download("^GSPC", period="2d", interval="1d", progress=False, auto_adjust=True)
            if not sp.empty and len(sp) >= 2:
                chg = float((sp.iloc[-1, 3] / sp.iloc[-2, 3] - 1) * 100)
                if chg > 0.5:   s += 0.3; r.append(f"S&P 500 +{chg:.1f}% → global risk-on")
                elif chg < -0.5:s -= 0.3; r.append(f"S&P 500 {chg:.1f}% → global risk-off")

            if symbol in ("GOLD","SILVER","CRUDEOIL","COPPER","BTC","ETH","SOL"):
                dxy = yf.download("DX-Y.NYB", period="2d", interval="1d",
                                   progress=False, auto_adjust=True)
                if not dxy.empty and len(dxy) >= 2:
                    dc = float((dxy.iloc[-1, 3] / dxy.iloc[-2, 3] - 1) * 100)
                    if dc > 0.3:    s -= 0.3; r.append(f"USD strong (+{dc:.1f}%) → {symbol} headwind")
                    elif dc < -0.3: s += 0.3; r.append(f"USD weak ({dc:.1f}%) → {symbol} tailwind")
        except Exception as e:
            logger.debug(f"Global signals: {e}")
            r.append("Global data temporarily unavailable")
        return max(-1.0, min(1.0, s)), r
