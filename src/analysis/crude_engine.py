"""
Crude Oil Intelligence Engine.
Predicts MCX crude opening direction using WTI/Brent overnight + DXY + news.
Free data only via yfinance.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from dataclasses import dataclass, field
from loguru import logger

IST = ZoneInfo("Asia/Kolkata")


@dataclass
class MCXPrediction:
    direction:        str
    confidence:       float
    expected_change:  float
    wti_change:       float
    brent_change:     float
    dxy_change:       float
    news_sentiment:   str
    news_score:       float
    reasoning:        list
    timestamp:        datetime
    risk_level:       str

    def __str__(self):
        arr = {"GAP_UP": "▲", "GAP_DOWN": "▼", "FLAT": "—"}
        return (
            f"{arr.get(self.direction,'—')} MCX CRUDE: {self.direction} "
            f"(conf={self.confidence:.0%}, ~{self.expected_change:+.1f}%) | "
            f"WTI={self.wti_change:+.2f}% Brent={self.brent_change:+.2f}% "
            f"News={self.news_sentiment}"
        )


class CrudeOilEngine:
    """
    Real-time crude oil intelligence for MCX trading.
    Uses yfinance for WTI (CL=F), Brent (BZ=F), DXY (DX-Y.NYB).
    Falls back gracefully if any data is unavailable.
    """

    STRONG_MOVE   = 0.8
    MODERATE_MOVE = 0.4
    DXY_WEIGHT    = 0.35

    def predict_mcx_open(self, news_sentiment: dict = None) -> MCXPrediction:
        now = datetime.now(IST)
        reasoning = []

        wti_chg,   wti_msg   = self._get_overnight_change("CL=F",     "WTI")
        brent_chg, brent_msg = self._get_overnight_change("BZ=F",     "Brent")
        dxy_chg,   dxy_msg   = self._get_overnight_change("DX-Y.NYB", "DXY")

        reasoning.extend([wti_msg, brent_msg, dxy_msg])

        wti_score   = self._to_score(wti_chg,   self.STRONG_MOVE)
        brent_score = self._to_score(brent_chg, self.STRONG_MOVE)
        dxy_impact  = -self._to_score(dxy_chg,  0.3) * self.DXY_WEIGHT

        news_score, news_label = 0.0, "NEUTRAL"
        if news_sentiment:
            news_score = news_sentiment.get("score", 0.0)
            news_label = news_sentiment.get("label", "NEUTRAL")
            if abs(news_score) > 0.1:
                reasoning.append(
                    f"News sentiment: {news_label} (score={news_score:+.2f})"
                )

        composite = (
            0.45 * wti_score +
            0.25 * brent_score +
            0.15 * dxy_impact +
            0.15 * news_score
        )

        agreed = (wti_chg > 0) == (brent_chg > 0)
        if not agreed:
            reasoning.append(
                f"WTI and Brent diverge — LOW confidence"
            )

        direction = (
            "GAP_UP"   if composite > 0.15 else
            "GAP_DOWN" if composite < -0.15 else
            "FLAT"
        )

        raw_conf   = min(1.0, abs(composite) * 1.8)
        confidence = round(raw_conf * (1.0 if agreed else 0.6), 3)

        expected_change = round(
            (0.85 * wti_chg + 0.10 * brent_chg) * 0.9, 2
        )

        if abs(wti_chg) > 2.0 or not agreed:
            risk_level = "HIGH"
            reasoning.append("HIGH RISK: large overnight move or divergence")
        elif abs(wti_chg) > 1.0:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return MCXPrediction(
            direction=direction, confidence=confidence,
            expected_change=expected_change,
            wti_change=wti_chg, brent_change=brent_chg, dxy_change=dxy_chg,
            news_sentiment=news_label, news_score=news_score,
            reasoning=reasoning, timestamp=now, risk_level=risk_level,
        )

    def get_crude_technicals(self, days_back: int = 60) -> dict:
        try:
            import yfinance as yf
            df = yf.download("CL=F", period=f"{days_back}d",
                             interval="1d", progress=False, auto_adjust=True)
            if df.empty or len(df) < 20:
                return {}

            df.columns = [str(c[0]).lower() if isinstance(c, tuple) else str(c).lower()
                          for c in df.columns]

            close = df["close"]
            price = float(close.iloc[-1])
            prev  = float(close.iloc[-2])
            chg   = (price - prev) / prev * 100

            # RSI
            delta  = close.diff()
            gain   = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
            loss   = (-delta).clip(lower=0).ewm(com=13, adjust=False).mean()
            rs     = gain / loss.replace(0, 1e-9)
            rsi    = float((100 - 100 / (1 + rs)).iloc[-1])

            # MACD
            ema12  = close.ewm(span=12, adjust=False).mean()
            ema26  = close.ewm(span=26, adjust=False).mean()
            macd_h = float((ema12 - ema26).iloc[-1])

            # ATR
            tr     = pd.concat([
                df["high"] - df["low"],
                (df["high"] - close.shift()).abs(),
                (df["low"]  - close.shift()).abs(),
            ], axis=1).max(axis=1)
            atr_val = float(tr.ewm(span=14, adjust=False).mean().iloc[-1])

            # BB
            sma20  = close.rolling(20).mean()
            std20  = close.rolling(20).std()
            bb_pct = float(((close - (sma20 - 2*std20)) / (4*std20 + 1e-9)).iloc[-1])

            return {
                "price":       round(price, 2),
                "change_pct":  round(chg, 2),
                "rsi":         round(rsi, 1),
                "macd_hist":   round(macd_h, 3),
                "bb_pct_b":    round(bb_pct, 3),
                "atr":         round(atr_val, 2),
                "atr_pct":     round(atr_val / price * 100, 2),
                "trend":       "BULLISH" if chg > 0 else "BEARISH",
            }
        except Exception as e:
            logger.warning(f"Crude technicals failed: {e}")
            return {}

    def get_correlated_assets(self) -> dict:
        import yfinance as yf
        assets = {
            "DXY (USD Index)": "DX-Y.NYB",
            "Gold":            "GC=F",
            "Natural Gas":     "NG=F",
            "Brent Crude":     "BZ=F",
            "S&P 500":         "^GSPC",
        }
        result = {}
        for name, ticker in assets.items():
            try:
                df = yf.download(ticker, period="5d", interval="1d",
                                 progress=False, auto_adjust=True)
                if not df.empty and len(df) >= 2:
                    close = df.iloc[:, 3]
                    price = float(close.iloc[-1])
                    prev  = float(close.iloc[-2])
                    chg   = (price - prev) / prev * 100
                    result[name] = {
                        "price":  round(price, 2),
                        "change": round(chg, 2),
                        "signal": "BULLISH" if chg > 0 else "BEARISH",
                    }
            except Exception:
                pass
        return result

    def _get_overnight_change(self, ticker: str, name: str) -> tuple:
        try:
            import yfinance as yf
            df = yf.download(ticker, period="5d", interval="1h",
                             progress=False, auto_adjust=True)
            if df.empty:
                return 0.0, f"{name}: no data"
            close      = df.iloc[:, 3]
            now_price  = float(close.iloc[-1])
            ago_idx    = max(0, len(close) - 8)
            ago_price  = float(close.iloc[ago_idx])
            chg        = (now_price - ago_price) / ago_price * 100
            strength   = ("strong" if abs(chg) > self.STRONG_MOVE
                          else "moderate" if abs(chg) > self.MODERATE_MOVE
                          else "small")
            direction  = "UP" if chg > 0 else "DOWN"
            msg = f"{name}: {chg:+.2f}% ({strength} {direction})"
            return round(chg, 3), msg
        except Exception as e:
            logger.warning(f"{name} overnight change failed: {e}")
            return 0.0, f"{name}: unavailable"

    def _to_score(self, pct_change: float, threshold: float) -> float:
        return max(-1.0, min(1.0, pct_change / threshold))
