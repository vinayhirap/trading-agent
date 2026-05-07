# trading-agent/src/analysis/regime_detector.py
"""
Market Regime Detector — HMM + Rule-Based 4-State System

4 Regimes:
  BULL_TREND   — ADX>25, price above EMA50, positive momentum
  BEAR_TREND   — ADX>25, price below EMA50, negative momentum
  RANGING_LOW  — ADX<18, low volatility, RSI 40-60
  RANGING_HIGH — ADX<18, high ATR ratio, wide BB

Integration:
  run_pipeline_v3.py  → gate signals through regime
  backtest_engine.py  → regime-conditional sizing
  risk_manager.py     → regime-aware stop multipliers
  app.py              → regime badge on every signal card

Usage:
    from src.analysis.regime_detector import RegimeDetector
    rd = RegimeDetector()
    result = rd.detect(features_series)
    signal, conf = result.gate_signal("BUY", 0.65)
"""
from __future__ import annotations
import json
import pickle
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger

ROOT      = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
HMM_PATH  = MODEL_DIR / "regime_hmm.pkl"
STATS_PATH = ROOT / "data" / "regime_stats.json"


# ── Regime enum ───────────────────────────────────────────────────────────────

class Regime(str, Enum):
    BULL_TREND   = "BULL_TREND"
    BEAR_TREND   = "BEAR_TREND"
    RANGING_LOW  = "RANGING_LOW"
    RANGING_HIGH = "RANGING_HIGH"


REGIME_PARAMS = {
    Regime.BULL_TREND: {
        "signal_multiplier": 1.0,
        "sl_multiplier":     1.0,
        "size_multiplier":   1.0,
        "allowed_signals":   ["BUY", "STRONG BUY"],
        "strategy":          "TREND_FOLLOWING",
        "description":       "Strong uptrend — trend signals reliable. Buy dips. Full size.",
    },
    Regime.BEAR_TREND: {
        "signal_multiplier": 0.7,
        "sl_multiplier":     1.0,
        "size_multiplier":   0.7,
        "allowed_signals":   ["SELL", "STRONG SELL"],
        "strategy":          "TREND_FOLLOWING",
        "description":       "Strong downtrend — avoid fresh longs. Short or stay in cash.",
    },
    Regime.RANGING_LOW: {
        "signal_multiplier": 0.85,
        "sl_multiplier":     0.8,
        "size_multiplier":   0.9,
        "allowed_signals":   ["BUY", "SELL", "STRONG BUY", "STRONG SELL"],
        "strategy":          "MEAN_REVERSION",
        "description":       "Sideways low-vol — mean reversion works. Trade RSI extremes.",
    },
    Regime.RANGING_HIGH: {
        "signal_multiplier": 0.6,
        "sl_multiplier":     1.5,
        "size_multiplier":   0.5,
        "allowed_signals":   [],
        "strategy":          "WAIT",
        "description":       "High-vol chop — reduce size, widen stops. Prefer waiting.",
    },
}

REGIME_FEATURE_COLS = [
    "adx", "di_diff", "atr_ratio", "atr_pct",
    "rsi_14", "bb_pct_b", "bb_width",
    "ema9_pct", "ema50_pct",
    "macd_hist", "macd_hist_chg",
    "vol_ratio", "obv_slope",
]


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RegimeResult:
    regime:             Regime
    confidence:         float
    signal_multiplier:  float
    sl_multiplier:      float
    size_multiplier:    float
    allowed_signals:    list
    strategy:           str
    description:        str
    adx:                float = 0.0
    atr_ratio:          float = 0.0
    rsi:                float = 50.0
    ema_slope:          float = 0.0
    supporting_factors: list  = field(default_factory=list)
    conflicting_factors:list  = field(default_factory=list)

    @property
    def is_trending(self) -> bool:
        return self.regime in (Regime.BULL_TREND, Regime.BEAR_TREND)

    @property
    def is_ranging(self) -> bool:
        return self.regime in (Regime.RANGING_LOW, Regime.RANGING_HIGH)

    def gate_signal(self, signal: str, confidence: float) -> tuple[str, float]:
        adj_conf = confidence * self.signal_multiplier
        if self.regime == Regime.BEAR_TREND and "BUY" in signal:
            return "HOLD", round(adj_conf * 0.5, 3)
        if self.regime == Regime.RANGING_HIGH and adj_conf < 0.60:
            return "HOLD", round(adj_conf, 3)
        return signal, round(adj_conf, 3)


# ── Rule-based detector ───────────────────────────────────────────────────────

class RuleBasedRegimeDetector:
    """No training needed. Works immediately from FeatureEngine output."""

    def detect(self, features: pd.Series) -> RegimeResult:
        def _f(key, default=0.0):
            v = features.get(key, default)
            try:
                v = float(v)
                return default if (v is None or np.isnan(v) or np.isinf(v)) else v
            except (TypeError, ValueError):
                return default

        adx       = _f("adx",          20)
        di_diff   = _f("di_diff",        0)
        atr_ratio = _f("atr_ratio",    1.0)
        rsi       = _f("rsi_14",       50)
        bb_width  = _f("bb_width",    0.04)
        ema9_pct  = _f("ema9_pct",      0)
        ema50_pct = _f("ema50_pct",     0)
        macd_h    = _f("macd_hist",     0)
        macd_chg  = _f("macd_hist_chg", 0)
        vol_ratio = _f("vol_ratio",   1.0)

        sup, con = [], []
        trend_score = 0
        vol_score   = 0

        # ── Trend signals ─────────────────────────────────────────────────────
        if adx > 30:
            trend_score += 3; sup.append(f"ADX {adx:.0f} — strong trend")
        elif adx > 25:
            trend_score += 2; sup.append(f"ADX {adx:.0f} — moderate trend")
        elif adx > 20:
            trend_score += 1
        else:
            trend_score -= 1; sup.append(f"ADX {adx:.0f} — ranging")

        if di_diff > 5:
            trend_score += 2; sup.append(f"+DI dominates ({di_diff:+.0f}) — bullish")
        elif di_diff < -5:
            trend_score -= 2; sup.append(f"-DI dominates ({di_diff:+.0f}) — bearish")

        if ema9_pct > 0 and ema50_pct > 0:
            trend_score += 2; sup.append("Above EMA9+EMA50 — bullish alignment")
        elif ema9_pct < 0 and ema50_pct < 0:
            trend_score -= 2; sup.append("Below EMA9+EMA50 — bearish alignment")
        else:
            con.append("Mixed EMA — no clean trend")

        if macd_h > 0 and macd_chg > 0:
            trend_score += 1; sup.append("MACD rising — momentum accelerating")
        elif macd_h < 0 and macd_chg < 0:
            trend_score -= 1; sup.append("MACD falling — downtrend strengthening")
        else:
            con.append("MACD not confirming direction")

        # ── Volatility signals ────────────────────────────────────────────────
        if atr_ratio > 2.5:
            vol_score += 3; sup.append(f"ATR {atr_ratio:.1f}× — extreme volatility")
        elif atr_ratio > 1.8:
            vol_score += 2; sup.append(f"ATR {atr_ratio:.1f}× — elevated volatility")
        elif atr_ratio > 1.3:
            vol_score += 1
        elif atr_ratio < 0.7:
            vol_score -= 1; sup.append(f"ATR {atr_ratio:.1f}× — compressed (breakout risk)")

        if bb_width > 0.08:
            vol_score += 1; sup.append(f"BB width {bb_width:.3f} — wide bands")
        elif bb_width < 0.02:
            vol_score -= 1; sup.append(f"BB width {bb_width:.3f} — squeeze forming")

        if vol_ratio > 2.0:
            vol_score += 1; sup.append(f"Volume {vol_ratio:.1f}× average — unusual")

        # ── Classification ────────────────────────────────────────────────────
        if adx > 20 and trend_score >= 3:
            regime   = Regime.BULL_TREND
            raw_conf = min(0.95, 0.55 + (trend_score - 3) * 0.08 + (adx - 25) * 0.005)
        elif adx > 20 and trend_score <= -3:
            regime   = Regime.BEAR_TREND
            raw_conf = min(0.95, 0.55 + (abs(trend_score) - 3) * 0.08 + (adx - 25) * 0.005)
        elif vol_score >= 2:
            regime   = Regime.RANGING_HIGH
            raw_conf = min(0.90, 0.55 + (vol_score - 2) * 0.1)
            con.append("High volatility but weak trend — choppy")
        else:
            regime   = Regime.RANGING_LOW
            raw_conf = min(0.85, 0.55 + max(0, 20 - adx) * 0.01)

        # RSI refinements
        if regime == Regime.BULL_TREND and rsi < 35:
            con.append(f"RSI {rsi:.0f} oversold in bull — pullback only")
            raw_conf *= 0.9
        if regime == Regime.BEAR_TREND and rsi > 65:
            con.append(f"RSI {rsi:.0f} overbought in bear — bounce risk")
            raw_conf *= 0.9
        if regime == Regime.RANGING_LOW and (rsi < 30 or rsi > 70):
            sup.append(f"RSI {rsi:.0f} extreme in ranging — high-prob reversal")
            raw_conf = min(raw_conf * 1.1, 0.90)

        p = REGIME_PARAMS[regime]
        return RegimeResult(
            regime=regime, confidence=round(raw_conf, 3),
            signal_multiplier=p["signal_multiplier"],
            sl_multiplier=p["sl_multiplier"],
            size_multiplier=p["size_multiplier"],
            allowed_signals=p["allowed_signals"],
            strategy=p["strategy"],
            description=p["description"],
            adx=adx, atr_ratio=atr_ratio, rsi=rsi, ema_slope=ema9_pct,
            supporting_factors=sup, conflicting_factors=con,
        )


# ── HMM detector ──────────────────────────────────────────────────────────────

class HMMRegimeDetector:
    def __init__(self, n_states: int = 4):
        self.n_states  = n_states
        self._model    = None
        self._scaler   = None
        self._trained  = False
        self._rule     = RuleBasedRegimeDetector()
        self._load()

    def _load(self):
        if HMM_PATH.exists():
            try:
                with open(HMM_PATH, "rb") as f:
                    saved = pickle.load(f)
                self._model   = saved.get("model")
                self._scaler  = saved.get("scaler")
                self._trained = True
                logger.info("HMM regime model loaded")
            except Exception as e:
                logger.warning(f"HMM load failed: {e}")

    def train(self, df_features: pd.DataFrame, symbol: str = "MULTI") -> dict:
        try:
            from hmmlearn import hmm
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return {"trained": False, "reason": "pip install hmmlearn"}

        avail = [c for c in REGIME_FEATURE_COLS if c in df_features.columns]
        if len(avail) < 4:
            return {"trained": False, "reason": f"Only {len(avail)} features"}

        X = df_features[avail].replace([np.inf, -np.inf], np.nan).fillna(0).values
        if len(X) < 100:
            return {"trained": False, "reason": "Need 100+ bars"}

        try:
            scaler   = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model    = hmm.GaussianHMM(
                n_components=self.n_states, covariance_type="diag",
                n_iter=200, random_state=42, verbose=False,
            )
            model.fit(X_scaled)
            self._model = model; self._scaler = scaler; self._trained = True
            with open(HMM_PATH, "wb") as f:
                pickle.dump({"model": model, "scaler": scaler}, f)
            logger.info(f"HMM trained: {len(X)} bars, {symbol}")
            return {"trained": True, "n_bars": len(X), "n_states": self.n_states}
        except Exception as e:
            return {"trained": False, "reason": str(e)}

    def detect(self, features: pd.Series) -> RegimeResult:
        rule_result = self._rule.detect(features)
        if not self._trained or self._model is None:
            return rule_result
        try:
            avail = [c for c in REGIME_FEATURE_COLS if c in features.index]
            X = np.array([[features.get(c, 0) for c in avail]], dtype=np.float32)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X_s = self._scaler.transform(X)
            probs = self._model.predict_proba(X_s)[0]
            hmm_conf = float(probs.max())
            # Blend: 60% rule-based + 40% HMM confidence
            rule_result.confidence = round(0.6 * rule_result.confidence + 0.4 * hmm_conf, 3)
            rule_result.supporting_factors.append(f"HMM confidence {hmm_conf:.0%}")
        except Exception:
            pass
        return rule_result


# ── Public API ────────────────────────────────────────────────────────────────

class RegimeDetector:
    """
    Drop-in replacement for src/models/regime_detector.py.
    HMM when trained, rule-based always available as fallback.
    """

    def __init__(self):
        self._det     = HMMRegimeDetector()
        self._history: list[RegimeResult] = []
        self._stats   = self._load_stats()

    def detect(self, features: pd.Series) -> RegimeResult:
        result = self._det.detect(features)
        self._history.append(result)
        if len(self._history) > 500:
            self._history = self._history[-500:]
        self._update_stats(result)
        return result

    def gate_signal(self, signal: str, confidence: float, features: pd.Series):
        """Returns (adj_signal, adj_confidence, RegimeResult)."""
        result = self.detect(features)
        adj_s, adj_c = result.gate_signal(signal, confidence)
        return adj_s, adj_c, result

    def get_regime_series(self, df: pd.DataFrame) -> pd.Series:
        results = []
        for i in range(len(df)):
            try:
                r = self._det._rule.detect(df.iloc[i])
                results.append(r.regime.value)
            except Exception:
                results.append(Regime.RANGING_LOW.value)
        return pd.Series(results, index=df.index, name="regime")

    def train(self, df_features: pd.DataFrame, symbol: str = "MULTI") -> dict:
        return self._det.train(df_features, symbol)

    def get_current_regime(self) -> Optional[RegimeResult]:
        return self._history[-1] if self._history else None

    def get_regime_distribution(self, n_bars: int = 50) -> dict:
        recent = self._history[-n_bars:]
        if not recent:
            return {}
        dist: dict[str, int] = {}
        for r in recent:
            dist[r.regime.value] = dist.get(r.regime.value, 0) + 1
        total = len(recent)
        return {k: round(v / total * 100, 1) for k, v in dist.items()}

    def get_stats(self) -> dict:
        return self._stats

    def _update_stats(self, result: RegimeResult):
        k = result.regime.value
        self._stats["counts"][k] = self._stats["counts"].get(k, 0) + 1
        self._stats["total"]         += 1
        self._stats["last_regime"]    = k
        self._stats["last_confidence"]= result.confidence
        if self._stats["total"] % 50 == 0:
            self._save_stats()

    def _load_stats(self) -> dict:
        try:
            if STATS_PATH.exists():
                return json.loads(STATS_PATH.read_text())
        except Exception:
            pass
        return {"counts": {}, "total": 0, "last_regime": None, "last_confidence": 0}

    def _save_stats(self):
        try:
            STATS_PATH.parent.mkdir(exist_ok=True)
            STATS_PATH.write_text(json.dumps(self._stats, indent=2))
        except Exception:
            pass


# ── Dashboard page ────────────────────────────────────────────────────────────

def render_regime_page():
    """
    Add to app.py sidebar + routing:
        "🔭 Regime Detector"
        ---
        elif page == "🔭 Regime Detector":
            from src.analysis.regime_detector import render_regime_page
            render_regime_page()
    """
    import streamlit as st
    import plotly.graph_objects as go

    st.header("🔭 Market Regime Detector")
    st.caption(
        "4-state HMM + rule-based hybrid. "
        "Gates every signal: BULL_TREND / BEAR_TREND / RANGING_LOW / RANGING_HIGH"
    )

    # Legend cards
    r1, r2, r3, r4 = st.columns(4)
    for col, name, bg, fg, desc in [
        (r1, "BULL_TREND",   "#003300", "#00ff88", "ADX>25, above EMA50. Full size. Trend follow."),
        (r2, "BEAR_TREND",   "#330000", "#ff4444", "ADX>25, below EMA50. Avoid longs. 70% size."),
        (r3, "RANGING_LOW",  "#1a1a2e", "#8888ff", "ADX<18, low vol. Mean reversion. 90% size."),
        (r4, "RANGING_HIGH", "#332200", "#ffaa00", "ADX<18, high vol. Choppy. Half size. WAIT."),
    ]:
        col.markdown(
            f'<div style="background:{bg};border:1px solid {fg};border-radius:6px;'
            f'padding:10px;text-align:center">'
            f'<div style="color:{fg};font-weight:700;font-size:11px">{name}</div>'
            f'<div style="color:#ccc;font-size:10px;margin-top:4px">{desc}</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    try:
        from src.data.manager import DataManager
        from src.data.models import Interval
        from src.features.feature_engine import FeatureEngine
    except ImportError as e:
        st.error(f"Import failed: {e}"); return

    WATCHLIST = ["NIFTY50", "BANKNIFTY", "GOLD", "CRUDEOIL", "BTC",
                 "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]

    selected = st.multiselect("Symbols", WATCHLIST, default=WATCHLIST[:5])
    days     = st.selectbox("Lookback days", [60, 90, 180, 365], index=1)

    if not selected:
        st.info("Select symbols."); return

    rd = RegimeDetector()
    dm = DataManager()
    fe = FeatureEngine()

    # Scan table
    st.subheader("Current Regime — Snapshot")
    rows = []
    prog = st.progress(0)
    for i, sym in enumerate(selected):
        prog.progress((i+1)/len(selected), text=f"Scanning {sym}...")
        try:
            df = dm.get_ohlcv(sym, Interval.D1, days_back=days)
            if df.empty or len(df) < 50: continue
            ft = fe.build(df, drop_na=False)
            if ft.empty: continue
            cur = rd.detect(ft.iloc[-1])
            rows.append({
                "Symbol":    sym,
                "Regime":    cur.regime.value,
                "Conf":      f"{cur.confidence:.0%}",
                "ADX":       f"{cur.adx:.0f}",
                "ATR×":      f"{cur.atr_ratio:.2f}",
                "RSI":       f"{cur.rsi:.0f}",
                "Sig Mult":  f"{cur.signal_multiplier:.0%}",
                "Size Mult": f"{cur.size_multiplier:.0%}",
                "SL Mult":   f"{cur.sl_multiplier:.1f}×",
                "Strategy":  cur.strategy,
            })
        except Exception as e:
            logger.warning(f"{sym}: {e}")
    prog.empty()

    if rows:
        df_r = pd.DataFrame(rows)
        _rc  = {"BULL_TREND":"color:#00ff88;font-weight:bold","BEAR_TREND":"color:#ff4444;font-weight:bold",
                "RANGING_LOW":"color:#8888ff","RANGING_HIGH":"color:#ffaa00"}
        st.dataframe(
            df_r.style.map(lambda v: _rc.get(v,""), subset=["Regime"]),
            use_container_width=True, hide_index=True,
        )

    # Deep dive
    st.divider()
    st.subheader("Deep Dive — Single Symbol")
    sel = st.selectbox("Symbol", selected)

    try:
        df  = dm.get_ohlcv(sel, Interval.D1, days_back=days)
        ft  = fe.build(df, drop_na=False)
        cur = rd.detect(ft.iloc[-1])

        _bg = {"BULL_TREND":("#003300","#00ff88"),"BEAR_TREND":("#330000","#ff4444"),
               "RANGING_LOW":("#1a1a2e","#8888ff"),"RANGING_HIGH":("#332200","#ffaa00")}
        bg_c, fg_c = _bg.get(cur.regime.value, ("#222","#fff"))
        st.markdown(
            f'<div style="background:{bg_c};border:1px solid {fg_c};border-radius:8px;'
            f'padding:14px 18px;margin-bottom:10px">'
            f'<div style="font-size:20px;font-weight:700;color:{fg_c}">{cur.regime.value}</div>'
            f'<div style="color:{fg_c};opacity:0.8;font-size:12px;margin-top:3px">{cur.description}</div>'
            f'</div>', unsafe_allow_html=True,
        )

        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Confidence",   f"{cur.confidence:.0%}")
        m2.metric("Signal Mult",  f"{cur.signal_multiplier:.0%}")
        m3.metric("Size Mult",    f"{cur.size_multiplier:.0%}")
        m4.metric("SL Mult",      f"{cur.sl_multiplier:.1f}×")
        m5.metric("Strategy",     cur.strategy)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Supporting:**")
            for t in cur.supporting_factors: st.success(t)
        with c2:
            st.markdown("**Conflicting:**")
            for t in cur.conflicting_factors: st.warning(t)

        # Gate simulator
        st.divider()
        st.subheader("Signal Gate Simulator")
        ts = st.selectbox("Signal", ["STRONG BUY","BUY","HOLD","SELL","STRONG SELL"])
        tc = st.slider("Confidence", 0.50, 0.95, 0.65, 0.05)
        adj_s, adj_c = cur.gate_signal(ts, tc)
        gc = "#00cc66" if adj_c >= tc*0.9 else "#ffaa00" if adj_c >= tc*0.6 else "#ff4444"
        st.markdown(
            f'<div style="background:rgba(0,0,0,0.2);border-radius:6px;padding:12px">'
            f'<b>{ts}</b> @ {tc:.0%} → '
            f'<span style="color:{gc};font-weight:700">{adj_s}</span> @ '
            f'<span style="color:{gc}">{adj_c:.0%}</span>'
            f' (regime gate ×{cur.signal_multiplier:.0%})</div>',
            unsafe_allow_html=True,
        )

        # Regime history chart
        st.divider()
        st.subheader("Regime History")
        step = max(1, len(ft) // 100)
        r_nums, r_labels, dates = [], [], []
        num_map = {Regime.BULL_TREND:2, Regime.RANGING_LOW:1, Regime.RANGING_HIGH:0, Regime.BEAR_TREND:-1}
        for j in range(0, len(ft), step):
            try:
                r = rd._det._rule.detect(ft.iloc[j])
                r_nums.append(num_map.get(r.regime, 0))
                r_labels.append(r.regime.value)
                dates.append(ft.index[j])
            except Exception:
                r_nums.append(0); r_labels.append("?"); dates.append(ft.index[j])

        prices = df["close"].iloc[::step].values[:len(dates)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=prices, name="Price", yaxis="y2",
            line=dict(color="#666", width=1), opacity=0.5,
        ))
        color_map = {2:"rgba(0,200,100,0.7)", 1:"rgba(100,100,255,0.7)",
                     0:"rgba(255,165,0,0.7)", -1:"rgba(255,50,50,0.7)"}
        label_map = {2:"Bull Trend", 1:"Ranging Low", 0:"Ranging High", -1:"Bear Trend"}
        for num in [2,1,0,-1]:
            xs = [dates[i] for i,v in enumerate(r_nums) if v==num]
            ys = [num]*len(xs)
            if xs:
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="markers",
                    marker=dict(color=color_map[num], size=9, symbol="square"),
                    name=label_map[num],
                ))
        fig.update_layout(
            height=320, template="plotly_dark",
            title=f"{sel} — Regime History ({days}d)",
            yaxis=dict(tickvals=[-1,0,1,2], ticktext=["Bear","Ranging↑","Ranging↓","Bull"], range=[-1.5,2.5]),
            yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Price"),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Deep dive error: {e}")

    # HMM training section
    st.divider()
    with st.expander("🔧 Train HMM (optional — needs: pip install hmmlearn)"):
        t_sym  = st.selectbox("Train symbol", WATCHLIST, key="hmm_sym")
        t_days = st.selectbox("Training bars", [500, 750, 1000], index=1, key="hmm_days")
        if st.button("Train HMM", type="secondary"):
            with st.spinner("Training..."):
                try:
                    df_t = dm.get_ohlcv(t_sym, Interval.D1, days_back=t_days)
                    ft_t = fe.build(df_t, drop_na=False)
                    res  = rd.train(ft_t, symbol=t_sym)
                    if res.get("trained"):
                        st.success(f"HMM trained on {res['n_bars']} bars. Model saved to models/regime_hmm.pkl")
                    else:
                        st.warning(f"Training failed: {res.get('reason')}")
                except Exception as e:
                    st.error(f"Error: {e}")