# trading-agent/src/analysis/options_oi.py
"""
NSE Options OI Analysis — PCR + Max Pain + OI Buildup

Why options data matters:
  - Options writers (market makers, institutions) are almost always right
  - Heavy call writing at a strike = resistance wall (price won't cross easily)
  - Heavy put writing at a strike = support floor (price won't fall below easily)
  - PCR > 1.3 = too many puts = contrarian bullish (put writers will defend)
  - PCR < 0.7 = too many calls = contrarian bearish (call writers will cap rally)
  - Max Pain = strike where maximum options expire worthless = magnet for price

Data source:
  NSE options chain API (free, public):
  https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY
  https://www.nseindia.com/api/option-chain-indices?symbol=BANKNIFTY

This is the same data displayed on NSE website — completely free.

Key signals:
  1. PCR (Put-Call Ratio) — sentiment gauge
  2. Max Pain — price magnet at expiry
  3. OI Buildup — where new positions are being added
  4. IV (Implied Volatility) — expected move size
  5. Change in OI — which strikes are being added/unwound

Usage:
    from src.analysis.options_oi import OptionsOIAnalyzer
    analyzer = OptionsOIAnalyzer()
    chain = analyzer.get_chain("NIFTY")
    signal = analyzer.get_signal("NIFTY")

Wire into app.py:
    elif page == "📊 Options OI":
        from src.analysis.options_oi import render_options_oi_page
        render_options_oi_page()
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Optional
import requests
import numpy as np
import pandas as pd
from loguru import logger

ROOT      = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache" / "options_oi"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_MIN = 5   # options data is time-sensitive — refresh every 5 minutes


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class StrikeData:
    strike:         float
    call_oi:        int
    call_oi_chg:    int      # change in OI from previous
    call_volume:    int
    call_iv:        float    # implied volatility %
    call_ltp:       float    # last traded price
    put_oi:         int
    put_oi_chg:     int
    put_volume:     int
    put_iv:         float
    put_ltp:        float
    is_atm:         bool = False


@dataclass
class OptionsChain:
    symbol:         str
    expiry:         str
    spot_price:     float
    atm_strike:     float
    strikes:        list[StrikeData] = field(default_factory=list)
    fetched_at:     str = ""

    @property
    def total_call_oi(self) -> int:
        return sum(s.call_oi for s in self.strikes)

    @property
    def total_put_oi(self) -> int:
        return sum(s.put_oi for s in self.strikes)

    @property
    def pcr(self) -> float:
        """Put-Call Ratio by OI. >1 = more puts = bearish sentiment."""
        if self.total_call_oi == 0:
            return 1.0
        return round(self.total_put_oi / self.total_call_oi, 3)

    @property
    def max_pain(self) -> float:
        """
        Strike at which maximum options expire worthless.
        Price tends to gravitate toward max pain near expiry.
        """
        if not self.strikes:
            return self.spot_price

        pain = {}
        for target_strike in [s.strike for s in self.strikes]:
            total_loss = 0
            for s in self.strikes:
                # Call holders lose if price < strike
                if target_strike < s.strike:
                    total_loss += s.call_oi * (s.strike - target_strike)
                # Put holders lose if price > strike
                if target_strike > s.strike:
                    total_loss += s.put_oi * (target_strike - s.strike)
            pain[target_strike] = total_loss

        return min(pain, key=pain.get) if pain else self.spot_price

    @property
    def max_pain_simple(self) -> float:
        """Simplified max pain calculation."""
        if not self.strikes:
            return self.spot_price
        best_strike = self.spot_price
        min_loss    = float("inf")
        for target in [s.strike for s in self.strikes]:
            loss = sum(
                max(0, (s.strike - target)) * s.call_oi +
                max(0, (target - s.strike)) * s.put_oi
                for s in self.strikes
            )
            if loss < min_loss:
                min_loss    = loss
                best_strike = target
        return best_strike

    @property
    def resistance_strikes(self) -> list[float]:
        """Strikes with highest call OI (resistance walls)."""
        sorted_by_call = sorted(
            [s for s in self.strikes if s.strike > self.spot_price],
            key=lambda s: s.call_oi, reverse=True,
        )
        return [s.strike for s in sorted_by_call[:3]]

    @property
    def support_strikes(self) -> list[float]:
        """Strikes with highest put OI (support floors)."""
        sorted_by_put = sorted(
            [s for s in self.strikes if s.strike < self.spot_price],
            key=lambda s: s.put_oi, reverse=True,
        )
        return [s.strike for s in sorted_by_put[:3]]

    @property
    def oi_buildup_signal(self) -> str:
        """
        Analyze where OI is being added.
        Call OI adding above spot = resistance building.
        Put OI adding above spot = bearish (shorts adding puts as hedge).
        """
        call_oi_above = sum(s.call_oi_chg for s in self.strikes
                            if s.strike > self.spot_price and s.call_oi_chg > 0)
        put_oi_above  = sum(s.put_oi_chg  for s in self.strikes
                            if s.strike > self.spot_price and s.put_oi_chg > 0)
        call_oi_below = sum(s.call_oi_chg for s in self.strikes
                            if s.strike < self.spot_price and s.call_oi_chg > 0)
        put_oi_below  = sum(s.put_oi_chg  for s in self.strikes
                            if s.strike < self.spot_price and s.put_oi_chg > 0)

        if put_oi_below > call_oi_above * 1.5:
            return "BULLISH"     # put writers defending below = bull
        if call_oi_above > put_oi_below * 1.5:
            return "BEARISH"     # call writers capping above = bear
        return "NEUTRAL"

    @property
    def expected_move(self) -> float:
        """Expected move based on ATM straddle price."""
        atm = next((s for s in self.strikes if s.is_atm), None)
        if atm:
            return atm.call_ltp + atm.put_ltp
        return 0.0


@dataclass
class OptionsSignal:
    symbol:          str
    spot:            float
    pcr:             float
    max_pain:        float
    max_pain_gap_pct:float   # % gap between spot and max pain
    pcr_signal:      str     # BULLISH / BEARISH / NEUTRAL
    oi_buildup:      str
    resistance:      list[float]
    support:         list[float]
    expected_move:   float
    verdict:         str     # overall verdict
    confidence:      float
    reasoning:       list[str]


# ── Options Analyzer ──────────────────────────────────────────────────────────

class OptionsOIAnalyzer:
    """
    Fetches and analyzes NSE options chain data.
    """

    NSE_HEADERS = {
        "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept":          "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer":         "https://www.nseindia.com/",
        "Connection":      "keep-alive",
    }

    SYMBOL_MAP = {
        "NIFTY":     "NIFTY",
        "NIFTY50":   "NIFTY",
        "BANKNIFTY": "BANKNIFTY",
        "FINNIFTY":  "FINNIFTY",
        "MIDCPNIFTY":"MIDCPNIFTY",
    }

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update(self.NSE_HEADERS)
        self._cookie_refreshed = False

    def get_chain(
        self,
        symbol:       str,
        n_strikes:    int = 10,
        expiry_index: int = 0,
        force_refresh:bool = False,
    ) -> Optional[OptionsChain]:
        """
        Fetch options chain from NSE.
        n_strikes: number of strikes each side of ATM
        expiry_index: 0=nearest, 1=next, 2=monthly
        """
        nse_sym  = self.SYMBOL_MAP.get(symbol.upper(), symbol.upper())
        cache_key = CACHE_DIR / f"chain_{nse_sym}_{expiry_index}.json"

        # Check cache
        if not force_refresh and cache_key.exists():
            age_min = (time.time() - cache_key.stat().st_mtime) / 60
            if age_min < CACHE_TTL_MIN:
                try:
                    return self._parse_cached(cache_key, nse_sym, n_strikes, expiry_index)
                except Exception:
                    pass

        # Fetch from NSE
        raw = self._fetch_nse(nse_sym)
        if raw:
            try:
                cache_key.write_text(json.dumps(raw))
            except Exception:
                pass
            return self._parse_chain(raw, nse_sym, n_strikes, expiry_index)

        # Try parsing stale cache
        if cache_key.exists():
            try:
                return self._parse_cached(cache_key, nse_sym, n_strikes, expiry_index)
            except Exception:
                pass

        return None

    def get_signal(self, symbol: str) -> Optional[OptionsSignal]:
        """
        High-level signal from options chain analysis.
        Returns None if data unavailable.
        """
        chain = self.get_chain(symbol)
        if not chain:
            return None
        return self._compute_signal(chain)

    # ── NSE fetch ─────────────────────────────────────────────────────────────

    def _fetch_nse(self, nse_sym: str) -> Optional[dict]:
        try:
            # Refresh cookies first
            if not self._cookie_refreshed:
                self._session.get("https://www.nseindia.com", timeout=10)
                time.sleep(0.5)
                self._cookie_refreshed = True

            url  = f"https://www.nseindia.com/api/option-chain-indices?symbol={nse_sym}"
            resp = self._session.get(url, timeout=15)

            if resp.status_code == 200:
                data = resp.json()
                if data.get("records"):
                    logger.info(f"Options chain fetched: {nse_sym}")
                    return data
            elif resp.status_code == 401:
                # Cookie expired — refresh
                self._cookie_refreshed = False
                self._session.get("https://www.nseindia.com", timeout=10)
                time.sleep(1)
                resp = self._session.get(url, timeout=15)
                if resp.status_code == 200:
                    return resp.json()

            logger.warning(f"NSE options chain returned {resp.status_code}")

        except requests.exceptions.Timeout:
            logger.warning(f"Options chain timeout for {nse_sym}")
        except Exception as e:
            logger.warning(f"Options chain fetch failed: {e}")

        return None

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse_cached(self, path: Path, symbol: str, n_strikes: int, expiry_index: int) -> OptionsChain:
        raw = json.loads(path.read_text())
        return self._parse_chain(raw, symbol, n_strikes, expiry_index)

    def _parse_chain(
        self, raw: dict, symbol: str, n_strikes: int, expiry_index: int
    ) -> Optional[OptionsChain]:
        try:
            records = raw.get("records", {})
            data    = records.get("data", [])
            if not data:
                return None

            # Spot price
            spot = float(records.get("underlyingValue", 0))
            if spot <= 0:
                # Try to extract from data
                for item in data[:5]:
                    ce = item.get("CE", {})
                    if ce.get("underlyingValue"):
                        spot = float(ce["underlyingValue"])
                        break

            # Expiry dates
            expiries = records.get("expiryDates", [])
            if not expiries:
                expiries = list({item.get("expiryDate","") for item in data if item.get("expiryDate")})
                expiries = sorted([e for e in expiries if e])

            expiry = expiries[min(expiry_index, len(expiries)-1)] if expiries else ""

            # Filter data for selected expiry
            expiry_data = [item for item in data if item.get("expiryDate") == expiry]
            if not expiry_data:
                expiry_data = data   # fallback: use all

            # ATM strike
            strikes_in_data = sorted({float(item.get("strikePrice", 0)) for item in expiry_data if item.get("strikePrice")})
            if not strikes_in_data:
                return None

            atm = min(strikes_in_data, key=lambda s: abs(s - spot))

            # Select n_strikes each side
            atm_idx = strikes_in_data.index(atm)
            lo      = max(0, atm_idx - n_strikes)
            hi      = min(len(strikes_in_data), atm_idx + n_strikes + 1)
            selected_strikes = set(strikes_in_data[lo:hi])

            # Build strike data
            strike_map = {}
            for item in expiry_data:
                sp = float(item.get("strikePrice", 0))
                if sp not in selected_strikes:
                    continue

                def _i(d, k):
                    v = d.get(k, 0)
                    try: return int(float(v))
                    except (TypeError, ValueError): return 0

                def _f(d, k):
                    v = d.get(k, 0)
                    try: return float(v)
                    except (TypeError, ValueError): return 0.0

                ce = item.get("CE", {})
                pe = item.get("PE", {})

                strike_map[sp] = StrikeData(
                    strike       = sp,
                    call_oi      = _i(ce, "openInterest"),
                    call_oi_chg  = _i(ce, "changeinOpenInterest"),
                    call_volume  = _i(ce, "totalTradedVolume"),
                    call_iv      = _f(ce, "impliedVolatility"),
                    call_ltp     = _f(ce, "lastPrice"),
                    put_oi       = _i(pe, "openInterest"),
                    put_oi_chg   = _i(pe, "changeinOpenInterest"),
                    put_volume   = _i(pe, "totalTradedVolume"),
                    put_iv       = _f(pe, "impliedVolatility"),
                    put_ltp      = _f(pe, "lastPrice"),
                    is_atm       = (sp == atm),
                )

            chain = OptionsChain(
                symbol      = symbol,
                expiry      = expiry,
                spot_price  = spot,
                atm_strike  = atm,
                strikes     = sorted(strike_map.values(), key=lambda s: s.strike),
                fetched_at  = datetime.now().isoformat(),
            )
            return chain

        except Exception as e:
            logger.error(f"Chain parse failed: {e}")
            return None

    # ── Signal computation ────────────────────────────────────────────────────

    def _compute_signal(self, chain: OptionsChain) -> OptionsSignal:
        pcr       = chain.pcr
        max_pain  = chain.max_pain_simple
        spot      = chain.spot_price
        gap_pct   = (max_pain - spot) / spot * 100 if spot > 0 else 0
        oi_signal = chain.oi_buildup_signal
        reasoning = []
        score     = 0.0

        # PCR signal
        if pcr > 1.5:
            pcr_signal = "STRONG BULLISH"
            score += 0.30
            reasoning.append(f"PCR {pcr:.2f} — very high put/call ratio. Extreme fear = contrarian buy.")
        elif pcr > 1.2:
            pcr_signal = "BULLISH"
            score += 0.15
            reasoning.append(f"PCR {pcr:.2f} — elevated puts. Put writers expect support.")
        elif pcr < 0.6:
            pcr_signal = "STRONG BEARISH"
            score -= 0.30
            reasoning.append(f"PCR {pcr:.2f} — too many calls. Call writers capping the rally.")
        elif pcr < 0.8:
            pcr_signal = "BEARISH"
            score -= 0.15
            reasoning.append(f"PCR {pcr:.2f} — bearish sentiment building.")
        else:
            pcr_signal = "NEUTRAL"
            reasoning.append(f"PCR {pcr:.2f} — balanced put/call ratio.")

        # Max pain signal
        if gap_pct > 1.0:
            score += 0.15
            reasoning.append(f"Max pain ₹{max_pain:,.0f} is {gap_pct:.1f}% above spot — price may drift up toward expiry.")
        elif gap_pct < -1.0:
            score -= 0.15
            reasoning.append(f"Max pain ₹{max_pain:,.0f} is {abs(gap_pct):.1f}% below spot — price may drift down toward expiry.")
        else:
            reasoning.append(f"Max pain ₹{max_pain:,.0f} close to spot — neutral expiry pressure.")

        # OI buildup
        if oi_signal == "BULLISH":
            score += 0.15
            reasoning.append("OI buildup: put writers adding below spot — support building.")
        elif oi_signal == "BEARISH":
            score -= 0.15
            reasoning.append("OI buildup: call writers adding above spot — resistance strengthening.")

        # Key levels
        resistance = chain.resistance_strikes
        support    = chain.support_strikes
        if resistance:
            reasoning.append(f"Key resistance: ₹{resistance[0]:,.0f} (highest call OI)")
        if support:
            reasoning.append(f"Key support: ₹{support[0]:,.0f} (highest put OI)")

        # Expected move
        exp_move = chain.expected_move
        if exp_move > 0:
            reasoning.append(f"ATM straddle price: ₹{exp_move:,.0f} — expected move ±₹{exp_move:,.0f} by expiry")

        # Verdict
        if score >= 0.35:
            verdict    = "BULLISH"
            confidence = min(0.82, 0.55 + score * 0.4)
        elif score >= 0.15:
            verdict    = "MILDLY BULLISH"
            confidence = 0.60
        elif score <= -0.35:
            verdict    = "BEARISH"
            confidence = min(0.82, 0.55 + abs(score) * 0.4)
        elif score <= -0.15:
            verdict    = "MILDLY BEARISH"
            confidence = 0.60
        else:
            verdict    = "NEUTRAL"
            confidence = 0.50

        return OptionsSignal(
            symbol           = chain.symbol,
            spot             = spot,
            pcr              = pcr,
            max_pain         = max_pain,
            max_pain_gap_pct = round(gap_pct, 2),
            pcr_signal       = pcr_signal,
            oi_buildup       = oi_signal,
            resistance       = resistance,
            support          = support,
            expected_move    = exp_move,
            verdict          = verdict,
            confidence       = round(confidence, 3),
            reasoning        = reasoning,
        )


# ── Dashboard page ────────────────────────────────────────────────────────────

def render_options_oi_page():
    """
    Add to app.py sidebar:
        "📊 Options OI"

    Add to routing:
        elif page == "📊 Options OI":
            from src.analysis.options_oi import render_options_oi_page
            render_options_oi_page()
    """
    import streamlit as st
    import plotly.graph_objects as go

    st.header("📊 Options OI Analysis")
    st.caption(
        "NSE options chain: PCR + Max Pain + OI buildup. "
        "Best edge for Bank Nifty weekly expiry trades. Data: NSE (free)."
    )

    analyzer = OptionsOIAnalyzer()

    # ── Controls ──────────────────────────────────────────────────────────────
    col_sym, col_exp, col_n, col_btn = st.columns([2, 1, 1, 1])
    with col_sym:
        symbol = st.selectbox("Index", ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"])
    with col_exp:
        expiry_idx = st.selectbox("Expiry", ["Weekly (nearest)", "Next week", "Monthly"],
                                  index=0)
        exp_map = {"Weekly (nearest)": 0, "Next week": 1, "Monthly": 2}
        exp_index = exp_map[expiry_idx]
    with col_n:
        n_strikes = st.selectbox("Strikes each side", [5, 8, 10, 15, 20], index=2)
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        refresh = st.button("🔄 Fetch Chain", type="primary")

    st.divider()

    # ── Fetch data ────────────────────────────────────────────────────────────
    with st.spinner(f"Fetching {symbol} options chain from NSE..."):
        chain = analyzer.get_chain(symbol, n_strikes=n_strikes,
                                   expiry_index=exp_index,
                                   force_refresh=refresh)

    if not chain:
        st.error(
            "NSE options chain unavailable. Possible reasons:\n"
            "- Market closed (NSE API only works during market hours)\n"
            "- NSE rate limiting — wait 30s and retry\n"
            "- Network issue"
        )
        st.info("💡 Options data works best during market hours (9:15 AM - 3:30 PM IST)")
        return

    # ── Signal banner ─────────────────────────────────────────────────────────
    signal = analyzer._compute_signal(chain)
    signal_colors = {
        "BULLISH":       ("#003300", "#00ff88"),
        "MILDLY BULLISH":("#002200", "#00cc66"),
        "NEUTRAL":       ("#1a1a2e", "#8888ff"),
        "MILDLY BEARISH":("#220000", "#ff6666"),
        "BEARISH":       ("#330000", "#ff4444"),
    }
    bg_c, fg_c = signal_colors.get(signal.verdict, ("#222","#fff"))

    st.markdown(
        f'<div style="background:{bg_c};border:1px solid {fg_c};border-radius:8px;'
        f'padding:14px 18px;margin-bottom:12px">'
        f'<div style="font-size:20px;font-weight:700;color:{fg_c}">'
        f'{symbol} Options Signal: {signal.verdict}</div>'
        f'<div style="color:{fg_c};opacity:0.85;font-size:12px;margin-top:4px">'
        f'Confidence: {signal.confidence:.0%} | '
        f'Expiry: {chain.expiry} | '
        f'Spot: ₹{chain.spot_price:,.2f} | '
        f'ATM: ₹{chain.atm_strike:,.0f}'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # ── KPI tiles ─────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    pcr_color  = "#00cc66" if chain.pcr > 1.1 else "#ff4444" if chain.pcr < 0.8 else "#888"
    pain_color = "#00cc66" if signal.max_pain_gap_pct > 0 else "#ff4444"

    k1.metric("PCR",           f"{chain.pcr:.3f}",    signal.pcr_signal)
    k2.metric("Max Pain",      f"₹{signal.max_pain:,.0f}",
              f"{signal.max_pain_gap_pct:+.1f}% from spot")
    k3.metric("Total Call OI", f"{chain.total_call_oi:,}")
    k4.metric("Total Put OI",  f"{chain.total_put_oi:,}")
    k5.metric("Expected Move", f"±₹{signal.expected_move:,.0f}" if signal.expected_move else "—")
    k6.metric("OI Buildup",    signal.oi_buildup)

    # ── Key levels ────────────────────────────────────────────────────────────
    lv1, lv2 = st.columns(2)
    with lv1:
        st.markdown("**🔴 Resistance (Call OI walls):**")
        for r in signal.resistance:
            diff = (r - chain.spot_price) / chain.spot_price * 100
            st.markdown(
                f'<span style="color:#ff6666">₹{r:,.0f}</span> '
                f'<span style="color:#888">({diff:+.1f}% from spot)</span>',
                unsafe_allow_html=True,
            )
    with lv2:
        st.markdown("**🟢 Support (Put OI floors):**")
        for s in signal.support:
            diff = (s - chain.spot_price) / chain.spot_price * 100
            st.markdown(
                f'<span style="color:#00cc66">₹{s:,.0f}</span> '
                f'<span style="color:#888">({diff:+.1f}% from spot)</span>',
                unsafe_allow_html=True,
            )

    # ── Reasoning ─────────────────────────────────────────────────────────────
    for r in signal.reasoning:
        if "support" in r.lower() or "bullish" in r.lower() or "buy" in r.lower():
            st.success(r)
        elif "resist" in r.lower() or "bearish" in r.lower() or "cap" in r.lower():
            st.error(r)
        else:
            st.info(r)

    st.divider()

    # ── OI chart ──────────────────────────────────────────────────────────────
    st.subheader("Open Interest — Call vs Put by Strike")

    strikes    = [s.strike for s in chain.strikes]
    call_ois   = [s.call_oi / 1000 for s in chain.strikes]   # convert to thousands
    put_ois    = [s.put_oi  / 1000 for s in chain.strikes]

    fig_oi = go.Figure()
    fig_oi.add_trace(go.Bar(
        x=strikes, y=call_ois,
        name="Call OI (thousands)",
        marker_color="#ff6666",
        opacity=0.8,
    ))
    fig_oi.add_trace(go.Bar(
        x=strikes, y=[-v for v in put_ois],   # negative so puts go down
        name="Put OI (thousands)",
        marker_color="#00cc66",
        opacity=0.8,
    ))

    # Mark spot, ATM, max pain
    fig_oi.add_vline(x=chain.spot_price, line_color="#ffffff", line_dash="dash",
                     annotation_text=f"Spot ₹{chain.spot_price:,.0f}")
    fig_oi.add_vline(x=signal.max_pain, line_color="#ffaa00", line_dash="dot",
                     annotation_text=f"Max Pain ₹{signal.max_pain:,.0f}")

    fig_oi.update_layout(
        height=380, template="plotly_dark",
        title=f"{symbol} Open Interest — Calls (red up) | Puts (green down)",
        xaxis_title="Strike Price",
        yaxis_title="OI (thousands)",
        barmode="overlay",
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig_oi, use_container_width=True)

    # ── OI Change chart ────────────────────────────────────────────────────────
    st.subheader("Change in OI — Where New Positions Are Being Added")

    call_chg = [s.call_oi_chg / 1000 for s in chain.strikes]
    put_chg  = [s.put_oi_chg  / 1000 for s in chain.strikes]

    fig_chg = go.Figure()
    call_colors_chg = ["#ff6666" if v >= 0 else "#ffaaaa" for v in call_chg]
    put_colors_chg  = ["#00cc66" if v >= 0 else "#aaffaa" for v in put_chg]

    fig_chg.add_trace(go.Bar(
        x=strikes, y=call_chg,
        name="Call OI Change",
        marker_color=call_colors_chg,
        opacity=0.7,
    ))
    fig_chg.add_trace(go.Bar(
        x=strikes, y=[-v for v in put_chg],
        name="Put OI Change",
        marker_color=put_colors_chg,
        opacity=0.7,
    ))
    fig_chg.add_vline(x=chain.spot_price, line_color="#fff", line_dash="dash")
    fig_chg.add_hline(y=0, line_color="#555")
    fig_chg.update_layout(
        height=280, template="plotly_dark",
        title="OI Change (fresh positions being added)",
        xaxis_title="Strike Price",
        yaxis_title="Change in OI (thousands)",
        barmode="overlay",
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig_chg, use_container_width=True)

    # ── IV smile ──────────────────────────────────────────────────────────────
    st.subheader("Implied Volatility Smile")
    call_ivs = [s.call_iv for s in chain.strikes]
    put_ivs  = [s.put_iv  for s in chain.strikes]

    if any(v > 0 for v in call_ivs):
        fig_iv = go.Figure()
        fig_iv.add_trace(go.Scatter(
            x=strikes, y=call_ivs, name="Call IV %",
            line=dict(color="#ff6666", width=2), mode="lines+markers",
        ))
        fig_iv.add_trace(go.Scatter(
            x=strikes, y=put_ivs, name="Put IV %",
            line=dict(color="#00cc66", width=2), mode="lines+markers",
        ))
        fig_iv.add_vline(x=chain.spot_price, line_color="#fff", line_dash="dash")
        fig_iv.update_layout(
            height=240, template="plotly_dark",
            title="IV Smile — High put IV = fear, High call IV = rally expectation",
            xaxis_title="Strike", yaxis_title="IV %",
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_iv, use_container_width=True)

    # ── Full chain table ───────────────────────────────────────────────────────
    st.subheader("Full Options Chain")

    rows = []
    for s in chain.strikes:
        is_atm = s.strike == chain.atm_strike
        rows.append({
            "Call OI":     f"{s.call_oi:,}",
            "Call Chg":    f"{s.call_oi_chg:+,}",
            "Call Vol":    f"{s.call_volume:,}",
            "Call IV%":    f"{s.call_iv:.1f}" if s.call_iv else "—",
            "Call LTP":    f"₹{s.call_ltp:.2f}" if s.call_ltp else "—",
            "Strike":      f"{'▶ ' if is_atm else ''}{s.strike:,.0f}{'◀ ATM' if is_atm else ''}",
            "Put LTP":     f"₹{s.put_ltp:.2f}" if s.put_ltp else "—",
            "Put IV%":     f"{s.put_iv:.1f}" if s.put_iv else "—",
            "Put Vol":     f"{s.put_volume:,}",
            "Put Chg":     f"{s.put_oi_chg:+,}",
            "Put OI":      f"{s.put_oi:,}",
            "_atm":        is_atm,
        })

    df_chain = pd.DataFrame(rows)
    display  = [c for c in df_chain.columns if c != "_atm"]

    def highlight_atm(row):
        if df_chain.loc[row.name, "_atm"]:
            return ["background-color:#1a1a00;font-weight:bold;color:#ffff00"] * len(row)
        return [""] * len(row)

    def color_oi_chg(val):
        if "+" in str(val) and val != "+0": return "color:#00cc66"
        if "-" in str(val): return "color:#ff4444"
        return ""

    st.dataframe(
        df_chain[display].style
            .apply(highlight_atm, axis=1)
            .map(color_oi_chg, subset=["Call Chg", "Put Chg"]),
        use_container_width=True, hide_index=True, height=420,
    )

    # ── PCR interpretation guide ───────────────────────────────────────────────
    st.divider()
    with st.expander("📚 PCR + Max Pain Interpretation Guide"):
        st.markdown("""
**PCR (Put-Call Ratio):**
- **PCR > 1.5** → Extreme fear. Put writers heavily defending. **Contrarian STRONG BUY.**
- **PCR 1.2-1.5** → Elevated fear. Bullish bias. Buy dips.
- **PCR 0.8-1.2** → Balanced. Neutral.
- **PCR 0.6-0.8** → Greed. Call writers capping rally. Bearish bias.
- **PCR < 0.6** → Extreme greed. **Contrarian STRONG SELL.** Rally likely to stall.

**Max Pain:**
- The strike where maximum options (both calls AND puts) expire worthless
- Institutions lose least money when price settles at max pain
- Market makers hedge positions to push price toward max pain near expiry
- Most powerful 1-2 days before expiry (Thursday for weekly, last Thursday of month)
- **If spot > max pain**: expect selling pressure pushing it down
- **If spot < max pain**: expect buying pressure pushing it up

**OI Buildup Rules:**
- **Fresh call OI above spot** → Resistance being built. Price may struggle to cross.
- **Fresh put OI below spot** → Support being built. Price may not fall below.
- **Put unwinding** → Bulls exiting. Bearish signal.
- **Call unwinding** → Bears exiting. Bullish signal.

**Bank Nifty Weekly Expiry Edge:**
- Thursday is Bank Nifty expiry day — OI signals are most powerful
- ATM straddle price on Thursday morning = expected move for the day
- If straddle is cheap (< ₹100) = sideways expected, sell straddle
- If straddle is expensive (> ₹300) = big move expected, buy breakout
        """)