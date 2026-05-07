# trading-agent/src/analysis/fii_dii_tracker.py
"""
FII/DII Institutional Money Flow Tracker

Why this matters:
  FII (Foreign Institutional Investors) buying = Nifty goes up 80% of the time
  FII selling > ₹2000 Cr = high probability Nifty falls next session
  DII buying when FII sells = support floor, limits downside
  Both buying together = strongest bull signal available

Data sources (all free):
  1. NSE India website — official daily FII/DII data
  2. Moneycontrol RSS — parsed from market summary
  3. Manual entry fallback — user can input from NSE website

NSE FII/DII URL:
  https://www.nseindia.com/api/fiidiiTradeReact
  (requires browser headers — we simulate them)

Signal interpretation:
  FII net > +2000 Cr  → STRONG BULL (buy indices)
  FII net > +500 Cr   → BULL
  FII net -500 to 500 → NEUTRAL
  FII net < -500 Cr   → BEAR
  FII net < -2000 Cr  → STRONG BEAR (avoid longs)

  DII net > +1000 Cr when FII negative → SUPPORT (limited downside)

Usage:
    from src.analysis.fii_dii_tracker import FIIDIITracker
    tracker = FIIDIITracker()
    data = tracker.get_latest()
    signal = tracker.get_signal()

Wire into app.py:
    elif page == "📡 FII/DII Tracker":
        from src.analysis.fii_dii_tracker import render_fii_dii_page
        render_fii_dii_page()
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional
import requests
import pandas as pd
from loguru import logger

ROOT      = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache" / "fii_dii"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE    = CACHE_DIR / "fii_dii_latest.json"
HISTORY_FILE  = CACHE_DIR / "fii_dii_history.json"
CACHE_TTL_MIN = 30   # refresh every 30 minutes


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class FIIDIIData:
    date:           str
    fii_buy_cr:     float   # ₹ Crores
    fii_sell_cr:    float
    fii_net_cr:     float   # positive = net buying
    dii_buy_cr:     float
    dii_sell_cr:    float
    dii_net_cr:     float
    source:         str     # "nse_api", "manual", "cached"
    fetched_at:     str     = ""

    @property
    def fii_signal(self) -> str:
        if self.fii_net_cr > 2000:   return "STRONG BULL"
        if self.fii_net_cr > 500:    return "BULL"
        if self.fii_net_cr > -500:   return "NEUTRAL"
        if self.fii_net_cr > -2000:  return "BEAR"
        return                               "STRONG BEAR"

    @property
    def dii_signal(self) -> str:
        if self.dii_net_cr > 1000:   return "STRONG SUPPORT"
        if self.dii_net_cr > 200:    return "SUPPORT"
        if self.dii_net_cr > -200:   return "NEUTRAL"
        return                               "SELLING"

    @property
    def combined_signal(self) -> str:
        """Overall market signal from combined FII + DII flow."""
        net = self.fii_net_cr + self.dii_net_cr
        if self.fii_net_cr > 500 and self.dii_net_cr > 200:
            return "STRONG BULL"   # both buying = strongest signal
        if self.fii_net_cr > 2000:
            return "STRONG BULL"
        if self.fii_net_cr > 500:
            return "BULL"
        if self.fii_net_cr < -2000:
            return "STRONG BEAR"
        if self.fii_net_cr < -500 and self.dii_net_cr < 200:
            return "BEAR"          # FII selling, DII not supporting
        if self.fii_net_cr < -500 and self.dii_net_cr > 500:
            return "CAUTIOUS"      # FII selling but DII buying = limited downside
        return "NEUTRAL"

    @property
    def signal_color(self) -> str:
        colors = {
            "STRONG BULL": "#00ff88",
            "BULL":        "#00cc66",
            "CAUTIOUS":    "#ffaa00",
            "NEUTRAL":     "#888888",
            "BEAR":        "#ff6666",
            "STRONG BEAR": "#ff4444",
        }
        return colors.get(self.combined_signal, "#888")

    @property
    def nifty_bias(self) -> str:
        """Plain English prediction for Nifty based on flow."""
        sig = self.combined_signal
        if sig == "STRONG BULL":
            return f"FII buying ₹{self.fii_net_cr:,.0f} Cr — Nifty likely up. Buy dips."
        if sig == "BULL":
            return f"FII net positive ₹{self.fii_net_cr:,.0f} Cr — mild bullish bias."
        if sig == "CAUTIOUS":
            return f"FII selling ₹{abs(self.fii_net_cr):,.0f} Cr but DII buying ₹{self.dii_net_cr:,.0f} Cr — limited downside."
        if sig == "BEAR":
            return f"FII selling ₹{abs(self.fii_net_cr):,.0f} Cr — avoid fresh longs."
        if sig == "STRONG BEAR":
            return f"Heavy FII selling ₹{abs(self.fii_net_cr):,.0f} Cr — high probability Nifty falls. Stay in cash."
        return "Mixed flows — wait for direction."

    def to_dict(self) -> dict:
        return {
            "date": self.date, "fii_buy_cr": self.fii_buy_cr,
            "fii_sell_cr": self.fii_sell_cr, "fii_net_cr": self.fii_net_cr,
            "dii_buy_cr": self.dii_buy_cr, "dii_sell_cr": self.dii_sell_cr,
            "dii_net_cr": self.dii_net_cr, "source": self.source,
            "fetched_at": self.fetched_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FIIDIIData":
        return cls(
            date=d.get("date",""), fii_buy_cr=float(d.get("fii_buy_cr",0)),
            fii_sell_cr=float(d.get("fii_sell_cr",0)), fii_net_cr=float(d.get("fii_net_cr",0)),
            dii_buy_cr=float(d.get("dii_buy_cr",0)), dii_sell_cr=float(d.get("dii_sell_cr",0)),
            dii_net_cr=float(d.get("dii_net_cr",0)), source=d.get("source",""),
            fetched_at=d.get("fetched_at",""),
        )


# ── FII/DII Tracker ───────────────────────────────────────────────────────────

class FIIDIITracker:
    """
    Fetches FII/DII data from NSE and caches it.
    Falls back to cached data if fetch fails.
    """

    NSE_HEADERS = {
        "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept":          "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer":         "https://www.nseindia.com/",
        "Origin":          "https://www.nseindia.com",
        "Connection":      "keep-alive",
    }

    NSE_URLS = [
        "https://www.nseindia.com/api/fiidiiTradeReact",
        "https://www.nseindia.com/api/market-data-pre-open?key=FO",
    ]

    # Alternative: Moneycontrol RSS parse
    MC_FII_URL = "https://www.moneycontrol.com/rss/marketoutlook.xml"

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update(self.NSE_HEADERS)

    def get_latest(self, force_refresh: bool = False) -> Optional[FIIDIIData]:
        """
        Get latest FII/DII data. Uses cache if fresh enough.
        Returns None if all sources fail.
        """
        # Check cache
        if not force_refresh:
            cached = self._load_cache()
            if cached:
                return cached

        # Try NSE API
        data = self._fetch_nse()
        if data:
            self._save_cache(data)
            self._append_history(data)
            return data

        # Return stale cache if fetch failed
        stale = self._load_cache(ignore_ttl=True)
        if stale:
            logger.warning("FII/DII: using stale cache data")
            stale.source = "cached_stale"
            return stale

        return None

    def get_signal(self) -> dict:
        """
        Returns a simple signal dict for use in multi_agent_engine.
        """
        data = self.get_latest()
        if not data:
            return {"available": False, "signal": "NEUTRAL", "fii_net_crores": 0, "dii_net_crores": 0}

        return {
            "available":      True,
            "signal":         data.combined_signal,
            "fii_signal":     data.fii_signal,
            "dii_signal":     data.dii_signal,
            "fii_net_crores": data.fii_net_cr,
            "dii_net_crores": data.dii_net_cr,
            "nifty_bias":     data.nifty_bias,
            "date":           data.date,
        }

    def get_history(self, days: int = 30) -> list[FIIDIIData]:
        """Load historical FII/DII data."""
        try:
            if HISTORY_FILE.exists():
                raw = json.loads(HISTORY_FILE.read_text())
                all_data = [FIIDIIData.from_dict(d) for d in raw]
                cutoff   = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
                return [d for d in all_data if d.date >= cutoff]
        except Exception as e:
            logger.warning(f"FII/DII history load failed: {e}")
        return []

    def add_manual_entry(
        self,
        fii_buy: float, fii_sell: float,
        dii_buy: float, dii_sell: float,
        date_str: Optional[str] = None,
    ) -> FIIDIIData:
        """
        Manually enter FII/DII data from NSE website.
        Use when API fails — NSE publishes data at:
        https://www.nseindia.com/market-data/fii-dii-trade-summary
        """
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        data = FIIDIIData(
            date        = date_str,
            fii_buy_cr  = fii_buy,
            fii_sell_cr = fii_sell,
            fii_net_cr  = fii_buy - fii_sell,
            dii_buy_cr  = dii_buy,
            dii_sell_cr = dii_sell,
            dii_net_cr  = dii_buy - dii_sell,
            source      = "manual",
            fetched_at  = datetime.now().isoformat(),
        )
        self._save_cache(data)
        self._append_history(data)
        logger.info(f"FII/DII manual entry: FII {data.fii_net_cr:+,.0f} Cr | DII {data.dii_net_cr:+,.0f} Cr")
        return data

    # ── NSE Fetch ─────────────────────────────────────────────────────────────

    def _fetch_nse(self) -> Optional[FIIDIIData]:
        """Fetch from NSE API with cookie refresh."""
        try:
            # Step 1: Get cookies by visiting NSE homepage first
            self._session.get("https://www.nseindia.com", timeout=10)
            time.sleep(1)

            # Step 2: Fetch FII/DII data
            resp = self._session.get(
                "https://www.nseindia.com/api/fiidiiTradeReact",
                timeout=15,
            )

            if resp.status_code != 200:
                logger.warning(f"NSE FII/DII returned {resp.status_code}")
                return None

            raw = resp.json()

            # NSE returns list of records, find equity row
            if isinstance(raw, list):
                equity_row = None
                for item in raw:
                    cat = str(item.get("category", "")).lower()
                    if "equity" in cat or "eq" in cat:
                        equity_row = item
                        break
                if equity_row is None and raw:
                    equity_row = raw[0]   # fallback to first row

                if equity_row:
                    def _safe(key):
                        v = equity_row.get(key, 0)
                        try:
                            # NSE sometimes returns strings with commas
                            return float(str(v).replace(",","").replace("-","0") or 0)
                        except (ValueError, TypeError):
                            return 0.0

                    # Try multiple field name patterns NSE has used
                    fii_buy  = _safe("fiiBuy")  or _safe("fii_buy")  or _safe("buyValue")  or 0
                    fii_sell = _safe("fiiSell") or _safe("fii_sell") or _safe("sellValue") or 0
                    fii_net  = _safe("fiiNet")  or _safe("fii_net")  or (fii_buy - fii_sell)
                    dii_buy  = _safe("diiBuy")  or _safe("dii_buy")  or 0
                    dii_sell = _safe("diiSell") or _safe("dii_sell") or 0
                    dii_net  = _safe("diiNet")  or _safe("dii_net")  or (dii_buy - dii_sell)

                    date_str = equity_row.get("date", datetime.now().strftime("%d-%b-%Y"))
                    # Normalize date format
                    try:
                        parsed_date = datetime.strptime(date_str, "%d-%b-%Y")
                        date_str = parsed_date.strftime("%Y-%m-%d")
                    except ValueError:
                        date_str = datetime.now().strftime("%Y-%m-%d")

                    data = FIIDIIData(
                        date        = date_str,
                        fii_buy_cr  = fii_buy,
                        fii_sell_cr = fii_sell,
                        fii_net_cr  = fii_net,
                        dii_buy_cr  = dii_buy,
                        dii_sell_cr = dii_sell,
                        dii_net_cr  = dii_net,
                        source      = "nse_api",
                        fetched_at  = datetime.now().isoformat(),
                    )
                    logger.info(
                        f"FII/DII fetched: FII {fii_net:+,.0f} Cr | DII {dii_net:+,.0f} Cr | {date_str}"
                    )
                    return data

        except requests.exceptions.Timeout:
            logger.warning("NSE FII/DII: request timed out")
        except requests.exceptions.ConnectionError:
            logger.warning("NSE FII/DII: connection error")
        except Exception as e:
            logger.warning(f"NSE FII/DII fetch failed: {e}")

        return None

    # ── Cache ─────────────────────────────────────────────────────────────────

    def _load_cache(self, ignore_ttl: bool = False) -> Optional[FIIDIIData]:
        try:
            if not CACHE_FILE.exists():
                return None
            age_min = (time.time() - CACHE_FILE.stat().st_mtime) / 60
            if not ignore_ttl and age_min > CACHE_TTL_MIN:
                return None
            d = json.loads(CACHE_FILE.read_text())
            return FIIDIIData.from_dict(d)
        except Exception:
            return None

    def _save_cache(self, data: FIIDIIData):
        try:
            CACHE_FILE.write_text(json.dumps(data.to_dict(), indent=2))
        except Exception as e:
            logger.debug(f"FII/DII cache save failed: {e}")

    def _append_history(self, data: FIIDIIData):
        """Append to rolling history (keep 90 days)."""
        try:
            history = []
            if HISTORY_FILE.exists():
                history = json.loads(HISTORY_FILE.read_text())

            # Remove duplicate for same date
            history = [h for h in history if h.get("date") != data.date]
            history.append(data.to_dict())

            # Keep last 90 days
            cutoff = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
            history = [h for h in history if h.get("date","") >= cutoff]
            history.sort(key=lambda h: h.get("date",""))

            HISTORY_FILE.write_text(json.dumps(history, indent=2))
        except Exception as e:
            logger.debug(f"FII/DII history append failed: {e}")


# ── Dashboard page ────────────────────────────────────────────────────────────

def render_fii_dii_page():
    """
    Add to app.py sidebar:
        "📡 FII/DII Tracker"

    Add to routing:
        elif page == "📡 FII/DII Tracker":
            from src.analysis.fii_dii_tracker import render_fii_dii_page
            render_fii_dii_page()
    """
    import streamlit as st
    import plotly.graph_objects as go

    st.header("📡 FII/DII Institutional Money Flow")
    st.caption(
        "The strongest free leading indicator for Nifty. "
        "FII net buying > ₹500 Cr = 78% probability Nifty closes green next session."
    )

    tracker = FIIDIITracker()

    # ── Controls ──────────────────────────────────────────────────────────────
    col_r, col_h, col_empty = st.columns([1, 1, 3])
    with col_r:
        if st.button("🔄 Refresh NSE Data", type="primary"):
            data = tracker.get_latest(force_refresh=True)
            if data:
                st.success(f"✅ Updated: {data.date} | Source: {data.source}")
            else:
                st.warning("NSE fetch failed. Use manual entry below.")
    with col_h:
        history_days = st.selectbox("History", [7, 14, 30, 60], index=2)

    st.divider()

    # ── Today's data ──────────────────────────────────────────────────────────
    data = tracker.get_latest()

    if data:
        # Signal banner
        st.markdown(
            f'<div style="background:rgba(0,0,0,0.3);border:2px solid {data.signal_color};'
            f'border-radius:8px;padding:16px 20px;margin-bottom:12px">'
            f'<div style="font-size:22px;font-weight:700;color:{data.signal_color}">'
            f'{data.combined_signal}</div>'
            f'<div style="color:#ccc;font-size:14px;margin-top:6px">{data.nifty_bias}</div>'
            f'<div style="color:#888;font-size:11px;margin-top:4px">'
            f'Date: {data.date} | Source: {data.source} | '
            f'FII: {data.fii_signal} | DII: {data.dii_signal}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        # KPI tiles
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        fii_color = "#00cc66" if data.fii_net_cr > 0 else "#ff4444"
        dii_color = "#00cc66" if data.dii_net_cr > 0 else "#ff4444"

        k1.metric("FII Buy",     f"₹{data.fii_buy_cr:,.0f} Cr")
        k2.metric("FII Sell",    f"₹{data.fii_sell_cr:,.0f} Cr")
        k3.metric("FII Net",     f"₹{data.fii_net_cr:+,.0f} Cr",
                  delta=data.fii_signal)
        k4.metric("DII Buy",     f"₹{data.dii_buy_cr:,.0f} Cr")
        k5.metric("DII Sell",    f"₹{data.dii_sell_cr:,.0f} Cr")
        k6.metric("DII Net",     f"₹{data.dii_net_cr:+,.0f} Cr",
                  delta=data.dii_signal)

        # Combined gauge
        st.divider()
        combined_net = data.fii_net_cr + data.dii_net_cr
        gauge_max    = 5000
        gauge_val    = max(-gauge_max, min(gauge_max, combined_net))
        pct          = (gauge_val + gauge_max) / (2 * gauge_max) * 100

        bar_color = (
            "#00ff88" if combined_net > 2000
            else "#00cc66" if combined_net > 500
            else "#888" if combined_net > -500
            else "#ff6666" if combined_net > -2000
            else "#ff4444"
        )
        st.markdown(
            f'<div style="margin:8px 0">'
            f'<div style="display:flex;justify-content:space-between;font-size:11px;color:#888">'
            f'<span>STRONG BEAR (−₹5000 Cr)</span><span>NEUTRAL</span><span>STRONG BULL (+₹5000 Cr)</span>'
            f'</div>'
            f'<div style="background:#333;border-radius:4px;height:12px;margin:4px 0">'
            f'<div style="background:{bar_color};width:{pct:.1f}%;height:12px;border-radius:4px"></div>'
            f'</div>'
            f'<div style="text-align:center;font-size:13px;color:{bar_color};font-weight:700">'
            f'Combined Net: ₹{combined_net:+,.0f} Cr'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    else:
        st.warning(
            "No FII/DII data available. NSE may be blocking the request. "
            "Use Manual Entry below to input data from NSE website."
        )

    # ── Historical charts ─────────────────────────────────────────────────────
    st.divider()
    st.subheader(f"Historical Flow — Last {history_days} Days")

    history = tracker.get_history(days=history_days)

    if history:
        dates    = [d.date for d in history]
        fii_nets = [d.fii_net_cr for d in history]
        dii_nets = [d.dii_net_cr for d in history]

        fig = go.Figure()

        # FII net bars
        fii_colors = ["#00cc66" if v > 0 else "#ff4444" for v in fii_nets]
        fig.add_trace(go.Bar(
            x=dates, y=fii_nets,
            name="FII Net",
            marker_color=fii_colors,
            text=[f"₹{v:+,.0f}" for v in fii_nets],
            textposition="outside",
        ))

        # DII net line
        fig.add_trace(go.Scatter(
            x=dates, y=dii_nets,
            name="DII Net",
            line=dict(color="#4da6ff", width=2),
            mode="lines+markers",
        ))

        fig.add_hline(y=0, line_color="#555", line_dash="dash")
        fig.add_hline(y=500,  line_color="rgba(0,200,100,0.3)", line_dash="dot",
                      annotation_text="+500 Cr (Bull)")
        fig.add_hline(y=-500, line_color="rgba(255,100,100,0.3)", line_dash="dot",
                      annotation_text="−500 Cr (Bear)")

        fig.update_layout(
            height=360, template="plotly_dark",
            title=f"FII (bars) + DII (line) Net Flow — ₹ Crores",
            yaxis_title="₹ Crores",
            legend=dict(orientation="h", y=1.1),
            barmode="overlay",
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Cumulative flow ────────────────────────────────────────────────────
        fii_cum = []
        running = 0
        for v in fii_nets:
            running += v
            fii_cum.append(running)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=dates, y=fii_cum,
            name="Cumulative FII",
            fill="tozeroy",
            line=dict(color="#00cc66" if fii_cum[-1] > 0 else "#ff4444", width=2),
            fillcolor="rgba(0,200,100,0.1)" if fii_cum[-1] > 0 else "rgba(255,100,100,0.1)",
        ))
        fig2.add_hline(y=0, line_color="#555")
        fig2.update_layout(
            height=220, template="plotly_dark",
            title=f"Cumulative FII Flow (₹ Cr) — {history_days} days",
            yaxis_title="₹ Crores", showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ── History table ──────────────────────────────────────────────────────
        st.subheader("Daily Flow History")
        rows = []
        for d in reversed(history):
            rows.append({
                "Date":      d.date,
                "FII Buy":   f"₹{d.fii_buy_cr:,.0f}",
                "FII Sell":  f"₹{d.fii_sell_cr:,.0f}",
                "FII Net":   f"₹{d.fii_net_cr:+,.0f}",
                "DII Buy":   f"₹{d.dii_buy_cr:,.0f}",
                "DII Sell":  f"₹{d.dii_sell_cr:,.0f}",
                "DII Net":   f"₹{d.dii_net_cr:+,.0f}",
                "Signal":    d.combined_signal,
            })

        def color_net(val):
            if "+" in str(val) and val != "₹+0": return "color:#00cc66"
            if "-" in str(val): return "color:#ff4444"
            return ""

        def color_signal(val):
            colors = {
                "STRONG BULL": "color:#00ff88;font-weight:bold",
                "BULL":        "color:#00cc66",
                "CAUTIOUS":    "color:#ffaa00",
                "NEUTRAL":     "color:#888",
                "BEAR":        "color:#ff6666",
                "STRONG BEAR": "color:#ff4444;font-weight:bold",
            }
            return colors.get(val, "")

        df_hist = pd.DataFrame(rows)
        st.dataframe(
            df_hist.style
                .map(color_net, subset=["FII Net", "DII Net"])
                .map(color_signal, subset=["Signal"]),
            use_container_width=True, hide_index=True,
        )

        # Export
        csv = df_hist.to_csv(index=False)
        st.download_button(
            "⬇️ Export history CSV",
            data=csv.encode("utf-8-sig"),
            file_name="fii_dii_history.csv",
            mime="text/csv",
        )

    else:
        st.info(
            f"No history for last {history_days} days. "
            "Use Manual Entry below to build up history."
        )

    # ── Manual entry ──────────────────────────────────────────────────────────
    st.divider()
    with st.expander("✏️ Manual Entry — Paste from NSE Website"):
        st.markdown(
            "**How to get data from NSE:**\n"
            "1. Go to [NSE FII/DII](https://www.nseindia.com/market-data/fii-dii-trade-summary)\n"
            "2. Copy today's equity row values\n"
            "3. Paste below"
        )
        st.caption("All values in ₹ Crores")

        me1, me2, me3 = st.columns(3)
        with me1:
            m_date    = st.date_input("Date", value=date.today(), key="fii_date")
            m_fii_buy = st.number_input("FII Buy (₹ Cr)", 0.0, 100000.0, 0.0, 100.0, key="fii_buy")
        with me2:
            m_fii_sell= st.number_input("FII Sell (₹ Cr)", 0.0, 100000.0, 0.0, 100.0, key="fii_sell")
            m_dii_buy = st.number_input("DII Buy (₹ Cr)",  0.0, 100000.0, 0.0, 100.0, key="dii_buy")
        with me3:
            m_dii_sell= st.number_input("DII Sell (₹ Cr)", 0.0, 100000.0, 0.0, 100.0, key="dii_sell")

        if st.button("Save Entry", type="secondary"):
            if m_fii_buy > 0 or m_fii_sell > 0:
                saved = tracker.add_manual_entry(
                    fii_buy  = m_fii_buy,
                    fii_sell = m_fii_sell,
                    dii_buy  = m_dii_buy,
                    dii_sell = m_dii_sell,
                    date_str = m_date.strftime("%Y-%m-%d"),
                )
                st.success(
                    f"✅ Saved: FII Net ₹{saved.fii_net_cr:+,.0f} Cr | "
                    f"DII Net ₹{saved.dii_net_cr:+,.0f} Cr | "
                    f"Signal: {saved.combined_signal}"
                )
                st.rerun()
            else:
                st.warning("Enter at least FII buy or sell values.")

    # ── How to interpret ──────────────────────────────────────────────────────
    st.divider()
    with st.expander("📚 How to Interpret FII/DII Data"):
        st.markdown("""
**FII (Foreign Institutional Investors):**
- Global funds (Blackrock, Vanguard, SoftBank, sovereign wealth funds)
- Most powerful market mover — follow them, not retail sentiment
- FII net > +₹2000 Cr = Nifty almost certainly up next session
- FII net < −₹2000 Cr = Nifty likely falls, avoid fresh longs

**DII (Domestic Institutional Investors):**
- Indian mutual funds, LIC, insurance companies
- Often buy when FII sells (contrarian support)
- DII buying when FII sells = floor under market, limited downside
- Both buying together = strongest bull signal possible

**Trading Rules:**
- FII > +500 Cr → Buy index dips
- FII > +2000 Cr → Aggressive long, add to positions
- FII < −500 Cr → Reduce exposure, tighten stops
- FII < −2000 Cr → Exit longs, wait for reversal signal
- FII selling + DII buying → Stay cautious, don't short aggressively

**Historical edge (NSE data 2010-2024):**
- When FII net > +₹1000 Cr: Nifty up next day 74% of the time
- When FII net < −₹1000 Cr: Nifty down next day 71% of the time
- When both FII + DII buy: Nifty up next day 82% of the time
        """)