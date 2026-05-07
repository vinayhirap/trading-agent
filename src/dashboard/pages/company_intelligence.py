# trading-agent/src/dashboard/pages/company_intelligence.py
"""
Company Intelligence — Bloomberg DES + FA + ANR + EEG + EQRV Equivalent

Bloomberg function map:
  DES  → Company Passport tab      (sector, cap, ratios, description)
  FA   → Financial Statements tab  (P&L, balance sheet, cash flow, 5yr)
  ANR  → Analyst Consensus tab     (buy/hold/sell, target price, revisions)
  EEG  → Earnings History tab      (EPS actual vs estimate, surprise %)
  EQRV → Peer Comparison tab       (valuation comps, sector peers)

Data sources:
  - Alpha Vantage (alphavantage_adapter.py)  — fundamentals, earnings, news
  - yfinance                                 — live price, recommendations, holders
  - Anthropic API                            — AI company brief

Usage in app.py:
    elif page == "🏢 Company Intelligence":
        from src.dashboard.pages.company_intelligence import render_company_intelligence
        render_company_intelligence()
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

# ── Symbol universe ────────────────────────────────────────────────────────────
NSE_UNIVERSE = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "SBIN", "WIPRO", "AXISBANK", "KOTAKBANK", "LT",
    "BAJFINANCE", "MARUTI", "SUNPHARMA", "BHARTIARTL",
    "TATASTEEL", "JSWSTEEL", "ONGC", "NTPC", "ASIANPAINT",
    "DRREDDY", "BAJAJFINSV", "ULTRACEMCO", "COALINDIA",
    "POWERGRID", "HINDALCO",
]

SECTOR_PEERS = {
    "Banking":    ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK"],
    "IT":         ["TCS", "INFY", "WIPRO", "TECHM", "HCLTECH"],
    "Energy":     ["RELIANCE", "ONGC", "BPCL", "IOC", "NTPC"],
    "Pharma":     ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "AUROPHARMA"],
    "Metals":     ["TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "COALINDIA"],
    "Auto":       ["MARUTI", "M&M", "TATAMOTORS", "BAJAJ-AUTO", "HEROMOTOCO"],
    "FMCG":       ["HINDUNILVR", "ITC", "NESTLEIND", "DABUR", "BRITANNIA"],
    "Infra":      ["LT", "ULTRACEMCO", "POWERGRID", "ADANIPORTS", "SIEMENS"],
    "Finance":    ["BAJFINANCE", "BAJAJFINSV", "HDFC", "CHOLAFIN", "MUTHOOTFIN"],
    "Telecom":    ["BHARTIARTL", "IDEA", "TATACOMM"],
    "Paint":      ["ASIANPAINT", "BERGERPAINTS", "KANSAINER", "INDIAMART"],
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _fmt_cr(val) -> str:
    """Format large number as ₹ Crores."""
    if val is None:
        return "—"
    try:
        v = float(val)
        if v >= 1e11:
            return f"₹{v/1e7:,.0f} Cr"
        if v >= 1e7:
            return f"₹{v/1e7:,.1f} Cr"
        return f"₹{v:,.0f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_pct(val, multiply=False) -> str:
    if val is None:
        return "—"
    try:
        v = float(val)
        if multiply:
            v *= 100
        return f"{v:.1f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_ratio(val, decimals=2) -> str:
    if val is None:
        return "—"
    try:
        return f"{float(val):.{decimals}f}×"
    except (TypeError, ValueError):
        return "—"


def _color_val(val, good_positive=True) -> str:
    """Return green/red color string based on value sign."""
    try:
        v = float(val)
        if v > 0:
            return "color:#00cc66" if good_positive else "color:#ff4444"
        if v < 0:
            return "color:#ff4444" if good_positive else "color:#00cc66"
    except (TypeError, ValueError):
        pass
    return ""


def _get_yf_data(symbol: str) -> dict:
    """
    Fetch yfinance supplementary data: live price, recommendations, holders, info.
    Returns dict with keys: price, info, recommendations, holders, sustainability
    """
    try:
        import yfinance as yf
        # Try NSE ticker first
        for ticker_suffix in [".NS", ".BO", ""]:
            try:
                ticker = yf.Ticker(f"{symbol}{ticker_suffix}")
                info   = ticker.info or {}
                if info.get("regularMarketPrice") or info.get("currentPrice"):
                    price = info.get("regularMarketPrice") or info.get("currentPrice")

                    # Recommendations
                    try:
                        recs = ticker.recommendations
                    except Exception:
                        recs = None

                    # Major holders
                    try:
                        holders = ticker.major_holders
                    except Exception:
                        holders = None

                    # Institutional holders
                    try:
                        inst_holders = ticker.institutional_holders
                    except Exception:
                        inst_holders = None

                    # Sustainability / ESG
                    try:
                        esg = ticker.sustainability
                    except Exception:
                        esg = None

                    return {
                        "price":            price,
                        "info":             info,
                        "recommendations":  recs,
                        "major_holders":    holders,
                        "inst_holders":     inst_holders,
                        "esg":              esg,
                        "ticker_used":      f"{symbol}{ticker_suffix}",
                    }
            except Exception:
                continue
    except ImportError:
        pass
    return {"price": None, "info": {}, "recommendations": None,
            "major_holders": None, "inst_holders": None, "esg": None}


def _get_ai_brief(symbol: str, overview: dict, api_key: str) -> str:
    """
    Generate Bloomberg BICO-style AI company brief using Anthropic API.
    """
    if not api_key or not overview:
        return ""

    try:
        import requests
        prompt = f"""You are a senior equity analyst. Write a concise Bloomberg BICO-style company brief for {symbol}.

Company data:
- Name: {overview.get('name','')}
- Sector: {overview.get('sector','')} / {overview.get('industry','')}
- Market Cap: {_fmt_cr(overview.get('market_cap'))}
- P/E Ratio: {overview.get('pe_ratio','N/A')}
- Profit Margin: {_fmt_pct(overview.get('profit_margin'), multiply=True)}
- ROE: {_fmt_pct(overview.get('return_on_equity'), multiply=True)}
- Revenue Growth YoY: {_fmt_pct(overview.get('revenue_growth_yoy'), multiply=True)}
- 52W Range: {overview.get('week_52_low','?')} - {overview.get('week_52_high','?')}
- Description: {(overview.get('description',''))[:300]}

Write 3 paragraphs:
1. Business overview and competitive positioning (2-3 sentences)
2. Key financial strengths and risks (2-3 sentences)
3. What to watch: catalysts and risks for the next 6-12 months (2-3 sentences)

Be specific, analytical, and actionable. Name competitors. Mention specific business segments."""

        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         api_key,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":      "claude-sonnet-4-20250514",
                "max_tokens": 600,
                "messages":   [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"].strip()
    except Exception as e:
        logger.warning(f"AI brief failed for {symbol}: {e}")
        return ""


# ── Tab renderers ──────────────────────────────────────────────────────────────

def _render_des_tab(symbol: str, overview: dict, yf_data: dict, api_key: str):
    """Bloomberg DES — Company Passport."""

    if not overview:
        st.warning(
            f"No Alpha Vantage data for {symbol}. "
            "Check if symbol is in NSE_TO_AV map in alphavantage_adapter.py. "
            "Showing yfinance data only."
        )
        _render_des_from_yfinance(symbol, yf_data)
        return

    info  = yf_data.get("info", {})
    price = yf_data.get("price")

    # ── Header ────────────────────────────────────────────────────────────────
    col_name, col_price = st.columns([3, 1])
    with col_name:
        st.markdown(f"## {overview.get('name', symbol)}")
        st.caption(
            f"{overview.get('exchange','')} | "
            f"{overview.get('sector','')} | "
            f"{overview.get('industry','')} | "
            f"{overview.get('country','')} | "
            f"{overview.get('currency','')}"
        )
    with col_price:
        if price:
            st.metric("Live Price", f"₹{price:,.2f}")
        target = overview.get("analyst_target")
        if target:
            upside = ((float(target) - float(price)) / float(price) * 100) if price else None
            upside_str = f"{upside:+.1f}%" if upside else ""
            st.metric("Analyst Target", f"₹{float(target):,.2f}", upside_str)

    st.divider()

    # ── Key metrics grid (6 tiles) ─────────────────────────────────────────────
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Market Cap",    _fmt_cr(overview.get("market_cap")))
    m2.metric("P/E Ratio",     f"{overview.get('pe_ratio','—')}" if overview.get("pe_ratio") else "—")
    m3.metric("EV/EBITDA",     _fmt_ratio(overview.get("ev_to_ebitda")))
    m4.metric("P/B Ratio",     _fmt_ratio(overview.get("price_to_book")))
    m5.metric("Profit Margin", _fmt_pct(overview.get("profit_margin"), multiply=True))
    m6.metric("ROE",           _fmt_pct(overview.get("return_on_equity"), multiply=True))

    m7, m8, m9, m10, m11, m12 = st.columns(6)
    m7.metric("Beta",          f"{overview.get('beta','—')}")
    m8.metric("Dividend Yield",_fmt_pct(overview.get("dividend_yield")))
    m9.metric("EPS (TTM)",     f"₹{overview.get('eps','—')}" if overview.get("eps") else "—")
    m10.metric("52W High",     f"₹{overview.get('week_52_high','—')}" if overview.get("week_52_high") else "—")
    m11.metric("52W Low",      f"₹{overview.get('week_52_low','—')}" if overview.get("week_52_low") else "—")
    m12.metric("Revenue (TTM)",_fmt_cr(overview.get("revenue_ttm")))

    st.divider()

    # ── 52W price position bar ─────────────────────────────────────────────────
    if price and overview.get("week_52_high") and overview.get("week_52_low"):
        try:
            lo   = float(overview["week_52_low"])
            hi   = float(overview["week_52_high"])
            pct  = (float(price) - lo) / (hi - lo) * 100
            bar_color = "#00cc66" if pct > 50 else "#ff4444"
            st.markdown(
                f"**52W Position:** ₹{lo:,.0f} "
                f'<span style="display:inline-block;width:200px;height:8px;'
                f'background:#333;border-radius:4px;vertical-align:middle;margin:0 8px">'
                f'<span style="display:block;width:{pct:.0f}%;height:8px;'
                f'background:{bar_color};border-radius:4px"></span></span>'
                f"₹{hi:,.0f} &nbsp; ({pct:.0f}% of range)",
                unsafe_allow_html=True,
            )
        except (TypeError, ValueError):
            pass

    st.divider()

    # ── Description ───────────────────────────────────────────────────────────
    desc = overview.get("description", "")
    if desc:
        with st.expander("📋 Company Description", expanded=True):
            st.write(desc)

    # ── AI Brief (Bloomberg BICO equivalent) ──────────────────────────────────
    if api_key:
        with st.expander("🤖 AI Company Brief (Bloomberg BICO equivalent)"):
            brief_key = f"ai_brief_{symbol}"
            if brief_key not in st.session_state:
                with st.spinner("Generating AI brief..."):
                    st.session_state[brief_key] = _get_ai_brief(symbol, overview, api_key)
            brief_text = st.session_state.get(brief_key, "")
            if brief_text:
                st.write(brief_text)
            else:
                st.caption("AI brief unavailable.")

    # ── Growth metrics ─────────────────────────────────────────────────────────
    st.subheader("Growth & Profitability")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Revenue Growth YoY",     _fmt_pct(overview.get("revenue_growth_yoy"), multiply=True))
    g2.metric("Earnings Growth YoY",    _fmt_pct(overview.get("earnings_growth_yoy"), multiply=True))
    g3.metric("Quarterly Rev Growth",   _fmt_pct(overview.get("quarterly_revenue_growth"), multiply=True))
    g4.metric("Quarterly EPS Growth",   _fmt_pct(overview.get("quarterly_earnings_growth"), multiply=True))

    op1, op2, op3, op4 = st.columns(4)
    op1.metric("Operating Margin",  _fmt_pct(overview.get("operating_margin"), multiply=True))
    op2.metric("ROA",               _fmt_pct(overview.get("return_on_assets"), multiply=True))
    op3.metric("EV/Revenue",        _fmt_ratio(overview.get("ev_to_revenue")))
    op4.metric("P/S Ratio",         _fmt_ratio(overview.get("price_to_sales")))

    # ── Moving averages vs price ──────────────────────────────────────────────
    if price:
        st.subheader("Technical Levels")
        t1, t2, t3 = st.columns(3)
        ma50  = overview.get("moving_avg_50")
        ma200 = overview.get("moving_avg_200")
        if ma50:
            pct50 = (float(price) - float(ma50)) / float(ma50) * 100
            t1.metric("vs 50-DMA",  f"₹{float(ma50):,.2f}", f"{pct50:+.1f}%")
        if ma200:
            pct200 = (float(price) - float(ma200)) / float(ma200) * 100
            t2.metric("vs 200-DMA", f"₹{float(ma200):,.2f}", f"{pct200:+.1f}%")
        t3.metric("Book Value/Share", f"₹{overview.get('book_value','—')}" if overview.get("book_value") else "—")


def _render_des_from_yfinance(symbol: str, yf_data: dict):
    """Fallback DES using only yfinance when AV data unavailable."""
    info = yf_data.get("info", {})
    if not info:
        st.error(f"No data available for {symbol}.")
        return

    st.markdown(f"## {info.get('longName', symbol)}")
    st.caption(f"{info.get('exchange','')} | {info.get('sector','')} | {info.get('industry','')}")

    price = yf_data.get("price") or info.get("currentPrice")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Price",       f"₹{float(price):,.2f}" if price else "—")
    m2.metric("Market Cap",  _fmt_cr(info.get("marketCap")))
    m3.metric("P/E (TTM)",   f"{info.get('trailingPE','—')}")
    m4.metric("P/B",         f"{info.get('priceToBook','—')}")
    m5.metric("Dividend %",  f"{info.get('dividendYield',0)*100:.2f}%" if info.get("dividendYield") else "—")

    desc = info.get("longBusinessSummary", "")
    if desc:
        with st.expander("Company Description", expanded=True):
            st.write(desc)


def _render_fa_tab(symbol: str):
    """Bloomberg FA — Financial Statements (P&L, Balance Sheet, Cash Flow)."""
    try:
        from src.data.adapters.alphavantage_adapter import (
            get_income_statement, get_balance_sheet, get_cash_flow
        )
    except ImportError:
        st.error("alphavantage_adapter not found. Copy it to src/data/adapters/")
        return

    tab_is, tab_bs, tab_cf = st.tabs(["📊 Income Statement", "🏦 Balance Sheet", "💰 Cash Flow"])

    # ── Income Statement ──────────────────────────────────────────────────────
    with tab_is:
        st.caption("Annual figures | Source: Alpha Vantage")
        with st.spinner("Loading income statement..."):
            df_is = get_income_statement(symbol, annual=True)

        if df_is.empty:
            st.warning("Income statement unavailable. Check symbol mapping in alphavantage_adapter.py")
        else:
            # Key metrics chart
            chart_cols = [c for c in ["totalRevenue", "grossProfit", "operatingIncome", "netIncome"]
                          if c in df_is.columns]
            if chart_cols and "fiscalDateEnding" in df_is.columns:
                fig = go.Figure()
                colors = ["#4da6ff", "#00cc66", "#ffaa00", "#ff6b6b"]
                labels = ["Revenue", "Gross Profit", "Operating Income", "Net Income"]
                for col, color, label in zip(chart_cols, colors, labels):
                    fig.add_trace(go.Bar(
                        x=df_is["fiscalDateEnding"].dt.strftime("%Y"),
                        y=df_is[col] / 1e7,  # Convert to Crores
                        name=label,
                        marker_color=color,
                    ))
                fig.update_layout(
                    height=320, template="plotly_dark",
                    title=f"{symbol} — P&L Trend (₹ Crores)",
                    yaxis_title="₹ Crores",
                    barmode="group",
                    legend=dict(orientation="h", y=1.1),
                )
                st.plotly_chart(fig, use_container_width=True)

            # Margin trend
            if "totalRevenue" in df_is.columns and "netIncome" in df_is.columns:
                df_is_plot = df_is.copy()
                df_is_plot["net_margin_%"] = (df_is_plot["netIncome"] / df_is_plot["totalRevenue"] * 100).round(1)
                if "grossProfit" in df_is.columns:
                    df_is_plot["gross_margin_%"] = (df_is_plot["grossProfit"] / df_is_plot["totalRevenue"] * 100).round(1)

                fig_m = go.Figure()
                for col, color, label in [
                    ("gross_margin_%", "#00cc66", "Gross Margin %"),
                    ("net_margin_%",   "#4da6ff",  "Net Margin %"),
                ]:
                    if col in df_is_plot.columns:
                        fig_m.add_trace(go.Scatter(
                            x=df_is_plot["fiscalDateEnding"].dt.strftime("%Y"),
                            y=df_is_plot[col],
                            name=label,
                            line=dict(color=color, width=2),
                            mode="lines+markers",
                        ))
                fig_m.update_layout(
                    height=220, template="plotly_dark",
                    title="Margin Trends (%)",
                    yaxis_title="%",
                )
                st.plotly_chart(fig_m, use_container_width=True)

            # Raw table
            display_map = {
                "fiscalDateEnding": "Year", "totalRevenue": "Revenue",
                "grossProfit": "Gross Profit", "operatingIncome": "Op. Income",
                "netIncome": "Net Income", "ebitda": "EBITDA",
                "eps": "EPS", "epsDiluted": "EPS (Diluted)",
            }
            show_cols = [c for c in display_map if c in df_is.columns]
            df_show = df_is[show_cols].copy()
            df_show.columns = [display_map[c] for c in show_cols]

            for col in df_show.columns:
                if col == "Year":
                    df_show[col] = pd.to_datetime(df_show[col]).dt.strftime("%Y")
                elif col in ("EPS", "EPS (Diluted)"):
                    df_show[col] = df_show[col].apply(lambda v: f"₹{v:.2f}" if pd.notna(v) else "—")
                else:
                    df_show[col] = df_show[col].apply(lambda v: _fmt_cr(v) if pd.notna(v) else "—")

            st.dataframe(df_show, use_container_width=True, hide_index=True)

        # Quarterly toggle
        st.divider()
        if st.checkbox("Show quarterly data", key=f"is_quarterly_{symbol}"):
            df_q = get_income_statement(symbol, annual=False)
            if not df_q.empty:
                show_cols_q = [c for c in display_map if c in df_q.columns]
                df_q_show = df_q[show_cols_q].copy()
                df_q_show.columns = [display_map[c] for c in show_cols_q]
                if "Year" in df_q_show.columns:
                    df_q_show["Year"] = pd.to_datetime(df_q_show["Year"]).dt.strftime("%Y-Q?")
                st.dataframe(df_q_show.head(8), use_container_width=True, hide_index=True)

    # ── Balance Sheet ─────────────────────────────────────────────────────────
    with tab_bs:
        st.caption("Annual figures | Source: Alpha Vantage")
        with st.spinner("Loading balance sheet..."):
            df_bs = get_balance_sheet(symbol, annual=True)

        if df_bs.empty:
            st.warning("Balance sheet unavailable.")
        else:
            key_bs = ["totalAssets", "totalCurrentAssets", "totalLiabilities",
                      "totalCurrentLiabilities", "totalShareholderEquity",
                      "longTermDebt", "shortLongTermDebtTotal", "cashAndCashEquivalentsAtCarryingValue"]
            show  = [c for c in key_bs if c in df_bs.columns]

            if show and "fiscalDateEnding" in df_bs.columns:
                labels_map = {
                    "totalAssets": "Total Assets",
                    "totalLiabilities": "Total Liabilities",
                    "totalShareholderEquity": "Equity",
                    "longTermDebt": "Long-term Debt",
                    "cashAndCashEquivalentsAtCarryingValue": "Cash",
                }
                fig_bs = go.Figure()
                colors = ["#4da6ff", "#ff6b6b", "#00cc66", "#ffaa00", "#aa88ff"]
                for col, color in zip([c for c in show if c in labels_map], colors):
                    fig_bs.add_trace(go.Bar(
                        x=df_bs["fiscalDateEnding"].dt.strftime("%Y"),
                        y=df_bs[col] / 1e7,
                        name=labels_map.get(col, col),
                        marker_color=color,
                    ))
                fig_bs.update_layout(
                    height=300, template="plotly_dark",
                    title=f"{symbol} — Balance Sheet (₹ Crores)",
                    yaxis_title="₹ Crores", barmode="group",
                    legend=dict(orientation="h", y=1.1),
                )
                st.plotly_chart(fig_bs, use_container_width=True)

            # Debt-to-equity trend
            if "longTermDebt" in df_bs.columns and "totalShareholderEquity" in df_bs.columns:
                df_bs["D/E"] = (df_bs["longTermDebt"] / df_bs["totalShareholderEquity"]).round(2)
                fig_de = go.Figure(go.Scatter(
                    x=df_bs["fiscalDateEnding"].dt.strftime("%Y"),
                    y=df_bs["D/E"],
                    mode="lines+markers",
                    line=dict(color="#ffaa00", width=2),
                    name="D/E Ratio",
                ))
                fig_de.add_hline(y=1.0, line_dash="dash", line_color="#555",
                                 annotation_text="D/E = 1×")
                fig_de.update_layout(height=180, template="plotly_dark",
                                     title="Debt-to-Equity Ratio", showlegend=False)
                st.plotly_chart(fig_de, use_container_width=True)

            display_bs = {
                "fiscalDateEnding": "Year", "totalAssets": "Total Assets",
                "totalCurrentAssets": "Current Assets",
                "totalLiabilities": "Total Liabilities",
                "totalCurrentLiabilities": "Current Liabilities",
                "totalShareholderEquity": "Equity",
                "longTermDebt": "LT Debt",
                "cashAndCashEquivalentsAtCarryingValue": "Cash",
            }
            show_b = [c for c in display_bs if c in df_bs.columns]
            df_b_show = df_bs[show_b].copy()
            df_b_show.columns = [display_bs[c] for c in show_b]
            if "Year" in df_b_show.columns:
                df_b_show["Year"] = pd.to_datetime(df_b_show["Year"]).dt.strftime("%Y")
            for col in df_b_show.columns:
                if col != "Year":
                    df_b_show[col] = df_b_show[col].apply(lambda v: _fmt_cr(v) if pd.notna(v) else "—")
            st.dataframe(df_b_show, use_container_width=True, hide_index=True)

    # ── Cash Flow ─────────────────────────────────────────────────────────────
    with tab_cf:
        st.caption("Annual figures | Source: Alpha Vantage")
        with st.spinner("Loading cash flow..."):
            df_cf = get_cash_flow(symbol, annual=True)

        if df_cf.empty:
            st.warning("Cash flow unavailable.")
        else:
            cf_cols = ["operatingCashflow", "capitalExpenditures", "freeCashFlow",
                       "cashflowFromFinancing", "dividendPayout"]
            show_cf = [c for c in cf_cols if c in df_cf.columns]

            if show_cf and "fiscalDateEnding" in df_cf.columns:
                fig_cf = go.Figure()
                cf_colors = {"operatingCashflow": "#00cc66", "freeCashFlow": "#4da6ff",
                             "capitalExpenditures": "#ff6b6b", "dividendPayout": "#ffaa00"}
                cf_labels = {"operatingCashflow": "Operating CF", "freeCashFlow": "Free CF",
                             "capitalExpenditures": "CapEx", "dividendPayout": "Dividends"}
                for col in show_cf:
                    fig_cf.add_trace(go.Bar(
                        x=df_cf["fiscalDateEnding"].dt.strftime("%Y"),
                        y=df_cf[col] / 1e7,
                        name=cf_labels.get(col, col),
                        marker_color=cf_colors.get(col, "#888"),
                    ))
                fig_cf.update_layout(
                    height=300, template="plotly_dark",
                    title=f"{symbol} — Cash Flow (₹ Crores)",
                    yaxis_title="₹ Crores", barmode="group",
                    legend=dict(orientation="h", y=1.1),
                )
                st.plotly_chart(fig_cf, use_container_width=True)

            display_cf = {
                "fiscalDateEnding": "Year",
                "operatingCashflow": "Operating CF",
                "capitalExpenditures": "CapEx",
                "freeCashFlow": "Free Cash Flow",
                "cashflowFromInvestment": "Investing CF",
                "cashflowFromFinancing": "Financing CF",
                "dividendPayout": "Dividends",
                "netBorrowings": "Net Borrowings",
            }
            show_c = [c for c in display_cf if c in df_cf.columns]
            df_c_show = df_cf[show_c].copy()
            df_c_show.columns = [display_cf[c] for c in show_c]
            if "Year" in df_c_show.columns:
                df_c_show["Year"] = pd.to_datetime(df_c_show["Year"]).dt.strftime("%Y")
            for col in df_c_show.columns:
                if col != "Year":
                    df_c_show[col] = df_c_show[col].apply(lambda v: _fmt_cr(v) if pd.notna(v) else "—")
            st.dataframe(df_c_show, use_container_width=True, hide_index=True)


def _render_anr_tab(symbol: str, overview: dict, yf_data: dict):
    """Bloomberg ANR — Analyst Recommendations + Consensus."""

    st.caption("Analyst ratings and target price consensus")

    # ── Consensus from Alpha Vantage ──────────────────────────────────────────
    if overview:
        strong_buy = overview.get("analyst_rating_strong_buy") or 0
        buy        = overview.get("analyst_rating_buy") or 0
        hold       = overview.get("analyst_rating_hold") or 0
        sell       = overview.get("analyst_rating_sell") or 0
        strong_sell= overview.get("analyst_rating_strong_sell") or 0
        total      = strong_buy + buy + hold + sell + strong_sell
        target     = overview.get("analyst_target")

        if total > 0:
            buy_pct  = round((strong_buy + buy) / total * 100)
            hold_pct = round(hold / total * 100)
            sell_pct = round((sell + strong_sell) / total * 100)

            # Consensus label
            if buy_pct >= 60:
                consensus, cons_color = "STRONG BUY", "#00cc66"
            elif buy_pct >= 45:
                consensus, cons_color = "BUY", "#66cc88"
            elif sell_pct >= 45:
                consensus, cons_color = "SELL", "#ff4444"
            elif sell_pct >= 30:
                consensus, cons_color = "HOLD/WEAK", "#ffaa00"
            else:
                consensus, cons_color = "HOLD", "#888"

            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(
                f'<div style="background:rgba(0,0,0,0.3);border:1px solid {cons_color};'
                f'border-radius:6px;padding:12px;text-align:center">'
                f'<div style="color:{cons_color};font-size:22px;font-weight:700">{consensus}</div>'
                f'<div style="color:#888;font-size:11px">{total} analysts</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            c2.metric("Buy / Strong Buy",  f"{buy_pct}%",  f"{strong_buy + buy} analysts")
            c3.metric("Hold",              f"{hold_pct}%", f"{hold} analysts")
            c4.metric("Sell / Strong Sell",f"{sell_pct}%", f"{sell + strong_sell} analysts")

            # Consensus bar
            fig_cons = go.Figure(go.Bar(
                x=[strong_buy, buy, hold, sell, strong_sell],
                y=["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"],
                orientation="h",
                marker_color=["#006600", "#00cc66", "#888888", "#cc4444", "#660000"],
                text=[str(v) for v in [strong_buy, buy, hold, sell, strong_sell]],
                textposition="outside",
            ))
            fig_cons.update_layout(
                height=220, template="plotly_dark",
                title="Analyst Rating Distribution",
                xaxis_title="Number of Analysts",
                showlegend=False,
            )
            st.plotly_chart(fig_cons, use_container_width=True)

            # Target price
            if target:
                price = yf_data.get("price")
                if price:
                    upside = (float(target) - float(price)) / float(price) * 100
                    up_color = "#00cc66" if upside > 0 else "#ff4444"
                    st.markdown(
                        f'<div style="background:rgba(0,0,0,0.2);border-radius:6px;'
                        f'padding:12px;margin:8px 0">'
                        f'<b>Consensus Target: ₹{float(target):,.2f}</b> | '
                        f'Current: ₹{float(price):,.2f} | '
                        f'<span style="color:{up_color}">Implied upside: {upside:+.1f}%</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.info("No analyst rating data from Alpha Vantage for this symbol.")

    # ── yfinance recommendations (recent history) ──────────────────────────────
    recs = yf_data.get("recommendations")
    if recs is not None and not recs.empty:
        st.subheader("Recent Broker Recommendations (yfinance)")
        st.caption("Individual broker ratings — shows who changed view recently")

        # Show last 20 recommendations
        recs_show = recs.copy()
        if len(recs_show) > 20:
            recs_show = recs_show.tail(20)

        # Color grade column
        def color_grade(val):
            val_s = str(val).upper()
            if any(w in val_s for w in ["STRONG BUY", "OUTPERFORM", "OVERWEIGHT", "BUY"]):
                return "color:#00cc66;font-weight:bold"
            if any(w in val_s for w in ["SELL", "UNDERPERFORM", "UNDERWEIGHT"]):
                return "color:#ff4444"
            return "color:#888"

        style_cols = [c for c in recs_show.columns if "grade" in c.lower() or "to grade" in c.lower()]
        if style_cols:
            st.dataframe(
                recs_show.style.map(color_grade, subset=style_cols),
                use_container_width=True,
            )
        else:
            st.dataframe(recs_show, use_container_width=True)

    st.divider()

    # ── News sentiment from Alpha Vantage ────────────────────────────────────
    st.subheader("📰 Recent News Sentiment (Alpha Vantage)")
    try:
        from src.data.adapters.alphavantage_adapter import get_news_sentiment
        with st.spinner("Loading news sentiment..."):
            df_news = get_news_sentiment(symbol, limit=10)

        if not df_news.empty:
            avg_score = df_news["overall_sentiment_score"].mean()
            sent_label = "Bullish" if avg_score > 0.15 else "Bearish" if avg_score < -0.15 else "Neutral"
            sent_color = "#00cc66" if avg_score > 0.15 else "#ff4444" if avg_score < -0.15 else "#888"

            ns1, ns2 = st.columns(2)
            ns1.metric("Avg Sentiment Score", f"{avg_score:.3f}")
            ns2.markdown(
                f'<span style="color:{sent_color};font-size:18px;font-weight:700">'
                f'{sent_label}</span>',
                unsafe_allow_html=True,
            )

            for _, row in df_news.head(8).iterrows():
                score = row.get("ticker_sentiment_score") or row.get("overall_sentiment_score", 0)
                label = row.get("ticker_sentiment_label") or row.get("overall_sentiment_label", "")
                icon  = "🟢" if score > 0.15 else "🔴" if score < -0.15 else "⚪"
                title = row.get("title", "")[:90]
                source = row.get("source", "")
                url    = row.get("url", "")

                if url:
                    st.markdown(f"{icon} **[{title}]({url})**")
                else:
                    st.markdown(f"{icon} **{title}**")
                st.caption(f"{source} | Score: {score:.3f} | {label}")
        else:
            st.info("No news sentiment data available.")
    except ImportError:
        st.warning("alphavantage_adapter not found.")
    except Exception as e:
        st.warning(f"News sentiment error: {e}")


def _render_eeg_tab(symbol: str):
    """Bloomberg EEG — Earnings Estimates vs Actuals (Surprise Analysis)."""

    st.caption("EPS actual vs estimate — beat/miss history | Source: Alpha Vantage")

    try:
        from src.data.adapters.alphavantage_adapter import get_earnings
    except ImportError:
        st.error("alphavantage_adapter not found.")
        return

    with st.spinner("Loading earnings data..."):
        earnings = get_earnings(symbol)

    df_q = earnings.get("quarterly", pd.DataFrame())
    df_a = earnings.get("annual",    pd.DataFrame())

    if df_q.empty and df_a.empty:
        st.warning("No earnings data available for this symbol.")
        return

    # ── Quarterly earnings chart ───────────────────────────────────────────────
    if not df_q.empty:
        df_plot = df_q.head(12).copy()
        df_plot = df_plot.dropna(subset=["reportedEPS"])

        if "fiscalDateEnding" in df_plot.columns:
            fig_earn = go.Figure()

            if "estimatedEPS" in df_plot.columns:
                fig_earn.add_trace(go.Bar(
                    x=df_plot["fiscalDateEnding"].dt.strftime("%Y-Q?"),
                    y=df_plot["estimatedEPS"],
                    name="Estimated EPS",
                    marker_color="#888888",
                    opacity=0.6,
                ))

            bar_colors = [
                "#00cc66" if (row.get("surprisePercentage") or 0) >= 0 else "#ff4444"
                for _, row in df_plot.iterrows()
            ]
            fig_earn.add_trace(go.Bar(
                x=df_plot["fiscalDateEnding"].dt.strftime("%Y-Q?"),
                y=df_plot["reportedEPS"],
                name="Reported EPS",
                marker_color=bar_colors,
            ))

            fig_earn.update_layout(
                height=300, template="plotly_dark",
                title=f"{symbol} — Quarterly EPS: Actual vs Estimate",
                yaxis_title="EPS (₹)",
                barmode="overlay",
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig_earn, use_container_width=True)

        # Surprise % chart
        if "surprisePercentage" in df_plot.columns:
            df_surprise = df_plot.dropna(subset=["surprisePercentage"])
            if not df_surprise.empty:
                surprise_colors = [
                    "#00cc66" if v >= 0 else "#ff4444"
                    for v in df_surprise["surprisePercentage"]
                ]
                fig_surp = go.Figure(go.Bar(
                    x=df_surprise["fiscalDateEnding"].dt.strftime("%Y-Q?"),
                    y=df_surprise["surprisePercentage"],
                    marker_color=surprise_colors,
                    text=[f"{v:+.1f}%" for v in df_surprise["surprisePercentage"]],
                    textposition="outside",
                ))
                fig_surp.add_hline(y=0, line_color="#555")
                fig_surp.update_layout(
                    height=220, template="plotly_dark",
                    title="EPS Surprise % (Beat = Green, Miss = Red)",
                    yaxis_title="Surprise %",
                    showlegend=False,
                )
                st.plotly_chart(fig_surp, use_container_width=True)

        # Table
        st.subheader("Quarterly Earnings Detail")
        display_q = df_q.head(12).copy()
        for col in ["reportedEPS", "estimatedEPS", "surprise"]:
            if col in display_q.columns:
                display_q[col] = display_q[col].apply(
                    lambda v: f"₹{v:.2f}" if pd.notna(v) else "—"
                )
        if "surprisePercentage" in display_q.columns:
            display_q["surprisePercentage"] = display_q["surprisePercentage"].apply(
                lambda v: f"{v:+.1f}%" if pd.notna(v) else "—"
            )
        if "fiscalDateEnding" in display_q.columns:
            display_q["fiscalDateEnding"] = display_q["fiscalDateEnding"].dt.strftime("%Y-%m-%d")

        def color_surprise(val):
            if "+" in str(val): return "color:#00cc66"
            if "-" in str(val) and "₹" not in str(val): return "color:#ff4444"
            return ""

        col_rename = {
            "fiscalDateEnding": "Quarter End", "reportedDate": "Report Date",
            "reportedEPS": "Reported EPS", "estimatedEPS": "Estimated EPS",
            "surprise": "Surprise", "surprisePercentage": "Surprise %",
        }
        show_cols = [c for c in col_rename if c in display_q.columns]
        display_q = display_q[show_cols].rename(columns=col_rename)

        if "Surprise %" in display_q.columns:
            st.dataframe(
                display_q.style.map(color_surprise, subset=["Surprise %"]),
                use_container_width=True, hide_index=True,
            )
        else:
            st.dataframe(display_q, use_container_width=True, hide_index=True)

    # Annual EPS trend
    if not df_a.empty:
        st.subheader("Annual EPS Trend")
        if "reportedEPS" in df_a.columns and "fiscalDateEnding" in df_a.columns:
            fig_ann = go.Figure(go.Scatter(
                x=df_a["fiscalDateEnding"].dt.strftime("%Y"),
                y=df_a["reportedEPS"],
                mode="lines+markers+text",
                text=[f"₹{v:.2f}" for v in df_a["reportedEPS"]],
                textposition="top center",
                line=dict(color="#4da6ff", width=2),
                marker=dict(size=8),
            ))
            fig_ann.update_layout(
                height=220, template="plotly_dark",
                title=f"{symbol} — Annual EPS",
                yaxis_title="EPS (₹)", showlegend=False,
            )
            st.plotly_chart(fig_ann, use_container_width=True)


def _render_eqrv_tab(symbol: str, overview: dict):
    """Bloomberg EQRV — Equity Relative Valuation (Peer Comparison)."""

    st.caption("Peer group valuation comparison | Source: Alpha Vantage")

    # Auto-detect sector peers
    sector = overview.get("sector", "") if overview else ""
    detected_peers = []
    for sector_key, peers in SECTOR_PEERS.items():
        if symbol in peers:
            detected_peers = [p for p in peers if p != symbol]
            break

    # Let user override peer group
    all_symbols = sorted(set(NSE_UNIVERSE + [symbol]))
    default_peers = detected_peers[:4] if detected_peers else [s for s in NSE_UNIVERSE[:5] if s != symbol]

    selected_peers = st.multiselect(
        "Select peer group (max 8):",
        options=[s for s in all_symbols if s != symbol],
        default=default_peers[:4],
        max_selections=8,
    )

    if not selected_peers:
        st.info("Select at least one peer to compare.")
        return

    if st.button(f"🔄 Load Peer Comparison ({len(selected_peers)} stocks)", type="primary"):
        with st.spinner(f"Loading peer data for {symbol} + {len(selected_peers)} peers..."):
            try:
                from src.data.adapters.alphavantage_adapter import get_peer_overview
                all_syms = [symbol] + selected_peers
                df_peers = get_peer_overview(all_syms)
                st.session_state[f"eqrv_{symbol}"] = df_peers
                st.success(f"Loaded {len(df_peers)} companies.")
            except ImportError:
                st.error("alphavantage_adapter not found.")
                return
            except Exception as e:
                st.error(f"Peer data error: {e}")
                return

    df_peers = st.session_state.get(f"eqrv_{symbol}", pd.DataFrame())

    if df_peers.empty:
        st.info("Click 'Load Peer Comparison' to fetch data.")
        return

    # ── Valuation comparison chart ─────────────────────────────────────────────
    metrics_to_chart = [
        ("pe_ratio",      "P/E Ratio",        "#4da6ff"),
        ("price_to_book", "Price/Book",        "#ffaa00"),
        ("ev_to_ebitda",  "EV/EBITDA",         "#00cc66"),
        ("profit_margin_%","Profit Margin %",  "#ff6b6b"),
        ("roe_%",         "ROE %",             "#aa88ff"),
    ]

    available = [(m, l, c) for m, l, c in metrics_to_chart if m in df_peers.columns]

    if available:
        chart_tabs = st.tabs([l for _, l, _ in available])
        for tab, (metric, label, color) in zip(chart_tabs, available):
            with tab:
                df_plot = df_peers[["symbol", metric]].dropna()
                if df_plot.empty:
                    st.info(f"No {label} data.")
                    continue

                # Highlight current symbol
                bar_colors = [
                    "#ffa500" if row["symbol"] == symbol else color
                    for _, row in df_plot.iterrows()
                ]

                fig = go.Figure(go.Bar(
                    x=df_plot["symbol"],
                    y=df_plot[metric],
                    marker_color=bar_colors,
                    text=[f"{v:.1f}" for v in df_plot[metric]],
                    textposition="outside",
                ))
                # Add median line
                median_val = df_plot[metric].median()
                fig.add_hline(y=median_val, line_dash="dash", line_color="#888",
                              annotation_text=f"Median: {median_val:.1f}")
                fig.update_layout(
                    height=280, template="plotly_dark",
                    title=f"{label} Comparison (orange = {symbol})",
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

    # ── Comparison table ───────────────────────────────────────────────────────
    st.subheader("Full Peer Comparison Table")

    display_cols = [
        "symbol", "name", "market_cap_cr", "pe_ratio", "forward_pe",
        "price_to_book", "ev_to_ebitda", "profit_margin_%", "roe_%",
        "dividend_yield_%", "beta", "analyst_target",
    ]
    show = [c for c in display_cols if c in df_peers.columns]
    df_show = df_peers[show].copy()

    col_rename = {
        "symbol": "Symbol", "name": "Company", "market_cap_cr": "Mkt Cap (Cr)",
        "pe_ratio": "P/E", "forward_pe": "Fwd P/E", "price_to_book": "P/B",
        "ev_to_ebitda": "EV/EBITDA", "profit_margin_%": "Net Margin %",
        "roe_%": "ROE %", "dividend_yield_%": "Div Yield %",
        "beta": "Beta", "analyst_target": "Target ₹",
    }
    df_show = df_show.rename(columns={c: col_rename.get(c, c) for c in df_show.columns})

    # Highlight selected symbol row
    def highlight_selected(row):
        if row.get("Symbol") == symbol:
            return ["background-color:#1a1a00;font-weight:bold"] * len(row)
        return [""] * len(row)

    numeric_cols = [c for c in df_show.columns
                    if c not in ("Symbol", "Company")
                    and df_show[c].dtype in ("float64", "int64")]

    def color_vs_median(val, col_series):
        if pd.isna(val):
            return ""
        med = col_series.median()
        if pd.isna(med) or med == 0:
            return ""
        if val > med * 1.2:
            return "color:#ff6b6b"  # expensive vs peers
        if val < med * 0.8:
            return "color:#00cc66"  # cheap vs peers
        return ""

    styled = df_show.style.apply(highlight_selected, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # vs-Median columns if available
    vs_med_cols = [c for c in df_peers.columns if "_vs_med" in c]
    if vs_med_cols:
        st.subheader("vs Sector Median")
        df_vs = df_peers[["symbol"] + vs_med_cols].copy()
        df_vs.columns = ["Symbol"] + [c.replace("_vs_med", " vs Median %") for c in vs_med_cols]

        def color_vs(val):
            if pd.isna(val):
                return ""
            return "color:#00cc66" if float(val) < 0 else "color:#ff6b6b"

        st.dataframe(
            df_vs.style.map(color_vs, subset=[c for c in df_vs.columns if "Median" in c]),
            use_container_width=True, hide_index=True,
        )


def _render_holders_tab(symbol: str, yf_data: dict):
    """Bloomberg HDS equivalent — Security Ownership."""

    st.caption("Institutional holdings and promoter data | Source: yfinance")

    major = yf_data.get("major_holders")
    inst  = yf_data.get("inst_holders")
    esg   = yf_data.get("esg")

    if major is not None and not major.empty:
        st.subheader("Major Holders")
        st.dataframe(major, use_container_width=True)
    else:
        st.info("Major holders data unavailable for this symbol.")

    if inst is not None and not inst.empty:
        st.subheader("Top Institutional Holders")
        inst_show = inst.copy()
        st.dataframe(inst_show, use_container_width=True, hide_index=True)

        # Chart top 10
        if "% Out" in inst_show.columns and "Holder" in inst_show.columns:
            top10 = inst_show.head(10)
            fig_pie = go.Figure(go.Pie(
                labels=top10["Holder"],
                values=top10["% Out"],
                hole=0.4,
                textinfo="label+percent",
            ))
            fig_pie.update_layout(
                height=300, template="plotly_dark",
                title="Top 10 Institutional Holdings",
                margin=dict(t=40, b=20, l=20, r=20),
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Institutional holder data unavailable.")

    # ESG
    if esg is not None and not esg.empty:
        st.subheader("ESG Scores (Bloomberg ESG equivalent)")
        st.dataframe(esg, use_container_width=True)
    else:
        st.caption("ESG data unavailable for this symbol via yfinance.")


# ── Main render function ───────────────────────────────────────────────────────

def render_company_intelligence():
    """
    Main entry point. Add to app.py:
        elif page == "🏢 Company Intelligence":
            from src.dashboard.pages.company_intelligence import render_company_intelligence
            render_company_intelligence()
    """
    st.header("🏢 Company Intelligence")
    st.caption(
        "Bloomberg DES + FA + ANR + EEG + EQRV equivalent for Indian markets. "
        "Data: Alpha Vantage + yfinance. Free, no subscription."
    )

    # ── Symbol selector ────────────────────────────────────────────────────────
    col_sym, col_cat, col_rate = st.columns([2, 2, 1])
    with col_sym:
        # Allow free text or dropdown
        symbol_input = st.text_input(
            "Symbol (NSE)", value="RELIANCE",
            help="NSE symbol e.g. RELIANCE, TCS, HDFCBANK. US stocks also work: AAPL, TSLA"
        ).strip().upper()

    with col_cat:
        symbol_preset = st.selectbox(
            "Or pick from universe",
            ["— type above —"] + NSE_UNIVERSE,
        )
        if symbol_preset != "— type above —":
            symbol_input = symbol_preset

    symbol = symbol_input or "RELIANCE"

    with col_rate:
        try:
            from src.data.adapters.alphavantage_adapter import get_rate_limit_status
            rl = get_rate_limit_status()
            st.metric("AV Calls Today", f"{rl['calls_today']}/25")
            if rl["calls_today"] >= 20:
                st.warning("Rate limit nearly hit — using cache")
        except ImportError:
            st.caption("AV adapter not loaded")

    # ── Load data ──────────────────────────────────────────────────────────────
    try:
        from config.settings import settings
        api_key = getattr(settings, "ANTHROPIC_API_KEY", None)
    except Exception:
        api_key = None

    cache_key = f"ci_{symbol}"
    if cache_key not in st.session_state or st.button("🔄 Refresh Data", key="ci_refresh"):
        with st.spinner(f"Loading {symbol} data from Alpha Vantage + yfinance..."):
            try:
                from src.data.adapters.alphavantage_adapter import get_overview
                overview = get_overview(symbol)
            except ImportError:
                st.error("alphavantage_adapter.py not found in src/data/adapters/. Copy it first.")
                return
            except Exception as e:
                st.warning(f"Alpha Vantage error: {e}")
                overview = {}

            yf_data = _get_yf_data(symbol)
            st.session_state[cache_key] = {"overview": overview, "yf_data": yf_data}
    else:
        cached = st.session_state[cache_key]
        overview = cached["overview"]
        yf_data  = cached["yf_data"]

    # ── Quick price banner ─────────────────────────────────────────────────────
    if overview or yf_data.get("price"):
        name   = overview.get("name", symbol) if overview else symbol
        price  = yf_data.get("price")
        sector = overview.get("sector", "") if overview else ""
        mktcap = overview.get("market_cap") if overview else None

        st.markdown(
            f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);'
            f'border-radius:8px;padding:12px 16px;margin-bottom:12px;display:flex;'
            f'justify-content:space-between;align-items:center">'
            f'<div><span style="font-size:18px;font-weight:700">{name}</span> '
            f'<span style="color:#888;font-size:13px">({symbol})</span> '
            f'<span style="background:rgba(77,166,255,0.15);color:#4da6ff;font-size:11px;'
            f'padding:2px 8px;border-radius:3px;margin-left:8px">{sector}</span></div>'
            f'<div style="font-size:20px;font-weight:700;color:#00cc66">'
            f'{"₹"+f"{price:,.2f}" if price else "—"}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Bloomberg function tabs ────────────────────────────────────────────────
    tab_des, tab_fa, tab_anr, tab_eeg, tab_eqrv, tab_hds = st.tabs([
        "📋 DES — Company",
        "📊 FA — Financials",
        "🎯 ANR — Analysts",
        "📈 EEG — Earnings",
        "⚖️ EQRV — Peer Comps",
        "🏦 HDS — Ownership",
    ])

    with tab_des:
        _render_des_tab(symbol, overview, yf_data, api_key)

    with tab_fa:
        _render_fa_tab(symbol)

    with tab_anr:
        _render_anr_tab(symbol, overview, yf_data)

    with tab_eeg:
        _render_eeg_tab(symbol)

    with tab_eqrv:
        _render_eqrv_tab(symbol, overview)

    with tab_hds:
        _render_holders_tab(symbol, yf_data)