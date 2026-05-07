# trading-agent/src/dashboard/pages/valuation_comps.py
"""
Valuation Comps — Bloomberg EQRV + KPIC Equivalent

Bloomberg functions replicated:
  EQRV — Equity Relative Valuation (P/E, EV/EBITDA, P/B, P/S peer comps)
  KPIC — KPI Comparison (margins, growth, ROE, ROA operational comps)

Features beyond the tab in company_intelligence.py:
  - Pre-built sector peer groups (Banking, IT, Energy, Pharma, Metals, Auto)
  - Bubble chart: P/E vs EPS growth (classic IB comp chart)
  - Percentile rankings within peer group
  - Undervalued / overvalued screener
  - Export to CSV

Usage in app.py:
    elif page == "⚖️ Valuation Comps":
        from src.dashboard.pages.valuation_comps import render_valuation_comps
        render_valuation_comps()
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

# ── Pre-built sector peer groups ──────────────────────────────────────────────

SECTOR_GROUPS = {
    "🏦 Banking & Finance": {
        "symbols": ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK",
                    "BAJFINANCE", "BAJAJFINSV", "INDUSINDBK", "BANDHANBNK"],
        "key_metrics": ["pe_ratio", "price_to_book", "return_on_equity", "profit_margin"],
        "benchmark": "HDFCBANK",
    },
    "💻 Information Technology": {
        "symbols": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "MPHASIS",
                    "LTIM", "COFORGE", "PERSISTENT"],
        "key_metrics": ["pe_ratio", "ev_to_ebitda", "profit_margin", "return_on_equity"],
        "benchmark": "TCS",
    },
    "⚡ Energy & Oil": {
        "symbols": ["RELIANCE", "ONGC", "BPCL", "IOC", "NTPC", "POWERGRID", "TATAPOWER"],
        "key_metrics": ["pe_ratio", "price_to_book", "dividend_yield", "ev_to_ebitda"],
        "benchmark": "RELIANCE",
    },
    "💊 Pharmaceuticals": {
        "symbols": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "AUROPHARMA",
                    "LUPIN", "TORNTPHARM", "ALKEM"],
        "key_metrics": ["pe_ratio", "ev_to_ebitda", "profit_margin", "return_on_equity"],
        "benchmark": "SUNPHARMA",
    },
    "⚙️ Metals & Mining": {
        "symbols": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "COALINDIA",
                    "NMDC", "SAIL", "JINDALSTEL"],
        "key_metrics": ["pe_ratio", "ev_to_ebitda", "price_to_book", "profit_margin"],
        "benchmark": "TATASTEEL",
    },
    "🚗 Automobiles": {
        "symbols": ["MARUTI", "M&M", "TATAMOTORS", "BAJAJ-AUTO", "HEROMOTOCO",
                    "EICHERMOT", "ASHOKLEY", "TVSMOTOR"],
        "key_metrics": ["pe_ratio", "ev_to_ebitda", "profit_margin", "return_on_equity"],
        "benchmark": "MARUTI",
    },
    "🏗️ Infrastructure & Cement": {
        "symbols": ["LT", "ULTRACEMCO", "SHREECEM", "AMBUJACEMENT", "ACC",
                    "ADANIPORTS", "GMRINFRA"],
        "key_metrics": ["pe_ratio", "ev_to_ebitda", "price_to_book", "return_on_equity"],
        "benchmark": "LT",
    },
    "🛒 FMCG & Consumer": {
        "symbols": ["HINDUNILVR", "ITC", "NESTLEIND", "DABUR", "BRITANNIA",
                    "MARICO", "GODREJCP", "EMAMILTD"],
        "key_metrics": ["pe_ratio", "price_to_book", "profit_margin", "return_on_equity"],
        "benchmark": "HINDUNILVR",
    },
    "📱 Telecom": {
        "symbols": ["BHARTIARTL", "IDEA", "TATACOMM", "INDUSTOWER"],
        "key_metrics": ["pe_ratio", "ev_to_ebitda", "price_to_book", "profit_margin"],
        "benchmark": "BHARTIARTL",
    },
}

METRIC_LABELS = {
    "pe_ratio":            "P/E Ratio",
    "forward_pe":          "Forward P/E",
    "price_to_book":       "Price/Book",
    "ev_to_ebitda":        "EV/EBITDA",
    "ev_to_revenue":       "EV/Revenue",
    "price_to_sales":      "Price/Sales",
    "profit_margin_%":     "Net Margin %",
    "roe_%":               "ROE %",
    "dividend_yield_%":    "Div Yield %",
    "beta":                "Beta",
    "market_cap_cr":       "Mkt Cap (Cr)",
    "analyst_target":      "Target ₹",
    "return_on_equity":    "ROE",
    "profit_margin":       "Net Margin",
    "return_on_assets":    "ROA",
    "revenue_growth_yoy":  "Rev Growth",
    "earnings_growth_yoy": "EPS Growth",
    "quarterly_earnings_growth": "Qtrly EPS Gr",
}

# Metrics where LOWER is better (for percentile ranking)
LOWER_IS_BETTER = {"pe_ratio", "forward_pe", "price_to_book", "ev_to_ebitda",
                   "ev_to_revenue", "price_to_sales", "beta"}


# ── Data fetcher ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_peer_data(symbols: tuple[str, ...]) -> pd.DataFrame:
    """Cached batch fetch from Alpha Vantage."""
    try:
        from src.data.adapters.alphavantage_adapter import get_peer_overview
        df = get_peer_overview(list(symbols))
        return df
    except ImportError:
        st.warning("alphavantage_adapter not found. Copy it to src/data/adapters/")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Peer data fetch error: {e}")
        return pd.DataFrame()


def _add_percentile_ranks(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Add percentile rank column for each metric."""
    df = df.copy()
    for metric in metrics:
        if metric not in df.columns:
            continue
        col_data = pd.to_numeric(df[metric], errors="coerce")
        if col_data.isna().all():
            continue
        if metric in LOWER_IS_BETTER:
            # Lower value = higher percentile (cheaper = better)
            df[f"{metric}_pct"] = col_data.rank(ascending=True, pct=True).round(2) * 100
        else:
            df[f"{metric}_pct"] = col_data.rank(ascending=False, pct=True).round(2) * 100
    return df


def _valuation_verdict(row: pd.Series, key_metrics: list[str]) -> tuple[str, str]:
    """
    Simple valuation verdict based on percentile ranks.
    Returns (verdict, color).
    """
    pct_cols = [f"{m}_pct" for m in key_metrics if f"{m}_pct" in row.index]
    if not pct_cols:
        return "N/A", "#888"

    avg_pct = float(row[pct_cols].mean())

    if avg_pct >= 75:
        return "CHEAP",     "#00cc66"
    elif avg_pct >= 55:
        return "FAIR",      "#66cc88"
    elif avg_pct >= 40:
        return "NEUTRAL",   "#888888"
    elif avg_pct >= 25:
        return "EXPENSIVE", "#ffaa00"
    else:
        return "VERY EXP.", "#ff4444"


# ── Chart builders ─────────────────────────────────────────────────────────────

def _bar_chart(df: pd.DataFrame, metric: str, label: str, highlight: str = "") -> go.Figure:
    """Horizontal bar chart for a single metric, sorted, with highlight."""
    df_plot = df[["symbol", metric]].dropna().sort_values(metric)
    if df_plot.empty:
        return go.Figure()

    median_val = df_plot[metric].median()
    colors = [
        "#ffa500" if row["symbol"] == highlight
        else "#00cc66" if (metric not in LOWER_IS_BETTER and row[metric] >= median_val)
             or (metric in LOWER_IS_BETTER and row[metric] <= median_val)
        else "#ff6b6b"
        for _, row in df_plot.iterrows()
    ]

    fig = go.Figure(go.Bar(
        x=df_plot[metric],
        y=df_plot["symbol"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}" for v in df_plot[metric]],
        textposition="outside",
    ))
    fig.add_vline(x=median_val, line_dash="dash", line_color="#888",
                  annotation_text=f"Median {median_val:.1f}", annotation_position="top right")
    fig.update_layout(
        height=max(220, len(df_plot) * 32),
        template="plotly_dark",
        title=f"{label} (orange = selected, green = above median)",
        xaxis_title=label, yaxis_title="",
        showlegend=False,
        margin=dict(l=100, r=60, t=40, b=20),
    )
    return fig


def _bubble_chart(df: pd.DataFrame, x_metric: str, y_metric: str,
                  size_metric: str, label_x: str, label_y: str,
                  highlight: str = "") -> go.Figure:
    """Bloomberg-style bubble chart: P/E vs EPS Growth, sized by market cap."""
    needed = [m for m in [x_metric, y_metric, size_metric, "symbol"] if m in df.columns]
    if len(needed) < 3:
        return go.Figure()

    df_plot = df[["symbol", x_metric, y_metric, size_metric]].dropna()
    if df_plot.empty:
        return go.Figure()

    sizes = df_plot[size_metric].clip(lower=1)
    sizes_norm = (sizes / sizes.max() * 50 + 10).values

    colors = ["#ffa500" if s == highlight else "#4da6ff" for s in df_plot["symbol"]]

    fig = go.Figure(go.Scatter(
        x=df_plot[x_metric],
        y=df_plot[y_metric],
        mode="markers+text",
        text=df_plot["symbol"],
        textposition="top center",
        textfont=dict(size=10),
        marker=dict(
            size=sizes_norm,
            color=colors,
            opacity=0.8,
            line=dict(color="white", width=0.5),
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            f"{label_x}: %{{x:.1f}}<br>"
            f"{label_y}: %{{y:.1f}}<br>"
            "<extra></extra>"
        ),
    ))

    # Add quadrant lines at medians
    med_x = df_plot[x_metric].median()
    med_y = df_plot[y_metric].median()
    fig.add_vline(x=med_x, line_dash="dot", line_color="#444",
                  annotation_text="Median", annotation_position="top right")
    fig.add_hline(y=med_y, line_dash="dot", line_color="#444")

    fig.update_layout(
        height=420, template="plotly_dark",
        title=f"{label_y} vs {label_x} (bubble size = market cap)",
        xaxis_title=label_x, yaxis_title=label_y,
        showlegend=False,
    )
    return fig


def _heatmap_chart(df: pd.DataFrame, metrics: list[str]) -> go.Figure:
    """Percentile rank heatmap across all metrics and symbols."""
    pct_cols = [f"{m}_pct" for m in metrics if f"{m}_pct" in df.columns]
    if not pct_cols or df.empty:
        return go.Figure()

    df_hm = df.set_index("symbol")[pct_cols].copy()
    df_hm.columns = [METRIC_LABELS.get(c.replace("_pct",""), c.replace("_pct","")) for c in pct_cols]

    z    = df_hm.values
    syms = df_hm.index.tolist()
    cols = df_hm.columns.tolist()

    fig = go.Figure(go.Heatmap(
        z=z,
        x=cols,
        y=syms,
        colorscale="RdYlGn",
        zmin=0, zmax=100,
        text=[[f"{v:.0f}th" if not np.isnan(v) else "—" for v in row] for row in z],
        texttemplate="%{text}",
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.0f}th percentile<extra></extra>",
        showscale=True,
    ))
    fig.update_layout(
        height=max(250, len(syms) * 30),
        template="plotly_dark",
        title="Percentile Rank Heatmap (green = cheap/good, red = expensive/weak)",
        margin=dict(l=100, r=60, t=50, b=50),
    )
    return fig


# ── Main render function ───────────────────────────────────────────────────────

def render_valuation_comps():
    """
    Add to app.py:
        elif page == "⚖️ Valuation Comps":
            from src.dashboard.pages.valuation_comps import render_valuation_comps
            render_valuation_comps()
    """
    st.header("⚖️ Valuation Comps")
    st.caption(
        "Bloomberg EQRV + KPIC equivalent. "
        "Peer group valuation and KPI comparison with percentile ranking."
    )

    # ── Sector selector ────────────────────────────────────────────────────────
    col_s, col_h, col_b = st.columns([2, 2, 1])
    with col_s:
        sector_choice = st.selectbox("Sector group", list(SECTOR_GROUPS.keys()))
    with col_h:
        highlight_sym = st.selectbox(
            "Highlight symbol",
            SECTOR_GROUPS[sector_choice]["symbols"],
            index=0,
        )
    with col_b:
        force_refresh = st.button("🔄 Refresh", type="primary")

    sector_data = SECTOR_GROUPS[sector_choice]
    symbols     = sector_data["symbols"]
    key_metrics = sector_data["key_metrics"]

    # Custom symbol override
    with st.expander("🔧 Customize peer group"):
        all_syms = sorted({s for g in SECTOR_GROUPS.values() for s in g["symbols"]})
        custom_syms = st.multiselect(
            "Override symbols (max 12)",
            all_syms,
            default=symbols[:6],
            max_selections=12,
        )
        if custom_syms:
            symbols = custom_syms

    st.divider()

    # ── Fetch data ────────────────────────────────────────────────────────────
    cache_key = f"vc_{sector_choice}_{','.join(symbols)}"
    if force_refresh and cache_key in st.session_state:
        del st.session_state[cache_key]

    if cache_key not in st.session_state:
        with st.spinner(f"Loading data for {len(symbols)} stocks... (Alpha Vantage, may take 30s)"):
            df_raw = _fetch_peer_data(tuple(symbols))
            if not df_raw.empty:
                # Add derived columns
                for col, orig, mult in [
                    ("profit_margin_%", "profit_margin", 100),
                    ("roe_%",           "return_on_equity", 100),
                    ("dividend_yield_%","dividend_yield", 100),
                ]:
                    if orig in df_raw.columns and col not in df_raw.columns:
                        df_raw[col] = pd.to_numeric(df_raw[orig], errors="coerce") * mult

                # Add percentile ranks
                rank_cols = key_metrics + ["profit_margin_%", "roe_%", "dividend_yield_%"]
                df_raw = _add_percentile_ranks(df_raw, rank_cols)

            st.session_state[cache_key] = df_raw

    df = st.session_state.get(cache_key, pd.DataFrame())

    if df.empty:
        st.warning(
            "No data loaded. Possible causes:\n"
            "- Alpha Vantage rate limit (25/day free) — wait until tomorrow\n"
            "- Symbols not in Alpha Vantage's India coverage\n"
            "- Check internet connection"
        )
        st.info("💡 Tip: US stocks (AAPL, TSLA, NVDA) work reliably on Alpha Vantage free tier.")
        return

    # ── Rate limit warning ─────────────────────────────────────────────────────
    try:
        from src.data.adapters.alphavantage_adapter import get_rate_limit_status
        rl = get_rate_limit_status()
        if rl["calls_today"] >= 20:
            st.warning(f"⚠️ Alpha Vantage: {rl['calls_today']}/25 daily calls used. Showing cached data.")
    except ImportError:
        pass

    # ── KPI summary row ───────────────────────────────────────────────────────
    st.subheader("Sector Summary")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Companies",    len(df))
    for col, metric, label in [
        (s2, "pe_ratio",      "Median P/E"),
        (s3, "price_to_book", "Median P/B"),
        (s4, "ev_to_ebitda",  "Median EV/EBITDA"),
        (s5, "profit_margin_%","Median Margin %"),
    ]:
        if metric in df.columns:
            med = pd.to_numeric(df[metric], errors="coerce").median()
            col.metric(label, f"{med:.1f}" if pd.notna(med) else "—")

    st.divider()

    # ── Main tabs ─────────────────────────────────────────────────────────────
    tab_table, tab_charts, tab_bubble, tab_heatmap, tab_screener = st.tabs([
        "📊 Comparison Table",
        "📈 Bar Charts",
        "🫧 Bubble Chart",
        "🌡 Heatmap",
        "🔍 Value Screener",
    ])

    # ══ TABLE ════════════════════════════════════════════════════════════════
    with tab_table:
        st.caption("All metrics | highlighted symbol in orange | sorted by market cap")

        display_cols_order = [
            "symbol", "name", "market_cap_cr",
            "pe_ratio", "forward_pe", "price_to_book", "ev_to_ebitda",
            "profit_margin_%", "roe_%", "dividend_yield_%",
            "beta", "analyst_target",
            "revenue_growth_yoy", "earnings_growth_yoy",
        ]
        show = [c for c in display_cols_order if c in df.columns]
        df_show = df[show].copy()

        # Rename
        df_show.columns = [METRIC_LABELS.get(c, c) for c in df_show.columns]

        # Format numerics
        for col in df_show.columns:
            if col in ("Symbol", "Company"):
                continue
            df_show[col] = pd.to_numeric(df_show[col], errors="coerce").round(2)

        # Sort by mkt cap
        if "Mkt Cap (Cr)" in df_show.columns:
            df_show = df_show.sort_values("Mkt Cap (Cr)", ascending=False)

        def highlight_row(row):
            sym_col = "Symbol" if "Symbol" in row.index else row.index[0]
            if row.get(sym_col, "") == highlight_sym:
                return ["background-color:#2a2000;font-weight:bold"] * len(row)
            return [""] * len(row)

        def color_pe(val):
            try:
                v = float(val)
                if v <= 0:   return "color:#888"
                if v < 15:   return "color:#00cc66"
                if v < 25:   return "color:#66cc88"
                if v < 40:   return "color:#ffaa00"
                return              "color:#ff4444"
            except (TypeError, ValueError):
                return ""

        def color_margin(val):
            try:
                v = float(val)
                if v >= 20:  return "color:#00cc66"
                if v >= 10:  return "color:#66cc88"
                if v >= 5:   return "color:#ffaa00"
                return              "color:#ff4444"
            except (TypeError, ValueError):
                return ""

        styled = df_show.style.apply(highlight_row, axis=1)
        if "P/E Ratio" in df_show.columns:
            styled = styled.map(color_pe, subset=["P/E Ratio"])
        if "Net Margin %" in df_show.columns:
            styled = styled.map(color_margin, subset=["Net Margin %"])

        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Export
        csv = df_show.to_csv(index=False)
        st.download_button(
            "⬇️ Export to CSV",
            data=csv.encode("utf-8-sig"),
            file_name=f"valuation_comps_{sector_choice[:15].replace(' ','_')}.csv",
            mime="text/csv",
        )

    # ══ BAR CHARTS ═══════════════════════════════════════════════════════════
    with tab_charts:
        all_chart_metrics = [
            ("pe_ratio",         "P/E Ratio"),
            ("price_to_book",    "Price/Book"),
            ("ev_to_ebitda",     "EV/EBITDA"),
            ("profit_margin_%",  "Net Margin %"),
            ("roe_%",            "ROE %"),
            ("dividend_yield_%", "Dividend Yield %"),
            ("beta",             "Beta"),
        ]
        available_chart = [(m, l) for m, l in all_chart_metrics if m in df.columns]

        if not available_chart:
            st.info("No chartable metrics available.")
        else:
            metric_sel = st.selectbox(
                "Select metric",
                [l for _, l in available_chart],
                key="vc_metric_sel",
            )
            metric_key = next(m for m, l in available_chart if l == metric_sel)
            fig = _bar_chart(df, metric_key, metric_sel, highlight=highlight_sym)
            st.plotly_chart(fig, use_container_width=True)

            # Show all metrics in grid
            st.divider()
            st.caption("All metrics — quick view")
            pairs = [(available_chart[i], available_chart[i+1] if i+1 < len(available_chart) else None)
                     for i in range(0, len(available_chart), 2)]
            for left, right in pairs:
                c1, c2 = st.columns(2)
                with c1:
                    f = _bar_chart(df, left[0], left[1], highlight_sym)
                    if f.data:
                        f.update_layout(height=220, title=left[1])
                        st.plotly_chart(f, use_container_width=True)
                with c2:
                    if right:
                        f = _bar_chart(df, right[0], right[1], highlight_sym)
                        if f.data:
                            f.update_layout(height=220, title=right[1])
                            st.plotly_chart(f, use_container_width=True)

    # ══ BUBBLE CHART ═════════════════════════════════════════════════════════
    with tab_bubble:
        st.caption(
            "Classic IB chart: P/E vs Growth, sized by market cap. "
            "Stocks in bottom-left quadrant = cheap + growing = interesting."
        )

        bubble_metrics = [(m, METRIC_LABELS.get(m, m)) for m in [
            "pe_ratio", "forward_pe", "price_to_book", "ev_to_ebitda",
            "profit_margin_%", "roe_%", "earnings_growth_yoy", "revenue_growth_yoy",
        ] if m in df.columns]

        bc1, bc2 = st.columns(2)
        with bc1:
            x_label = st.selectbox("X axis", [l for _, l in bubble_metrics], index=0, key="bx")
            x_metric = next(m for m, l in bubble_metrics if l == x_label)
        with bc2:
            y_label = st.selectbox("Y axis", [l for _, l in bubble_metrics],
                                   index=min(4, len(bubble_metrics)-1), key="by")
            y_metric = next(m for m, l in bubble_metrics if l == y_label)

        size_metric = "market_cap_cr" if "market_cap_cr" in df.columns else (
            df.select_dtypes(include=[np.number]).columns[0] if len(df.select_dtypes(include=[np.number]).columns) else None
        )
        if size_metric:
            fig_b = _bubble_chart(df, x_metric, y_metric, size_metric,
                                  x_label, y_label, highlight=highlight_sym)
            st.plotly_chart(fig_b, use_container_width=True)

            # Quadrant analysis
            df_plot = df[["symbol", x_metric, y_metric]].dropna()
            if not df_plot.empty:
                med_x = df_plot[x_metric].median()
                med_y = df_plot[y_metric].median()

                cheap_growing = df_plot[
                    (df_plot[x_metric] <= med_x) & (df_plot[y_metric] >= med_y)
                ]["symbol"].tolist()
                expensive_weak = df_plot[
                    (df_plot[x_metric] >= med_x) & (df_plot[y_metric] <= med_y)
                ]["symbol"].tolist()

                q1, q2 = st.columns(2)
                with q1:
                    if cheap_growing:
                        st.success(f"🟢 Cheap + Good: {', '.join(cheap_growing)}")
                    else:
                        st.info("No stocks in cheap+good quadrant")
                with q2:
                    if expensive_weak:
                        st.error(f"🔴 Expensive + Weak: {', '.join(expensive_weak)}")
        else:
            st.info("Market cap data needed for bubble sizing.")

    # ══ HEATMAP ══════════════════════════════════════════════════════════════
    with tab_heatmap:
        st.caption(
            "Percentile rank heatmap. Green = cheap/strong (top percentile), "
            "Red = expensive/weak (bottom percentile). "
            "Shows at a glance who is best and worst on each metric."
        )

        hm_metrics = [m for m in [
            "pe_ratio", "price_to_book", "ev_to_ebitda",
            "profit_margin_%", "roe_%", "dividend_yield_%",
            "beta", "earnings_growth_yoy",
        ] if m in df.columns]

        if hm_metrics:
            # Ensure percentile columns exist
            df_hm = _add_percentile_ranks(df, hm_metrics)
            fig_hm = _heatmap_chart(df_hm, hm_metrics)
            if fig_hm.data:
                st.plotly_chart(fig_hm, use_container_width=True)

            # Overall cheapness score
            st.divider()
            st.subheader("Overall Value Score (avg percentile rank)")
            pct_cols = [f"{m}_pct" for m in hm_metrics if f"{m}_pct" in df_hm.columns]
            if pct_cols and "symbol" in df_hm.columns:
                df_score = df_hm[["symbol"] + pct_cols].copy()
                df_score["avg_score"] = df_score[pct_cols].mean(axis=1).round(1)
                df_score = df_score[["symbol", "avg_score"]].sort_values("avg_score", ascending=False)

                # Add verdict
                def get_verdict(score):
                    if score >= 75: return "CHEAP 🟢"
                    elif score >= 55: return "FAIR 🟡"
                    elif score >= 40: return "NEUTRAL ⚪"
                    elif score >= 25: return "EXPENSIVE 🟠"
                    else: return "VERY EXP. 🔴"

                df_score["Verdict"] = df_score["avg_score"].apply(get_verdict)
                df_score.columns = ["Symbol", "Value Score (0-100)", "Verdict"]

                def color_score(val):
                    try:
                        v = float(val)
                        if v >= 75: return "color:#00cc66;font-weight:bold"
                        if v >= 55: return "color:#66cc88"
                        if v >= 40: return "color:#888"
                        if v >= 25: return "color:#ffaa00"
                        return "color:#ff4444"
                    except (TypeError, ValueError):
                        return ""

                st.dataframe(
                    df_score.style.map(color_score, subset=["Value Score (0-100)"]),
                    use_container_width=True, hide_index=True,
                )
        else:
            st.info("No percentile-rankable metrics available.")

    # ══ SCREENER ════════════════════════════════════════════════════════════
    with tab_screener:
        st.caption("Filter for undervalued stocks using valuation criteria")

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            max_pe   = st.number_input("Max P/E ratio",   0.0, 200.0, 30.0, 5.0)
            min_roe  = st.number_input("Min ROE %",       0.0, 100.0, 10.0, 5.0)
        with sc2:
            max_pb   = st.number_input("Max P/B ratio",   0.0, 20.0,   5.0, 0.5)
            min_marg = st.number_input("Min Net Margin %", 0.0, 100.0,  5.0, 1.0)
        with sc3:
            max_beta = st.number_input("Max Beta",        0.0,  5.0,   1.5, 0.1)
            min_div  = st.number_input("Min Div Yield %", 0.0, 20.0,   0.0, 0.5)

        if st.button("🔍 Run Screener", type="primary"):
            df_screen = df.copy()
            filters_applied = []

            if "pe_ratio" in df_screen.columns and max_pe > 0:
                col = pd.to_numeric(df_screen["pe_ratio"], errors="coerce")
                df_screen = df_screen[(col <= max_pe) | col.isna()]
                filters_applied.append(f"P/E ≤ {max_pe}")

            if "price_to_book" in df_screen.columns and max_pb > 0:
                col = pd.to_numeric(df_screen["price_to_book"], errors="coerce")
                df_screen = df_screen[(col <= max_pb) | col.isna()]
                filters_applied.append(f"P/B ≤ {max_pb}")

            if "roe_%" in df_screen.columns and min_roe > 0:
                col = pd.to_numeric(df_screen["roe_%"], errors="coerce")
                df_screen = df_screen[(col >= min_roe) | col.isna()]
                filters_applied.append(f"ROE ≥ {min_roe}%")

            if "profit_margin_%" in df_screen.columns and min_marg > 0:
                col = pd.to_numeric(df_screen["profit_margin_%"], errors="coerce")
                df_screen = df_screen[(col >= min_marg) | col.isna()]
                filters_applied.append(f"Margin ≥ {min_marg}%")

            if "beta" in df_screen.columns and max_beta < 5:
                col = pd.to_numeric(df_screen["beta"], errors="coerce")
                df_screen = df_screen[(col <= max_beta) | col.isna()]
                filters_applied.append(f"Beta ≤ {max_beta}")

            if "dividend_yield_%" in df_screen.columns and min_div > 0:
                col = pd.to_numeric(df_screen["dividend_yield_%"], errors="coerce")
                df_screen = df_screen[(col >= min_div) | col.isna()]
                filters_applied.append(f"Div ≥ {min_div}%")

            st.session_state["vc_screen_result"] = df_screen
            st.session_state["vc_screen_filters"] = filters_applied

        results = st.session_state.get("vc_screen_result", pd.DataFrame())
        filters = st.session_state.get("vc_screen_filters", [])

        if not results.empty:
            st.success(f"✅ {len(results)} stocks passed: {' | '.join(filters)}")
            show_r = [c for c in [
                "symbol", "name", "pe_ratio", "price_to_book",
                "ev_to_ebitda", "profit_margin_%", "roe_%",
                "dividend_yield_%", "beta", "analyst_target",
            ] if c in results.columns]
            df_r = results[show_r].copy()
            df_r.columns = [METRIC_LABELS.get(c, c) for c in df_r.columns]
            st.dataframe(df_r, use_container_width=True, hide_index=True)
        elif filters:
            st.warning("No stocks matched all filters. Relax the criteria.")
        else:
            st.info("Set filters and click 'Run Screener'.")