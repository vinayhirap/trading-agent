# trading-agent/src/dashboard/pages/nerve_center.py
"""
Global Finance Nerve Center — Bloomberg BICO/ALTD + Nerve Center Guide

Implements the full Nerve Center architecture from the guide:
  1. News ingestion (NewsAPI + RSS feeds already in news_fetcher.py)
  2. Anthropic AI impact analysis per headline
  3. Color-coded heatmap across markets, sectors, assets
  4. Morning Analyst Brief (auto-generated)
  5. Shock Feed ranked by magnitude, not recency

Impact scoring:
  -5 = severe negative   (deep red #8B0000)
  -3 = significant neg   (red #cc3333)
  -1 = mild negative     (light red #ffaaaa)
   0 = neutral           (gray)
  +1 = mild positive     (light green)
  +3 = significant pos   (green)
  +5 = strong positive   (deep green)

Usage in app.py:
    elif page == "🧠 Nerve Center":
        from src.dashboard.pages.nerve_center import render_nerve_center
        render_nerve_center()
"""
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
import streamlit as st
import pandas as pd
import requests
from loguru import logger

# ── Constants ─────────────────────────────────────────────────────────────────

MARKETS  = ["US", "India", "China", "EU", "Japan", "UK", "EM", "Middle_East"]
SECTORS  = ["Banking", "Technology", "Pharma", "Energy", "Metals",
            "Real_Estate", "FMCG", "Auto", "Telecom", "Infrastructure"]
ASSETS   = ["Equities", "Bonds", "Gold", "Crude_Oil", "USD_Index",
            "Crypto", "INR_USD", "VIX"]

NEWS_TYPES = [
    "Monetary Policy", "Geopolitical", "Earnings", "Regulatory",
    "Commodity Shock", "Trade Policy", "Currency", "Tech Disruption",
]

CACHE_DIR = Path("data/cache/nerve_center")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Color scale for heatmap cells ─────────────────────────────────────────────

def score_to_color(score: Optional[float], alpha: float = 0.85) -> str:
    if score is None:
        return f"rgba(50,50,50,{alpha})"
    if score >= 4:   return f"rgba(0,120,0,{alpha})"
    if score >= 2:   return f"rgba(0,180,0,{alpha})"
    if score >= 0.5: return f"rgba(100,200,100,{alpha})"
    if score >= -0.5:return f"rgba(80,80,80,{alpha})"
    if score >= -2:  return f"rgba(220,80,80,{alpha})"
    if score >= -4:  return f"rgba(180,0,0,{alpha})"
    return                   f"rgba(100,0,0,{alpha})"


def score_to_text_color(score: Optional[float]) -> str:
    if score is None:
        return "#888"
    return "#ffffff" if abs(score) > 0.5 else "#aaa"


def score_to_arrow(score: Optional[float]) -> str:
    if score is None or abs(score) < 0.5:
        return "—"
    if score >= 3:   return "▲▲"
    if score >= 1:   return "▲"
    if score <= -3:  return "▼▼"
    return                   "▼"


# ── AI Impact Analyzer ───────────────────────────────────────────────────────

IMPACT_SYSTEM_PROMPT = """You are a Global Markets Intelligence Analyst. The user will give you news headlines.
For EACH headline, return a JSON object with exactly this structure:

{
  "headline": "the news headline",
  "source": "source name",
  "type": "Monetary Policy | Geopolitical | Earnings | Regulatory | Commodity Shock | Trade Policy | Currency | Tech Disruption",
  "impacts": {
    "markets": {"US": 0, "India": 0, "China": 0, "EU": 0, "Japan": 0, "UK": 0, "EM": 0, "Middle_East": 0},
    "sectors": {"Banking": 0, "Technology": 0, "Pharma": 0, "Energy": 0, "Metals": 0, "Real_Estate": 0, "FMCG": 0, "Auto": 0, "Telecom": 0, "Infrastructure": 0},
    "assets":  {"Equities": 0, "Bonds": 0, "Gold": 0, "Crude_Oil": 0, "USD_Index": 0, "Crypto": 0, "INR_USD": 0, "VIX": 0}
  },
  "analysis": "2-3 sentence explanation of impact chains including 2nd and 3rd order effects",
  "analyst_brief": "Specific actionable tasks an analyst at a top bank would do because of this news"
}

Scores: -5=severe negative, -3=significant negative, -1=mild negative, 0=neutral, +1=mild positive, +3=significant positive, +5=strong positive.

RULES:
- Think about 2nd and 3rd order effects, not just obvious ones
- India-specific: always consider INR, FPI flows, RBI response
- Be specific in analysis — name companies, not just sectors
- Analyst brief must have ACTIONABLE tasks, not observations
- Return ONLY valid JSON array, no markdown, no preamble"""


def _analyze_headlines_with_ai(headlines: list[dict], api_key: str) -> list[dict]:
    """
    Call Anthropic API to analyze headlines.
    headlines: list of dicts with 'title', 'source' keys
    Returns list of impact dicts.
    """
    if not headlines or not api_key:
        return []

    headlines_text = "\n".join(
        f"- [{h.get('source','Unknown')}] {h['title']}"
        for h in headlines[:10]  # cap at 10 per call
    )

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         api_key,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":      "claude-sonnet-4-20250514",
                "max_tokens": 4000,
                "system":     IMPACT_SYSTEM_PROMPT,
                "messages": [{
                    "role":    "user",
                    "content": f"Analyze these headlines:\n{headlines_text}\n\nReturn a JSON array of impact objects, one per headline.",
                }],
            },
            timeout=45,
        )
        resp.raise_for_status()
        content = resp.json()["content"][0]["text"].strip()

        # Strip markdown fences if present
        content = content.replace("```json", "").replace("```", "").strip()

        # Parse JSON
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            parsed = [parsed]
        return parsed

    except json.JSONDecodeError as e:
        logger.error(f"Nerve Center: AI response not valid JSON: {e}")
        return []
    except Exception as e:
        logger.error(f"Nerve Center AI call failed: {e}")
        return []


def _load_cached_analysis(cache_key: str) -> Optional[list]:
    path = CACHE_DIR / f"{cache_key}.json"
    if path.exists():
        age_mins = (time.time() - path.stat().st_mtime) / 60
        if age_mins < 30:  # 30-minute cache
            try:
                return json.loads(path.read_text())
            except Exception:
                pass
    return None


def _save_analysis_cache(cache_key: str, data: list):
    path = CACHE_DIR / f"{cache_key}.json"
    try:
        path.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logger.debug(f"Cache save failed: {e}")


def _get_news_items(news_api_key: str, n: int = 15) -> list[dict]:
    """
    Fetch news from multiple sources for Nerve Center.
    Returns list of {'title': ..., 'source': ..., 'url': ...}
    """
    items = []

    # 1. NewsAPI
    if news_api_key:
        try:
            resp = requests.get(
                "https://newsapi.org/v2/top-headlines",
                params={
                    "category": "business",
                    "language": "en",
                    "pageSize": min(n, 10),
                    "apiKey":   news_api_key,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                for a in resp.json().get("articles", []):
                    title = a.get("title", "").strip()
                    if title and "[Removed]" not in title:
                        items.append({
                            "title":  title,
                            "source": a.get("source", {}).get("name", "NewsAPI"),
                            "url":    a.get("url", ""),
                        })
        except Exception as e:
            logger.warning(f"NewsAPI fetch failed: {e}")

    # 2. RSS fallback — Economic Times + Moneycontrol
    if len(items) < 5:
        try:
            import feedparser
            rss_feeds = [
                ("https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms", "Economic Times"),
                ("https://www.moneycontrol.com/rss/marketoutlook.xml", "Moneycontrol"),
            ]
            for url, source in rss_feeds:
                try:
                    feed = feedparser.parse(url)
                    for entry in feed.entries[:5]:
                        title = entry.get("title", "").strip()
                        if title:
                            items.append({
                                "title":  title,
                                "source": source,
                                "url":    entry.get("link", ""),
                            })
                except Exception:
                    continue
        except ImportError:
            pass

    # Deduplicate
    seen = set()
    unique = []
    for item in items:
        key = item["title"][:50].lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)

    return unique[:n]


# ── Morning Brief Generator ────────────────────────────────────────────────────

def _generate_morning_brief(impacts: list[dict]) -> dict:
    """
    From list of impact dicts, extract top 3 by magnitude + net sentiment.
    """
    if not impacts:
        return {"top_stories": [], "net_sentiment": "NEUTRAL", "risk_mode": "RISK-ON"}

    # Score each impact by total absolute magnitude
    scored = []
    for imp in impacts:
        total_abs = 0
        for category in ["markets", "sectors", "assets"]:
            for v in (imp.get("impacts", {}).get(category, {}) or {}).values():
                try:
                    total_abs += abs(float(v))
                except (TypeError, ValueError):
                    pass
        scored.append((total_abs, imp))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_3 = [imp for _, imp in scored[:3]]

    # Net sentiment: sum of all India + Equities scores
    india_scores = []
    equity_scores = []
    for imp in impacts:
        markets = imp.get("impacts", {}).get("markets", {}) or {}
        assets  = imp.get("impacts", {}).get("assets",  {}) or {}
        try:
            india_scores.append(float(markets.get("India", 0)))
        except (TypeError, ValueError):
            pass
        try:
            equity_scores.append(float(assets.get("Equities", 0)))
        except (TypeError, ValueError):
            pass

    avg_india  = sum(india_scores)  / len(india_scores)  if india_scores  else 0
    avg_equity = sum(equity_scores) / len(equity_scores) if equity_scores else 0
    net_score  = (avg_india + avg_equity) / 2

    if net_score > 1.0:
        net_label = "BULLISH"
        risk_mode = "RISK-ON"
    elif net_score > 0.2:
        net_label = "MILDLY BULLISH"
        risk_mode = "RISK-ON"
    elif net_score < -1.0:
        net_label = "BEARISH"
        risk_mode = "RISK-OFF"
    elif net_score < -0.2:
        net_label = "MILDLY BEARISH"
        risk_mode = "RISK-OFF"
    else:
        net_label = "NEUTRAL"
        risk_mode = "NEUTRAL"

    return {
        "top_stories": top_3,
        "net_sentiment": net_label,
        "net_score": round(net_score, 2),
        "risk_mode": risk_mode,
        "total_stories": len(impacts),
    }


# ── Aggregate impact across all stories ───────────────────────────────────────

def _aggregate_impacts(impacts: list[dict]) -> dict:
    """Sum all scores across all headlines to get net aggregate view."""
    agg = {
        "markets": {m: 0.0 for m in MARKETS},
        "sectors": {s: 0.0 for s in SECTORS},
        "assets":  {a: 0.0 for a in ASSETS},
    }
    count = len(impacts)
    if count == 0:
        return agg

    for imp in impacts:
        for cat in ["markets", "sectors", "assets"]:
            for key, val in (imp.get("impacts", {}).get(cat, {}) or {}).items():
                if key in agg[cat]:
                    try:
                        agg[cat][key] += float(val)
                    except (TypeError, ValueError):
                        pass

    # Normalize to average
    for cat in agg:
        for key in agg[cat]:
            agg[cat][key] = round(agg[cat][key] / count, 2)

    return agg


# ── HTML Heatmap renderer ──────────────────────────────────────────────────────

def _render_heatmap_html(data: dict, title: str) -> str:
    """
    Generate HTML heatmap table for markets/sectors/assets.
    data: {label: score} dict
    """
    cells = ""
    for label, score in data.items():
        bg    = score_to_color(score)
        tc    = score_to_text_color(score)
        arrow = score_to_arrow(score)
        score_str = f"{score:+.1f}" if score is not None else "—"
        display_label = label.replace("_", " ")
        cells += f"""
        <div style="background:{bg};color:{tc};border-radius:6px;padding:10px 8px;
                    text-align:center;min-width:90px;border:1px solid rgba(255,255,255,0.08)">
          <div style="font-size:11px;opacity:0.85;margin-bottom:3px">{display_label}</div>
          <div style="font-size:18px;font-weight:700">{arrow}</div>
          <div style="font-size:12px;margin-top:2px">{score_str}</div>
        </div>"""

    return f"""
    <div style="margin-bottom:8px;font-size:11px;color:#888;letter-spacing:1px;
                text-transform:uppercase">{title}</div>
    <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:20px">{cells}</div>"""


# ── Main render function ───────────────────────────────────────────────────────

def render_nerve_center():
    """
    Main entry point. Call from app.py:
        from src.dashboard.pages.nerve_center import render_nerve_center
        render_nerve_center()
    """
    st.header("🧠 Global Finance Nerve Center")
    st.caption(
        "Real-time news → AI impact analysis → Market heatmap. "
        "Powered by Anthropic Claude. Stories ranked by impact magnitude, not recency."
    )

    # ── Settings ──────────────────────────────────────────────────────────────
    try:
        from config.settings import settings
        api_key      = getattr(settings, "ANTHROPIC_API_KEY", None)
        news_api_key = getattr(settings, "NEWS_API_KEY", None)
    except Exception:
        api_key      = None
        news_api_key = None

    if not api_key:
        st.error(
            "ANTHROPIC_API_KEY not configured. "
            "Add it to your .env file to enable AI impact analysis."
        )
        st.info("You can still view the Shock Feed with manual headlines below.")

    # ── Controls ──────────────────────────────────────────────────────────────
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 1, 1])
    with col_ctrl1:
        view_mode = st.radio(
            "View",
            ["Impact Map", "Shock Feed", "Morning Brief", "Manual Analysis"],
            horizontal=True,
        )
    with col_ctrl2:
        n_headlines = st.selectbox("Headlines", [5, 10, 15, 20], index=1)
    with col_ctrl3:
        force_refresh = st.button("🔄 Refresh News", type="primary")

    st.divider()

    # ── Fetch + analyze news ──────────────────────────────────────────────────
    cache_key = f"nc_{datetime.now().strftime('%Y%m%d_%H')}"

    if force_refresh:
        # Clear cache
        for f in CACHE_DIR.glob(f"{cache_key}*.json"):
            f.unlink(missing_ok=True)

    with st.spinner("Fetching headlines and running AI impact analysis..."):
        # Try cache first
        impacts = _load_cached_analysis(cache_key)

        if not impacts:
            # Fetch news
            news_items = _get_news_items(news_api_key, n=n_headlines)

            if not news_items:
                st.warning("No news fetched. Check NEWS_API_KEY or internet connection.")
                news_items = []

            # AI analysis
            if news_items and api_key:
                impacts = _analyze_headlines_with_ai(news_items, api_key)
                if impacts:
                    _save_analysis_cache(cache_key, impacts)
            else:
                impacts = []

    if not impacts and view_mode != "Manual Analysis":
        st.info(
            "No analysis yet. Click 'Refresh News' or use Manual Analysis tab "
            "to paste headlines directly."
        )

    # ── KPI bar ───────────────────────────────────────────────────────────────
    brief = _generate_morning_brief(impacts)
    agg   = _aggregate_impacts(impacts)

    if impacts:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Stories Analyzed", len(impacts))
        k2.metric("Net Sentiment",    brief["net_sentiment"])
        k3.metric("Market Regime",    brief["risk_mode"])

        # Most impacted market
        if agg["markets"]:
            top_market = max(agg["markets"], key=lambda k: abs(agg["markets"][k]))
            k4.metric("Most Impacted", top_market.replace("_", " "),
                      f"{agg['markets'][top_market]:+.1f}")

    # ══════════════════════════════════════════════
    # VIEW: IMPACT MAP
    # ══════════════════════════════════════════════
    if view_mode == "Impact Map":
        if not impacts:
            st.info("No data to display.")
        else:
            # Story selector
            story_options = ["AGGREGATE (all stories)"] + [
                f"{imp.get('headline','?')[:70]}..." for imp in impacts
            ]
            selected_story = st.selectbox("Select story to isolate (or view aggregate)", story_options)

            if selected_story == "AGGREGATE (all stories)":
                display_impacts = agg
                st.caption("Showing **net aggregate impact** of all analyzed stories.")
            else:
                idx = story_options.index(selected_story) - 1
                sel = impacts[idx]
                display_impacts = sel.get("impacts", {})
                # Show analysis for selected story
                analysis = sel.get("analysis", "")
                brief_text = sel.get("analyst_brief", "")
                if analysis:
                    st.info(f"📊 **Analysis:** {analysis}")
                if brief_text:
                    st.success(f"🎯 **Analyst Action:** {brief_text}")

            # Render heatmaps
            markets_html = _render_heatmap_html(
                {m: display_impacts.get("markets", {}).get(m) for m in MARKETS},
                "🌍 Markets"
            )
            sectors_html = _render_heatmap_html(
                {s: display_impacts.get("sectors", {}).get(s) for s in SECTORS},
                "🏭 Sectors"
            )
            assets_html = _render_heatmap_html(
                {a: display_impacts.get("assets", {}).get(a) for a in ASSETS},
                "📊 Assets"
            )

            legend_html = """
            <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px;font-size:11px">
              <span>Scale: </span>
              <span style="background:rgba(100,0,0,0.8);color:white;padding:2px 8px;border-radius:3px">▼▼ Severe (-5)</span>
              <span style="background:rgba(180,0,0,0.8);color:white;padding:2px 8px;border-radius:3px">▼ Negative (-3)</span>
              <span style="background:rgba(80,80,80,0.8);color:white;padding:2px 8px;border-radius:3px">— Neutral</span>
              <span style="background:rgba(0,180,0,0.8);color:white;padding:2px 8px;border-radius:3px">▲ Positive (+3)</span>
              <span style="background:rgba(0,120,0,0.8);color:white;padding:2px 8px;border-radius:3px">▲▲ Strong (+5)</span>
            </div>"""

            st.markdown(
                legend_html + markets_html + sectors_html + assets_html,
                unsafe_allow_html=True,
            )

    # ══════════════════════════════════════════════
    # VIEW: SHOCK FEED
    # ══════════════════════════════════════════════
    elif view_mode == "Shock Feed":
        st.caption("Stories ranked by **total impact magnitude** — most market-moving on top.")

        if not impacts:
            st.info("No stories analyzed yet.")
        else:
            # Sort by magnitude
            def _magnitude(imp: dict) -> float:
                total = 0
                for cat in ["markets", "sectors", "assets"]:
                    for v in (imp.get("impacts", {}).get(cat, {}) or {}).values():
                        try:
                            total += abs(float(v))
                        except (TypeError, ValueError):
                            pass
                return total

            sorted_impacts = sorted(impacts, key=_magnitude, reverse=True)

            for i, imp in enumerate(sorted_impacts):
                mag    = _magnitude(imp)
                story  = imp.get("headline", "Unknown")
                source = imp.get("source", "")
                ntype  = imp.get("type", "")
                analysis = imp.get("analysis", "")

                # Top affected markets/assets as colored tags
                all_scores = {}
                for cat in ["markets", "sectors", "assets"]:
                    for k, v in (imp.get("impacts", {}).get(cat, {}) or {}).items():
                        try:
                            all_scores[k] = float(v)
                        except (TypeError, ValueError):
                            pass

                top_affected = sorted(
                    [(k, v) for k, v in all_scores.items() if abs(v) >= 2],
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )[:6]

                tags_html = ""
                for label, score in top_affected:
                    bg    = score_to_color(score, 0.7)
                    arrow = score_to_arrow(score)
                    tags_html += f'<span style="background:{bg};color:white;padding:2px 8px;border-radius:3px;font-size:11px;margin-right:4px">{arrow} {label.replace("_"," ")}</span>'

                magnitude_bar_pct = min(mag / 50 * 100, 100)
                mag_color = "#ff4444" if mag > 30 else "#ffaa44" if mag > 15 else "#44aaff"

                with st.expander(f"#{i+1} [{ntype}] {story[:100]}... | Magnitude: {mag:.0f}"):
                    st.markdown(f"**Source:** {source} | **Type:** {ntype}")
                    st.progress(magnitude_bar_pct / 100, text=f"Impact magnitude: {mag:.0f}")
                    if tags_html:
                        st.markdown(f"**Key impacts:** {tags_html}", unsafe_allow_html=True)
                    if analysis:
                        st.info(f"**Analysis:** {analysis}")
                    brief_text = imp.get("analyst_brief", "")
                    if brief_text:
                        st.success(f"**Analyst action:** {brief_text}")

                    # Mini heatmap for this story
                    imp_data = imp.get("impacts", {})
                    mini_html = _render_heatmap_html(
                        {m: imp_data.get("markets", {}).get(m) for m in MARKETS[:4]},
                        "Key Markets"
                    )
                    st.markdown(mini_html, unsafe_allow_html=True)

    # ══════════════════════════════════════════════
    # VIEW: MORNING ANALYST BRIEF
    # ══════════════════════════════════════════════
    elif view_mode == "Morning Brief":
        ist_time = datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC")

        # Risk mode banner
        risk_colors = {
            "RISK-ON":  ("#003300", "#00ff88"),
            "RISK-OFF": ("#330000", "#ff4444"),
            "NEUTRAL":  ("#1a1a2e", "#8888ff"),
        }
        bg_c, fg_c = risk_colors.get(brief["risk_mode"], ("#1a1a2e", "#ffffff"))
        net_score  = brief.get("net_score", 0)

        st.markdown(
            f'<div style="background:{bg_c};border:1px solid {fg_c};border-radius:8px;'
            f'padding:16px 20px;margin-bottom:16px">'
            f'<div style="font-size:11px;letter-spacing:2px;color:{fg_c};opacity:0.7">MORNING BRIEF — {ist_time}</div>'
            f'<div style="font-size:24px;font-weight:700;color:{fg_c};margin:4px 0">'
            f'{brief["risk_mode"]} | {brief["net_sentiment"]}</div>'
            f'<div style="font-size:13px;color:{fg_c};opacity:0.8">'
            f'Net India/Equity score: {net_score:+.2f} | '
            f'Based on {brief["total_stories"]} analyzed stories</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        top_stories = brief.get("top_stories", [])
        if not top_stories:
            st.info("No stories analyzed yet. Click 'Refresh News'.")
        else:
            st.subheader("3 Things That Matter Before Your Morning Meeting")

            for i, story in enumerate(top_stories[:3]):
                headline  = story.get("headline", "")
                source    = story.get("source", "")
                ntype     = story.get("type", "")
                analysis  = story.get("analysis", "")
                brief_txt = story.get("analyst_brief", "")

                # India + Equities score for this story
                india_score  = float((story.get("impacts", {}).get("markets", {}) or {}).get("India", 0))
                equity_score = float((story.get("impacts", {}).get("assets",  {}) or {}).get("Equities", 0))
                story_bg     = score_to_color((india_score + equity_score) / 2, 0.3)

                st.markdown(
                    f'<div style="border-left:3px solid {score_to_color((india_score+equity_score)/2)};'
                    f'padding:12px 16px;margin:8px 0;background:{story_bg};border-radius:0 6px 6px 0">'
                    f'<div style="font-size:11px;color:#888;letter-spacing:1px">{i+1}. {ntype} | {source}</div>'
                    f'<div style="font-size:15px;font-weight:600;margin:4px 0">{headline[:120]}</div>'
                    f'<div style="font-size:13px;color:#ccc;margin-top:6px">{analysis}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                if brief_txt:
                    st.caption(f"🎯 **Action:** {brief_txt}")

                # Impact tags for this story
                top_impacts = {}
                for cat in ["markets", "sectors", "assets"]:
                    for k, v in (story.get("impacts", {}).get(cat, {}) or {}).items():
                        try:
                            top_impacts[k] = float(v)
                        except (TypeError, ValueError):
                            pass

                top_tags = sorted(
                    [(k, v) for k, v in top_impacts.items() if abs(v) >= 2],
                    key=lambda x: abs(x[1]), reverse=True,
                )[:5]
                if top_tags:
                    tags = " ".join(
                        f'<span style="background:{score_to_color(v,0.7)};color:white;'
                        f'padding:2px 7px;border-radius:3px;font-size:10px;margin-right:3px">'
                        f'{score_to_arrow(v)} {k.replace("_"," ")}</span>'
                        for k, v in top_tags
                    )
                    st.markdown(tags, unsafe_allow_html=True)

                st.divider()

    # ══════════════════════════════════════════════
    # VIEW: MANUAL ANALYSIS
    # ══════════════════════════════════════════════
    elif view_mode == "Manual Analysis":
        st.caption("Paste today's headlines to run AI impact analysis without waiting for news fetch.")

        manual_text = st.text_area(
            "Paste headlines (one per line):",
            height=200,
            placeholder=(
                "Fed keeps rates unchanged, signals 2 cuts in 2025\n"
                "RBI holds repo rate at 6.5%, changes stance to neutral\n"
                "Crude oil rises 3% on Middle East tensions\n"
                "India Q3 GDP beats estimates at 7.2%"
            ),
        )

        if st.button("Analyze Headlines", type="primary"):
            if not manual_text.strip():
                st.warning("Paste at least one headline.")
            elif not api_key:
                st.error("ANTHROPIC_API_KEY required for AI analysis.")
            else:
                lines = [l.strip() for l in manual_text.strip().split("\n") if l.strip()]
                headline_dicts = [{"title": l, "source": "Manual"} for l in lines]

                with st.spinner(f"Analyzing {len(lines)} headlines..."):
                    manual_impacts = _analyze_headlines_with_ai(headline_dicts, api_key)

                if manual_impacts:
                    st.session_state["nc_manual_impacts"] = manual_impacts
                    st.success(f"✅ {len(manual_impacts)} headlines analyzed.")
                else:
                    st.error("Analysis failed. Check API key and try again.")

        # Display manual results
        manual_impacts = st.session_state.get("nc_manual_impacts", [])
        if manual_impacts:
            st.divider()
            manual_agg = _aggregate_impacts(manual_impacts)
            manual_brief = _generate_morning_brief(manual_impacts)

            st.markdown(f"**Net Sentiment:** {manual_brief['net_sentiment']} | **Regime:** {manual_brief['risk_mode']}")

            markets_html = _render_heatmap_html(
                {m: manual_agg.get("markets", {}).get(m) for m in MARKETS},
                "🌍 Markets Impact"
            )
            sectors_html = _render_heatmap_html(
                {s: manual_agg.get("sectors", {}).get(s) for s in SECTORS},
                "🏭 Sectors Impact"
            )
            assets_html = _render_heatmap_html(
                {a: manual_agg.get("assets", {}).get(a) for a in ASSETS},
                "📊 Assets Impact"
            )
            st.markdown(markets_html + sectors_html + assets_html, unsafe_allow_html=True)

            st.divider()
            st.subheader("Story Analysis")
            for imp in manual_impacts:
                with st.expander(imp.get("headline", "")[:80]):
                    st.markdown(f"**Type:** {imp.get('type','')}")
                    st.info(imp.get("analysis", ""))
                    if imp.get("analyst_brief"):
                        st.success(f"**Action:** {imp['analyst_brief']}")