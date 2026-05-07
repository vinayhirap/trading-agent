# trading-agent — Alerts page + integration snippets for app.py
# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Copy src/alerts/price_alert_manager.py  ← price_alert_manager.py
#
# STEP 2: Add cached resource to app.py:
#
# @st.cache_resource
# def get_alert_manager():
#     try:
#         from src.alerts.price_alert_manager import alert_manager
#         alert_manager.start()
#         return alert_manager
#     except Exception:
#         return None
#
# STEP 3: Call once near top of app.py (outside page blocks):
#   get_alert_manager()
#
# STEP 4: "🔔 Alerts" already in sidebar — paste the elif block below
# ═══════════════════════════════════════════════════════════════════════════════
import streamlit as st
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def render_alerts_page(get_alert_manager, get_price_store=None):
    """
    Full Alerts dashboard page.
    Call as: render_alerts_page(get_alert_manager)
    Or paste the elif block directly into app.py page routing.
    """
    st.header("🔔 Price Alerts")
    st.caption("Set price targets and RSI alerts — fires instantly to Telegram")

    am = get_alert_manager()
    if am is None:
        st.error("Alert manager failed to load. Check src/alerts/price_alert_manager.py")
        st.stop()

    # ── KPI row ────────────────────────────────────────────────────────────────
    active_alerts = am.get_alerts()
    history_all   = am.get_history(limit=999)
    today_str     = datetime.now(IST).strftime("%d %b")
    fired_today   = sum(1 for h in history_all if today_str in h.get("fired_at", ""))

    k1, k2, k3 = st.columns(3)
    k1.metric("Active Alerts",  len(active_alerts))
    k2.metric("Fired Today",    fired_today)
    k3.metric("All Time Fired", len(history_all))

    st.divider()

    # ── Add new alert ──────────────────────────────────────────────────────────
    st.subheader("➕ Add New Alert")

    try:
        from src.data.models import ALL_SYMBOLS
        sym_list = sorted(ALL_SYMBOLS.keys())
    except Exception:
        sym_list = [
            "NIFTY50","BANKNIFTY","RELIANCE","TCS","HDFCBANK",
            "GOLD","CRUDEOIL","BTC","USDINR","SBIN","INFY",
        ]

    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        default_idx = sym_list.index("NIFTY50") if "NIFTY50" in sym_list else 0
        alert_sym = st.selectbox("Symbol", sym_list, index=default_idx, key="alert_sym")

    with col2:
        alert_type = st.selectbox(
            "Alert Type",
            ["ABOVE", "BELOW", "PCT_CHANGE", "RSI_OVERSOLD", "RSI_OVERBOUGHT"],
            key="alert_type",
        )

    with col3:
        if alert_type in ("RSI_OVERSOLD", "RSI_OVERBOUGHT"):
            label    = "RSI Level (0-100)"
            default  = 30.0 if alert_type == "RSI_OVERSOLD" else 70.0
            step     = 1.0
        elif alert_type == "PCT_CHANGE":
            label    = "% Move Trigger"
            default  = 2.0
            step     = 0.5
        else:
            label    = "Price Level (₹)"
            try:
                from src.streaming.price_store import price_store
                cur = price_store.get(alert_sym) or 0
                default = float(cur) if cur else 100.0
            except Exception:
                default = 100.0
            step = 1.0

        threshold = st.number_input(
            label, value=default, step=step, format="%.2f", key="alert_threshold"
        )

    col4, col5 = st.columns([3, 1])
    with col4:
        note = st.text_input(
            "Note (optional)",
            placeholder="e.g. Nifty key support, BTC breakout",
            key="alert_note",
        )
    with col5:
        fire_once = st.checkbox("Delete after firing", value=True, key="alert_fire_once")

    # Description
    descs = {
        "ABOVE":          f"🔔 Fires when {alert_sym} rises ABOVE ₹{threshold:,.1f}",
        "BELOW":          f"🔔 Fires when {alert_sym} falls BELOW ₹{threshold:,.1f}",
        "PCT_CHANGE":     f"🔔 Fires when {alert_sym} moves ±{threshold:.1f}% from current price",
        "RSI_OVERSOLD":   f"🔔 Fires when {alert_sym} RSI(14) ≤ {threshold:.0f} (oversold)",
        "RSI_OVERBOUGHT": f"🔔 Fires when {alert_sym} RSI(14) ≥ {threshold:.0f} (overbought)",
    }
    st.caption(descs.get(alert_type, ""))

    if st.button("🔔 Add Alert", type="primary", use_container_width=True):
        try:
            aid = am.add_alert(
                symbol     = alert_sym,
                alert_type = alert_type,
                threshold  = threshold,
                note       = note,
                fire_once  = fire_once,
            )
            st.success(f"✅ Alert `{aid}` added — will fire to Telegram when triggered.")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to add alert: {e}")

    st.divider()

    # ── Active alerts ──────────────────────────────────────────────────────────
    st.subheader(f"📋 Active Alerts ({len(active_alerts)})")

    if not active_alerts:
        st.info("No active alerts. Add one above ↑")
    else:
        # Header row
        h1, h2, h3, h4, h5 = st.columns([1.5, 2, 2, 3, 0.8])
        h1.markdown("**Symbol**")
        h2.markdown("**Type**")
        h3.markdown("**Threshold**")
        h4.markdown("**Note**")
        h5.markdown("**Del**")

        for alert in active_alerts:
            c1, c2, c3, c4, c5 = st.columns([1.5, 2, 2, 3, 0.8])
            c1.markdown(f"**{alert['symbol']}**")
            c2.markdown(alert["alert_type"])

            if alert["alert_type"] in ("RSI_OVERSOLD", "RSI_OVERBOUGHT"):
                thresh_str = f"RSI {alert['threshold']:.0f}"
            elif alert["alert_type"] == "PCT_CHANGE":
                thresh_str = f"±{alert['threshold']:.1f}%"
            else:
                thresh_str = f"₹{alert['threshold']:,.1f}"
            c3.code(thresh_str)
            c4.markdown(
                f"<small style='color:#aaa'>{alert.get('note','')[:45]}</small>",
                unsafe_allow_html=True,
            )
            if c5.button("🗑", key=f"del_{alert['id']}"):
                am.delete_alert(alert["id"])
                st.rerun()

        # Price distance table
        st.divider()
        st.caption("📏 Current price vs alert level:")
        dist_rows = []
        for alert in active_alerts[:10]:
            if alert["alert_type"] not in ("ABOVE", "BELOW"):
                continue
            try:
                from src.streaming.price_store import price_store
                cur = price_store.get(alert["symbol"]) or 0
                if cur:
                    dist = (cur - alert["threshold"]) / alert["threshold"] * 100
                    gap  = alert["threshold"] - cur if alert["alert_type"] == "ABOVE" else cur - alert["threshold"]
                    dist_rows.append({
                        "Symbol":  alert["symbol"],
                        "Type":    alert["alert_type"],
                        "Target":  f"₹{alert['threshold']:,.1f}",
                        "Current": f"₹{cur:,.1f}",
                        "Gap":     f"₹{abs(gap):,.1f}",
                        "Dist %":  f"{abs(dist):.1f}%",
                        "Status":  "🔴 CLOSE" if abs(dist) < 0.5 else ("🟡" if abs(dist) < 2 else "🟢"),
                    })
            except Exception:
                pass

        if dist_rows:
            df_dist = pd.DataFrame(dist_rows)
            st.dataframe(df_dist, use_container_width=True, hide_index=True)

    st.divider()

    # ── Alert history ──────────────────────────────────────────────────────────
    st.subheader("📜 Fired Alert History")

    history = am.get_history(limit=25)
    if not history:
        st.info("No alerts have fired yet.")
    else:
        rows = []
        for h in history:
            rows.append({
                "Fired At":  h.get("fired_at", ""),
                "Symbol":    h.get("symbol", ""),
                "Type":      h.get("alert_type", ""),
                "Threshold": h.get("threshold", ""),
                "Actual":    f"{float(h.get('actual_value', 0)):.2f}",
                "Note":      h.get("note", ""),
            })
        df_hist = pd.DataFrame(rows)
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

        if st.button("🗑️ Clear History", type="secondary"):
            am.clear_history()
            st.rerun()

    st.divider()

    # ── Actions row ────────────────────────────────────────────────────────────
    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        if st.button("🔄 Check Alerts Now", use_container_width=True):
            with st.spinner("Checking all alerts against live prices..."):
                fired = am.check_now()
            if fired:
                st.success(f"✅ {len(fired)} alert(s) triggered — check Telegram!")
                for f in fired:
                    st.info(f"🔔 {f.symbol} {f.alert_type} @ {f.actual_value:.2f}")
                st.rerun()
            else:
                st.info("No alerts triggered at current prices.")

    with col_btn2:
        if st.button("📱 Test Telegram", use_container_width=True):
            try:
                from src.alerts.telegram_sender import TelegramSender
                now = datetime.now(IST).strftime("%H:%M IST")
                ok  = TelegramSender().send_message(
                    f"🧪 <b>Alert System Test</b>\n"
                    f"Price Alert Manager is active! {now}"
                )
                if ok:
                    st.success("✅ Telegram test message sent!")
                else:
                    st.error("❌ Telegram send failed — check TELEGRAM_BOT_TOKEN in .env")
            except Exception as e:
                st.error(f"Telegram test error: {e}")


# ── app.py elif block (copy-paste) ────────────────────────────────────────────
ELIF_BLOCK = """
elif page == "🔔 Alerts":
    from src.dashboard.alerts_page import render_alerts_page
    render_alerts_page(get_alert_manager)
"""
# OR inline — replace the existing empty Alerts elif with render_alerts_page() call


if __name__ == "__main__":
    print("Copy this file to: src/dashboard/alerts_page.py")
    print("Then use: render_alerts_page(get_alert_manager) in app.py")