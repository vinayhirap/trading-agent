"""
patch_prices.py — Fix price ticking speed and consistency

Problems fixed:
1. Prices stuck at 5-6s age → target <1s
2. Price mismatch between header ticker and signal cards
3. FALLBACK_REFRESH_SECONDS=10 causing yfinance block on every get()
4. Signal card prices use stale candle close, not live price_store price

Run from project root:
  python patch_prices.py
"""
from pathlib import Path

ROOT = Path(".")

# ── Fix 1: price_store.py ─────────────────────────────────────────────────────
def fix_price_store():
    path = ROOT / "src/streaming/price_store.py"
    if not path.exists():
        print("[price_store] NOT FOUND"); return

    src = path.read_text(encoding="utf-8")
    changed = False

    # 1a. FALLBACK_REFRESH_SECONDS: 10 → 2
    if "FALLBACK_REFRESH_SECONDS = 10" in src:
        src = src.replace("FALLBACK_REFRESH_SECONDS = 10", "FALLBACK_REFRESH_SECONDS = 2")
        print("[price_store] ✅ FALLBACK_REFRESH_SECONDS 10→2")
        changed = True
    elif "FALLBACK_REFRESH_SECONDS = 2" in src:
        print("[price_store] FALLBACK_REFRESH_SECONDS already 2 — skip")
    else:
        print("[price_store] ⚠ FALLBACK_REFRESH_SECONDS not found")

    # 1b. Background refresh interval_seconds in start_background_refresh
    if "interval_seconds: int = 10" in src:
        src = src.replace("interval_seconds: int = 10", "interval_seconds: int = 3")
        print("[price_store] ✅ default interval_seconds 10→3")
        changed = True
    else:
        print("[price_store] default interval_seconds already changed — skip")

    if changed:
        path.write_text(src, encoding="utf-8")
        print("[price_store] ✅ saved")

# ── Fix 2: main.py ────────────────────────────────────────────────────────────
def fix_main():
    path = ROOT / "web/main.py"
    if not path.exists():
        print("[main.py] NOT FOUND"); return

    src = path.read_text(encoding="utf-8")
    changed = False

    # 2a. broadcast loop sleep: 1 → 0.1
    if "await asyncio.sleep(1)\n" in src and "price_broadcast_loop" in src:
        # Only replace the one in price_broadcast_loop
        src = src.replace(
            "        await asyncio.sleep(1)\n",
            "        await asyncio.sleep(0.1)\n",
            1
        )
        print("[main.py] ✅ broadcast sleep 1→0.1s")
        changed = True
    elif "await asyncio.sleep(0.1)" in src:
        print("[main.py] broadcast sleep already 0.1 — skip")
    else:
        print("[main.py] ⚠ broadcast sleep pattern not found")

    # 2b. start_background_refresh interval: 10 → 3
    if "interval_seconds=10" in src:
        src = src.replace("interval_seconds=10", "interval_seconds=3", 1)
        print("[main.py] ✅ background refresh 10→3s")
        changed = True
    elif "interval_seconds=3" in src:
        print("[main.py] background refresh already 3s — skip")

    # 2c. Fix PRICE_META fallback in _get_prices to use "none" not "usd_to_inr"
    old_meta = '{"convert": "usd_to_inr", "label": ""}'
    new_meta = '{"convert": "none", "label": ""}'
    if old_meta in src:
        src = src.replace(old_meta, new_meta)
        print("[main.py] ✅ PRICE_META fallback fixed to 'none'")
        changed = True
    else:
        print("[main.py] PRICE_META fallback already correct — skip")

    if changed:
        path.write_text(src, encoding="utf-8")
        print("[main.py] ✅ saved")

# ── Fix 3: realtime_advisor.py — price consistency ───────────────────────────
def fix_realtime_advisor():
    """
    Make signal card prices match header ticker by reading from price_store
    instead of using the candle close price.
    """
    path = ROOT / "src/analysis/realtime_advisor.py"
    if not path.exists():
        print("[realtime_advisor] NOT FOUND"); return

    src = path.read_text(encoding="utf-8")

    # 3a. BAR_CACHE_TTL: 20 → 5
    if "BAR_CACHE_TTL_SECONDS     = 20.0" in src:
        src = src.replace(
            "BAR_CACHE_TTL_SECONDS     = 20.0",
            "BAR_CACHE_TTL_SECONDS     = 5.0"
        )
        print("[realtime_advisor] ✅ BAR_CACHE_TTL 20→5s")
    else:
        print("[realtime_advisor] BAR_CACHE_TTL already changed — skip")

    # 3b. Use price_store live price instead of candle close for the advice.price field
    # Find where RealtimeAdvice is constructed and override price with price_store value
    marker = "# USE_LIVE_PRICE_PATCHED"
    if marker in src:
        print("[realtime_advisor] live price patch already applied — skip")
    else:
        old = "        return RealtimeAdvice(\n            symbol               = symbol,"
        new = """        # Use live price_store price for consistency with header ticker
        live_px = price_store.get(symbol, fallback=False)
        if live_px and live_px > 0:
            close = live_px  # overwrite candle close with live tick # USE_LIVE_PRICE_PATCHED

        return RealtimeAdvice(
            symbol               = symbol,"""
        if old in src:
            src = src.replace(old, new, 1)
            print("[realtime_advisor] ✅ live price_store price used for advice.price")
        else:
            print("[realtime_advisor] ⚠ RealtimeAdvice constructor pattern not found — skip")

    path.write_text(src, encoding="utf-8")
    print("[realtime_advisor] ✅ saved")

# ── Fix 4: base.html — JS price formatter consistency ───────────────────────
def fix_base_html():
    """
    The JS fmtDisplay() in base.html must match Python convert_price() exactly.
    Also ensure the WebSocket message handler updates ALL price elements on page.
    """
    path = ROOT / "web/templates/base.html"
    if not path.exists():
        print("[base.html] NOT FOUND"); return

    src = path.read_text(encoding="utf-8")

    # Ensure pingWS keeps connection alive so prices don't go stale
    if "setInterval(pingWS, 15000)" in src:
        src = src.replace("setInterval(pingWS, 15000)", "setInterval(pingWS, 5000)")
        print("[base.html] ✅ ping interval 15→5s")
    else:
        print("[base.html] ping already fast — skip")

    # Ensure wsDelay resets fast on reconnect
    if "wsDelay = Math.min(wsDelay * 1.5, 15000)" in src:
        src = src.replace(
            "wsDelay = Math.min(wsDelay * 1.5, 15000)",
            "wsDelay = Math.min(wsDelay * 1.5, 5000)"
        )
        print("[base.html] ✅ max WS reconnect delay 15→5s")

    path.write_text(src, encoding="utf-8")
    print("[base.html] ✅ saved")

# ── Fix 5: dashboard.html — signal card prices use WebSocket price ───────────
def fix_dashboard():
    path = ROOT / "web/templates/dashboard.html"
    if not path.exists():
        print("[dashboard.html] NOT FOUND"); return

    src = path.read_text(encoding="utf-8")

    # Check if signal card price is already using WebSocket price
    marker = "// SIG_CARD_LIVE_PRICE_PATCHED"
    if marker in src:
        print("[dashboard.html] signal card live price already patched — skip")
        return

    # Replace the price line in signal card update to use WebSocket prices
    old = """    const price = s.price ? fmtDisplay(sym, s.price * (["BTC","ETH","SOL"].includes(sym) ? (data.usdinr||92.46) : 1), "") : "—";"""
    new = """    // Use WebSocket live price for consistency with header ticker // SIG_CARD_LIVE_PRICE_PATCHED
    const wsInfo = (typeof latestPrices !== 'undefined' && latestPrices[sym]) ? latestPrices[sym] : null;
    const price = wsInfo ? fmtDisplay(sym, wsInfo.display, wsInfo.label) :
                  s.price ? fmtDisplay(sym, s.price * (["BTC","ETH","SOL"].includes(sym) ? (data.usdinr||92.46) : 1), "") : "—";"""

    if old in src:
        src = src.replace(old, new, 1)
        print("[dashboard.html] ✅ signal card uses WebSocket live price")
    else:
        print("[dashboard.html] ⚠ signal card price pattern not found — skip")
        return

    # Also make sure latestPrices is updated from WebSocket event
    if "let latestPrices = {};" not in src:
        old_kpi_prev = "let kpiPrev = {};"
        new_kpi_prev = "let kpiPrev = {};\nlet latestPrices = {};"
        if old_kpi_prev in src:
            src = src.replace(old_kpi_prev, new_kpi_prev, 1)
            print("[dashboard.html] ✅ latestPrices variable added")

    # Update latestPrices on WebSocket event
    old_ws_event = 'window.addEventListener("prices", (e) => {\n  const { prices, usdinr } = e.detail;'
    new_ws_event = 'window.addEventListener("prices", (e) => {\n  const { prices, usdinr } = e.detail;\n  latestPrices = prices; // keep live price cache updated'
    if old_ws_event in src and "latestPrices = prices" not in src:
        src = src.replace(old_ws_event, new_ws_event, 1)
        print("[dashboard.html] ✅ latestPrices updated on WebSocket event")

    path.write_text(src, encoding="utf-8")
    print("[dashboard.html] ✅ saved")

# ── Fix 6: scanner.html — use WebSocket prices in table ──────────────────────
def fix_scanner():
    path = ROOT / "web/templates/scanner.html"
    if not path.exists():
        print("[scanner.html] NOT FOUND"); return

    src = path.read_text(encoding="utf-8")

    marker = "// SCANNER_LIVE_PRICE_PATCHED"
    if marker in src:
        print("[scanner.html] already patched — skip"); return

    # Update latestPrices in scanner
    old = "let allRows = [], sortCol = \"conf\", sortDir = -1;\nlet latestPrices = {};"
    if old not in src:
        old2 = "let allRows = [], sortCol = \"conf\", sortDir = -1;"
        if old2 in src:
            src = src.replace(old2, old2 + "\nlet latestPrices = {}; // SCANNER_LIVE_PRICE_PATCHED", 1)
            print("[scanner.html] ✅ latestPrices added")

    # Wire WebSocket event if not already there
    if 'window.addEventListener("prices"' not in src:
        ws_listener = '''
window.addEventListener("prices", e => {
  latestPrices = e.detail.prices;
});
'''
        # Insert before loadAll function
        src = src.replace("async function loadAll()", ws_listener + "async function loadAll()", 1)
        print("[scanner.html] ✅ WebSocket price listener added")

    # Use live price in table rows
    old_px = '    const px    = r.price ? `₹${(r.price).toLocaleString("en-IN",{maximumFractionDigits:2})}` : "—";'
    new_px = '''    // Use live WebSocket price for consistency // SCANNER_LIVE_PRICE_PATCHED
    const wsInfo = latestPrices[r.sym];
    const px = wsInfo ? fmtDisplay(r.sym, wsInfo.display, wsInfo.label) :
               r.price ? `₹${(r.price).toLocaleString("en-IN",{maximumFractionDigits:2})}` : "—";'''
    if old_px in src:
        src = src.replace(old_px, new_px, 1)
        print("[scanner.html] ✅ table uses live WebSocket price")

    path.write_text(src, encoding="utf-8")
    print("[scanner.html] ✅ saved")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("Price tick + consistency patcher")
    print("=" * 55)

    fix_price_store()
    print()
    fix_main()
    print()
    fix_realtime_advisor()
    print()
    fix_base_html()
    print()
    fix_dashboard()
    print()
    fix_scanner()

    print()
    print("=" * 55)
    print("Done. Clear pycache and restart:")
    print()
    print("  Get-ChildItem -Recurse -Filter __pycache__ -Directory | Remove-Item -Recurse -Force")
    print("  python run.py")
    print()
    print("Expected after restart:")
    print("  - Header ticker updates every ~0.5s visibly")
    print("  - Signal card prices match header ticker exactly")
    print("  - All prices show age <3s")
    print("=" * 55)

if __name__ == "__main__":
    main()