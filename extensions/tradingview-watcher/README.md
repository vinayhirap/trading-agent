# TradingView Watcher Bridge

This optional Chrome extension reports the TradingView chart you are actively watching to the local trading agent at `http://localhost:8502/chart-context`.

What it does:
- reads the current TradingView chart URL
- extracts the visible symbol and best-effort timeframe
- sends that chart context to the local agent every 2 seconds

What it does not do:
- it does not scrape private TradingView internals
- it does not guarantee access to TradingView-owned indicator values
- it does not replace the backend signal engine

Use it for:
- keeping the local advisor focused on the chart you are currently watching
- linking your TradingView browsing workflow to dashboard and Telegram signals

Load it in Chrome:
1. Open `chrome://extensions`
2. Turn on `Developer mode`
3. Click `Load unpacked`
4. Select this folder: `extensions/tradingview-watcher`

The local Streamlit stack must already be running, because the bridge posts to the Flask sidecar on port `8502`.
