(function () {
  const POST_URL = "http://localhost:8502/chart-context";

  function normalizeSymbol(rawSymbol) {
    const value = (rawSymbol || "").toUpperCase().trim();
    const mapping = {
      "NSE:NIFTY": "NIFTY50",
      "NSE:NIFTY50": "NIFTY50",
      "NSE:BANKNIFTY": "BANKNIFTY",
      "BITSTAMP:BTCUSD": "BTC",
      "BINANCE:BTCUSDT": "BTC",
      "BITSTAMP:ETHUSD": "ETH",
      "BINANCE:ETHUSDT": "ETH",
      "COMEX:GC1!": "GOLD",
      "NYMEX:CL1!": "CRUDEOIL"
    };
    return mapping[value] || value.split(":").pop() || value;
  }

  function extractRawSymbol() {
    const url = new URL(window.location.href);
    return (
      url.searchParams.get("symbol") ||
      document.querySelector("[data-symbol-short]")?.getAttribute("data-symbol-short") ||
      document.querySelector("[data-name='legend-source-item']")?.textContent ||
      ""
    );
  }

  function extractTimeframe() {
    return (
      document.querySelector("[data-name='header-intervals'] button[aria-pressed='true']")?.textContent ||
      document.querySelector("[data-value] [aria-checked='true']")?.textContent ||
      ""
    );
  }

  async function pushContext() {
    const rawSymbol = extractRawSymbol();
    if (!rawSymbol) {
      return;
    }

    const payload = {
      symbol: normalizeSymbol(rawSymbol),
      raw_symbol: rawSymbol,
      timeframe: extractTimeframe(),
      url: window.location.href,
      title: document.title,
      source: "tradingview-extension",
      captured_at: new Date().toISOString()
    };

    try {
      await fetch(POST_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
    } catch (err) {
      console.debug("TradingView watcher bridge unavailable:", err);
    }
  }

  let lastHref = window.location.href;
  setInterval(() => {
    if (window.location.href !== lastHref) {
      lastHref = window.location.href;
      setTimeout(pushContext, 800);
    }
    pushContext();
  }, 2000);

  window.addEventListener("load", () => setTimeout(pushContext, 1200));
})();
