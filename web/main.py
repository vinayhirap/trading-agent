# trading-agent/web/main.py
"""
AI Trading Agent — FastAPI Web Server

Stack:
  FastAPI + Uvicorn  (async HTTP + WebSocket)
  Jinja2             (HTML templates)
  Alpine.js          (reactive frontend, no build step)
  TradingView Charts (professional real-time charts)

Run:
  cd trading-agent
  python web/main.py

Or production:
  uvicorn web.main:app --host 0.0.0.0 --port 8010 --reload

Access:
  Dashboard:  http://localhost:8010
  API docs:   http://localhost:8010/docs
  WebSocket:  ws://localhost:8010/ws/prices
"""
import sys
import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config.settings import settings

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]   # project root
WEB  = Path(__file__).resolve().parent       # web/ folder

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Background service startup ────────────────────────────────────────────────
def _start_background_services():
    """Start all trading system background threads."""
    import threading, time as t

    def _run():
        t.sleep(2)
        try:
            from src.streaming.price_store import price_store
            symbols = [
                "NIFTY50","BANKNIFTY","NIFTYIT","SENSEX",
                "GOLD","SILVER","CRUDEOIL","COPPER","NATURALGAS",
                "BTC","ETH","SOL","BNB","XRP",
                "USDINR","EURUSD","GBPUSD",
                "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK",
                "SBIN","WIPRO","AXISBANK","KOTAKBANK","LT",
                "BAJFINANCE","MARUTI","SUNPHARMA","BHARTIARTL","TATASTEEL",
            ]
            price_store.warm_up(symbols)
            price_store.start_background_refresh(symbols, interval_seconds=10)
            print("✅ PriceStore: background refresh started")
        except Exception as e:
            print(f"⚠️  PriceStore: {e}")
 
        try:
            from src.analysis.realtime_advisor import realtime_advisor
            if realtime_advisor is not None:
                realtime_advisor.start()
                print("✅ RealtimeAdvisor: started")
            else:
                print("⚠️  RealtimeAdvisor: failed to initialize")

        except Exception as e:
            print(f"⚠️  RealtimeAdvisor: {e}")

        try:
            from src.alerts.daily_summary import summary_scheduler
            summary_scheduler.start()
            print("✅ DailySummaryScheduler: started")
        except Exception as e:
            print(f"⚠️  DailySummaryScheduler: {e}")

        try:
            from src.alerts.price_alert_manager import alert_manager
            alert_manager.start()
            print("✅ PriceAlertManager: started")
        except Exception as e:
            print(f"⚠️  PriceAlertManager: {e}")

        try:
            from src.streaming.mcx_token_manager import mcx_token_manager
            mcx_token_manager.start()
            print("✅ MCXTokenManager: auto-rollover active")
        except Exception as e:
            print(f"⚠️  MCXTokenManager: {e}")

        try:
            from src.streaming.angel_one_ticker import angel_one_ticker
            started = angel_one_ticker.start()
            if started:
                print("✅ AngelOneTicker: WebSocket streaming started")
            else:
                print("⚠️  AngelOneTicker: credentials missing — using yfinance only")
        except Exception as e:
            print(f"⚠️  AngelOneTicker: {e}")

        try:
            from src.streaming.event_bus import event_bus
            import threading as _t
            _t.Thread(target=event_bus.start, daemon=True, name="EventBus").start()
            print("✅ EventBus: started")
        except Exception as e:
            print(f"⚠️  EventBus: {e}")
           
    threading.Thread(target=_run, daemon=True, name="BGServices").start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background services when server starts."""
    _start_background_services()

    # ── Broadcast loops ──────────────────────────────────────
    async def safe_price_loop():
        try:
            await price_broadcast_loop()
        except Exception as e:
            print(f"❌ price_broadcast_loop crashed: {e}")
            import traceback; traceback.print_exc()

    async def safe_signal_loop():
        try:
            await signal_broadcast_loop()
        except Exception as e:
            print(f"❌ signal_broadcast_loop crashed: {e}")

    asyncio.create_task(safe_price_loop())
    asyncio.create_task(safe_signal_loop())

    # ── Angel One tick callback ───────────────────────────────
    try:
        from src.streaming.price_store import price_store as _ps
        from src.streaming.price_store import PRICE_META as _PM, convert_price as _cp, ACTIVE_EXPIRY as _AE
        _loop = asyncio.get_event_loop()
        def _on_angel_tick(sym, raw_price):
            try:
                usdinr = _ps.get("USDINR", fallback=False) or 92.46
                if not (70 < usdinr < 120):
                    usdinr = 92.46
                meta    = _PM.get(sym, {"convert": "none", "label": ""})
                display = _cp(sym, raw_price, usdinr)
                payload = {"type": "prices", "usdinr": usdinr, "prices": {
                    sym: {
                        "raw":     round(raw_price, 4),
                        "display": round(display, 2),
                        "label":   meta.get("label", ""),
                        "convert": meta.get("convert", "none"),
                        "age":     0.0,
                        "stale":   False,
                        "expiry":  _AE.get(sym, ""),
                    }
                }}
                if manager.price_clients:
                    _loop.call_soon_threadsafe(
                        lambda p=payload: asyncio.ensure_future(manager.broadcast_prices(p))
                    )
            except Exception:
                pass
        _ps.register_tick_callback(_on_angel_tick)
    except Exception:
        pass

    yield


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    # FIX: redirect_slashes=True so /ws/prices/ → /ws/prices works for WebSocket too
    # FastAPI cannot redirect WebSocket connections; instead we register BOTH routes below.
    redirect_slashes = False,
    title       = "AI Trading Agent",
    description = "Real-time AI-powered trading dashboard for Indian markets",
    version     = "3.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# Static files + templates
_static_dir = WEB / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")
templates = Jinja2Templates(directory=WEB / "templates")


# ── API Key Security Middleware (P4-B) ────────────────────────────────────────
# If DASHBOARD_API_KEY is set in .env, all /api/* requests must include:
#   Header:  X-API-Key: <your_key>
#   OR Query: ?api_key=<your_key>
# Page routes (HTML) are always accessible — no auth needed for local use.
# Set DASHBOARD_API_KEY in .env only if you expose this on a network.
@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    from config.settings import settings
    required_key = getattr(settings, "DASHBOARD_API_KEY", None)
    # Only protect /api/* routes, and only when key is configured
    if required_key and request.url.path.startswith("/api/"):
        provided = (
            request.headers.get("X-API-Key")
            or request.headers.get("x-api-key")
            or request.query_params.get("api_key")
        )
        if not provided or provided != required_key:
            return JSONResponse(
                status_code=401,
                content={"ok": False, "error": "Unauthorized — X-API-Key header required"},
            )
    return await call_next(request)


# ── WebSocket connection manager ──────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.price_clients:  list[WebSocket] = []
        self.signal_clients: list[WebSocket] = []

    async def connect_prices(self, ws: WebSocket):
        await ws.accept()
        self.price_clients.append(ws)

    async def connect_signals(self, ws: WebSocket):
        await ws.accept()
        self.signal_clients.append(ws)

    def disconnect(self, ws: WebSocket):
        self.price_clients  = [c for c in self.price_clients  if c != ws]
        self.signal_clients = [c for c in self.signal_clients if c != ws]

    async def broadcast_prices(self, data: dict):
        msg  = json.dumps(data)
        dead = []
        for client in self.price_clients:
            try:
                await client.send_text(msg)
            except Exception:
                dead.append(client)
        for d in dead:
            self.price_clients = [c for c in self.price_clients if c != d]

    async def broadcast_signals(self, data: dict):
        msg  = json.dumps(data)
        dead = []
        for client in self.signal_clients:
            try:
                await client.send_text(msg)
            except Exception:
                dead.append(client)
        for d in dead:
            self.signal_clients = [c for c in self.signal_clients if c != d]


manager = ConnectionManager()


# ── Background price broadcaster ──────────────────────────────────────────────
async def price_broadcast_loop():
    await asyncio.sleep(15)
    while True:
        if manager.price_clients:
            try:
                data = await _get_prices_async()
                if data.get("prices"):
                    await manager.broadcast_prices(data)
            except Exception:
                pass
        await asyncio.sleep(1)

async def signal_broadcast_loop():
    """Push signals to all connected clients every 30 seconds."""
    while True:
        if manager.signal_clients:
            try:
                data = await asyncio.get_event_loop().run_in_executor(None, _get_signals)
                await manager.broadcast_signals(data)
            except Exception:
                pass
        await asyncio.sleep(15)


# ── Data helpers (sync, run in executor) ──────────────────────────────────────
def _get_prices() -> dict:
    try:
        from src.streaming.price_store import (
            price_store, PRICE_META, ACTIVE_EXPIRY, convert_price
        )
        usdinr = price_store.get("USDINR", fallback=False) or 92.46
        if not (70 < usdinr < 120):
            usdinr = 92.46
        all_raw = price_store.get_all()
        result  = {}
        for sym, raw in all_raw.items():
            if not raw or raw <= 0:
                continue
            meta    = PRICE_META.get(sym, {"convert": "none", "label": ""})
            display = convert_price(sym, raw, usdinr)
            age     = price_store.age_seconds(sym) or 0
            result[sym] = {
                "raw":     round(raw, 4),
                "display": round(display, 2),
                "label":   meta.get("label", ""),
                "convert": meta.get("convert", "none"),
                "age":     round(age, 1),
                "stale":   age > 300,
                "expiry":  ACTIVE_EXPIRY.get(sym, ""),
            }
        return {"type": "prices", "usdinr": usdinr, "prices": result}
    except Exception as e:
        return {"type": "prices", "usdinr": 92.46, "prices": {}, "error": str(e)}


async def _get_prices_async() -> dict:
    """Fast async price fetch — only reads price_store, no heavy imports."""
    try:
        from src.streaming.price_store import (
            price_store, PRICE_META, ACTIVE_EXPIRY, convert_price
        )
        usdinr = price_store.get("USDINR", fallback=False) or 92.46
        if not (70 < usdinr < 120):
            usdinr = 92.46
        all_raw = price_store.get_all()
        result  = {}
        for sym, raw in all_raw.items():
            if not raw or raw <= 0:
                continue
            meta    = PRICE_META.get(sym, {"convert": "none", "label": ""})
            display = convert_price(sym, raw, usdinr)
            result[sym] = {
                "raw":     round(raw, 4),
                "display": round(display, 2),
                "label":   meta.get("label", ""),
                "convert": meta.get("convert", "none"),
                "age":     0.0,
                "stale":   False,
                "expiry":  ACTIVE_EXPIRY.get(sym, ""),
            }
        return {"type": "prices", "usdinr": usdinr, "prices": result}
    except Exception as e:
        return {"type": "prices", "usdinr": 92.46, "prices": {}, "error": str(e)}

def _get_signals() -> dict:
    try:
        from src.analysis.realtime_advisor import realtime_advisor
        return {"type": "signals", "signals": realtime_advisor.get_all()}
    except Exception as e:
        return {"type": "signals", "signals": {}, "error": str(e)}


def _market_status() -> dict:
    """Safe market status — always returns a complete dict, never raises."""
    result = {
        "nse_open":      False,
        "mcx_open":      True,
        "is_holiday":    False,
        "holiday_name":  "",
        "secs_to_close": 0,
        "secs_to_open":  0,
    }
    try:
        from src.utils.market_hours import market_hours
        s = market_hours.get_full_status()
        if s is not None:
            result.update({
                "nse_open":      bool(getattr(s, "nse_tradeable",  False)),
                "mcx_open":      bool(getattr(s, "mcx_tradeable",  True)),
                "is_holiday":    bool(getattr(s, "is_nse_holiday", False)),
                "holiday_name":  str(getattr(s, "holiday_name", "") or ""),
                "secs_to_close": int(getattr(s, "secs_to_nse_close", 0) or 0),
                "secs_to_open":  int(getattr(s, "secs_to_nse_open",  0) or 0),
            })
    except Exception:
        pass   # silently use defaults — server must not 500 because of this
    return result


def _template_context(request: Request) -> dict:
    """Base context passed to every template — never raises."""
    try:
        market = _market_status()
    except Exception:
        market = {
            "nse_open": False, "mcx_open": True,
            "is_holiday": False, "holiday_name": "",
            "secs_to_close": 0, "secs_to_open": 0,
        }
    return {
        "request": request,
        "market":  market,
        "usdinr":  92.46,
    }


# ── WebSocket endpoints ───────────────────────────────────────────────────────
# FIX: register BOTH /ws/prices and /ws/prices/ so trailing-slash clients work.
# FastAPI cannot redirect WebSocket connections — must handle both explicitly.

async def _ws_prices_handler(websocket: WebSocket):
    await manager.connect_prices(websocket)
    try:
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                pass  # keep alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


@app.websocket("/ws/prices")
async def ws_prices(websocket: WebSocket):
    await _ws_prices_handler(websocket)


@app.websocket("/ws/prices/")
async def ws_prices_slash(websocket: WebSocket):
    await _ws_prices_handler(websocket)


async def _ws_signals_handler(websocket: WebSocket):
    await manager.connect_signals(websocket)
    try:
        data = await asyncio.get_event_loop().run_in_executor(None, _get_signals)
        await websocket.send_text(json.dumps(data))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


@app.websocket("/ws/signals")
async def ws_signals(websocket: WebSocket):
    await _ws_signals_handler(websocket)


@app.websocket("/ws/signals/")
async def ws_signals_slash(websocket: WebSocket):
    await _ws_signals_handler(websocket)


# ── Favicon — suppress 404 noise ──────────────────────────────────────────────
from fastapi.responses import Response

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


# ── Page routes ───────────────────────────────────────────────────────────────
PAGES = [
    ("dashboard", "/",            "📊", "Dashboard",       "Market overview, live prices, signals"),
    ("scanner",   "/scanner",     "🔍", "Signal Scanner",  "Full multi-symbol signal scanner"),
    ("charts",    "/charts",      "📈", "Charts",          "Live TradingView charts with AI overlay"),
    ("news",      "/news",        "📰", "News Feed",       "Real-time news with sentiment analysis"),
    ("crude",     "/crude",       "🛢️", "Crude Intel",     "WTI/Brent analysis + MCX prediction"),
    ("ai",        "/ai",          "🤖", "AI Insights",     "Ask AI about market conditions"),
    ("events",    "/events",      "🌍", "Event Monitor",   "Global macro events"),
    ("timing",    "/timing",      "⏰", "Market Timing",   "Session analysis and intraday patterns"),
    ("trade",     "/trade",       "📟", "Trade",           "Paper and live trade execution"),
    ("alerts",    "/alerts",      "🔔", "Alerts",          "Price and RSI alerts → Telegram"),
    ("portfolio", "/portfolio",   "💼", "Portfolio",       "Paper portfolio P&L and positions"),
    ("accuracy",  "/accuracy",    "🎯", "Accuracy",        "Prediction vs actual tracking"),
    ("backtest",  "/backtest",    "📊", "Backtesting",     "Walk-forward backtest engine"),
    ("status",    "/status",      "⚙️", "System Status",   "Models, APIs, data cache"),
    ("nerve_center",  "/nerve-center",  "🧠", "Nerve Center",    "Central command and control"),
    ("fii_dii",       "/fii-dii",       "📡", "FII/DII",         "Institutional money flow tracker"),
    ("options_oi",    "/options-oi",    "📐", "Options OI",      "Open interest and max pain"),
    ("regime",        "/regime",        "🌐", "Market Regime",   "HMM regime detection"),
    ("system_health", "/system-health", "💊", "System Health",   "Phase roadmap and module status"),
]

def page_view(template_name: str):
    async def view(request: Request):
        try:
            ctx = _template_context(request)
            ctx["active_page"] = template_name
            ctx["pages"]       = PAGES
            # Starlette 0.36+: request is first positional arg
            return templates.TemplateResponse(request, f"{template_name}.html", ctx)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            return HTMLResponse(
                f"""<!doctype html>
<html>
<head><title>Template Error — {template_name}</title></head>
<body style="background:#06060c;color:#e2e2f0;font-family:monospace;padding:40px">
  <h2 style="color:#ff3366">⚠ Template Error — {template_name}.html</h2>
  <pre style="color:#ffaa00;background:#1a1a2e;padding:16px;border-radius:8px;
              overflow:auto;margin-top:16px">{e}</pre>
  <details style="margin-top:16px">
    <summary style="color:#8888aa;cursor:pointer">Full traceback</summary>
    <pre style="color:#666688;background:#0d0d1a;padding:12px;border-radius:6px;
                margin-top:8px;overflow:auto">{tb}</pre>
  </details>
  <p style="margin-top:20px;color:#8888aa">
    Check <code>web/templates/{template_name}.html</code>
  </p>
  <a href="/" style="color:#4488ff">← Home</a>
</body>
</html>""",
                status_code=500
            )
    view.__name__ = f"view_{template_name}"
    return view


app.add_route("/",          page_view("dashboard"),  methods=["GET"])
app.add_route("/scanner",   page_view("scanner"),    methods=["GET"])
app.add_route("/charts",    page_view("charts"),     methods=["GET"])
app.add_route("/news",      page_view("news"),       methods=["GET"])
app.add_route("/crude",     page_view("crude"),      methods=["GET"])
app.add_route("/ai",        page_view("ai"),         methods=["GET"])
app.add_route("/events",    page_view("events"),     methods=["GET"])
app.add_route("/timing",    page_view("timing"),     methods=["GET"])
app.add_route("/trade",     page_view("trade"),      methods=["GET"])
app.add_route("/alerts",    page_view("alerts"),     methods=["GET"])
app.add_route("/portfolio", page_view("portfolio"),  methods=["GET"])
app.add_route("/accuracy",  page_view("accuracy"),   methods=["GET"])
app.add_route("/backtest",  page_view("backtest"),   methods=["GET"])
app.add_route("/status",    page_view("status"),     methods=["GET"])
app.add_route("/nerve-center",  page_view("nerve_center"),  methods=["GET"])
app.add_route("/fii-dii",       page_view("fii_dii"),       methods=["GET"])
app.add_route("/options-oi",    page_view("options_oi"),    methods=["GET"])
app.add_route("/regime",        page_view("regime"),        methods=["GET"])
app.add_route("/system-health", page_view("system_health"), methods=["GET"])

# ── REST API ──────────────────────────────────────────────────────────────────
from fastapi import APIRouter
api = APIRouter(prefix="/api", tags=["API"])


@api.get("/prices")
async def api_prices():
    return await _get_prices_async()


@api.get("/signals")
async def api_signals(symbols: str = "NIFTY50,BANKNIFTY,GOLD,BTC,ETH"):
    syms = [s.strip().upper() for s in symbols.split(",")]
    try:
        from src.analysis.realtime_advisor import realtime_advisor
        all_sig = realtime_advisor.get_all()
        return {"ok": True, "signals": {s: all_sig.get(s, {}) for s in syms}}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@api.get("/market-status")
async def api_market_status():
    return {"ok": True, "status": _market_status()}


@api.get("/portfolio")
async def api_portfolio(source: str = "paper"):
    """
    source=paper  → PaperBroker (default)
    source=groww  → Groww live holdings (requires Groww data in groww_holdings.json)
    source=all    → Both combined in one response
    """
    result = {}
    try:
        from src.execution.paper_broker import PaperBroker
        from config.settings import settings
        broker  = PaperBroker(initial_capital=settings.INITIAL_CAPITAL)
        summary = broker.get_portfolio_summary()
        history = broker.get_trade_history()
        summary["recent_trades"] = history[-20:][::-1]  # last 20, newest first
        result["paper"] = summary
    except Exception as e:
        result["paper"] = {"error": str(e)}

    if source in ("groww", "all"):
        try:
            from src.portfolio.groww_connector import GrowwConnector
            gc      = GrowwConnector()
            gp      = gc.get_portfolio_with_live_prices()
            result["groww"] = {
                "total_invested":  gp.total_invested,
                "total_value":     gp.total_value,
                "total_pnl":       gp.total_pnl,
                "total_pnl_pct":   gp.total_pnl_pct,
                "fetched_at":      gp.fetched_at,
                "source":          gp.source,
                "holdings": [
                    {
                        "symbol":          h.symbol,
                        "name":            h.name,
                        "quantity":        h.quantity,
                        "avg_buy_price":   h.avg_buy_price,
                        "current_price":   h.current_price,
                        "invested_value":  h.invested_value,
                        "current_value":   h.current_value,
                        "pnl":             h.pnl,
                        "pnl_pct":         h.pnl_pct,
                    }
                    for h in gp.holdings
                ],
            }
        except Exception as e:
            result["groww"] = {"error": str(e)}

    if source == "paper":
        return {"ok": True, "portfolio": result["paper"]}
    elif source == "groww":
        return {"ok": True, "portfolio": result.get("groww", {}), "source": "groww"}
    else:
        return {"ok": True, "paper": result.get("paper", {}), "groww": result.get("groww", {})}


@api.get("/news")
async def api_news(limit: int = 20):
    try:
        from src.news.news_intelligence import NewsIntelligence
        ni = NewsIntelligence()
        
        # Get full linked articles
        articles = ni.get_all_linked(max_age=120)
        sent = ni.get_market_sentiment()
        
        # Build articles list news.html expects
        art_list = []
        for a in articles[:limit]:
            score = getattr(a, 'sentiment', 0)
            art_list.append({
                "title":        getattr(a, 'title', ''),
                "source":       getattr(a, 'source', 'Market News'),
                "sentiment":    getattr(a, 'sentiment_label', 'NEUTRAL'),
                "score":        score,
                "published_at": getattr(a, 'published_at', None),
                "url":          getattr(a, 'url', '#'),
                "symbols":      getattr(a, 'linked_symbols', []),
                "age_mins":     getattr(a, 'age_minutes', 0),
            })
        
        sent["articles"] = art_list
        return {"ok": True, "sentiment": sent}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@api.post("/signal-action")
async def api_signal_action(request: Request):
    body   = await request.json()
    symbol = body.get("symbol", "NIFTY50")
    try:
        from src.analysis.realtime_advisor import realtime_advisor
        return {"ok": True, "signal": realtime_advisor.get(symbol)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@api.post("/alert/add")
async def api_alert_add(request: Request):
    body = await request.json()
    try:
        from src.alerts.price_alert_manager import alert_manager
        aid = alert_manager.add_alert(
            symbol     = body["symbol"],
            alert_type = body["type"],
            threshold  = float(body["threshold"]),
            note       = body.get("note", ""),
            fire_once  = body.get("fire_once", True),
        )
        return {"ok": True, "id": aid}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@api.post("/alert/delete")
async def api_alert_delete(request: Request):
    body = await request.json()
    try:
        from src.alerts.price_alert_manager import alert_manager
        ok = alert_manager.delete_alert(body["id"])
        return {"ok": ok}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── API: ACCURACY TRACKING ────────────────────────────────────────────────────
@api.get("/accuracy")
async def api_accuracy(symbol: str = None, asset_class: str = None):
    try:
        from src.analysis.learning_engine_v2 import LearningEngineV2
        le   = LearningEngineV2()

        # Overall stats
        overall = le.get_accuracy_stats(
            symbol=symbol or None,
            asset_class=asset_class or None
        )

        # Per-asset-class breakdown (always include even if filtered)
        by_ac = {}
        for ac in ("index", "equity", "futures", "crypto"):
            s = le.get_accuracy_stats(asset_class=ac)
            if s.get("total", 0) > 0:
                by_ac[ac] = s

        # Pending (unresolved) count
        all_preds = le._db.get("predictions", [])
        pending   = sum(1 for p in all_preds if not p.get("resolved"))

        # Expected value at 1:1.5 R:R
        acc = overall.get("accuracy", 0)
        ev  = round(acc * 1.5 - (1 - acc), 4)  # EV = p*reward - (1-p)*risk

        # Recent signals (last 20 resolved)
        resolved = sorted(
            [p for p in all_preds if p.get("resolved")],
            key=lambda x: x.get("timestamp", ""),
            reverse=True,
        )[:20]

        return {
            "ok":         True,
            "overall":    overall,
            "by_ac":      by_ac,
            "pending":    pending,
            "total_all":  len(all_preds),
            "ev":         ev,
            "break_even": 0.40,
            "recent":     resolved,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@api.get("/alerts")
async def api_alerts_list():
    try:
        from src.alerts.price_alert_manager import alert_manager
        return {"ok": True, "alerts": alert_manager.get_alerts()}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@api.post("/backtest/run")
async def api_backtest(request: Request):
    body = await request.json()
    try:
        import datetime
        from src.backtesting.backtest_engine import BacktestEngine
        from src.backtesting.backtest_report import BacktestReport
        engine = BacktestEngine(
            initial_capital = float(body.get("capital", 10000)),
            risk_per_trade  = float(body.get("risk", 0.01)),
        )
        start  = datetime.date.fromisoformat(body["start"])
        end    = datetime.date.fromisoformat(body["end"])
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: engine.run(body["symbol"], start, end, verbose=False)
        )
        BacktestReport.compute_metrics(result)
        summary = BacktestReport.summary_dict(result)

        # Add numeric fields for charts (summary_dict returns formatted strings)
        r = result
        summary["_total_return_pct"]  = round(r.total_return_pct, 2)
        summary["_annual_return"]     = round(r.annual_return, 2)
        summary["_benchmark_return"]  = round(r.benchmark_return, 2)
        summary["_max_drawdown_pct"]  = round(r.max_drawdown_pct, 2)
        summary["_sharpe_ratio"]      = round(r.sharpe_ratio, 3)
        summary["_win_rate"]          = round(r.win_rate * 100, 1)
        summary["_n_trades"]          = r.n_trades
        summary["_final_capital"]     = round(r.final_capital, 2)
        summary["_initial_capital"]   = round(r.initial_capital, 2)

        # Equity curve — downsample to max 200 points for performance
        try:
            ec = r.equity_curve
            if hasattr(ec, "reset_index"):
                ec_df = ec.reset_index()
                ec_df.columns = ["date", "value"]
                n = len(ec_df)
                step = max(1, n // 200)
                sampled = ec_df.iloc[::step]
                summary["_equity_curve"] = [
                    {"date": str(row["date"])[:10], "value": round(float(row["value"]), 2)}
                    for _, row in sampled.iterrows()
                ]
            else:
                summary["_equity_curve"] = []
        except Exception:
            summary["_equity_curve"] = []

        # Grade
        try:
            grade, grade_note = BacktestReport.grade(result)
            summary["_grade"]      = grade
            summary["_grade_note"] = grade_note
        except Exception:
            summary["_grade"] = "—"
            summary["_grade_note"] = ""

        return {"ok": True, "result": summary}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@api.post("/trade/execute")
async def api_trade_execute(request: Request):
    body = await request.json()
    mode = body.get("mode", "paper").lower()
    try:
        import uuid
        from src.risk.models import TradeOrder, OrderSide, OrderStatus
        from config.settings import settings

        order_kwargs = dict(
            order_id     = str(uuid.uuid4())[:8],
            symbol       = body["symbol"],
            side         = OrderSide.BUY if body["side"] == "BUY" else OrderSide.SELL,
            quantity     = int(body["quantity"]),
            order_type   = "MARKET",
            status       = OrderStatus.APPROVED,
            stop_loss    = float(body.get("sl", 0)),
            target_price = float(body.get("target", 0)),
            signal       = body.get("signal", "MANUAL"),
            confidence   = float(body.get("confidence", 0.5)),
        )

        def _send_trade_telegram(sym, side, qty, fill_price, sl, target, charges, is_live, order_id):
            try:
                from src.alerts.daily_summary import NotificationPrefs
                if not NotificationPrefs.load().is_enabled("trade_executed"):
                    return
                from src.alerts.telegram_sender import TelegramSender
                TelegramSender().send_trade(
                    sym, side, qty, fill_price, sl, target,
                    charges, is_live=is_live, order_id=order_id,
                )
            except Exception as _te:
                logger.debug(f"Trade Telegram failed: {_te}")

        if mode == "live":
            try:
                from src.brokers.angel_one_live import angel_one_live
                order      = TradeOrder(**order_kwargs)
                # angel_one_live.place_order expects positional args, not TradeOrder
                angel_id = angel_one_live.place_order(
                    symbol     = order_kwargs["symbol"],
                    side       = body["side"],
                    quantity   = order_kwargs["quantity"],
                    price      = float(body.get("price", 0)),
                    stop_loss  = order_kwargs["stop_loss"],
                    target     = order_kwargs["target_price"],
                )
                fill_price = float(body.get("price", 0))
                charges    = 0  # Angel One returns only order_id; charges unknown at placement
                record     = type("R", (), {"fill_price": fill_price, "total_charges": charges, "order_id": angel_id or order.order_id})()
                _send_trade_telegram(
                    order_kwargs["symbol"], body["side"], order_kwargs["quantity"],
                    fill_price, order_kwargs["stop_loss"], order_kwargs["target_price"],
                    charges, is_live=True, order_id=order.order_id,
                )
                return {"ok": True, "mode": "live", "fill_price": fill_price,
                        "charges": charges, "order_id": order.order_id}
            except Exception as e:
                return {"ok": False, "mode": "live", "error": f"Live order failed: {e}"}
        else:
            from src.execution.paper_broker import PaperBroker
            broker = PaperBroker(initial_capital=settings.INITIAL_CAPITAL)
            order  = TradeOrder(**order_kwargs)
            record = broker.execute(order, current_price=float(body["price"]))
            _send_trade_telegram(
                order_kwargs["symbol"], body["side"], order_kwargs["quantity"],
                record.fill_price, order_kwargs["stop_loss"], order_kwargs["target_price"],
                record.total_charges, is_live=False, order_id=order.order_id,
            )
            return {"ok": True, "mode": "paper", "fill_price": record.fill_price,
                    "charges": record.total_charges, "order_id": order.order_id}
    except Exception as e:
        return {"ok": False, "mode": mode, "error": str(e)}

@api.get("/config-status")
async def api_config_status():
    try:
        from config.settings import settings as s
        import os
        return {"ok": True, "configured": {
            "anthropic":  bool(os.environ.get("ANTHROPIC_API_KEY") or getattr(s, "ANTHROPIC_API_KEY", None)),
            "news_api":   bool(getattr(s, "NEWS_API_KEY", None)),
            "gnews":      bool(getattr(s, "GNEWS_API_KEY", None)),
            "coinswitch": bool(getattr(s, "COINSWITCH_API_KEY", None)),
            "telegram":   bool(getattr(s, "TELEGRAM_BOT_TOKEN", None)),
            "angel":      bool(getattr(s, "ANGEL_API_KEY", None)),
        }}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# MCX_TOKENS_ROUTE_FIXED
@api.get("/mcx-tokens")
async def api_mcx_tokens():
    try:
        from src.streaming.mcx_token_manager import mcx_token_manager
        return {"ok": True, "status": mcx_token_manager.get_status()}
    except Exception as e:
        return {"ok": False, "error": str(e)}




# ── API: FII/DII ──────────────────────────────────────────────────────────────
@api.get("/fii-dii")
async def api_fii_dii():
    try:
        from src.analysis.fii_dii_tracker import FIIDIITracker
        tracker = FIIDIITracker()
        latest  = tracker.get_latest()
        history = tracker.get_history(days=30)
        return {
            "ok": True,
            "data": {
                "today":   latest.to_dict() if latest else None,
                "history": [d.to_dict() for d in history],
                "source":  latest.source if latest else "unavailable",
            }
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@api.post("/fii-dii/add")
async def api_fii_dii_add(request: Request):
    body = await request.json()
    try:
        from src.analysis.fii_dii_tracker import FIIDIITracker
        tracker = FIIDIITracker()
        result  = tracker.add_manual_entry(
            fii_buy  = float(body.get("fii_buy",  0)),
            fii_sell = float(body.get("fii_sell", 0)),
            dii_buy  = float(body.get("dii_buy",  0)),
            dii_sell = float(body.get("dii_sell", 0)),
            date_str = body.get("date"),
        )
        return {"ok": True, "result": result.to_dict()}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── API: OPTIONS OI ───────────────────────────────────────────────────────────
@api.get("/options-oi")
async def api_options_oi(symbol: str = "NIFTY50"):
    try:
        from src.analysis.options_oi import OptionsOIAnalyzer
        analyzer = OptionsOIAnalyzer()
        chain    = await asyncio.get_event_loop().run_in_executor(
            None, lambda: analyzer.get_chain(symbol)
        )
        if not chain:
            return {"ok": False, "error": "NSE chain unavailable — market hours only"}

        # Serialize chain to dict (OptionsChain has no to_dict — build manually)
        strikes_list = []
        for s in chain.strikes:
            strikes_list.append({
                "strike":       s.strike,
                "call_oi":      s.call_oi,
                "call_oi_chg":  s.call_oi_chg,
                "call_vol":     s.call_volume,
                "call_iv":      s.call_iv,
                "call_ltp":     s.call_ltp,
                "put_oi":       s.put_oi,
                "put_oi_chg":   s.put_oi_chg,
                "put_vol":      s.put_volume,
                "put_iv":       s.put_iv,
                "put_ltp":      s.put_ltp,
                "is_atm":       s.is_atm,
            })

        # Compute DTE
        dte = None
        try:
            from datetime import datetime as _dt
            expiry_dt = _dt.strptime(chain.expiry, "%d-%b-%Y")
            dte = (expiry_dt - _dt.now()).days
        except Exception:
            pass

        return {
            "ok": True,
            "chain": {
                "symbol":     chain.symbol,
                "expiry":     chain.expiry,
                "spot":       chain.spot_price,
                "atm_strike": chain.atm_strike,
                "pcr":        chain.pcr,
                "max_pain":   chain.max_pain_simple,
                "dte":        dte,
                "iv_rank":    None,   # not computed — placeholder
                "strikes":    strikes_list,
            }
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── API: REGIME ───────────────────────────────────────────────────────────────
@api.get("/regime")
async def api_regime(symbol: str = "NIFTY50"):
    try:
        from src.analysis.regime_detector import RegimeDetector
        from src.data.manager import DataManager
        from src.data.models import Interval
        from src.features.feature_engine import FeatureEngine

        dm = DataManager()
        fe = FeatureEngine()
        rd = RegimeDetector()

        df = await asyncio.get_event_loop().run_in_executor(
            None, lambda: dm.get_ohlcv(symbol, Interval.D1, days_back=120)
        )
        feats  = fe.build(df, drop_na=False)
        result = rd.detect(feats.iloc[-1])   # detect() takes pd.Series

        return {
            "ok": True,
            "regime": {
                "regime":           result.regime.value,
                "confidence":       result.confidence,
                "signal_multiplier":result.signal_multiplier,
                "sl_multiplier":    result.sl_multiplier,
                "size_multiplier":  result.size_multiplier,
                "strategy":         result.strategy,
                "description":      result.description,
                "method":           "HMM+Rule" if result.confidence > 0.7 else "Rule-based",
                "trade_verdict":    (
                    "TRADE LONG"  if result.regime.value == "BULL_TREND"  else
                    "TRADE SHORT" if result.regime.value == "BEAR_TREND"  else
                    "MEAN REVERT" if result.regime.value == "RANGING_LOW" else
                    "REDUCE SIZE"
                ),
                "features": {
                    "adx":      result.adx,
                    "atr_ratio":result.atr_ratio,
                    "rsi_14":   result.rsi,
                    "ema_slope":result.ema_slope,
                },
                "supporting": result.supporting_factors,
                "conflicting":result.conflicting_factors,
            }
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── API: SYSTEM HEALTH ────────────────────────────────────────────────────────
@api.get("/system-health")
async def api_system_health():
    import importlib, platform, json as _json
    from pathlib import Path as _Path

    ROOT = _Path(__file__).resolve().parents[1]

    PHASES = [
        {
            "name": "Phase 1 — Intelligence",
            "status": "complete",
            "items": [
                {"name": "Alpha Vantage adapter",       "path": "src/data/adapters/alphavantage_adapter.py"},
                {"name": "Nerve Center",                "path": "src/dashboard/pages/nerve_center.py"},
                {"name": "Company Intelligence",        "path": "src/dashboard/pages/company_intelligence.py"},
            ],
        },
        {
            "name": "Phase 2 — Regime & Valuation",
            "status": "complete",
            "items": [
                {"name": "Regime Detector (HMM)",       "path": "src/analysis/regime_detector.py"},
                {"name": "Valuation Comps",             "path": "src/dashboard/pages/valuation_comps.py"},
            ],
        },
        {
            "name": "Phase 3 — Agents & Flow",
            "status": "complete",
            "items": [
                {"name": "FII/DII Tracker",             "path": "src/analysis/fii_dii_tracker.py"},
                {"name": "Options OI (PCR + MaxPain)",  "path": "src/analysis/options_oi.py"},
                {"name": "Market Alerts",               "path": "src/alerts/market_alerts.py"},
            ],
        },
        {
            "name": "Phase 4 — Optimization",
            "status": "complete",
            "items": [
                {"name": "Optuna Tuner (Bayesian HPO)", "path": "src/prediction/optuna_tuner.py"},
                {"name": "System Health Dashboard",     "path": "web/templates/system_health.html"},
            ],
        },
        {
            "name": "Phase 5 — Production",
            "status": "complete",
            "items": [
                {"name": "RL Position Sizer (DQN)",     "path": "src/prediction/rl_position_sizer.py"},
                {"name": "Event-Driven Architecture",   "path": "src/streaming/event_bus.py"},
                {"name": "Angel One Live Trading",      "path": "src/brokers/angel_one_live.py"},
            ],
        },
    ]

    for phase in PHASES:
        done = 0
        for item in phase["items"]:
            item["exists"] = (ROOT / item["path"]).exists()
            if item["exists"]:
                done += 1
        phase["done"]  = done
        phase["total"] = len(phase["items"])

    # Quick module check
    QUICK_MODULES = [
        "src.data.manager",
        "src.features.feature_engine",
        "src.analysis.realtime_advisor",
        "src.analysis.regime_detector",
        "src.analysis.fii_dii_tracker",
        "src.analysis.options_oi",
        "src.analysis.multi_agent_engine",
    ]
    ok_count = 0
    for mod in QUICK_MODULES:
        try:
            importlib.import_module(mod)
            ok_count += 1
        except Exception:
            pass

    # Models
    model_dir = ROOT / "models"
    models    = []
    for ac in ["index", "equity", "futures", "crypto", "regime_hmm"]:
        fname   = f"ensemble_{ac}.pkl" if ac != "regime_hmm" else "regime_hmm.pkl"
        p       = model_dir / fname
        trained = p.exists()
        import datetime as _datetime
        models.append({
            "asset":      ac,
            "trained":    trained,
            "size":       f"{p.stat().st_size // 1024} KB" if trained else "—",
            "last_train": _datetime.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M") if trained else "—",
        })
    trained_ct = sum(1 for m in models if m["trained"])

    # Accuracy
    accuracy = []
    val_path = ROOT / "data" / "asset_validation.json"
    if val_path.exists():
        try:
            val      = _json.loads(val_path.read_text())
            baseline = {"index": 43.2, "equity": 37.5, "futures": 40.3, "crypto": 0.0}
            for ac, r in val.items():
                acc = r.get("accuracy", 0)
                accuracy.append({
                    "asset":       ac,
                    "accuracy":    f"{acc:.1f}%",
                    "vs_baseline": f"{acc - baseline.get(ac, 0):+.1f}%",
                    "folds":       r.get("n_folds", "—"),
                })
        except Exception:
            pass

    # AV rate limit
    av_rate = {"calls_today": 0, "calls_remaining": 25, "daily_limit": 25}
    try:
        from src.data.adapters.alphavantage_adapter import get_rate_limit_status
        av_rate = get_rate_limit_status()
    except Exception:
        pass

    # APIs configured
    apis = {}
    try:
        import os
        from config.settings import settings as _s
        apis = {
            "anthropic":     bool(os.environ.get("ANTHROPIC_API_KEY") or getattr(_s, "ANTHROPIC_API_KEY", None)),
            "news_api":      bool(getattr(_s, "NEWS_API_KEY", None)),
            "gnews":         bool(getattr(_s, "GNEWS_API_KEY", None)),
            "coinswitch":    bool(getattr(_s, "COINSWITCH_API_KEY", None)),
            "telegram":      bool(getattr(_s, "TELEGRAM_BOT_TOKEN", None)),
            "angel":         bool(getattr(_s, "ANGEL_API_KEY", None)),
            "alpha_vantage": bool(getattr(_s, "ALPHA_VANTAGE_KEY", None)),
        }
    except Exception:
        pass

    # Cache stats
    cache = {"files": 0, "total_mb": 0, "oldest": "—", "newest": "—"}
    cache_dir = ROOT / "data" / "processed"
    if cache_dir.exists():
        files = list(cache_dir.glob("*.parquet"))
        if files:
            import datetime as _dt2
            mtimes = [f.stat().st_mtime for f in files]
            cache  = {
                "files":    len(files),
                "total_mb": round(sum(f.stat().st_size for f in files) / 1e6, 1),
                "oldest":   _dt2.datetime.fromtimestamp(min(mtimes)).strftime("%Y-%m-%d"),
                "newest":   _dt2.datetime.fromtimestamp(max(mtimes)).strftime("%Y-%m-%d"),
            }

    # System info
    sys_info = {
        "python":   platform.python_version(),
        "platform": platform.system() + " " + platform.release(),
        "root":     ROOT.name,
        "ram":      "—",
    }
    try:
        import psutil
        mem = psutil.virtual_memory()
        sys_info["ram"] = f"{mem.percent:.0f}% ({mem.available / 1e9:.1f} GB free)"
    except ImportError:
        pass

    # Key packages
    PKGS = [
        ("fastapi",      "fastapi"),
        ("uvicorn",      "uvicorn"),
        ("xgboost",      "xgboost"),
        ("lightgbm",     "lightgbm"),
        ("optuna",       "optuna"),
        ("pandas",       "pandas"),
        ("numpy",        "numpy"),
        ("scikit-learn", "sklearn"),
        ("hmmlearn",     "hmmlearn"),
    ]
    packages = []
    for pkg_name, import_name in PKGS:
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "?")
            packages.append({"name": pkg_name, "version": ver, "ok": True})
        except ImportError:
            packages.append({"name": pkg_name, "version": "NOT INSTALLED", "ok": False})

    return {
        "ok": True,
        "health": {
            "modules_ok":     ok_count,
            "modules_total":  len(QUICK_MODULES),
            "models_trained": trained_ct,
            "models_total":   len(models),
            "av_calls":       av_rate.get("calls_today", 0),
            "av_limit":       av_rate.get("daily_limit", 25),
            "apis_ok":        sum(1 for v in apis.values() if v),
            "apis_total":     len(apis),
            "phases":         PHASES,
            "models":         models,
            "accuracy":       accuracy,
            "av_rate":        av_rate,
            "apis":           apis,
            "cache":          cache,
            "sys_info":       sys_info,
            "packages":       packages,
        }
    }


@api.get("/system-health/modules")
async def api_system_health_modules():
    import importlib

    GROUPS = {
        "Core": {
            "Data Manager":        "src.data.manager",
            "Feature Engine":      "src.features.feature_engine",
            "Ensemble Model":      "src.prediction.ensemble_model",
            "Backtest Engine":     "src.backtesting.backtest_engine",
            "Risk Manager":        "src.risk.risk_manager",
            "Paper Broker":        "src.execution.paper_broker",
        },
        "Analysis Engines": {
            "Realtime Advisor":    "src.analysis.realtime_advisor",
            "Regime Detector":     "src.analysis.regime_detector",
            "FII/DII Tracker":     "src.analysis.fii_dii_tracker",
            "Options OI":          "src.analysis.options_oi",
            "Multi-Agent Engine":  "src.analysis.multi_agent_engine",
        },
        "Intelligence (Phase 1)": {
            "Alpha Vantage":       "src.data.adapters.alphavantage_adapter",
            "News Intelligence":   "src.news.news_intelligence",
            "Price Alert Manager": "src.alerts.price_alert_manager",
            "Daily Summary":       "src.alerts.daily_summary",
        },
    }

    result = {}
    for grp, mods in GROUPS.items():
        rows = []
        for name, path in mods.items():
            try:
                importlib.import_module(path)
                rows.append({"name": name, "path": path, "ok": True, "error": ""})
            except Exception as e:
                rows.append({"name": name, "path": path, "ok": False, "error": str(e)[:80]})
        result[grp] = rows

    total = sum(len(v) for v in result.values())
    ok_ct = sum(1 for v in result.values() for m in v if m["ok"])
    return {"ok": True, "groups": result, "summary": {"total": total, "ok": ok_ct}}


# ── API: NOTIFICATION PREFERENCES ────────────────────────────────────────────
@api.get("/notification-prefs")
async def api_notif_prefs_get():
    try:
        from src.alerts.daily_summary import NotificationPrefs
        prefs = NotificationPrefs.load()
        return {"ok": True, "prefs": prefs.all()}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@api.post("/notification-prefs")
async def api_notif_prefs_set(request: Request):
    body = await request.json()
    try:
        from src.alerts.daily_summary import NotificationPrefs
        prefs    = NotificationPrefs.load()
        category = body.get("category")
        enabled  = bool(body.get("enabled", True))
        if not category:
            return {"ok": False, "error": "category required"}
        ok = prefs.set(category, enabled)
        if ok:
            return {"ok": True, "category": category, "enabled": enabled}
        return {"ok": False, "error": f"Unknown category: {category}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@api.post("/notification-test")
async def api_notif_test(request: Request):
    """Send a test message to verify Telegram is working."""
    try:
        from src.alerts.telegram_sender import TelegramSender
        from datetime import datetime
        from zoneinfo import ZoneInfo
        IST = ZoneInfo("Asia/Kolkata")
        now = datetime.now(IST).strftime("%H:%M IST")
        sender = TelegramSender()
        ok = sender.send_message(
            f"✅ <b>Test Message</b>\n"
            f"<i>AI Trading Agent — Telegram is working</i>\n"
            f"<code>{now}</code>"
        )
        return {"ok": ok, "message": "Test sent" if ok else "Send failed — check TELEGRAM_BOT_TOKEN"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── API: OPTUNA HPO ───────────────────────────────────────────────────────────
_optuna_jobs: dict = {}  # asset_class → {"status","best_accuracy","n_trials","duration_s","tuned"}

@api.get("/optuna/status")
async def api_optuna_status():
    try:
        from src.prediction.optuna_tuner import OptunaTuner
        results = {}
        for ac in ("index", "equity", "futures", "crypto"):
            job = _optuna_jobs.get(ac, {})
            results[ac] = {
                "tuned":         job.get("tuned", False),
                "job_status":    job.get("status", "idle"),
                "best_accuracy": job.get("best_accuracy", None),
                "n_trials":      job.get("n_trials", 0),
                "duration_s":    job.get("duration_s", 0),
            }
        return {"ok": True, "results": results}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@api.post("/optuna/run")
async def api_optuna_run(request: Request):
    body = await request.json()
    ac   = body.get("asset_class", "all")
    n    = int(body.get("n_trials", 30))
    try:
        import threading, time as _t
        from src.prediction.optuna_tuner import OptunaTuner

        def _tune(asset_class: str):
            _optuna_jobs[asset_class] = {"status": "running", "tuned": False, "n_trials": 0}
            t0 = _t.time()
            try:
                tuner  = OptunaTuner(asset_class=asset_class)
                study  = tuner.run(n_trials=n)
                best   = study.best_trial
                dur    = round(_t.time() - t0, 1)
                acc    = round(best.value * 100, 2) if best and best.value else None
                _optuna_jobs[asset_class] = {
                    "status": "complete", "tuned": True,
                    "best_accuracy": acc, "n_trials": len(study.trials),
                    "duration_s": dur,
                }
            except Exception as ex:
                _optuna_jobs[asset_class] = {"status": f"error: {ex}", "tuned": False, "n_trials": 0}

        targets = ["index", "equity", "futures", "crypto"] if ac == "all" else [ac]
        for t in targets:
            threading.Thread(target=_tune, args=(t,), daemon=True).start()

        return {"ok": True, "message": f"Tuning started for: {', '.join(targets)} ({n} trials each)"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── API: RL POSITION SIZER ────────────────────────────────────────────────────
@api.post("/rl-sizer")
async def api_rl_sizer(request: Request):
    body = await request.json()
    try:
        from src.prediction.rl_position_sizer import RLPositionSizer
        from config.settings import settings
        sizer  = RLPositionSizer()
        state  = {
            "regime":     body.get("regime", "trending"),
            "confidence": float(body.get("confidence", 0.6)),
            "atr_pct":    float(body.get("atr_pct", 0.015)),
            "rsi":        float(body.get("rsi", 50)),
            "win_rate":   float(body.get("win_rate", 0.55)),
            "drawdown":   float(body.get("drawdown", 0.0)),
        }
        size_pct = sizer.get_position_size(state)
        capital  = getattr(settings, "INITIAL_CAPITAL", 1_000_000)
        size_inr = round(capital * size_pct / 100, 2)
        return {
            "ok": True,
            "size_pct": round(size_pct, 2),
            "size_inr": size_inr,
            "capital":  capital,
            "state":    state,
        }
    except Exception as e:
        # Fallback: Kelly-like heuristic when RL model not trained
        try:
            conf    = float(body.get("confidence", 0.6))
            dd      = float(body.get("drawdown",   0.0))
            size    = max(1.0, min(5.0, (conf - 0.5) * 20 * (1 - dd)))
            from config.settings import settings
            capital = getattr(settings, "INITIAL_CAPITAL", 1_000_000)
            return {
                "ok": True,
                "size_pct": round(size, 2),
                "size_inr": round(capital * size / 100, 2),
                "capital":  capital,
                "fallback": True,
                "note": "Heuristic sizing — RL model not yet trained",
            }
        except Exception as e2:
            return {"ok": False, "error": str(e2)}


# ── API: ECONOMIC CALENDAR ───────────────────────────────────────────────────
@api.get("/events")
async def api_events():
    """
    Returns upcoming economic events:
    1. NSE F&O expiry dates (calculated, always accurate)
    2. RBI MPC meeting schedule (from known 2026 calendar)
    3. US economic events from Investing.com RSS
    """
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
    now = datetime.now(IST)
    events = []

    # ── NSE F&O expiry dates (last Thursday of each month) ───────────────────
    NSE_HOLIDAYS_2026 = {
        "2026-01-26", "2026-03-31", "2026-04-14", "2026-04-17",
        "2026-05-01", "2026-08-15", "2026-10-02", "2026-10-26",
        "2026-11-05", "2026-11-20", "2026-12-25",
    }
    for month_offset in range(0, 6):
        year  = now.year + (now.month + month_offset - 1) // 12
        month = (now.month + month_offset - 1) % 12 + 1
        # Last Thursday of month
        import calendar
        last_day = calendar.monthrange(year, month)[1]
        d = datetime(year, month, last_day, tzinfo=IST)
        while d.weekday() != 3:   # 3 = Thursday
            d -= timedelta(days=1)
        # If holiday, move to Wednesday
        if d.strftime("%Y-%m-%d") in NSE_HOLIDAYS_2026:
            d -= timedelta(days=1)
        if d.date() >= now.date():
            events.append({
                "title":   "NSE Monthly F&O Expiry",
                "date":    d.strftime("%Y-%m-%d"),
                "impact":  "HIGH",
                "country": "IN",
                "note":    "Monthly derivatives expiry — high OI unwinding, elevated volatility near close",
                "source":  "calculated",
            })

    # ── RBI MPC 2026 meeting schedule ────────────────────────────────────────
    RBI_MPC_2026 = [
        ("2026-06-04", "RBI MPC Decision", "Rate decision — CPI and growth trajectory"),
        ("2026-08-06", "RBI MPC Decision", "Post-monsoon inflation outlook — key for NBFCs"),
        ("2026-10-07", "RBI MPC Decision", "Pre-festive policy stance"),
        ("2026-12-03", "RBI MPC Decision", "Year-end monetary policy review"),
    ]
    for date_str, title, note in RBI_MPC_2026:
        if date_str >= now.strftime("%Y-%m-%d"):
            events.append({
                "title":   title,
                "date":    date_str,
                "impact":  "HIGH",
                "country": "IN",
                "note":    note,
                "source":  "rbi_calendar",
            })

    # ── US Events from RSS ───────────────────────────────────────────────────
    try:
        import feedparser, re
        from datetime import timezone
        RSS_URLS = [
            "https://feeds.content.dowjones.io/public/rss/mw_topstories",
            "https://feeds.marketwatch.com/marketwatch/marketpulse/",
        ]
        US_KEYWORDS = [
            "fed ", "fomc", "interest rate", "cpi", "inflation", "jobs report",
            "nonfarm", "gdp", "payroll", "pce", "powell", "yellen",
            "treasury", "fed rate", "rate decision",
        ]
        seen = set()
        for url in RSS_URLS:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:15]:
                    title = entry.get("title", "")
                    summary = entry.get("summary", "")
                    text = (title + " " + summary).lower()
                    if not any(kw in text for kw in US_KEYWORDS):
                        continue
                    if title in seen:
                        continue
                    seen.add(title)
                    pub = entry.get("published_parsed")
                    if pub:
                        dt = datetime(*pub[:6], tzinfo=timezone.utc).astimezone(IST)
                        date_str = dt.strftime("%Y-%m-%d")
                    else:
                        date_str = now.strftime("%Y-%m-%d")
                    # Classify impact
                    high_kw = ["fed ", "fomc", "rate decision", "interest rate"]
                    impact  = "HIGH" if any(k in text for k in high_kw) else "MEDIUM"
                    events.append({
                        "title":   title[:80],
                        "date":    date_str,
                        "impact":  impact,
                        "country": "US",
                        "note":    summary[:120] if summary else "",
                        "source":  "rss",
                    })
                    if len([e for e in events if e["source"] == "rss"]) >= 8:
                        break
            except Exception:
                continue
    except ImportError:
        pass

    # ── Sort by date ─────────────────────────────────────────────────────────
    events.sort(key=lambda x: x["date"])

    # ── Days-until annotation ─────────────────────────────────────────────────
    for e in events:
        try:
            event_date = datetime.strptime(e["date"], "%Y-%m-%d").replace(tzinfo=IST)
            days_left  = (event_date.date() - now.date()).days
            e["days_left"] = days_left
        except Exception:
            e["days_left"] = 999

    high   = [e for e in events if e["impact"] == "HIGH"]
    medium = [e for e in events if e["impact"] == "MEDIUM"]
    low    = [e for e in events if e["impact"] == "LOW"]

    return {
        "ok":     True,
        "events": events,
        "counts": {"high": len(high), "medium": len(medium), "low": len(low)},
    }


# ── API: EVENT BUS STATUS ─────────────────────────────────────────────────────
@api.get("/event-bus")
async def api_event_bus():
    try:
        from src.streaming.event_bus import event_bus
        return {
            "ok": True,
            "running":       getattr(event_bus, "_running", False),
            "queue_size":    getattr(event_bus, "_queue", None) and event_bus._queue.qsize() or 0,
            "handler_count": len(getattr(event_bus, "_handlers", {})),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


app.include_router(api)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════╗
║  AI Trading Agent — FastAPI v3                       ║
║                                                      ║
║  Dashboard:  http://localhost:8010                   ║
║  API Docs:   http://localhost:8010/docs              ║
║  WebSocket:  ws://localhost:8010/ws/prices           ║
╚══════════════════════════════════════════════════════╝
    """)
    uvicorn.run(
        "web.main:app",
        host             = "0.0.0.0",
        port             = 8010,
        reload           = False,
        log_level        = "info",
        ws_ping_interval = 20,
        ws_ping_timeout  = 20,
    )