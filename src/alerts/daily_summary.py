# trading-agent/src/alerts/daily_summary.py
"""
Daily Telegram Summary — fires at 3:30 PM IST every trading day.

Contents:
  - Market overview: Nifty, BankNifty, Gold, Crude, BTC with % change
  - Top signals from watchlist (BUY/SELL with confidence)
  - Paper portfolio P&L
  - Running model accuracy (LearningEngineV2)
  - Tomorrow's watchlist

Two usage modes:
  1. Background thread (started from app.py via DailySummaryScheduler)
  2. Standalone:
       python src/alerts/daily_summary.py --now       # send immediately
       python src/alerts/daily_summary.py --schedule  # run loop forever

Add to app.py cached resources:
    @st.cache_resource
    def get_summary_scheduler():
        try:
            from src.alerts.daily_summary import summary_scheduler
            summary_scheduler.start()
            return summary_scheduler
        except Exception:
            return None

Then call get_summary_scheduler() once near the top of app.py (outside any page block).
"""
import sys
import time
import argparse
import threading
from datetime import datetime, date
from pathlib import Path
from loguru import logger
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IST            = ZoneInfo("Asia/Kolkata")
SUMMARY_HOUR   = 15   # 3 PM IST
SUMMARY_MINUTE = 30   # :30

WATCHLIST = [
    "NIFTY50", "BANKNIFTY", "RELIANCE", "TCS",
    "HDFCBANK", "GOLD", "CRUDEOIL", "BTC",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _is_holiday() -> bool:
    try:
        from src.utils.market_hours import market_hours
        s = market_hours.get_full_status()
        return s.is_nse_holiday or datetime.now(IST).weekday() >= 5
    except Exception:
        return datetime.now(IST).weekday() >= 5


def _fetch_price_change(symbol: str) -> tuple[float, float]:
    """(price, change_pct) — both 0 on failure."""
    try:
        from src.data.manager import DataManager
        from src.data.models import Interval
        dm = DataManager()
        df = dm.get_ohlcv(symbol, Interval.D1, days_back=5)
        if df.empty or len(df) < 2:
            return 0.0, 0.0
        price = float(df["close"].iloc[-1])
        prev  = float(df["close"].iloc[-2])
        return price, (price - prev) / prev * 100 if prev > 0 else 0.0
    except Exception:
        return 0.0, 0.0


def _scan_signals() -> list[dict]:
    """Rule-based signal scan for WATCHLIST."""
    out = []
    try:
        from src.data.manager import DataManager
        from src.data.models import Interval
        from src.features.feature_engine import FeatureEngine
        dm = DataManager()
        fe = FeatureEngine()
        for sym in WATCHLIST:
            try:
                df = dm.get_ohlcv(sym, Interval.D1, days_back=250)
                if df.empty or len(df) < 50:
                    continue
                ft = fe.build(df, drop_na=False)
                if ft.empty:
                    continue
                lat = ft.iloc[-1].to_dict()

                score = 0
                rsi   = lat.get("rsi_14", 50)
                mh    = lat.get("macd_hist", 0)
                mc    = lat.get("macd_hist_chg", 0)
                e9    = lat.get("ema9_pct", 0)
                e50   = lat.get("ema50_pct", 0)

                if rsi < 35:   score += 2
                elif rsi > 65: score -= 2
                elif rsi > 55: score += 1
                elif rsi < 45: score -= 1

                if mh > 0 and mc > 0:   score += 2
                elif mh < 0 and mc < 0: score -= 2

                if e9 > 0 and e50 > 0:   score += 2
                elif e9 < 0 and e50 < 0: score -= 2

                if score >= 3:    bias, conf = "STRONG BUY",  0.78
                elif score >= 1:  bias, conf = "BUY",          0.62
                elif score <= -3: bias, conf = "STRONG SELL",  0.78
                elif score <= -1: bias, conf = "SELL",          0.62
                else:             bias, conf = "NEUTRAL",       0.45

                out.append({
                    "symbol": sym,
                    "bias":   bias,
                    "conf":   conf,
                    "price":  float(df["close"].iloc[-1]),
                    "rsi":    rsi,
                    "score":  score,
                })
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Signal scan error: {e}")
    return out


def _portfolio_summary() -> dict:
    try:
        from src.execution.paper_broker import PaperBroker
        from config.settings import settings
        return PaperBroker(initial_capital=settings.INITIAL_CAPITAL).get_portfolio_summary()
    except Exception:
        return {}


def _accuracy_stats() -> dict:
    try:
        from src.analysis.learning_engine_v2 import LearningEngineV2
        return LearningEngineV2().get_accuracy_stats()
    except Exception:
        return {}


# ── Message builder ────────────────────────────────────────────────────────────

def build_summary() -> str:
    now   = datetime.now(IST)
    today = now.strftime("%d %b %Y, %A")
    t_str = now.strftime("%H:%M IST")

    msg = [
        f"📊 <b>Daily Trading Summary</b>",
        f"<i>{today} | {t_str}</i>",
        "",
        "━━━ 📈 <b>Market Overview</b>",
    ]

    for sym, label in [
        ("NIFTY50",   "Nifty 50"),
        ("BANKNIFTY", "Bank Nifty"),
        ("GOLD",      "Gold"),
        ("CRUDEOIL",  "Crude Oil"),
        ("BTC",       "Bitcoin"),
        ("USDINR",    "USD/INR"),
    ]:
        price, chg = _fetch_price_change(sym)
        if price > 0:
            icon = "🟢" if chg >= 0 else "🔴"
            sign = "+" if chg >= 0 else ""
            msg.append(f"{icon} <b>{label}</b>: ₹{price:,.1f} <code>({sign}{chg:.2f}%)</code>")

    msg += ["", "━━━ 🎯 <b>Today's Signals</b>"]
    signals = _scan_signals()
    actionable = [s for s in signals if s["bias"] != "NEUTRAL"]
    actionable.sort(key=lambda x: abs(x["score"]), reverse=True)

    if actionable:
        for s in actionable[:6]:
            if "STRONG BUY"  in s["bias"]: icon = "🚀"
            elif "BUY"       in s["bias"]: icon = "🟢"
            elif "STRONG SELL" in s["bias"]: icon = "💥"
            elif "SELL"      in s["bias"]: icon = "🔴"
            else:                           icon = "⚪"
            msg.append(
                f"{icon} <b>{s['symbol']}</b>: {s['bias']} "
                f"<code>{s['conf']:.0%}</code> | RSI {s['rsi']:.0f}"
            )
    else:
        msg.append("⚪ No strong signals — market ranging.")

    msg += ["", "━━━ 💼 <b>Portfolio P&L</b>"]
    ps = _portfolio_summary()
    if ps:
        pnl     = ps.get("total_pnl", 0)
        pnl_pct = ps.get("total_pnl_pct", 0)
        cash    = ps.get("cash", 0)
        n_pos   = ps.get("n_positions", 0)
        charges = ps.get("total_charges", 0)
        sign    = "+" if pnl >= 0 else ""
        icon    = "📈" if pnl >= 0 else "📉"
        msg.append(f"{icon} P&L: <code>{sign}₹{pnl:,.0f} ({sign}{pnl_pct:.2f}%)</code>")
        msg.append(f"💵 Cash: <code>₹{cash:,.0f}</code> | Positions: <code>{n_pos}</code>")
        msg.append(f"💸 Charges paid: <code>₹{charges:.2f}</code>")
    else:
        msg.append("⚠️ Portfolio data unavailable.")

    msg += ["", "━━━ 🎓 <b>Model Accuracy</b>"]
    acc = _accuracy_stats()
    if acc and acc.get("total", 0) > 5:
        total    = acc.get("total", 0)
        accuracy = acc.get("accuracy", 0) * 100
        status   = "✅" if accuracy >= 55 else ("🟡" if accuracy >= 45 else "🔴")
        msg.append(
            f"{status} <code>{accuracy:.1f}%</code> over {total} predictions"
        )
        msg.append("ℹ️ Break-even at 1:1.5 R:R = 40%")
    else:
        msg.append("ℹ️ Not enough predictions yet.")

    msg += ["", "━━━ 🔭 <b>Watch Tomorrow</b>"]
    buys  = sorted([s for s in signals if "BUY"  in s["bias"]], key=lambda x: x["conf"], reverse=True)
    sells = sorted([s for s in signals if "SELL" in s["bias"]], key=lambda x: x["conf"], reverse=True)
    if buys:
        msg.append("🟢 <b>Long:</b> " + " | ".join(
            f"{s['symbol']} <code>{s['conf']:.0%}</code>" for s in buys[:4]
        ))
    if sells:
        msg.append("🔴 <b>Short:</b> " + " | ".join(
            f"{s['symbol']} <code>{s['conf']:.0%}</code>" for s in sells[:4]
        ))
    if not buys and not sells:
        msg.append("⚪ No directional signals — wait for setup.")

    msg += [
        "",
        "━━━━━━━━━━━━━━━━━━",
        "🤖 <i>AI Trading Agent | Auto at 3:30 PM IST</i>",
    ]
    return "\n".join(msg)


# ── Send ───────────────────────────────────────────────────────────────────────

def send_daily_summary(force: bool = False) -> bool:
    """Build and send. Returns True on success."""
    if not force and _is_holiday():
        logger.info("Daily summary: skipping holiday/weekend")
        return False
    try:
        text = build_summary()
        from src.alerts.telegram_sender import TelegramSender
        sender = TelegramSender()
        ok = sender.send_message(text)
        if ok:
            logger.info("✅ Daily Telegram summary sent")
        else:
            logger.error("❌ Daily summary: send failed")
        return ok
    except Exception as e:
        logger.error(f"Daily summary error: {e}")
        return False


def send_signal_alert(symbol: str, signal: dict, usdinr: float = 92.46) -> bool:
    """
    Send a rich signal alert to Telegram with SL, Target, exit conditions.
    Called from Signal Scanner or ActionEngine when a signal fires.

    Example:
        from src.alerts.daily_summary import send_signal_alert
        send_signal_alert("NIFTY50", {
            "bias": "BUY",
            "price": 24353.55,
            "confidence": 0.72,
            "regime": "TRENDING_UP",
            "hold_type": "INTRADAY",
            "action": "ENTER",
            "session_label": "Trend Window",
            "entry_timing": "ENTER NOW",
            "reasons": ["RSI 68 bullish", "ADX 38 strong trend"],
        })
    """
    try:
        from src.alerts.signal_formatter import format_signal_telegram
        from src.alerts.telegram_sender import TelegramSender
        msg    = format_signal_telegram(symbol, signal, usdinr=usdinr)
        sender = TelegramSender()
        ok     = sender.send_message(msg)
        if ok:
            logger.info(f"✅ Signal alert sent: {symbol} {signal.get('bias','')}")
        return ok
    except Exception as e:
        logger.error(f"Signal alert error: {e}")
        return False


# ── Scheduler ──────────────────────────────────────────────────────────────────

class DailySummaryScheduler:
    """
    Background thread that fires send_daily_summary() at 15:30 IST.
    Checks every 60 seconds. Sends once per day maximum.

    Usage in app.py:
        @st.cache_resource
        def get_summary_scheduler():
            try:
                from src.alerts.daily_summary import summary_scheduler
                summary_scheduler.start()
                return summary_scheduler
            except Exception:
                return None

        # Call once outside any page block:
        get_summary_scheduler()
    """

    def __init__(self):
        self._running   = False
        self._thread    = None
        self._last_sent: date = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="DailySummary"
        )
        self._thread.start()
        logger.info(
            f"DailySummaryScheduler started — "
            f"fires at {SUMMARY_HOUR:02d}:{SUMMARY_MINUTE:02d} IST"
        )

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            try:
                now   = datetime.now(IST)
                today = now.date()
                if (
                    now.hour   == SUMMARY_HOUR and
                    now.minute == SUMMARY_MINUTE and
                    self._last_sent != today
                ):
                    logger.info("DailySummaryScheduler: firing...")
                    if send_daily_summary():
                        self._last_sent = today
            except Exception as e:
                logger.error(f"DailySummaryScheduler error: {e}")
            time.sleep(60)


# Singleton — import and call .start() once
summary_scheduler = DailySummaryScheduler()


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Daily Telegram summary")
    ap.add_argument("--now",      action="store_true", help="Send immediately")
    ap.add_argument("--schedule", action="store_true", help="Run loop forever")
    args = ap.parse_args()

    if args.now:
        print("Sending now...")
        ok = send_daily_summary(force=True)
        print("✅ Done!" if ok else "❌ Failed — check TELEGRAM_BOT_TOKEN in .env")

    elif args.schedule:
        print(f"Scheduler running — fires at {SUMMARY_HOUR:02d}:{SUMMARY_MINUTE:02d} IST daily")
        print("Ctrl+C to stop.")
        summary_scheduler.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            summary_scheduler.stop()
            print("Stopped.")
    else:
        ap.print_help()