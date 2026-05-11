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

ROOT = Path(__file__).resolve().parents[2]  # src/alerts → src → project root
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

def _bar(value: float, max_val: float, width: int = 10, fill: str = "█", empty: str = "░") -> str:
    """Mini text progress bar for Telegram."""
    if max_val == 0:
        return empty * width
    pct   = min(1.0, abs(value) / max_val)
    filled = round(pct * width)
    return fill * filled + empty * (width - filled)


def _fetch_fii_dii() -> dict:
    try:
        from src.analysis.fii_dii_tracker import FIIDIITracker
        tracker = FIIDIITracker()
        data    = tracker.get_latest()
        if data:
            return {"fii": data.fii_net_cr, "dii": data.dii_net_cr, "source": data.source}
    except Exception:
        pass
    return {"fii": 0, "dii": 0, "source": "unavailable"}


def _fetch_regime() -> str:
    try:
        from src.analysis.regime_detector import RegimeDetector
        from src.data.manager import DataManager
        from src.data.models import Interval
        dm     = DataManager()
        df     = dm.get_ohlcv("NIFTY50", Interval.D1, days_back=120)
        if df.empty:
            return "UNKNOWN"
        rd     = RegimeDetector()
        result = rd.detect(df["close"])
        return getattr(result, "regime", "UNKNOWN")
    except Exception:
        return "UNKNOWN"


def build_summary() -> str:
    now   = datetime.now(IST)
    today = now.strftime("%d %b %Y")
    dow   = now.strftime("%A")
    t_str = now.strftime("%H:%M IST")

    # ── Gather all data upfront ───────────────────────────────────────────
    market_data = {}
    for sym, label, fmt in [
        ("NIFTY50",   "NIFTY",    "pts"),
        ("BANKNIFTY", "BNKN",     "pts"),
        ("SENSEX",    "SENSEX",   "pts"),
        ("GOLD",      "GOLD",     "₹/10g"),
        ("SILVER",    "SILVER",   "₹/kg"),
        ("CRUDEOIL",  "CRUDE",    "₹/bbl"),
        ("BTC",       "BTC",      "₹"),
        ("ETH",       "ETH",      "₹"),
        ("USDINR",    "USD/INR",  "₹"),
    ]:
        price, chg = _fetch_price_change(sym)
        if price > 0:
            market_data[sym] = {"label": label, "price": price, "chg": chg, "fmt": fmt}

    signals = _scan_signals()
    ps      = _portfolio_summary()
    acc     = _accuracy_stats()
    fii_dii = _fetch_fii_dii()
    regime  = _fetch_regime()

    # ── Header ────────────────────────────────────────────────────────────
    regime_icons = {
        "TRENDING_UP": "📈 TRENDING UP", "TRENDING_DOWN": "📉 TRENDING DOWN",
        "RANGING": "↔️ RANGING", "VOLATILE": "⚡ VOLATILE", "UNKNOWN": "◼ UNKNOWN",
    }
    regime_label = regime_icons.get(regime.upper(), f"◼ {regime}")

    lines = [
        "╔══════════════════════════╗",
        f"║  📊 <b>AI TRADING TERMINAL</b>  ║",
        "╚══════════════════════════╝",
        f"<code>{today} {dow}  {t_str}</code>",
        f"<b>Regime:</b> {regime_label}",
        "",
    ]

    # ── Market Overview ───────────────────────────────────────────────────
    lines.append("─── <b>MARKET CLOSE</b> ──────────────")
    indices = ["NIFTY50", "BANKNIFTY", "SENSEX"]
    for sym in indices:
        d = market_data.get(sym)
        if not d:
            continue
        arrow = "▲" if d["chg"] >= 0 else "▼"
        sign  = "+" if d["chg"] >= 0 else ""
        chg_str = f"{arrow}{sign}{d['chg']:.2f}%"
        lines.append(
            f"<code>{d['label']:<8}</code> <b>{d['price']:>10,.1f}</b>  "
            f"<code>{chg_str}</code>"
        )

    lines.append("")
    commodities = ["GOLD", "SILVER", "CRUDEOIL"]
    lines.append("─── <b>COMMODITIES</b> ───────────────")
    for sym in commodities:
        d = market_data.get(sym)
        if not d:
            continue
        arrow = "▲" if d["chg"] >= 0 else "▼"
        sign  = "+" if d["chg"] >= 0 else ""
        lines.append(
            f"<code>{d['label']:<8}</code> <b>{d['price']:>10,.1f}</b>  "
            f"<code>{arrow}{sign}{d['chg']:.2f}%</code>"
        )

    lines.append("")
    lines.append("─── <b>CRYPTO &amp; FX</b> ──────────────")
    for sym in ["BTC", "ETH", "USDINR"]:
        d = market_data.get(sym)
        if not d:
            continue
        arrow = "▲" if d["chg"] >= 0 else "▼"
        sign  = "+" if d["chg"] >= 0 else ""
        lines.append(
            f"<code>{d['label']:<8}</code> <b>{d['price']:>10,.1f}</b>  "
            f"<code>{arrow}{sign}{d['chg']:.2f}%</code>"
        )

    # ── FII/DII ───────────────────────────────────────────────────────────
    lines.append("")
    lines.append("─── <b>FII / DII FLOW</b> ────────────")
    fii = fii_dii.get("fii", 0)
    dii = fii_dii.get("dii", 0)
    if fii != 0 or dii != 0:
        fii_bar   = _bar(fii, 5000, width=8)
        dii_bar   = _bar(dii, 5000, width=8)
        fii_icon  = "🟢" if fii >= 0 else "🔴"
        dii_icon  = "🟢" if dii >= 0 else "🔴"
        fii_sign  = "+" if fii >= 0 else ""
        dii_sign  = "+" if dii >= 0 else ""
        fii_label = "BUYING" if fii >= 0 else "SELLING"
        dii_label = "BUYING" if dii >= 0 else "SELLING"
        combined  = fii + dii
        if fii > 0 and dii > 0:
            flow_sig = "🟢 <b>STRONG BULL</b> — both institutional buying"
        elif fii > 500:
            flow_sig = "📈 <b>FII LED</b> — foreign buying, moderate bullish"
        elif fii < -500 and dii > 0:
            flow_sig = "⚪ <b>NEUTRAL</b> — DII absorbing FII selling"
        elif fii < -2000:
            flow_sig = "🔴 <b>BEAR ALERT</b> — heavy FII outflow"
        else:
            flow_sig = "⚪ <b>NEUTRAL</b> — mixed institutional flow"
        lines += [
            f"FII  {fii_icon} <code>{fii_bar}</code> <b>{fii_sign}₹{abs(fii):,.0f} Cr</b> {fii_label}",
            f"DII  {dii_icon} <code>{dii_bar}</code> <b>{dii_sign}₹{abs(dii):,.0f} Cr</b> {dii_label}",
            f"Net  <code>{'+'if combined>=0 else ''}₹{combined:,.0f} Cr combined</code>",
            flow_sig,
        ]
    else:
        lines.append("<i>⚠️ FII/DII data unavailable (market closed)</i>")

    # ── Signals ───────────────────────────────────────────────────────────
    lines.append("")
    lines.append("─── <b>SIGNALS</b> ───────────────────")
    actionable = [s for s in signals if s["bias"] != "NEUTRAL"]
    actionable.sort(key=lambda x: (x["conf"], abs(x["score"])), reverse=True)

    if actionable:
        for s in actionable[:8]:
            bias = s["bias"]
            if   "STRONG BUY"  in bias: icon, clr = "🚀", "▲▲"
            elif "BUY"         in bias: icon, clr = "🟢", "▲ "
            elif "STRONG SELL" in bias: icon, clr = "🔴", "▼▼"
            elif "SELL"        in bias: icon, clr = "🟠", "▼ "
            else:                       icon, clr = "⚪", "  "
            rsi_bar = _bar(s["rsi"] - 50, 30, width=5)
            lines.append(
                f"{icon} <b>{s['symbol']:<12}</b> <code>{clr} {s['conf']:.0%}</code>"
                f"  RSI <code>{s['rsi']:.0f}</code>"
            )
    else:
        lines.append("⚪ <i>No strong signals today — all markets ranging</i>")

    # ── Portfolio ─────────────────────────────────────────────────────────
    lines.append("")
    lines.append("─── <b>PORTFOLIO</b> ─────────────────")
    if ps:
        pnl     = ps.get("total_pnl", 0)
        pnl_pct = ps.get("total_pnl_pct", 0)
        cash    = ps.get("cash", 0)
        n_pos   = ps.get("n_positions", 0)
        charges = ps.get("total_charges", 0)
        equity  = ps.get("total_equity", cash)
        pnl_bar = _bar(pnl_pct, 5.0, width=8)
        pnl_icon = "📈" if pnl >= 0 else "📉"
        sign     = "+" if pnl >= 0 else ""
        lines += [
            f"<code>P&amp;L    </code> {pnl_icon} <b><code>{sign}₹{pnl:>10,.0f}</code></b>  <code>{sign}{pnl_pct:.2f}%</code>",
            f"<code>Equity  ₹{equity:>12,.0f}</code>",
            f"<code>Cash    ₹{cash:>12,.0f}  [{n_pos} pos]</code>",
            f"<code>Charges ₹{charges:>12,.2f}</code>",
        ]
    else:
        lines.append("<i>Portfolio data unavailable</i>")

    # ── Accuracy ──────────────────────────────────────────────────────────
    lines.append("")
    lines.append("─── <b>MODEL ACCURACY</b> ────────────")
    if acc and acc.get("total", 0) > 5:
        total    = acc.get("total", 0)
        accuracy = acc.get("accuracy", 0) * 100
        status   = "✅" if accuracy >= 55 else ("🟡" if accuracy >= 45 else "🔴")
        acc_bar  = _bar(accuracy - 40, 20, width=8)
        lines += [
            f"{status} <b><code>{accuracy:.1f}%</code></b> accuracy  ({total} predictions)",
            f"<code>Break-even 40% @ 1:1.5 R:R</code>",
            f"<code>EV bar: {acc_bar}  {accuracy:.1f}%</code>",
        ]
    else:
        lines.append("<i>Insufficient predictions — keep running signals</i>")

    # ── Watchlist for Tomorrow ────────────────────────────────────────────
    lines.append("")
    lines.append("─── <b>WATCH TOMORROW</b> ────────────")
    buys  = sorted([s for s in signals if "BUY"  in s["bias"]], key=lambda x: x["conf"], reverse=True)
    sells = sorted([s for s in signals if "SELL" in s["bias"]], key=lambda x: x["conf"], reverse=True)
    if buys:
        buy_str = "  ".join(f"<b>{s['symbol']}</b> <code>{s['conf']:.0%}</code>" for s in buys[:4])
        lines.append(f"🟢 Long:  {buy_str}")
    if sells:
        sell_str = "  ".join(f"<b>{s['symbol']}</b> <code>{s['conf']:.0%}</code>" for s in sells[:4])
        lines.append(f"🔴 Short: {sell_str}")
    if not buys and not sells:
        lines.append("⚪ <i>No setups — wait for breakout confirmation</i>")

    # ── Footer ────────────────────────────────────────────────────────────
    lines += [
        "",
        "──────────────────────────────",
        "🤖 <i>AI Trading Agent  |  15:30 IST daily</i>",
    ]

    return "\n".join(lines)


# ── Notification Preferences ───────────────────────────────────────────────────

import json as _json

_PREFS_FILE = Path(__file__).resolve().parents[2] / "data" / "notification_prefs.json"

# Default categories — all enabled by default
_DEFAULT_PREFS = {
    "daily_summary":   {"enabled": True,  "label": "Daily Summary",    "desc": "3:30 PM market wrap"},
    "signal_alert":    {"enabled": True,  "label": "Signal Alerts",    "desc": "BUY/SELL signals"},
    "price_alert":     {"enabled": True,  "label": "Price Alerts",     "desc": "Threshold crossings"},
    "trade_executed":  {"enabled": True,  "label": "Trade Executed",   "desc": "Paper & live fills"},
    "regime_change":   {"enabled": True,  "label": "Regime Change",    "desc": "HMM regime switches"},
    "fii_dii_extreme": {"enabled": True,  "label": "FII/DII Extreme",  "desc": "Flow > ±2000 Cr"},
    "system_alert":    {"enabled": True,  "label": "System Alerts",    "desc": "Errors & warnings"},
}


class NotificationPrefs:
    """Persist per-category notification enable/disable state to JSON."""

    @classmethod
    def load(cls) -> "NotificationPrefs":
        obj = cls()
        try:
            if _PREFS_FILE.exists():
                saved = _json.loads(_PREFS_FILE.read_text())
                # Merge saved into defaults (new categories always enabled by default)
                for k, v in _DEFAULT_PREFS.items():
                    obj._prefs[k] = dict(v)
                    if k in saved and "enabled" in saved[k]:
                        obj._prefs[k]["enabled"] = bool(saved[k]["enabled"])
            else:
                for k, v in _DEFAULT_PREFS.items():
                    obj._prefs[k] = dict(v)
        except Exception:
            for k, v in _DEFAULT_PREFS.items():
                obj._prefs[k] = dict(v)
        return obj

    def __init__(self):
        self._prefs: dict = {}

    def is_enabled(self, category: str) -> bool:
        return self._prefs.get(category, {}).get("enabled", True)

    def set(self, category: str, enabled: bool) -> bool:
        if category not in self._prefs:
            return False
        self._prefs[category]["enabled"] = enabled
        return self._save()

    def all(self) -> dict:
        return {k: dict(v) for k, v in self._prefs.items()}

    def _save(self) -> bool:
        try:
            _PREFS_FILE.parent.mkdir(parents=True, exist_ok=True)
            _PREFS_FILE.write_text(_json.dumps(self._prefs, indent=2))
            return True
        except Exception as e:
            logger.error(f"NotificationPrefs save failed: {e}")
            return False


# ── Send ───────────────────────────────────────────────────────────────────────

def send_daily_summary(force: bool = False) -> bool:
    """Build and send. Returns True on success."""
    if not force and _is_holiday():
        logger.info("Daily summary: skipping holiday/weekend")
        return False
    if not force and not NotificationPrefs.load().is_enabled("daily_summary"):
        logger.info("Daily summary: disabled via notification prefs")
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
    Respects notification_prefs — will no-op if signal_alert is disabled.
    """
    if not NotificationPrefs.load().is_enabled("signal_alert"):
        logger.debug(f"Signal alert suppressed (disabled): {symbol}")
        return False
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