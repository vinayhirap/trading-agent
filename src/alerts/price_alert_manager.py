# trading-agent/src/alerts/price_alert_manager.py
"""
Price Alert Manager — watches live prices and fires Telegram alerts.

Alert types supported:
  ABOVE        — price crosses above a level   (e.g. Nifty > 23000)
  BELOW        — price crosses below a level   (e.g. Gold < 8500)
  PCT_CHANGE   — % move from reference price   (e.g. BTC moves ±3%)
  RSI_OVERSOLD — RSI drops below threshold     (e.g. RSI < 30)
  RSI_OVERBOUGHT — RSI rises above threshold   (e.g. RSI > 70)

Persistence: alerts stored in data/price_alerts.json
Checks every 30 seconds via background thread (started from app.py)

Usage in app.py:
    @st.cache_resource
    def get_alert_manager():
        try:
            from src.alerts.price_alert_manager import alert_manager
            alert_manager.start()
            return alert_manager
        except Exception:
            return None

    get_alert_manager()  # call once at startup

Dashboard Alerts page calls:
    am = get_alert_manager()
    am.add_alert(symbol, alert_type, threshold, note)
    am.get_alerts()       # list all
    am.delete_alert(id)   # remove one
    am.get_history()      # fired alerts
"""
import json
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict
from loguru import logger
from zoneinfo import ZoneInfo

ROOT       = Path(__file__).resolve().parents[2]  # src/alerts → src → project root
DATA_DIR   = ROOT / "data"
ALERTS_FILE = DATA_DIR / "price_alerts.json"
HISTORY_FILE = DATA_DIR / "alert_history.json"
IST        = ZoneInfo("Asia/Kolkata")

CHECK_INTERVAL = 30   # seconds between price checks
MAX_HISTORY    = 200  # keep last N fired alerts


# ── Alert types ────────────────────────────────────────────────────────────────

ALERT_TYPES = {
    "ABOVE":           "Price rises above threshold",
    "BELOW":           "Price falls below threshold",
    "PCT_CHANGE":      "Price moves ±% from reference",
    "RSI_OVERSOLD":    "RSI drops below threshold",
    "RSI_OVERBOUGHT":  "RSI rises above threshold",
}


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class PriceAlert:
    id:           str
    symbol:       str
    alert_type:   str         # ABOVE / BELOW / PCT_CHANGE / RSI_OVERSOLD / RSI_OVERBOUGHT
    threshold:    float       # price level or % or RSI value
    note:         str = ""    # user note, e.g. "Nifty breakout level"
    ref_price:    float = 0.0 # for PCT_CHANGE: price when alert was created
    active:       bool = True
    created_at:   str = ""
    last_checked: str = ""
    fire_once:    bool = True  # delete after firing?

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PriceAlert":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class FiredAlert:
    alert_id:    str
    symbol:      str
    alert_type:  str
    threshold:   float
    actual_value:float
    note:        str
    fired_at:    str
    message:     str


# ── Manager ────────────────────────────────────────────────────────────────────

class PriceAlertManager:
    """
    Background price alert checker.
    Loads alerts from JSON, checks live prices, fires Telegram on trigger.
    """

    def __init__(self):
            self._alerts:  list[PriceAlert] = []
            self._history: list[dict]       = []
            self._running  = False
            self._thread   = None
            self._lock     = threading.Lock()
            self._dm       = None   # lazy init — reused across all RSI checks
            DATA_DIR.mkdir(exist_ok=True)
            self._load()

    # ── Public API ─────────────────────────────────────────────────────────────

    def add_alert(
        self,
        symbol:     str,
        alert_type: str,
        threshold:  float,
        note:       str  = "",
        fire_once:  bool = True,
    ) -> str:
        """
        Add a new price alert. Returns the alert ID.

        Args:
            symbol:     e.g. "NIFTY50", "GOLD", "BTC"
            alert_type: one of ALERT_TYPES keys
            threshold:  price level, % change, or RSI value
            note:       optional human label
            fire_once:  if True, alert is deleted after first trigger
        """
        if alert_type not in ALERT_TYPES:
            raise ValueError(f"Unknown alert_type '{alert_type}'. Use: {list(ALERT_TYPES)}")

        # For PCT_CHANGE, capture current price as reference
        ref_price = 0.0
        if alert_type == "PCT_CHANGE":
            ref_price = self._get_price(symbol) or 0.0

        alert = PriceAlert(
            id         = str(uuid.uuid4())[:8],
            symbol     = symbol.upper(),
            alert_type = alert_type,
            threshold  = float(threshold),
            note       = note or f"{symbol} {alert_type} {threshold}",
            ref_price  = ref_price,
            active     = True,
            created_at = datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"),
            fire_once  = fire_once,
        )
        with self._lock:
            self._alerts.append(alert)
            self._save()
        logger.info(f"Alert added: {alert.symbol} {alert.alert_type} @ {alert.threshold}")
        return alert.id

    def delete_alert(self, alert_id: str) -> bool:
        """Remove alert by ID. Returns True if found."""
        with self._lock:
            before = len(self._alerts)
            self._alerts = [a for a in self._alerts if a.id != alert_id]
            removed = len(self._alerts) < before
            if removed:
                self._save()
        return removed

    def get_alerts(self) -> list[dict]:
        """Return all active alerts as list of dicts."""
        with self._lock:
            return [a.to_dict() for a in self._alerts if a.active]

    def get_all_alerts(self) -> list[dict]:
        """Return ALL alerts including inactive."""
        with self._lock:
            return [a.to_dict() for a in self._alerts]

    def get_history(self, limit: int = 50) -> list[dict]:
        """Return recent fired alert history."""
        with self._lock:
            return list(reversed(self._history[-limit:]))

    def clear_history(self):
        with self._lock:
            self._history = []
            self._save()

    def start(self):
        """Start background check loop (daemon thread)."""
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="PriceAlertManager"
        )
        self._thread.start()
        logger.info(
            f"PriceAlertManager started — "
            f"checking every {CHECK_INTERVAL}s | "
            f"{len(self._alerts)} alert(s) loaded"
        )

    def stop(self):
        self._running = False

    def check_now(self) -> list[FiredAlert]:
        """Manually trigger one check cycle. Returns list of fired alerts."""
        return self._check_all()

    # ── Background loop ────────────────────────────────────────────────────────

    def _loop(self):
        while self._running:
            try:
                fired = self._check_all()
                if fired:
                    logger.info(f"PriceAlertManager: {len(fired)} alert(s) fired")
            except Exception as e:
                logger.error(f"PriceAlertManager loop error: {e}")
            time.sleep(CHECK_INTERVAL)

    def _check_all(self) -> list[FiredAlert]:
        """Check all active alerts against live prices."""
        fired_list = []
        with self._lock:
            active = [a for a in self._alerts if a.active]

        if not active:
            return fired_list

        # Batch fetch prices for unique symbols
        symbols = list({a.symbol for a in active})
        prices  = {}
        rsi_map = {}
        for sym in symbols:
            try:
                prices[sym]  = self._get_price(sym)
                rsi_map[sym] = self._get_rsi(sym)
            except Exception:
                pass

        to_deactivate = []
        for alert in active:
            price = prices.get(alert.symbol, 0)
            rsi   = rsi_map.get(alert.symbol, 50)
            if not price:
                continue

            fired, actual_val = self._check_alert(alert, price, rsi)

            if fired:
                fa = self._fire_alert(alert, actual_val, price)
                fired_list.append(fa)
                if alert.fire_once:
                    to_deactivate.append(alert.id)
                else:
                    # Update last_checked to avoid re-firing every cycle
                    alert.last_checked = datetime.now(IST).isoformat()

        # Deactivate fire_once alerts
        if to_deactivate:
            with self._lock:
                for aid in to_deactivate:
                    for a in self._alerts:
                        if a.id == aid:
                            a.active = False
                self._save()

        return fired_list

    def _check_alert(
        self, alert: PriceAlert, price: float, rsi: float
    ) -> tuple[bool, float]:
        """
        Check if alert condition is met.
        Returns (triggered, actual_value).
        """
        t = alert.alert_type

        if t == "ABOVE":
            return price >= alert.threshold, price

        elif t == "BELOW":
            return price <= alert.threshold, price

        elif t == "PCT_CHANGE":
            if alert.ref_price <= 0:
                return False, price
            pct = abs((price - alert.ref_price) / alert.ref_price * 100)
            return pct >= alert.threshold, pct

        elif t == "RSI_OVERSOLD":
            return rsi <= alert.threshold, rsi

        elif t == "RSI_OVERBOUGHT":
            return rsi >= alert.threshold, rsi

        return False, price

    def _fire_alert(
        self, alert: PriceAlert, actual_val: float, price: float
    ) -> FiredAlert:
        """Build message, send to Telegram, record in history."""
        now_str = datetime.now(IST).strftime("%d %b %Y %H:%M IST")

        # Build message
        type_labels = {
            "ABOVE":          f"📈 Price crossed ABOVE ₹{alert.threshold:,.1f}",
            "BELOW":          f"📉 Price fell BELOW ₹{alert.threshold:,.1f}",
            "PCT_CHANGE":     f"⚡ Price moved {actual_val:.1f}% from ref ₹{alert.ref_price:,.1f}",
            "RSI_OVERSOLD":   f"🟢 RSI OVERSOLD: {actual_val:.0f} ≤ {alert.threshold:.0f}",
            "RSI_OVERBOUGHT": f"🔴 RSI OVERBOUGHT: {actual_val:.0f} ≥ {alert.threshold:.0f}",
        }
        trigger_text = type_labels.get(alert.alert_type, f"Alert triggered @ {actual_val:.2f}")

        msg = (
            f"🔔 <b>PRICE ALERT TRIGGERED</b>\n"
            f"<b>{alert.symbol}</b> | {now_str}\n\n"
            f"{trigger_text}\n"
            f"Current price: <code>₹{price:,.1f}</code>\n"
        )
        if alert.note:
            msg += f"\n📝 <i>{alert.note}</i>"

        # Send Telegram (respects per-category notification prefs)
        try:
            from src.alerts.daily_summary import NotificationPrefs
            if NotificationPrefs.load().is_enabled("price_alert"):
                from src.alerts.telegram_sender import make_telegram_sender_from_settings
                from config.settings import settings
                make_telegram_sender_from_settings(settings).send_message(msg)
                logger.info(f"🔔 Alert fired: {alert.symbol} {alert.alert_type} @ {actual_val:.2f}")
            else:
                logger.debug(f"🔔 Alert fired (Telegram suppressed): {alert.symbol}")
        except Exception as e:
            logger.error(f"Alert Telegram send failed: {e}")
            
        # Record in history
        fa = FiredAlert(
            alert_id    = alert.id,
            symbol      = alert.symbol,
            alert_type  = alert.alert_type,
            threshold   = alert.threshold,
            actual_value= actual_val,
            note        = alert.note,
            fired_at    = now_str,
            message     = msg,
        )
        with self._lock:
            self._history.append(asdict(fa))
            # Trim history
            if len(self._history) > MAX_HISTORY:
                self._history = self._history[-MAX_HISTORY:]
            self._save()

        return fa

    # ── Price / RSI fetchers ───────────────────────────────────────────────────

    def _get_price(self, symbol: str) -> float:
        """Get latest price from PriceStore → yfinance fallback."""
        try:
            from src.streaming.price_store import price_store
            p = price_store.get(symbol)
            if p and p > 0:
                return float(p)
        except Exception:
            pass
        try:
            from src.data.manager import DataManager
            from src.data.models import Interval
            dm = DataManager()
            df = dm.get_ohlcv(symbol, Interval.D1, days_back=3)
            if not df.empty:
                return float(df["close"].iloc[-1])
        except Exception:
            pass
        return 0.0

    def _get_rsi(self, symbol: str) -> float:
        """Get latest RSI(14) for symbol."""
        try:
            from src.data.manager import DataManager
            from src.data.models import Interval
            from src.features.feature_engine import FeatureEngine
            if self._dm is None:
                self._dm = DataManager()
            dm = self._dm
            df = dm.get_ohlcv(symbol, Interval.D1, days_back=60)
            if df.empty or len(df) < 20:
                return 50.0
            fe = FeatureEngine()
            ft = fe.build(df, drop_na=False)
            if ft.empty or "rsi_14" not in ft.columns:
                return 50.0
            return float(ft["rsi_14"].iloc[-1])
        except Exception:
            return 50.0

    # ── Persistence ────────────────────────────────────────────────────────────

    def _save(self):
        try:
            data = {
                "alerts":  [a.to_dict() for a in self._alerts],
                "history": self._history,
            }
            ALERTS_FILE.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"PriceAlertManager save failed: {e}")

    def _load(self):
        try:
            if ALERTS_FILE.exists():
                data = json.loads(ALERTS_FILE.read_text())
                self._alerts  = [PriceAlert.from_dict(d) for d in data.get("alerts",  [])]
                self._history = data.get("history", [])
                logger.info(
                    f"PriceAlertManager loaded: "
                    f"{len(self._alerts)} alerts, {len(self._history)} history"
                )
        except Exception as e:
            logger.warning(f"PriceAlertManager load failed (starting fresh): {e}")
            self._alerts  = []
            self._history = []


# Module-level singleton
alert_manager = PriceAlertManager()