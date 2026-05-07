# trading-agent/src/alerts/alert_manager.py
"""
Alert Manager — monitors live prices and triggers alerts.

Alert types:
  - PRICE_ABOVE    : price crosses above threshold
  - PRICE_BELOW    : price crosses below threshold
  - RSI_OVERBOUGHT : RSI > level (default 70)
  - RSI_OVERSOLD   : RSI < level (default 30)
  - SIGNAL_CHANGE  : BUY or SELL signal generated
  - NEWS_SENTIMENT : sentiment score crosses threshold

Storage: JSON file (no database needed).
Each alert fires once then goes TRIGGERED (can be re-armed).
"""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional
from loguru import logger

ALERTS_DB = Path("data/alerts.json")


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class Alert:
    id:           str
    symbol:       str
    alert_type:   str        # PRICE_ABOVE | PRICE_BELOW | RSI_OVERBOUGHT |
                             # RSI_OVERSOLD | SIGNAL_CHANGE | NEWS_SENTIMENT
    threshold:    float      # price level, RSI level, sentiment score
    label:        str        # human-readable name, e.g. "Nifty 22500 breakout"
    created_at:   str
    status:       str        # ACTIVE | TRIGGERED | DISABLED
    notify_email: bool       = True
    notify_ui:    bool       = True
    triggered_at: Optional[str]   = None
    triggered_val: Optional[float] = None
    message:      Optional[str]   = None
    # For SIGNAL_CHANGE: which signals to watch
    watch_signals: list = field(default_factory=lambda: ["BUY", "SELL"])


@dataclass
class AlertFire:
    alert_id:   str
    symbol:     str
    alert_type: str
    label:      str
    message:    str
    value:      float
    threshold:  float
    fired_at:   str


class AlertManager:
    """
    Checks live market data against registered alerts.
    Fires notifications via UI toast and/or email.
    """

    def __init__(self, email_sender=None, telegram_sender=None):
        ALERTS_DB.parent.mkdir(parents=True, exist_ok=True)
        self._db           = self._load()
        self._email        = email_sender
        self._telegram     = telegram_sender
        self._fired_this_session: set = set()
        logger.info(f"AlertManager | {len(self._db['alerts'])} alerts loaded")

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def add_alert(
        self,
        symbol:       str,
        alert_type:   str,
        threshold:    float,
        label:        str       = "",
        notify_email: bool      = True,
        notify_ui:    bool      = True,
        watch_signals: list     = None,
    ) -> str:
        aid = str(uuid.uuid4())[:8]
        if not label:
            label = f"{symbol} {alert_type} {threshold}"
        alert = Alert(
            id            = aid,
            symbol        = symbol.upper(),
            alert_type    = alert_type.upper(),
            threshold     = threshold,
            label         = label,
            created_at    = datetime.now(timezone.utc).isoformat(),
            status        = "ACTIVE",
            notify_email  = notify_email,
            notify_ui     = notify_ui,
            watch_signals = watch_signals or ["BUY", "SELL"],
        )
        self._db["alerts"].append(asdict(alert))
        self._save()
        logger.info(f"Alert added: {aid} — {label}")
        return aid

    def remove_alert(self, alert_id: str) -> bool:
        before = len(self._db["alerts"])
        self._db["alerts"] = [a for a in self._db["alerts"] if a["id"] != alert_id]
        if len(self._db["alerts"]) < before:
            self._save()
            return True
        return False

    def rearm_alert(self, alert_id: str) -> bool:
        for a in self._db["alerts"]:
            if a["id"] == alert_id:
                a["status"]        = "ACTIVE"
                a["triggered_at"]  = None
                a["triggered_val"] = None
                a["message"]       = None
                self._save()
                self._fired_this_session.discard(alert_id)
                return True
        return False

    def disable_alert(self, alert_id: str) -> bool:
        for a in self._db["alerts"]:
            if a["id"] == alert_id:
                a["status"] = "DISABLED"
                self._save()
                return True
        return False

    def get_all_alerts(self) -> list:
        return self._db["alerts"]

    def get_active_alerts(self) -> list:
        return [a for a in self._db["alerts"] if a["status"] == "ACTIVE"]

    def get_fired_alerts(self) -> list:
        return [a for a in self._db["alerts"] if a["status"] == "TRIGGERED"]

    # ── Check engine ─────────────────────────────────────────────────────────

    def check_all(self, prices: dict, features: dict = None,
                  signals: dict = None, sentiment: dict = None) -> list[AlertFire]:
        """
        Main check loop. Call this every refresh cycle.

        Args:
            prices:    {symbol: float}  — latest prices
            features:  {symbol: {rsi_14: float, ...}}  — latest indicators
            signals:   {symbol: "BUY"|"SELL"|"HOLD"}  — latest signals
            sentiment: {symbol: float}  — news sentiment scores (-1 to +1)

        Returns:
            List of AlertFire objects that fired this cycle.
        """
        fired  = []
        active = self.get_active_alerts()

        for alert in active:
            aid    = alert["id"]
            sym    = alert["symbol"]
            atype  = alert["alert_type"]

            # Skip if already fired this session
            if aid in self._fired_this_session:
                continue

            fire: Optional[AlertFire] = None

            try:
                if atype == "PRICE_ABOVE":
                    px = prices.get(sym)
                    if px and px > alert["threshold"]:
                        fire = AlertFire(
                            alert_id   = aid,
                            symbol     = sym,
                            alert_type = atype,
                            label      = alert["label"],
                            message    = f"{sym} price ₹{px:,.2f} crossed ABOVE ₹{alert['threshold']:,.2f}",
                            value      = px,
                            threshold  = alert["threshold"],
                            fired_at   = datetime.now(timezone.utc).isoformat(),
                        )

                elif atype == "PRICE_BELOW":
                    px = prices.get(sym)
                    if px and px < alert["threshold"]:
                        fire = AlertFire(
                            alert_id   = aid,
                            symbol     = sym,
                            alert_type = atype,
                            label      = alert["label"],
                            message    = f"{sym} price ₹{px:,.2f} dropped BELOW ₹{alert['threshold']:,.2f}",
                            value      = px,
                            threshold  = alert["threshold"],
                            fired_at   = datetime.now(timezone.utc).isoformat(),
                        )

                elif atype == "RSI_OVERBOUGHT":
                    rsi = (features or {}).get(sym, {}).get("rsi_14")
                    if rsi and rsi > alert["threshold"]:
                        fire = AlertFire(
                            alert_id   = aid,
                            symbol     = sym,
                            alert_type = atype,
                            label      = alert["label"],
                            message    = f"{sym} RSI {rsi:.1f} is OVERBOUGHT (>{alert['threshold']:.0f})",
                            value      = rsi,
                            threshold  = alert["threshold"],
                            fired_at   = datetime.now(timezone.utc).isoformat(),
                        )

                elif atype == "RSI_OVERSOLD":
                    rsi = (features or {}).get(sym, {}).get("rsi_14")
                    if rsi and rsi < alert["threshold"]:
                        fire = AlertFire(
                            alert_id   = aid,
                            symbol     = sym,
                            alert_type = atype,
                            label      = alert["label"],
                            message    = f"{sym} RSI {rsi:.1f} is OVERSOLD (<{alert['threshold']:.0f})",
                            value      = rsi,
                            threshold  = alert["threshold"],
                            fired_at   = datetime.now(timezone.utc).isoformat(),
                        )

                elif atype == "SIGNAL_CHANGE":
                    sig = (signals or {}).get(sym)
                    if sig and sig in alert.get("watch_signals", ["BUY", "SELL"]):
                        px  = prices.get(sym, 0)
                        fire = AlertFire(
                            alert_id   = aid,
                            symbol     = sym,
                            alert_type = atype,
                            label      = alert["label"],
                            message    = f"{sym} signal changed to {sig} at ₹{px:,.2f}",
                            value      = px,
                            threshold  = 0,
                            fired_at   = datetime.now(timezone.utc).isoformat(),
                        )

                elif atype == "NEWS_SENTIMENT":
                    score = (sentiment or {}).get(sym)
                    if score is not None:
                        thr = alert["threshold"]
                        if (thr > 0 and score > thr) or (thr < 0 and score < thr):
                            direction = "BULLISH" if score > 0 else "BEARISH"
                            fire = AlertFire(
                                alert_id   = aid,
                                symbol     = sym,
                                alert_type = atype,
                                label      = alert["label"],
                                message    = f"{sym} news sentiment {direction}: {score:+.2f} (threshold {thr:+.2f})",
                                value      = score,
                                threshold  = thr,
                                fired_at   = datetime.now(timezone.utc).isoformat(),
                            )

            except Exception as e:
                logger.warning(f"Alert check error [{aid}]: {e}")
                continue

            if fire:
                self._trigger(alert, fire)
                fired.append(fire)

                # Email notification
                if alert.get("notify_email") and self._email:
                    try:
                        self._email.send_alert(fire)
                    except Exception as e:
                        logger.warning(f"Email send failed: {e}")

                # Telegram notification (always send if configured, regardless of notify_email flag)
                if self._telegram and self._telegram.is_configured:
                    try:
                        self._telegram.send_alert(fire)
                    except Exception as e:
                        logger.warning(f"Telegram send failed: {e}")

        if fired:
            logger.info(f"AlertManager: {len(fired)} alert(s) fired")
        return fired

    # ── Fire history ──────────────────────────────────────────────────────────

    def get_fire_history(self, n: int = 50) -> list:
        history = self._db.get("fire_history", [])
        return sorted(history, key=lambda x: x["fired_at"], reverse=True)[:n]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _trigger(self, alert: dict, fire: AlertFire):
        alert["status"]        = "TRIGGERED"
        alert["triggered_at"]  = fire.fired_at
        alert["triggered_val"] = fire.value
        alert["message"]       = fire.message
        self._fired_this_session.add(alert["id"])

        # Append to fire history
        self._db.setdefault("fire_history", []).append(asdict(fire))
        # Keep last 200 history entries
        self._db["fire_history"] = self._db["fire_history"][-200:]
        self._save()
        logger.info(f"ALERT FIRED: {fire.message}")

    def _load(self) -> dict:
        if ALERTS_DB.exists():
            try:
                with open(ALERTS_DB) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"alerts": [], "fire_history": [], "version": 1}

    def _save(self):
        with open(ALERTS_DB, "w") as f:
            json.dump(self._db, f, indent=2)
