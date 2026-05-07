# trading-agent/src/alerts/telegram_sender.py
"""
Telegram Alert Sender — Bot API with automatic signal enrichment.
Every signal message (regardless of source) is intercepted at _send()
and converted from old bare format to rich format with SL/Target/exits.
"""
import re
import requests
from datetime import datetime, timezone
from loguru import logger

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"

ALERT_EMOJIS = {
    "PRICE_ABOVE":    "📈", "PRICE_BELOW":    "📉",
    "RSI_OVERBOUGHT": "🔴", "RSI_OVERSOLD":   "🟢",
    "SIGNAL_CHANGE":  "🎯", "NEWS_SENTIMENT": "📰",
    "AUTO_EXIT":      "🚨", "DAILY_SUMMARY":  "📊",
    "TRADE_EXECUTED": "✅", "SYSTEM":         "⚙️",
}

REGIME_EMOJIS = {
    "TRENDING_UP": "📈", "TRENDING_DOWN": "📉",
    "RANGING": "↔️",    "VOLATILE": "⚡",
}


def _enrich_if_signal(text: str) -> str:
    """
    Intercepts old bare signal format (from realtime_advisor.py) and
    converts it to the rich format with SL, Target, R:R, exit conditions.

    Old format detected by: has "Action:" + "Confidence:" but NOT "Stop Loss"
    Works for ALL symbols: BTC, NIFTY50, CRUDEOIL, etc.
    """
    if not text:
        return text
    # Already rich — skip
    if any(x in text for x in ["Stop Loss", "Trade Levels", "Risk:Reward"]):
        return text
    # Not a signal — skip
    if "Action:" not in text or "Confidence:" not in text:
        return text

    try:
        # Parse symbol
        sym_m = re.search(r"<b>(\w+)</b>", text)
        if not sym_m:
            sym_m = re.search(r"(?:System\s+)?([A-Z]{2,12})\s+Action:", text)
        if not sym_m:
            return text
        symbol = sym_m.group(1).strip()

        # Parse fields
        act_m   = re.search(r"Action:\s*`?(\w+)`?", text)
        conf_m  = re.search(r"Confidence:\s*`?(\d+)%`?", text)
        price_m = re.search(r"Price:\s*`?([\d,\.]+)`?", text)
        regime_m= re.search(r"Regime:\s*`?(\w+)`?", text)
        sess_m  = re.search(r"Session:\s*`?([^`\n]+?)`?\s*(?:Timing|$|\|)", text)
        time_m  = re.search(r"Timing:\s*`?([^`\n]+?)`?\s*(?:Regime|$|\|)", text)
        reason_m= re.search(r"Reason:\s*(.+?)(?:Warning:|\s{2,}\d{2}:\d{2}|$)", text, re.DOTALL)
        warn_m  = re.search(r"Warning:\s*(.+?)(?:\s{2,}\d{2}:\d{2}|$)", text, re.DOTALL)

        if not act_m or not price_m:
            return text

        action  = act_m.group(1).strip().upper()
        price   = float(price_m.group(1).replace(",", ""))
        conf    = float(conf_m.group(1)) / 100 if conf_m else 0.5
        regime  = regime_m.group(1).strip() if regime_m else "RANGING"
        session = sess_m.group(1).strip() if sess_m else "Standard"
        timing  = time_m.group(1).strip() if time_m else "ENTER"
        reason  = reason_m.group(1).strip().replace("\n", " ")[:120] if reason_m else ""
        warning = warn_m.group(1).strip().replace("\n", " ")[:80] if warn_m else ""

        # Get live USDINR
        usdinr = 92.46
        try:
            from src.streaming.price_store import price_store as _ps
            _u = _ps.get("USDINR", fallback=True)
            if _u and 70 < float(_u) < 120:
                usdinr = float(_u)
        except Exception:
            pass

        # Determine hold type from session
        s_lower = session.lower()
        if any(x in s_lower for x in ["crypto", "24x7"]):
            hold_type = "SWING"
        elif any(x in s_lower for x in ["open", "close", "intraday"]):
            hold_type = "INTRADAY"
        elif any(x in s_lower for x in ["positional", "week"]):
            hold_type = "POSITIONAL"
        else:
            hold_type = "SWING"

        from src.alerts.signal_formatter import format_signal_telegram
        reasons = []
        if reason:  reasons.append(reason)
        if warning: reasons.append(f"⚠️ {warning}")

        return format_signal_telegram(symbol, {
            "bias":          action,
            "price":         price,
            "confidence":    conf,
            "regime":        regime,
            "hold_type":     hold_type,
            "action":        "ENTER",
            "session_label": session,
            "entry_timing":  timing,
            "reasons":       reasons,
        }, usdinr=usdinr)

    except Exception as e:
        logger.debug(f"Signal enrichment skipped: {e}")
        return text


class TelegramSender:
    """Sends formatted alert messages to Telegram. Signals auto-enriched."""

    TIMEOUT = 10

    def __init__(self, bot_token: str = None, chat_id: str = None):
        # Auto-load from settings if not provided
        if not bot_token or not chat_id:
            try:
                from config.settings import settings
                bot_token = bot_token or getattr(settings, "TELEGRAM_BOT_TOKEN", "") or ""
                chat_id   = chat_id   or getattr(settings, "TELEGRAM_CHAT_ID",   "") or ""
            except Exception:
                pass

        self.bot_token = (bot_token or "").strip()
        self.chat_id   = (chat_id   or "").strip()
        self._enabled  = bool(self.bot_token and self.chat_id)

        if self._enabled:
            logger.info(f"TelegramSender ready | chat_id={self.chat_id}")
        else:
            logger.info("TelegramSender: not configured (add to .env)")

    @property
    def is_configured(self) -> bool:
        return self._enabled

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send any text message — signals auto-enriched with SL/Target."""
        if not self._enabled:
            return False
        return self._send(text)

    def send_alert(self, fire) -> bool:
        if not self._enabled:
            return False
        return self._send(self._format_alert(fire))

    def send_trade(self, symbol, side, quantity, price, sl, target,
                   charges, is_live=False, order_id="") -> bool:
        if not self._enabled:
            return False
        mode     = "🔴 LIVE" if is_live else "📄 Paper"
        side_sym = "▲ BUY"  if side == "BUY" else "▼ SELL"
        rr       = abs(target - price) / abs(price - sl) if abs(price - sl) > 0 else 0
        now      = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        msg = (
            f"✅ <b>Trade Executed</b> [{mode}]\n\n"
            f"<b>{side_sym} {quantity}× {symbol}</b>\n"
            f"Entry:   <code>₹{price:,.2f}</code>\n"
            f"Stop SL: <code>₹{sl:,.2f}</code>\n"
            f"Target:  <code>₹{target:,.2f}</code>\n"
            f"R:R:     <code>1:{rr:.1f}</code>\n"
            f"Charges: <code>₹{charges:.2f}</code>\n"
        )
        if order_id:
            msg += f"Order ID: <code>{order_id}</code>\n"
        msg += f"\n<i>{now}</i>"
        return self._send(msg)

    def send_auto_exit(self, symbol, reason, price, pnl) -> bool:
        if not self._enabled:
            return False
        pnl_sym = "🟢 +" if pnl >= 0 else "🔴 "
        now     = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        msg = (
            f"🚨 <b>Auto Exit Triggered</b>\n\n"
            f"<b>{symbol}</b> @ <code>₹{price:,.2f}</code>\n"
            f"Reason: {reason}\n"
            f"P&L: {pnl_sym}<code>₹{abs(pnl):,.0f}</code>\n\n"
            f"<i>{now}</i>"
        )
        return self._send(msg)

    def send_daily_summary(self, portfolio_value, daily_pnl, n_trades,
                           n_positions, total_charges, accuracy_pct=None) -> bool:
        if not self._enabled:
            return False
        pnl_sym  = "🟢 +" if daily_pnl >= 0 else "🔴 "
        date_str = datetime.now(timezone.utc).strftime("%d %b %Y")
        msg = (
            f"📊 <b>Daily Summary — {date_str}</b>\n\n"
            f"Portfolio Value: <code>₹{portfolio_value:,.0f}</code>\n"
            f"Today's P&L:     {pnl_sym}<code>₹{abs(daily_pnl):,.0f}</code>\n"
            f"Trades today:    <code>{n_trades}</code>\n"
            f"Open positions:  <code>{n_positions}</code>\n"
            f"Charges paid:    <code>₹{total_charges:.2f}</code>\n"
        )
        if accuracy_pct is not None:
            msg += f"Model accuracy:  <code>{accuracy_pct:.1f}%</code>\n"
        return self._send(msg)

    def send_regime_change(self, symbol, old_regime, new_regime, adx, confidence) -> bool:
        if not self._enabled:
            return False
        old_icon = REGIME_EMOJIS.get(old_regime, "")
        new_icon = REGIME_EMOJIS.get(new_regime, "")
        now      = datetime.now(timezone.utc).strftime("%H:%M UTC")
        msg = (
            f"🔄 <b>Regime Change — {symbol}</b>\n\n"
            f"{old_icon} {old_regime.replace('_',' ')} → "
            f"{new_icon} <b>{new_regime.replace('_',' ')}</b>\n"
            f"ADX: <code>{adx:.1f}</code> | Confidence: <code>{confidence:.0%}</code>\n\n"
            f"<i>Adjust strategy accordingly</i>\n<i>{now}</i>"
        )
        return self._send(msg)

    def send_system_message(self, text: str) -> bool:
        if not self._enabled:
            return False
        msg = f"⚙️ <b>System</b>\n\n{text}\n\n<i>{datetime.now(timezone.utc).strftime('%H:%M UTC')}</i>"
        return self._send(msg)

    def send_test(self) -> tuple:
        if not self._enabled:
            return False, "Not configured. Add TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID to .env"
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        msg = (
            f"✅ <b>AI Trading Agent — Connected!</b>\n\n"
            f"Signals now include SL + Target + exit conditions.\n\n"
            f"<i>Sent at: {now}</i>"
        )
        ok = self._send(msg)
        return (True, "✅ Test sent") if ok else (False, "❌ Send failed")

    def verify_connection(self) -> tuple:
        if not self._enabled:
            return False, "Not configured"
        try:
            url  = TELEGRAM_API.format(token=self.bot_token, method="getMe")
            resp = requests.get(url, timeout=self.TIMEOUT)
            data = resp.json()
            if data.get("ok"):
                bot_name = data["result"].get("first_name", "Unknown")
                username = data["result"].get("username", "")
                return True, f"✅ Bot: {bot_name} (@{username})"
            return False, f"❌ {data.get('description', 'Unknown error')}"
        except Exception as e:
            return False, f"❌ Connection error: {e}"

    def _send(self, text: str) -> bool:
        """
        THE single send point — ALL messages pass through here.
        Old-format signal messages are enriched before sending.
        """
        # ── Intercept and enrich old bare signal format ─────────────────────
        text = _enrich_if_signal(text)

        try:
            url  = TELEGRAM_API.format(token=self.bot_token, method="sendMessage")
            data = {
                "chat_id":                  self.chat_id,
                "text":                     text,
                "parse_mode":               "HTML",
                "disable_web_page_preview": True,
            }
            resp   = requests.post(url, data=data, timeout=self.TIMEOUT)
            result = resp.json()
            if result.get("ok"):
                return True
            logger.warning(f"Telegram send failed: {result.get('description')}")
            return False
        except requests.Timeout:
            logger.warning("Telegram send timed out")
            return False
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False

    def _format_alert(self, fire) -> str:
        if hasattr(fire, "__dataclass_fields__"):
            from dataclasses import asdict
            f = asdict(fire)
        elif hasattr(fire, "__dict__"):
            f = fire.__dict__
        else:
            f = dict(fire)
        atype   = f.get("alert_type", "ALERT")
        symbol  = f.get("symbol",     "")
        message = f.get("message",    "Alert triggered")
        val     = f.get("value",      0)
        thr     = f.get("threshold",  0)
        fired   = f.get("fired_at",   "")[:16].replace("T", " ")
        label   = f.get("label",      "")
        icon     = ALERT_EMOJIS.get(atype, "🔔")
        type_str = atype.replace("_", " ").title()
        if atype in ("PRICE_ABOVE", "PRICE_BELOW"):
            val_str = f"₹{val:,.2f}"; thr_str = f"₹{thr:,.2f}"
        elif "RSI" in atype:
            val_str = f"{val:.1f}"; thr_str = f"{thr:.0f}"
        elif atype == "NEWS_SENTIMENT":
            val_str = f"{val:+.2f}"; thr_str = f"{thr:+.2f}"
        else:
            val_str = str(round(val, 2)); thr_str = str(round(thr, 2))
        return (
            f"{icon} <b>{type_str}</b>\n\n"
            f"<b>{symbol}</b> — {label}\n\n"
            f"{message}\n\n"
            f"Value:     <code>{val_str}</code>\n"
            f"Threshold: <code>{thr_str}</code>\n"
            f"Time:      <code>{fired} UTC</code>"
        )


def make_telegram_sender_from_settings(settings) -> TelegramSender:
    token   = getattr(settings, "TELEGRAM_BOT_TOKEN", None) or ""
    chat_id = getattr(settings, "TELEGRAM_CHAT_ID",   None) or ""
    return TelegramSender(bot_token=token, chat_id=chat_id)