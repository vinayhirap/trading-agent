"""
market_alerts.py
Regime-change and FII/DII extreme event Telegram alerts.
Called from realtime_advisor and fii_dii_tracker.
"""
from __future__ import annotations
from loguru import logger


REGIME_ICONS: dict[str, str] = {
    "TRENDING_UP":   "📈",
    "TRENDING_DOWN": "📉",
    "RANGING":       "↔️",
    "VOLATILE":      "⚡",
}


def send_regime_change_alert(symbol: str, old_regime: str, new_regime: str,
                              confidence: float, rsi: float) -> bool:
    """Send Telegram alert when HMM regime switches for a symbol."""
    try:
        from src.alerts.daily_summary import NotificationPrefs
        if not NotificationPrefs.load().is_enabled("regime_change"):
            return False
        from src.alerts.telegram_sender import TelegramSender
        old_icon = REGIME_ICONS.get(old_regime, "◼")
        new_icon = REGIME_ICONS.get(new_regime, "◼")
        old_label = old_regime.replace("_", " ")
        new_label = new_regime.replace("_", " ")
        conf_str  = str(round(confidence * 100)) + "%"
        rsi_str   = str(round(rsi, 1))
        lines = [
            "🌐 <b>Regime Change: " + symbol + "</b>",
            old_icon + " " + old_label + " → " + new_icon + " " + new_label,
            "<code>Confidence: " + conf_str + "  |  RSI: " + rsi_str + "</code>",
        ]
        ok = TelegramSender().send_message("\n".join(lines))
        if ok:
            logger.info("Regime change alert sent: " + symbol + " " + old_regime + "->" + new_regime)
        return ok
    except Exception as e:
        logger.debug("Regime change alert failed (" + symbol + "): " + str(e))
        return False


def send_fii_dii_extreme_alert(fii_net: float, dii_net: float, date_str: str) -> bool:
    """Send Telegram alert when FII or DII flow exceeds ±2000 Cr (extreme reading)."""
    try:
        from src.alerts.daily_summary import NotificationPrefs
        if not NotificationPrefs.load().is_enabled("fii_dii_extreme"):
            return False
        from src.alerts.telegram_sender import TelegramSender

        combined = fii_net + dii_net

        # Determine signal
        if fii_net > 2000 and dii_net > 0:
            signal = "🟢 <b>EXTREME BULL FLOW</b> — FII + DII both buying heavily"
        elif fii_net < -2000 and dii_net < 0:
            signal = "🔴 <b>EXTREME BEAR FLOW</b> — FII + DII both selling"
        elif fii_net < -2000:
            signal = "🔴 <b>HEAVY FII SELLING</b> — watch for Nifty weakness"
        elif fii_net > 2000:
            signal = "🟢 <b>HEAVY FII BUYING</b> — Nifty likely bullish"
        elif abs(dii_net) > 2000:
            signal = "📡 <b>EXTREME DII FLOW</b> — domestic institutions moving large"
        else:
            signal = "📡 <b>EXTREME INSTITUTIONAL FLOW</b>"

        fii_sign = "+" if fii_net >= 0 else ""
        dii_sign = "+" if dii_net >= 0 else ""
        comb_sign = "+" if combined >= 0 else ""

        lines = [
            "📡 <b>FII/DII Extreme Alert</b>  <code>" + date_str + "</code>",
            "",
            "FII  <b><code>" + fii_sign + str(round(fii_net)) + " Cr</code></b>",
            "DII  <b><code>" + dii_sign + str(round(dii_net)) + " Cr</code></b>",
            "Net  <code>" + comb_sign + str(round(combined)) + " Cr combined</code>",
            "",
            signal,
        ]
        ok = TelegramSender().send_message("\n".join(lines))
        if ok:
            logger.info("FII/DII extreme alert sent: FII=" + str(round(fii_net)) + " DII=" + str(round(dii_net)))
        return ok
    except Exception as e:
        logger.debug("FII/DII extreme alert failed: " + str(e))
        return False