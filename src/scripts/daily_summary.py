# trading-agent/scripts/daily_summary.py
"""
Daily Summary Script — run at market close (3:45 PM IST)

Sends a Telegram summary of:
  - Portfolio value and P&L
  - Open positions
  - Model accuracy
  - Charges paid today

Windows Task Scheduler setup:
  1. Open Task Scheduler → Create Basic Task
  2. Trigger: Daily at 15:45
  3. Action: Start a program
  4. Program: python
  5. Arguments: C:\\Users\\vinay.hirap\\Downloads\\trading-agent\\trading-agent\\scripts\\daily_summary.py
  6. Start in: C:\\Users\\vinay.hirap\\Downloads\\trading-agent\\trading-agent

Or run manually: python scripts/daily_summary.py
"""
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loguru import logger

def main():
    logger.info("Running daily summary...")

    try:
        from config.settings import settings
        from src.alerts.telegram_sender import make_telegram_sender_from_settings
        from src.execution.paper_broker import PaperBroker
        from src.analysis.learning_engine import LearningEngine
        from src.utils.market_hours import market_hours

        tg = make_telegram_sender_from_settings(settings)
        if not tg.is_configured:
            logger.warning("Telegram not configured — skipping summary")
            return

        broker  = PaperBroker(initial_capital=settings.INITIAL_CAPITAL)
        le      = LearningEngine()
        summary = broker.get_portfolio_summary()
        stats   = le.get_accuracy_stats()
        mh      = market_hours.get_status()

        ok = tg.send_daily_summary(
            portfolio_value = summary["total_value"],
            daily_pnl       = summary["total_pnl"],
            n_trades        = summary["n_trades"],
            n_positions     = summary["n_positions"],
            total_charges   = summary["total_charges"],
            accuracy_pct    = stats.get("accuracy_pct"),
        )

        if ok:
            logger.info("Daily summary sent via Telegram")
        else:
            logger.error("Daily summary send failed")

        # Also send open positions detail if any
        if summary["n_positions"] > 0 and ok:
            pos_text = "📋 <b>Open Positions</b>\n\n"
            for pos in summary["positions"]:
                pnl_sym = "🟢 +" if pos["pnl"] >= 0 else "🔴 "
                pos_text += (
                    f"<b>{pos['symbol']}</b> × {pos['quantity']}\n"
                    f"  Entry: ₹{pos['entry']:,.2f} | "
                    f"Current: ₹{pos['current']:,.2f}\n"
                    f"  P&L: {pnl_sym}₹{abs(pos['pnl']):,.0f} "
                    f"({pos['pnl_pct']:+.2f}%)\n\n"
                )
            tg._send(pos_text)

    except Exception as e:
        logger.error(f"Daily summary failed: {e}")
        raise


if __name__ == "__main__":
    main()