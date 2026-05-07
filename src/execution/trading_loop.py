# trading-agent/src/execution/trading_loop.py
"""
The main trading loop — runs during market hours, checks signals,
evaluates risk, and executes paper trades.

Run this script during market hours (9:15–15:30 IST).
It polls every N minutes and acts on high-confidence signals.
"""
import time
from datetime import datetime, timezone
from loguru import logger

from src.data.manager import DataManager
from src.data.models import Interval
from src.features.feature_engine import FeatureEngine
from src.prediction.pipeline import PredictionPipeline
from src.risk.engine import RiskEngine
from src.risk.models import PortfolioState, OrderStatus
from src.execution.paper_broker import PaperBroker
from src.utils.market_hours import market_hours
from config.settings import settings

WATCHLIST = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
POLL_INTERVAL_SECONDS = 1   # check every 1 minute


class TradingLoop:
    """
    Orchestrates the full signal → risk → execution cycle.

    Safety features:
    - Checks market hours before every cycle
    - Respects risk engine gates on every signal
    - Paper trading only by default
    - Full audit log via loguru
    """

    def __init__(
        self,
        watchlist:        list[str] = None,
        poll_seconds:     int       = POLL_INTERVAL_SECONDS,
        min_confidence:   float     = 0.55,
        initial_capital:  float     = None,
    ):
        self.watchlist       = watchlist or WATCHLIST
        self.poll_seconds    = poll_seconds
        self.min_confidence  = min_confidence

        capital = initial_capital or settings.INITIAL_CAPITAL
        self.dm      = DataManager()
        self.engine  = FeatureEngine()
        self.broker  = PaperBroker(initial_capital=capital)

        portfolio = PortfolioState(capital=capital, cash_available=capital)
        self.risk  = RiskEngine(portfolio=portfolio,
                                min_confidence=min_confidence)

        # Load or train models for each symbol
        self.pipelines: dict[str, PredictionPipeline] = {}
        for sym in self.watchlist:
            self.pipelines[sym] = PredictionPipeline(
                sym, Interval.D1, horizon=5,
                min_confidence=min_confidence,
            )

        logger.info(f"TradingLoop ready | watchlist={self.watchlist} | "
                    f"poll={poll_seconds}s | capital=INR{capital:,.0f}")

    def train_all(self, days_back: int = 730) -> None:
        """Train models for all watchlist symbols. Run once before the loop."""
        for sym in self.watchlist:
            logger.info(f"Training {sym}...")
            try:
                result = self.pipelines[sym].train_and_validate(days_back=days_back)
                logger.info(f"{sym} trained | acc={result.accuracy:.1%} | "
                            f"folds={len(result.fold_results)}")
            except Exception as e:
                logger.error(f"Training failed for {sym}: {e}")

    def run_once(self) -> list[dict]:
        """
        Single scan cycle — check all watchlist symbols and act on signals.
        Returns list of actions taken this cycle.
        """
        status = market_hours.get_status()
        actions = []

        if not status["tradeable"]:
            logger.info(f"Market not tradeable: {market_hours.format_status()}")
            return actions

        logger.info(f"Scan cycle | {market_hours.format_status()} | "
                    f"positions={self.risk.portfolio.open_position_count}")

        for sym in self.watchlist:
            try:
                action = self._evaluate_symbol(sym)
                if action:
                    actions.append(action)
            except Exception as e:
                logger.error(f"Error evaluating {sym}: {e}")

        # Update existing positions with current prices
        self._update_positions()

        # Print portfolio summary
        summary = self.broker.get_portfolio_summary()
        logger.info(
            f"Portfolio | value=INR{summary['total_value']:,.0f} | "
            f"P&L=INR{summary['total_pnl']:+,.0f} ({summary['total_pnl_pct']:+.2f}%) | "
            f"charges=INR{summary['total_charges']:.0f}"
        )
        return actions

    def run(self) -> None:
        """
        Continuous loop — run during market hours.
        Press Ctrl+C to stop gracefully.
        """
        logger.info("Starting trading loop — press Ctrl+C to stop")
        self.risk.reset_daily_state()

        try:
            while True:
                self.run_once()
                next_check = datetime.now(timezone.utc)
                logger.info(f"Next scan in {self.poll_seconds}s...")
                time.sleep(self.poll_seconds)
        except KeyboardInterrupt:
            logger.info("Trading loop stopped by user")
            summary = self.broker.get_portfolio_summary()
            logger.info(f"Final P&L: INR{summary['total_pnl']:+,.0f} "
                        f"({summary['total_pnl_pct']:+.2f}%)")

    def _evaluate_symbol(self, symbol: str) -> dict | None:
        """Get signal and evaluate risk for one symbol."""
        try:
            signal = self.pipelines[symbol].get_signal()
        except RuntimeError:
            logger.debug(f"{symbol}: no trained model — skipping")
            return None

        price  = self.dm.get_latest_price(symbol)
        if price <= 0:
            return None

        # Get ATR for stop calculation
        df = self.dm.get_ohlcv(symbol, Interval.D1, days_back=30)
        from src.features.indicators import atr as calc_atr
        atr_val = calc_atr(df).iloc[-1] if not df.empty else price * 0.015

        order = self.risk.evaluate(
            symbol=symbol,
            prediction=signal,
            entry_price=price,
            atr=float(atr_val),
            check_market_hours=True,
        )

        result = {
            "symbol":     symbol,
            "signal":     str(signal),
            "order_status": order.status.value,
        }

        if order.status == OrderStatus.APPROVED:
            record = self.broker.execute(order, current_price=price)
            self.risk.record_fill(order)
            result["execution"] = {
                "fill_price":   record.fill_price,
                "quantity":     record.quantity,
                "total_charges":record.total_charges,
            }
            logger.success(
                f"TRADE EXECUTED: {symbol} | {signal.signal.name} | "
                f"qty={order.quantity} @ INR{record.fill_price:.2f} | "
                f"charges=INR{record.total_charges:.0f}"
            )

        return result

    def _update_positions(self) -> None:
        """Update all open positions with current prices."""
        for sym in list(self.risk.portfolio.positions.keys()):
            price = self.dm.get_latest_price(sym)
            if price > 0:
                df = self.dm.get_ohlcv(sym, Interval.D1, days_back=30)
                from src.features.indicators import atr as calc_atr
                atr_val = calc_atr(df).iloc[-1] if not df.empty else price * 0.015
                result = self.risk.update_position_price(sym, price, float(atr_val))
                if result["action"] in ("stop_hit", "target_hit"):
                    logger.info(f"{sym} exit [{result['action']}]: "
                                f"P&L=INR{result.get('pnl', 0):+.2f}")