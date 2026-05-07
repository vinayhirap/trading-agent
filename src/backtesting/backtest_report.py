# trading-agent/src/backtesting/backtest_report.py
"""
Backtest Report — Metrics computation and analysis.

Computes all standard quant trading metrics from a BacktestResult:
  - Return metrics:  total return, annualised return, alpha vs benchmark
  - Risk metrics:    Sharpe, Sortino, Calmar, max drawdown, VaR
  - Trade metrics:   win rate, profit factor, expectancy, avg hold
  - Cost analysis:   total charges, charge drag on returns

All metrics use industry-standard formulas.
Risk-free rate: 6.5% (approximate Indian T-bill rate).
"""
import numpy as np
import pandas as pd
from loguru import logger

RISK_FREE_RATE = 0.065   # 6.5% per annum (Indian T-bill proxy)
TRADING_DAYS   = 252


class BacktestReport:
    """
    Computes and formats backtest metrics.

    Usage:
        BacktestReport.compute_metrics(result)   # fills result in-place
        summary = BacktestReport.summary_dict(result)
        BacktestReport.print_full(result)
    """

    @staticmethod
    def compute_metrics(result) -> None:
        """Fill all metric fields on a BacktestResult in-place."""
        r = result

        if r.equity_curve.empty or len(r.equity_curve) < 2:
            return

        # ── Return metrics ─────────────────────────────────────────────────
        r.total_return_pct = round(
            (r.final_capital / r.initial_capital - 1) * 100, 2
        )

        # Benchmark return
        if not r.benchmark.empty:
            r.benchmark_return = round(
                (float(r.benchmark.iloc[-1]) / float(r.benchmark.iloc[0]) - 1) * 100, 2
            )

        # Annualised return
        n_days  = (r.end_date - r.start_date).days
        n_years = max(n_days / 365, 0.01)
        r.annual_return = round(
            ((r.final_capital / r.initial_capital) ** (1 / n_years) - 1) * 100, 2
        )

        # ── Daily returns ──────────────────────────────────────────────────
        daily_ret = r.daily_returns
        if daily_ret.empty and not r.equity_curve.empty:
            daily_ret = r.equity_curve.pct_change().dropna()
            r.daily_returns = daily_ret

        if daily_ret.empty:
            return

        # ── Risk metrics ───────────────────────────────────────────────────
        mean_daily = float(daily_ret.mean())
        std_daily  = float(daily_ret.std())

        # Sharpe ratio (annualised)
        if std_daily > 0:
            daily_rf  = RISK_FREE_RATE / TRADING_DAYS
            r.sharpe_ratio = round(
                (mean_daily - daily_rf) / std_daily * np.sqrt(TRADING_DAYS), 2
            )

        # Sortino ratio (downside deviation only)
        downside = daily_ret[daily_ret < 0]
        if len(downside) > 1:
            downside_std = float(downside.std())
            if downside_std > 0:
                daily_rf  = RISK_FREE_RATE / TRADING_DAYS
                r.sortino_ratio = round(
                    (mean_daily - daily_rf) / downside_std * np.sqrt(TRADING_DAYS), 2
                )

        # Max drawdown
        rolling_max = r.equity_curve.expanding().max()
        drawdown    = (r.equity_curve - rolling_max) / rolling_max
        r.max_drawdown_pct = round(float(drawdown.min()) * 100, 2)

        # Calmar ratio (annual return / max drawdown)
        if r.max_drawdown_pct != 0:
            r.calmar_ratio = round(
                r.annual_return / abs(r.max_drawdown_pct), 2
            )

        # ── Trade metrics ──────────────────────────────────────────────────
        if not r.trades:
            return

        r.n_trades  = len(r.trades)
        r.n_winners = sum(1 for t in r.trades if t.is_winner)
        r.n_losers  = r.n_trades - r.n_winners
        r.win_rate  = round(r.n_winners / r.n_trades, 3) if r.n_trades > 0 else 0

        pnls     = [t.net_pnl for t in r.trades]
        winners  = [p for p in pnls if p > 0]
        losers   = [p for p in pnls if p < 0]

        r.avg_win  = round(float(np.mean(winners)), 2) if winners else 0
        r.avg_loss = round(float(np.mean(losers)),  2) if losers  else 0

        # Profit factor: gross profit / gross loss
        gross_profit = sum(winners)
        gross_loss   = abs(sum(losers))
        r.profit_factor = round(
            gross_profit / gross_loss if gross_loss > 0 else float("inf"), 2
        )

        # Expectancy (expected P&L per trade)
        r.expectancy = round(
            r.win_rate * r.avg_win + (1 - r.win_rate) * r.avg_loss, 2
        )

        # Average hold period
        holds = []
        for t in r.trades:
            if t.entry_date and t.exit_date:
                holds.append((t.exit_date - t.entry_date).days)
        r.avg_hold_bars = round(float(np.mean(holds)), 1) if holds else 0

        # Total charges
        r.total_charges = round(sum(t.charges for t in r.trades), 2)

    @staticmethod
    def summary_dict(result) -> dict:
        """Return metrics as a flat dict for display."""
        r = result
        return {
            "symbol":            r.symbol,
            "period":            f"{r.start_date} → {r.end_date}",
            "total_return":      f"{r.total_return_pct:+.1f}%",
            "annual_return":     f"{r.annual_return:+.1f}%",
            "benchmark_return":  f"{r.benchmark_return:+.1f}%",
            "alpha":             f"{r.total_return_pct - r.benchmark_return:+.1f}%",
            "sharpe_ratio":      f"{r.sharpe_ratio:.2f}",
            "sortino_ratio":     f"{r.sortino_ratio:.2f}",
            "calmar_ratio":      f"{r.calmar_ratio:.2f}",
            "max_drawdown":      f"{r.max_drawdown_pct:.1f}%",
            "win_rate":          f"{r.win_rate:.0%}",
            "profit_factor":     f"{r.profit_factor:.2f}",
            "expectancy":        f"₹{r.expectancy:+.0f}",
            "n_trades":          r.n_trades,
            "n_winners":         r.n_winners,
            "n_losers":          r.n_losers,
            "avg_win":           f"₹{r.avg_win:+.0f}",
            "avg_loss":          f"₹{r.avg_loss:+.0f}",
            "avg_hold_days":     f"{r.avg_hold_bars:.0f}d",
            "total_charges":     f"₹{r.total_charges:.0f}",
            "initial_capital":   f"₹{r.initial_capital:,.0f}",
            "final_capital":     f"₹{r.final_capital:,.0f}",
        }

    @staticmethod
    def grade(result) -> tuple[str, str]:
        """
        Grade the backtest result: A+ to F.
        Returns (grade, explanation).
        """
        r   = result
        pts = 0

        # Sharpe
        if r.sharpe_ratio >= 1.5:   pts += 3
        elif r.sharpe_ratio >= 1.0: pts += 2
        elif r.sharpe_ratio >= 0.5: pts += 1

        # Win rate
        if r.win_rate >= 0.55:      pts += 2
        elif r.win_rate >= 0.48:    pts += 1

        # Profit factor
        if r.profit_factor >= 1.8:  pts += 2
        elif r.profit_factor >= 1.3:pts += 1

        # Alpha (beat benchmark)
        alpha = r.total_return_pct - r.benchmark_return
        if alpha >= 10:             pts += 2
        elif alpha >= 0:            pts += 1

        # Max drawdown penalty
        if r.max_drawdown_pct < -30: pts -= 2
        elif r.max_drawdown_pct < -20:pts -= 1

        grade_map = {
            (10, 100): ("A+", "Exceptional. Rare in real markets. Double-check for overfitting."),
            (8,  10):  ("A",  "Excellent. Strong risk-adjusted returns and edge."),
            (6,  8):   ("B+", "Good. Consistent edge with manageable risk."),
            (4,  6):   ("B",  "Acceptable. Modest edge, needs improvement."),
            (2,  4):   ("C",  "Marginal. Edge barely covers costs."),
            (-10, 2):  ("F",  "No edge detected. Do not trade this strategy."),
        }
        for (lo, hi), (g, exp) in grade_map.items():
            if lo <= pts < hi:
                return g, exp
        return "F", "No edge detected."

    @staticmethod
    def trade_log_df(result) -> pd.DataFrame:
        """Return trades as a formatted DataFrame."""
        if not result.trades:
            return pd.DataFrame()
        rows = []
        for t in result.trades:
            rows.append({
                "Date":     t.entry_date,
                "Exit":     t.exit_date,
                "Side":     t.side,
                "Entry":    f"₹{t.entry_price:,.2f}",
                "Exit Px":  f"₹{t.exit_price:,.2f}",
                "Qty":      t.quantity,
                "Signal":   t.entry_signal,
                "Conf":     f"{t.signal_conf:.0%}",
                "P&L":      f"₹{t.net_pnl:+,.0f}",
                "Charges":  f"₹{t.charges:.0f}",
                "Reason":   t.exit_reason,
                "Result":   "✅ WIN" if t.is_winner else "❌ LOSS",
            })
        return pd.DataFrame(rows)