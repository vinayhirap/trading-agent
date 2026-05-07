# trading-agent/src/backtesting/backtest_engine.py
"""
Backtesting Engine v3 — Ideal System

Improvements vs v2:
  1. Confidence gate 55% (was 50%)
  2. Trailing stop loss — moves SL up after +1% profit
  3. Daily loss limit — halts after -2% day loss
  4. Cache-first data loading
  5. max(1, qty) — fixes BTC/NIFTY 0-trade bug
  6. Regime-aware signal (ADX gate)
"""
import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[3]

BROKERAGE_PER_ORDER  = 20.0
BROKERAGE_PCT        = 0.0003
STT_DELIVERY         = 0.001
EXCHANGE_TXN_CHARGE  = 0.0000345
SEBI_CHARGE          = 0.000001
GST_RATE             = 0.18
STAMP_DUTY           = 0.00015
SLIPPAGE_PCT         = 0.0005

MIN_CONFIDENCE       = 0.55
DAILY_LOSS_LIMIT_PCT = 0.02
TRAILING_TRIGGER_PCT = 0.01
TRAILING_STEP_PCT    = 0.005


@dataclass
class BacktestTrade:
    trade_id:      str
    symbol:        str
    side:          str
    entry_date:    date
    exit_date:     Optional[date]
    entry_price:   float
    exit_price:    float
    quantity:      int
    entry_signal:  str
    exit_reason:   str
    signal_conf:   float
    stop_loss:     float
    target:        float
    gross_pnl:     float
    charges:       float
    net_pnl:       float
    hold_bars:     int
    is_winner:     bool
    sl_adjustments:int = 0


@dataclass
class BacktestResult:
    symbol:           str
    start_date:       date
    end_date:         date
    initial_capital:  float
    final_capital:    float
    trades:           list  = field(default_factory=list)
    equity_curve:     object = None
    benchmark:        object = None
    daily_returns:    object = None
    total_return_pct: float  = 0.0
    benchmark_return: float  = 0.0
    sharpe_ratio:     float  = 0.0
    sortino_ratio:    float  = 0.0
    calmar_ratio:     float  = 0.0
    max_drawdown_pct: float  = 0.0
    win_rate:         float  = 0.0
    profit_factor:    float  = 0.0
    avg_hold_bars:    float  = 0.0
    total_charges:    float  = 0.0
    n_trades:         int    = 0
    n_winners:        int    = 0
    n_losers:         int    = 0
    avg_win:          float  = 0.0
    avg_loss:         float  = 0.0
    expectancy:       float  = 0.0
    annual_return:    float  = 0.0
    halted_days:      int    = 0


class BacktestEngine:

    def __init__(
        self,
        initial_capital: float = 10_000,
        risk_per_trade:  float = 0.01,
        atr_sl_mult:     float = 1.5,
        target_rr:       float = 2.0,
        use_trailing:    bool  = True,
        use_daily_limit: bool  = True,
        min_confidence:  float = MIN_CONFIDENCE,
    ):
        self.initial_capital = initial_capital
        self.risk_per_trade  = risk_per_trade
        self.atr_sl_mult     = atr_sl_mult
        self.target_rr       = target_rr
        self.use_trailing    = use_trailing
        self.use_daily_limit = use_daily_limit
        self.min_confidence  = min_confidence

    def run(self, symbol, start_date, end_date, df=None, verbose=True):
        logger.info(
            f"Backtest v3: {symbol} | {start_date}→{end_date} | "
            f"₹{self.initial_capital:,.0f} | "
            f"conf≥{self.min_confidence:.0%} | "
            f"trailing={'on' if self.use_trailing else 'off'}"
        )

        if df is None:
            df = self._load_data(symbol, start_date, end_date)

        if df is None or len(df) < 50:
            return self._empty_result(symbol, start_date, end_date)

        df = df.loc[
            (df.index.date >= start_date) &
            (df.index.date <= end_date)
        ].copy()

        if len(df) < 30:
            return self._empty_result(symbol, start_date, end_date)

        featured = self._compute_features(df)
        if featured is None or featured.empty:
            return self._empty_result(symbol, start_date, end_date)

        benchmark = self._build_benchmark(df, self.initial_capital)
        trades, equity_curve, halted_days = self._simulate(symbol, df, featured)

        daily_returns = pd.Series(dtype=float)
        if equity_curve is not None and len(equity_curve) > 1:
            daily_returns = equity_curve.pct_change().dropna()

        final_cap = float(equity_curve.iloc[-1]) if equity_curve is not None \
                    else self.initial_capital

        result = BacktestResult(
            symbol=symbol, start_date=start_date, end_date=end_date,
            initial_capital=self.initial_capital, final_capital=final_cap,
            trades=trades, equity_curve=equity_curve,
            benchmark=benchmark, daily_returns=daily_returns,
            halted_days=halted_days,
        )

        from src.backtesting.backtest_report import BacktestReport
        BacktestReport.compute_metrics(result)

        if verbose:
            self._print_summary(result)
        return result

    def run_multi(self, symbols, start_date, end_date):
        results = {}
        for sym in symbols:
            try:
                r = self.run(sym, start_date, end_date, verbose=False)
                results[sym] = r
                logger.info(
                    f"  {sym}: {r.total_return_pct:+.1f}% | "
                    f"Sharpe {r.sharpe_ratio:.2f} | WR {r.win_rate:.0%}"
                )
            except Exception as e:
                logger.error(f"  {sym}: FAILED — {e}")
        return results

    # ── Simulation ────────────────────────────────────────────────────────────

    def _simulate(self, symbol, df, featured):
        cash          = self.initial_capital
        position      = None
        trades        = []
        equity        = {}
        trade_num     = 0
        WARMUP        = 50
        halted_days   = 0
        bars          = featured.index.tolist()

        current_day       = None
        day_start_capital = cash
        day_halted        = False

        for i, idx in enumerate(bars):
            bar   = df.loc[idx]
            feat  = featured.loc[idx]
            price = float(bar["close"])
            low   = float(bar["low"])
            high  = float(bar["high"])
            today = idx.date() if hasattr(idx, "date") else idx

            # New day
            if today != current_day:
                current_day       = today
                day_start_capital = cash + (position["quantity"] * price if position else 0)
                day_halted        = False

            if i < WARMUP:
                equity[idx] = cash
                continue

            # Trailing stop update
            if position and self.use_trailing and i > position["entry_bar"] + 1:
                profit_pct = (price - position["entry_price"]) / position["entry_price"]
                if profit_pct >= TRAILING_TRIGGER_PCT:
                    steps  = int(profit_pct / TRAILING_STEP_PCT)
                    new_sl = round(position["entry_price"] * (1 + (steps - 1) * TRAILING_STEP_PCT), 2)
                    if new_sl > position["sl"]:
                        position["sl"]           = new_sl
                        position["sl_adjustments"] = position.get("sl_adjustments", 0) + 1

            # SL / Target exit
            if position:
                exit_price, reason = None, None
                if position["side"] == "BUY":
                    if low  <= position["sl"]:     exit_price, reason = position["sl"],     "SL_HIT"
                    elif high >= position["target"]:exit_price, reason = position["target"], "TARGET_HIT"
                else:
                    if high >= position["sl"]:     exit_price, reason = position["sl"],     "SL_HIT"
                    elif low  <= position["target"]:exit_price, reason = position["target"], "TARGET_HIT"

                if reason:
                    trade, cash, _ = self._close(position, exit_price, today, reason, cash, i)
                    trades.append(trade)
                    trade_num += 1
                    position   = None

            # Daily loss limit
            portfolio_val = cash + (position["quantity"] * price if position else 0)
            if (self.use_daily_limit and not day_halted
                    and portfolio_val < day_start_capital * (1 - DAILY_LOSS_LIMIT_PCT)):
                day_halted = True
                halted_days += 1

            # New signal
            if not day_halted:
                signal, conf = self._signal(feat)

                if position is None and "HOLD" not in signal and conf >= self.min_confidence:
                    if cash > 100:
                        atr          = float(feat.get("atr_14", price * 0.015))
                        sl, tgt, qty = self._size(price, signal, atr, cash)
                        if qty >= 1:
                            charges   = self._charges(price, qty, "BUY")
                            total_cost = qty * price + charges
                            if total_cost <= cash:
                                cash -= total_cost
                                position = {
                                    "symbol": symbol, "side": "BUY" if "BUY" in signal else "SELL",
                                    "entry_price": price, "entry_date": today, "entry_bar": i,
                                    "quantity": qty, "sl": sl, "target": tgt,
                                    "signal": signal, "confidence": conf,
                                    "trade_id": f"{symbol}_{trade_num}",
                                    "entry_charges": charges, "sl_adjustments": 0,
                                }

                elif position and i > position["entry_bar"] + 2 and conf >= 0.65:
                    flip = (
                        ("SELL" in signal and position["side"] == "BUY") or
                        ("BUY"  in signal and position["side"] == "SELL")
                    )
                    if flip:
                        trade, cash, _ = self._close(position, price, today, "SIGNAL_FLIP", cash, i)
                        trades.append(trade); trade_num += 1; position = None

            equity[idx] = cash + (position["quantity"] * price if position else 0)

        # Force close at end
        if position:
            lp = float(df["close"].iloc[-1])
            ld = df.index[-1].date() if hasattr(df.index[-1], "date") else df.index[-1]
            trade, cash, _ = self._close(position, lp, ld, "END_OF_TEST", cash, len(bars) - 1)
            trades.append(trade)

        return trades, pd.Series(equity), halted_days

    def _close(self, pos, exit_price, exit_date, reason, cash, exit_bar):
        qty    = pos["quantity"]
        entry  = pos["entry_price"]
        side   = pos["side"]

        if reason == "SL_HIT":
            slip       = exit_price * SLIPPAGE_PCT
            exit_price = (exit_price - slip) if side == "BUY" else (exit_price + slip)
            exit_price = round(exit_price, 2)

        exit_charges = self._charges(exit_price, qty, "SELL")
        cash        += qty * exit_price - exit_charges
        gross        = (exit_price - entry) * qty if side == "BUY" else (entry - exit_price) * qty
        all_ch       = pos.get("entry_charges", 0) + exit_charges
        net          = gross - all_ch
        hold         = exit_bar - pos.get("entry_bar", exit_bar)

        return BacktestTrade(
            trade_id=pos["trade_id"], symbol=pos.get("symbol",""),
            side=side, entry_date=pos["entry_date"], exit_date=exit_date,
            entry_price=round(entry, 2), exit_price=round(exit_price, 2),
            quantity=qty, entry_signal=pos["signal"], exit_reason=reason,
            signal_conf=pos["confidence"], stop_loss=pos["sl"], target=pos["target"],
            gross_pnl=round(gross, 2), charges=round(all_ch, 2), net_pnl=round(net, 2),
            hold_bars=hold, is_winner=net > 0,
            sl_adjustments=pos.get("sl_adjustments", 0),
        ), cash, net

    def _signal(self, feat):
        try:
            rsi    = float(feat.get("rsi_14",       50))
            macd_h = float(feat.get("macd_hist",     0))
            macd_c = float(feat.get("macd_hist_chg", 0))
            e9     = float(feat.get("ema9_pct",      0))
            e50    = float(feat.get("ema50_pct",     0))
            adx    = float(feat.get("adx",           20))
            di_d   = float(feat.get("di_diff",        0))
            bb     = float(feat.get("bb_pct_b",     0.5))
            atr_r  = float(feat.get("atr_ratio",     1.0))
            obv    = float(feat.get("obv_slope",     0))

            if atr_r > 3.0:
                return "HOLD", 0.35

            score = 0
            if adx > 25:
                if   e9 > 0 and e50 > 0: score += 3
                elif e9 < 0 and e50 < 0: score -= 3
                elif e9 > 0:             score += 1
                elif e9 < 0:             score -= 1
                if   di_d > 0:           score += 2
                elif di_d < 0:           score -= 2
                if   macd_h > 0 and macd_c > 0: score += 2
                elif macd_h < 0 and macd_c < 0: score -= 2
                elif macd_h > 0:                 score += 1
                elif macd_h < 0:                 score -= 1
                if obv > 0:  score += 1
                elif obv < 0:score -= 1
                if adx > 35: score = int(score * 1.3)
                if   score >= 7:  return "STRONG BUY",  0.78
                elif score >= 4:  return "BUY",          0.62
                elif score <= -7: return "STRONG SELL",  0.78
                elif score <= -4: return "SELL",          0.62
                return "HOLD", 0.40

            elif adx < 18:
                if   rsi < 30:   score += 3
                elif rsi < 38:   score += 2
                elif rsi > 70:   score -= 3
                elif rsi > 62:   score -= 2
                if   bb < 0.05:  score += 2
                elif bb < 0.15:  score += 1
                elif bb > 0.95:  score -= 2
                elif bb > 0.85:  score -= 1
                if macd_h < 0 and macd_c > 0:  score += 2
                elif macd_h > 0 and macd_c < 0: score -= 2
                if   score >= 5:  return "STRONG BUY",  0.72
                elif score >= 3:  return "BUY",          0.59
                elif score <= -5: return "STRONG SELL",  0.72
                elif score <= -3: return "SELL",          0.59
                return "HOLD", 0.42

            else:
                if e9 > 0 and macd_h > 0 and macd_c > 0: return "BUY",  0.56
                if e9 < 0 and macd_h < 0 and macd_c < 0: return "SELL", 0.56
                return "HOLD", 0.38

        except Exception:
            return "HOLD", 0.40

    def _size(self, price, signal, atr, cash):
        sl_dist = max(atr * self.atr_sl_mult, price * 0.005)
        if "BUY" in signal:
            sl, tgt = round(price - sl_dist, 2), round(price + sl_dist * self.target_rr, 2)
        else:
            sl, tgt = round(price + sl_dist, 2), round(price - sl_dist * self.target_rr, 2)
        qty = int((cash * self.risk_per_trade) / sl_dist)
        if qty < 1:
            qty = max(1, int(cash * 0.05 / price))
        qty = min(max(1, qty), max(1, int(cash * 0.95 / price)))
        return sl, tgt, qty

    def _charges(self, price, qty, side):
        v  = price * qty
        b  = min(BROKERAGE_PER_ORDER, v * BROKERAGE_PCT)
        return round(
            b + (v * STT_DELIVERY if side == "BUY" else 0)
            + v * EXCHANGE_TXN_CHARGE + v * SEBI_CHARGE
            + (v * STAMP_DUTY if side == "BUY" else 0)
            + (b + v * EXCHANGE_TXN_CHARGE) * GST_RATE,
            2
        )

    def _load_data(self, symbol, start, end):
        try:
            from src.data.store import LocalDataStore
            cached = LocalDataStore().load(f"{symbol}_1d.parquet")
            if cached is not None and len(cached) > 100:
                return cached
        except Exception:
            pass
        try:
            from src.data.manager import DataManager
            from src.data.models import Interval
            df = DataManager().get_ohlcv(symbol, Interval.D1, days_back=(end-start).days+120)
            if not df.empty:
                return df
        except Exception:
            pass
        try:
            import yfinance as yf
            from src.data.models import ALL_SYMBOLS
            info = ALL_SYMBOLS.get(symbol)
            base = info.symbol if info else symbol
            buf  = start - timedelta(days=120)
            for t in [base, f"{symbol}.NS", f"{symbol}.BO", f"{symbol}-USD", symbol]:
                try:
                    raw = yf.download(t, start=buf.isoformat(),
                                      end=(end+timedelta(1)).isoformat(),
                                      interval="1d", progress=False,
                                      auto_adjust=True, timeout=15)
                    if not raw.empty:
                        raw.columns = [c.lower() for c in raw.columns]
                        raw.index   = pd.to_datetime(raw.index, utc=True)
                        return raw
                except Exception:
                    continue
        except Exception as e:
            logger.error(f"All sources failed for {symbol}: {e}")
        return None

    def _compute_features(self, df):
        try:
            from src.features.feature_engine import FeatureEngine
            return FeatureEngine("1d").build(df, drop_na=False)
        except Exception as e:
            logger.error(f"Features failed: {e}")
            return None

    def _build_benchmark(self, df, capital):
        return (df["close"] * capital / float(df["close"].iloc[0])).rename("benchmark")

    def _empty_result(self, symbol, start, end):
        return BacktestResult(
            symbol=symbol, start_date=start, end_date=end,
            initial_capital=self.initial_capital, final_capital=self.initial_capital,
        )

    def _print_summary(self, r):
        logger.info("=" * 55)
        logger.info(f"BACKTEST v3: {r.symbol}")
        logger.info(f"Return:       {r.total_return_pct:+.1f}% (BH: {r.benchmark_return:+.1f}%)")
        logger.info(f"Annual:       {r.annual_return:+.1f}%")
        logger.info(f"Sharpe:       {r.sharpe_ratio:.2f}")
        logger.info(f"Max DD:       {r.max_drawdown_pct:.1f}%")
        logger.info(f"Win rate:     {r.win_rate:.0%} ({r.n_winners}W/{r.n_losers}L)")
        logger.info(f"PF:           {r.profit_factor:.2f}")
        logger.info(f"Expectancy:   ₹{r.expectancy:+.0f}/trade")
        logger.info(f"Trades:       {r.n_trades} (halted {r.halted_days} days)")
        logger.info(f"Charges:      ₹{r.total_charges:.2f}")
        logger.info(f"Final:        ₹{r.final_capital:,.2f}")
        logger.info("=" * 55)