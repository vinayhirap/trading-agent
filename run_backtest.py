RUN_BACKTEST_SCRIPT = '''#!/usr/bin/env python
# trading-agent/run_backtest.py
"""
Backtest CLI — run from terminal
 
Usage:
    python run_backtest.py                              # NIFTY50, 2023
    python run_backtest.py --symbol RELIANCE            # single symbol
    python run_backtest.py --symbol GOLD --start 2022-01-01
    python run_backtest.py --multi Indices              # all indices
    python run_backtest.py --capital 50000 --risk 1.5  # custom params
"""
import sys
import argparse
from datetime import date
from pathlib import Path
 
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
 
from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.backtest_report import BacktestReport
 
UNIVERSE = {
    "Indices":     ["NIFTY50","BANKNIFTY","SENSEX"],
    "Large Cap":   ["RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","SBIN"],
    "Commodities": ["GOLD","SILVER","CRUDEOIL"],
    "Crypto":      ["BTC","ETH"],
}
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",  default="NIFTY50")
    parser.add_argument("--start",   default="2023-01-01")
    parser.add_argument("--end",     default=str(date.today()))
    parser.add_argument("--capital", type=float, default=10000)
    parser.add_argument("--risk",    type=float, default=1.0)
    parser.add_argument("--rr",      type=float, default=2.0)
    parser.add_argument("--atr",     type=float, default=1.5)
    parser.add_argument("--multi",   default=None,
                        help="Run all symbols in a category")
    args = parser.parse_args()
 
    engine = BacktestEngine(
        initial_capital = args.capital,
        risk_per_trade  = args.risk / 100,
        target_rr       = args.rr,
        atr_sl_mult     = args.atr,
    )
 
    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)
 
    if args.multi:
        symbols = UNIVERSE.get(args.multi, [args.symbol])
        print(f"\\nRunning backtest for {len(symbols)} symbols...")
        results = engine.run_multi(symbols, start, end)
        print("\\n" + "="*60)
        print(f"{'Symbol':<12} {'Return':>8} {'Sharpe':>8} {'WinRate':>8} {'Grade':>6}")
        print("="*60)
        for sym, r in results.items():
            grade, _ = BacktestReport.grade(r)
            print(f"{sym:<12} {r.total_return_pct:>+7.1f}% {r.sharpe_ratio:>8.2f} {r.win_rate:>8.0%} {grade:>6}")
    else:
        result = engine.run(args.symbol, start, end, verbose=True)
        if result.n_trades > 0:
            grade, note = BacktestReport.grade(result)
            print(f"\\nGrade: {grade} — {note}")
            summary = BacktestReport.summary_dict(result)
            import json
            out = ROOT / "data" / f"backtest_{args.symbol}_{args.start}.json"
            out.parent.mkdir(exist_ok=True)
            out.write_text(json.dumps(summary, indent=2))
            print(f"Results saved → {out}")
 
if __name__ == "__main__":
    main()
'''