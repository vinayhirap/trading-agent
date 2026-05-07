# trading-agent/tests/scratch.py
    # scratch.py — run this in your venv to see it all working
from src.utils.market_hours import market_hours
from src.data.manager import DataManager
from src.features.feature_engine import FeatureEngine
from src.data.models import Interval

print(market_hours.format_status())

dm = DataManager()
df = dm.get_ohlcv("RELIANCE", Interval.D1, days_back=300)

engine = FeatureEngine()
featured = engine.build(df)
latest = featured.iloc[-1]

print(f"\nRELIANCE — latest bar features:")
print(f"  RSI-14:    {latest['rsi_14']:.1f}")
print(f"  MACD hist: {latest['macd_hist']:.2f}")
print(f"  ATR%:      {latest['atr_pct']:.3f}  ({latest['atr_pct']*100:.1f}% of price)")
print(f"  BB %B:     {latest['bb_pct_b']:.2f}  (0=lower band, 1=upper band)")
print(f"  ADX:       {latest['adx']:.1f}  (>25 = trending)")
print(f"  Squeeze:   {'YES — low vol. coiling' if latest['squeeze'] else 'no'}")