# trading-agent/run_pipeline.py
"""
Full pipeline demo on real RELIANCE data.
Run: python run_pipeline.py
"""
from src.prediction.pipeline import PredictionPipeline
from src.data.models import Interval
from src.utils.market_hours import market_hours

print(market_hours.format_status())
print("=" * 60)

pipe = PredictionPipeline(
    symbol="RELIANCE",
    interval=Interval.D1,
    horizon=5,
    min_confidence=0.52,
)

print("\nTraining + walk-forward validation on 2 years of RELIANCE data...")
result = pipe.train_and_validate(days_back=730, train_months=12, test_months=1)

print("\n" + "=" * 60)
print("WALK-FORWARD RESULTS")
print("=" * 60)
for fold in result.fold_results:
    print(f"  Fold {fold['fold']:>2} | test={fold['test_month']} | "
          f"train_bars={fold['train_bars']:>4} | acc={fold['accuracy']:.1%}")

print(f"\nOverall accuracy : {result.accuracy:.1%}")
print(f"Summary          : {result.summary()}")
print(f"\nClassification report:\n{result.report}")

print("\n" + "=" * 60)
print("LIVE SIGNAL")
print("=" * 60)
signal = pipe.get_signal()
print(f"\nRELIANCE next-5-day forecast:")
print(f"  {signal}")
print(f"\nTop driving features:")
for feat, info in signal.top_features.items():
    print(f"  {feat:<20} value={info['value']:>8.4f}  importance={info['importance']:.1%}")

pipe.print_feature_importance()