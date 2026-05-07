from src.prediction.pipeline import PredictionPipeline
from src.data.models import Interval

pipe = PredictionPipeline('RELIANCE.NS', Interval.D1, horizon=5, min_confidence=0.52)

result = pipe.train_ensemble(days_back=1095, train_months=12, test_months=1)

print(f"Ensemble accuracy: {result['accuracy']:.1%} over {result['n_folds']} folds")
print("Fold breakdown:")

for f in result["fold_results"]:
    print(f"  Fold {f['fold']}: {f['test_month']} acc={f['accuracy']:.1%}")