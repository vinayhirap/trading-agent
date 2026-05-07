# diagnose.py
import sys, os, json
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

print("=" * 60)
print("TRADING AGENT — FULL SYSTEM DIAGNOSIS")
print("=" * 60)

# 1. Python version
print(f"\n[1] Python: {sys.version}")

# 2. Models
print("\n[2] Model files:")
models_dir = ROOT / "models"
for name in ["ensemble_index","ensemble_equity","ensemble_futures","ensemble_crypto"]:
    p = models_dir / f"{name}.pkl"
    if p.exists():
        print(f"  ✅ {name}.pkl ({p.stat().st_size//1024} KB)")
    else:
        print(f"  ❌ {name}.pkl — MISSING")

# 3. Load models with joblib
print("\n[3] Joblib load test:")
try:
    import joblib
    for name in ["ensemble_index","ensemble_equity","ensemble_futures","ensemble_crypto"]:
        p = models_dir / f"{name}.pkl"
        if p.exists():
            try:
                bundle = joblib.load(p)
                keys = list(bundle.keys()) if isinstance(bundle, dict) else type(bundle)
                xgb = bundle.get("global_xgb") if isinstance(bundle, dict) else None
                lgb = bundle.get("global_lgb") if isinstance(bundle, dict) else None
                feats = len(bundle.get("feature_cols", [])) if isinstance(bundle, dict) else 0
                print(f"  ✅ {name}: keys={keys} xgb={xgb is not None} lgb={lgb is not None} feats={feats}")
            except Exception as e:
                print(f"  ❌ {name}: {e}")
except ImportError:
    print("  ❌ joblib not installed — run: pip install joblib")

# 4. Price store
print("\n[4] Price store:")
try:
    from src.streaming.price_store import price_store
    px = price_store.get_all()
    print(f"  Symbols cached: {len(px)}")
    if px:
        for sym, p in list(px.items())[:3]:
            age = price_store.age_seconds(sym)
            print(f"  {sym}: {p:.2f} | {age:.0f}s old")
    else:
        print("  ⚠️  Empty — server not running or warmup incomplete")
except Exception as e:
    print(f"  ❌ {e}")

# 5. Realtime advisor
print("\n[5] Realtime advisor:")
try:
    from src.analysis.realtime_advisor import realtime_advisor
    print(f"  Models loaded: {list(realtime_advisor._models.keys())}")
    print(f"  Watchlist: {realtime_advisor._watchlist}")
    print(f"  Running: {realtime_advisor._running}")
    print(f"  Advice count: {len(realtime_advisor._advice)}")
except Exception as e:
    print(f"  ❌ {e}")

# 6. Settings / .env
print("\n[6] Config / .env:")
try:
    from config.settings import settings
    print(f"  INITIAL_CAPITAL: ₹{settings.INITIAL_CAPITAL:,.0f}")
    print(f"  ANTHROPIC_API_KEY: {'✅ set' if os.environ.get('ANTHROPIC_API_KEY') or getattr(settings,'ANTHROPIC_API_KEY',None) else '❌ missing'}")
    print(f"  TELEGRAM_BOT_TOKEN: {'✅ set' if settings.TELEGRAM_BOT_TOKEN else '❌ missing'}")
    print(f"  NEWS_API_KEY: {'✅ set' if settings.NEWS_API_KEY else '❌ missing'}")
    print(f"  ANGEL_API_KEY: {'✅ set' if settings.ANGEL_API_KEY else '❌ missing'}")
except Exception as e:
    print(f"  ❌ {e}")

# 7. API endpoints (if server running)
print("\n[7] API endpoints (requires server running on 8010):")
try:
    import requests
    for url, label in [
        ("/api/prices",                   "prices"),
        ("/api/market-status",            "market-status"),
        ("/api/signals?symbols=NIFTY50",  "signals NIFTY50"),
        ("/api/signals?symbols=BTC",      "signals BTC"),
        ("/api/config-status",            "config-status"),
    ]:
        try:
            r = requests.get(f"http://localhost:8010{url}", timeout=5)
            d = r.json()
            if url.startswith("/api/signals"):
                sigs = d.get("signals", {})
                sym = list(sigs.keys())[0] if sigs else None
                if sym:
                    s = sigs[sym]
                    print(f"  ✅ {label}: {s.get('recommended_instrument','?')} conf={s.get('confidence','?')} source={s.get('source','?')}")
                else:
                    print(f"  ⚠️  {label}: empty signals")
            elif url == "/api/prices":
                n = len(d.get("prices", {}))
                print(f"  ✅ {label}: {n} symbols")
            else:
                print(f"  ✅ {label}: {json.dumps(d)[:80]}")
        except requests.ConnectionError:
            print(f"  ⚠️  {label}: server not running")
        except Exception as e:
            print(f"  ❌ {label}: {e}")
except ImportError:
    print("  ❌ requests not installed")

# 8. Anthropic API direct test
print("\n[8] Anthropic API direct test:")
try:
    import requests as req
    key = os.environ.get("ANTHROPIC_API_KEY") or getattr(__import__('config.settings', fromlist=['settings']).settings, 'ANTHROPIC_API_KEY', None)
    if not key:
        print("  ❌ No API key found")
    else:
        r = req.post("https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": "claude-haiku-4-5-20251001", "max_tokens": 20, "messages": [{"role":"user","content":"say ok"}]},
            timeout=10)
        d = r.json()
        if "error" in d:
            print(f"  ❌ API error: {d['error']['type']} — {d['error']['message']}")
        else:
            print(f"  ✅ API working: {d['content'][0]['text']}")
except Exception as e:
    print(f"  ❌ {e}")

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)