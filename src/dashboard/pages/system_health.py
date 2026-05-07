# trading-agent/src/dashboard/pages/system_health.py
"""
System Health & Evolution Dashboard

Tracks everything in one place:
  - All module status (loaded / failed / version)
  - Model performance vs baseline
  - API rate limits and usage
  - Data cache status
  - Optuna tuning results
  - Phase completion tracker
  - Build log summary

Wire into app.py:
    elif page == "🏥 System Health":
        from src.dashboard.pages.system_health import render_system_health
        render_system_health()
"""
import json
import sys
import time
import importlib
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[3]


# ── Module registry ───────────────────────────────────────────────────────────

MODULES = {
    "Core": {
        "Data Manager":         "src.data.manager",
        "Feature Engine":       "src.features.feature_engine",
        "Ensemble Model":       "src.prediction.ensemble_model",
        "Backtest Engine":      "src.backtesting.backtest_engine",
        "Risk Manager":         "src.risk.risk_manager",
        "Paper Broker":         "src.execution.paper_broker",
    },
    "Intelligence (Phase 1)": {
        "Alpha Vantage Adapter":"src.data.adapters.alphavantage_adapter",
        "Nerve Center":         "src.dashboard.pages.nerve_center",
        "Company Intelligence": "src.dashboard.pages.company_intelligence",
    },
    "Regime & Valuation (Phase 2)": {
        "Regime Detector":      "src.analysis.regime_detector",
        "Valuation Comps":      "src.dashboard.pages.valuation_comps",
    },
    "Phase 3 — Agents & Flow": {
        "Multi-Agent Engine":   "src.analysis.multi_agent_engine",
        "FII/DII Tracker":      "src.analysis.fii_dii_tracker",
        "Options OI":           "src.analysis.options_oi",
    },
    "Phase 4 — Optimization": {
        "Optuna Tuner":         "src.prediction.optuna_tuner",
    },
    "Analysis Engines": {
        "Hybrid Engine":        "src.analysis.hybrid_engine",
        "Event Engine":         "src.analysis.event_engine",
        "Behavior Model":       "src.analysis.behavior_model",
        "Prediction Engine":    "src.analysis.prediction_engine",
        "Action Engine":        "src.analysis.action_engine",
    },
    "Data & Alerts": {
        "News Intelligence":    "src.news.news_intelligence",
        "Latest News Service":  "src.news.latest_news_service",
        "Price Alert Manager":  "src.alerts.price_alert_manager",
        "Daily Summary":        "src.alerts.daily_summary",
    },
}

# ── Phase tracker ─────────────────────────────────────────────────────────────

PHASES = {
    "Phase 1 — Intelligence": {
        "status": "complete",
        "items": [
            ("Alpha Vantage adapter", "src/data/adapters/alphavantage_adapter.py"),
            ("Nerve Center (news → heatmap)", "src/dashboard/pages/nerve_center.py"),
            ("Company Intelligence (DES/FA/ANR)", "src/dashboard/pages/company_intelligence.py"),
        ],
    },
    "Phase 2 — Regime & Valuation": {
        "status": "complete",
        "items": [
            ("Regime Detector (HMM + rule-based)", "src/analysis/regime_detector.py"),
            ("Valuation Comps (EQRV + KPIC)", "src/dashboard/pages/valuation_comps.py"),
        ],
    },
    "Phase 3 — Agents & Flow": {
        "status": "complete",
        "items": [
            ("Multi-Agent Engine (4 agents)", "src/analysis/multi_agent_engine.py"),
            ("FII/DII Tracker (institutional flow)", "src/analysis/fii_dii_tracker.py"),
            ("Options OI (PCR + max pain)", "src/analysis/options_oi.py"),
        ],
    },
    "Phase 4 — Optimization": {
        "status": "in_progress",
        "items": [
            ("Optuna Tuner (Bayesian HPO)", "src/prediction/optuna_tuner.py"),
            ("System Health Dashboard", "src/dashboard/pages/system_health.py"),
        ],
    },
    "Phase 5 — Production (Future)": {
        "status": "planned",
        "items": [
            ("RL Position Sizer (DQN)", "src/prediction/rl_sizer.py"),
            ("Event-Driven Architecture (asyncio)", "src/streaming/event_bus.py"),
            ("Angel One Live Trading", "src/brokers/angel_one_live.py"),
            ("Mobile-Responsive CSS", "src/dashboard/mobile.py"),
        ],
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_module(mod_path: str) -> tuple[bool, str]:
    """Try importing a module. Returns (success, error_msg)."""
    try:
        importlib.import_module(mod_path)
        return True, ""
    except ImportError as e:
        return False, str(e)[:60]
    except Exception as e:
        return False, f"Error: {str(e)[:60]}"


def _check_file(rel_path: str) -> bool:
    """Check if a file exists in project."""
    return (ROOT / rel_path).exists()


def _get_model_status() -> list[dict]:
    """Check which models are trained and their sizes."""
    model_dir = ROOT / "models"
    rows = []
    for ac in ["index", "equity", "futures", "crypto"]:
        path = model_dir / f"ensemble_{ac}.pkl"
        exists = path.exists()
        size_kb = round(path.stat().st_size / 1024, 0) if exists else 0
        mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M") if exists else "—"
        rows.append({
            "Asset Class": ac,
            "Status":      "✅ Trained" if exists else "❌ Not trained",
            "Size":        f"{size_kb:.0f} KB" if exists else "—",
            "Last Train":  mtime,
        })

    # HMM regime model
    hmm_path = model_dir / "regime_hmm.pkl"
    rows.append({
        "Asset Class": "regime_hmm",
        "Status":      "✅ Trained" if hmm_path.exists() else "❌ Not trained",
        "Size":        f"{hmm_path.stat().st_size/1024:.0f} KB" if hmm_path.exists() else "—",
        "Last Train":  datetime.fromtimestamp(hmm_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M") if hmm_path.exists() else "—",
    })
    return rows


def _get_optuna_results() -> list[dict]:
    """Load Optuna tuning results."""
    optuna_dir = ROOT / "data" / "optuna"
    rows = []
    for ac in ["index", "equity", "futures", "crypto"]:
        path = optuna_dir / f"best_params_{ac}.json"
        if path.exists():
            try:
                d = json.loads(path.read_text())
                rows.append({
                    "Asset":    ac,
                    "Accuracy": f"{d.get('best_accuracy', 0):.1f}%",
                    "Trials":   d.get("n_trials", 0),
                    "Duration": f"{d.get('duration_s', 0):.0f}s",
                    "Date":     d.get("timestamp", "")[:10],
                })
            except Exception:
                pass
    return rows


def _get_av_rate_limit() -> dict:
    """Get Alpha Vantage rate limit status."""
    try:
        from src.data.adapters.alphavantage_adapter import get_rate_limit_status
        return get_rate_limit_status()
    except Exception:
        return {"calls_today": 0, "calls_remaining": 25, "daily_limit": 25, "pct_used": 0}


def _get_cache_stats() -> dict:
    """Get data cache statistics."""
    cache_dir = ROOT / "data" / "processed"
    if not cache_dir.exists():
        return {"files": 0, "total_mb": 0}
    files = list(cache_dir.glob("*.parquet"))
    total_bytes = sum(f.stat().st_size for f in files)
    return {
        "files":    len(files),
        "total_mb": round(total_bytes / 1e6, 1),
        "oldest":   min((datetime.fromtimestamp(f.stat().st_mtime) for f in files),
                        default=datetime.now()).strftime("%Y-%m-%d") if files else "—",
        "newest":   max((datetime.fromtimestamp(f.stat().st_mtime) for f in files),
                        default=datetime.now()).strftime("%Y-%m-%d") if files else "—",
    }


def _get_validation_results() -> dict:
    """Load asset validation results from run_pipeline_v3."""
    val_path = ROOT / "data" / "asset_validation.json"
    if val_path.exists():
        try:
            return json.loads(val_path.read_text())
        except Exception:
            pass
    return {}


# ── Main render function ──────────────────────────────────────────────────────

def render_system_health():
    st.header("🏥 System Health & Evolution")
    st.caption(
        "Full system diagnostics — module status, model performance, "
        "API limits, cache, Optuna results, and phase completion tracker."
    )

    # ── Overall health score ──────────────────────────────────────────────────
    st.subheader("System Score")

    all_modules = {k: v for group in MODULES.values() for k, v in group.items()}
    loaded = sum(1 for mod in all_modules.values() if _check_module(mod)[0])
    total  = len(all_modules)
    score  = round(loaded / total * 100)

    score_color = "#00cc66" if score >= 80 else "#ffaa00" if score >= 60 else "#ff4444"
    st.markdown(
        f'<div style="background:rgba(0,0,0,0.2);border-radius:8px;padding:16px;'
        f'margin-bottom:16px;display:flex;align-items:center;gap:20px">'
        f'<div style="font-size:48px;font-weight:700;color:{score_color}">{score}%</div>'
        f'<div>'
        f'<div style="font-size:14px;color:#ccc">{loaded}/{total} modules loaded</div>'
        f'<div style="font-size:12px;color:#888;margin-top:4px">'
        f'{"Excellent — production ready" if score>=90 else "Good — minor modules missing" if score>=70 else "Needs attention — core modules failing"}'
        f'</div></div></div>',
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    models = _get_model_status()
    trained = sum(1 for m in models if "✅" in m["Status"])
    av      = _get_av_rate_limit()
    cache   = _get_cache_stats()

    k1.metric("Models Trained",  f"{trained}/{len(models)}")
    k2.metric("AV Calls Today",  f"{av['calls_today']}/{av['daily_limit']}",
              f"{av['calls_remaining']} remaining")
    k3.metric("Cache Files",     f"{cache['files']}",
              f"{cache['total_mb']} MB")
    k4.metric("Modules Loaded",  f"{loaded}/{total}")

    st.divider()

    # ── Phase completion tracker ───────────────────────────────────────────────
    st.subheader("🗺️ Evolution Roadmap")

    for phase_name, phase_data in PHASES.items():
        status = phase_data["status"]
        items  = phase_data["items"]

        # Check which files actually exist
        existing = [_check_file(path) for _, path in items]
        done     = sum(existing)
        total_i  = len(items)
        pct      = done / total_i * 100

        status_config = {
            "complete":    ("#003300", "#00ff88", "✅ COMPLETE"),
            "in_progress": ("#332200", "#ffaa00", "🔄 IN PROGRESS"),
            "planned":     ("#1a1a2e", "#8888ff", "📅 PLANNED"),
        }
        bg_c, fg_c, label = status_config.get(status, ("#222","#fff","?"))

        with st.expander(
            f"{label} — {phase_name} ({done}/{total_i} files)",
            expanded=(status == "in_progress"),
        ):
            # Progress bar
            st.markdown(
                f'<div style="background:#333;border-radius:3px;height:6px;margin-bottom:10px">'
                f'<div style="background:{fg_c};width:{pct:.0f}%;height:6px;border-radius:3px"></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            for (item_name, item_path), exists in zip(items, existing):
                icon = "✅" if exists else "⬜"
                st.markdown(
                    f'{icon} `{item_path}` — {item_name}',
                    unsafe_allow_html=True,
                )

    st.divider()

    # ── Module status ──────────────────────────────────────────────────────────
    st.subheader("🔌 Module Status")

    for group_name, group_modules in MODULES.items():
        with st.expander(group_name, expanded=(group_name == "Core")):
            rows = []
            for name, mod_path in group_modules.items():
                ok, err = _check_module(mod_path)
                rows.append({
                    "Module":  name,
                    "Status":  "✅ Loaded" if ok else "❌ Failed",
                    "Path":    mod_path,
                    "Error":   err if not ok else "",
                })
            df_mods = pd.DataFrame(rows)

            def color_status(val):
                if "✅" in str(val): return "color:#00cc66"
                if "❌" in str(val): return "color:#ff4444"
                return ""

            st.dataframe(
                df_mods.style.map(color_status, subset=["Status"]),
                use_container_width=True, hide_index=True,
            )

    st.divider()

    # ── Model status ───────────────────────────────────────────────────────────
    st.subheader("🤖 Trained Models")

    model_rows = _get_model_status()
    df_models  = pd.DataFrame(model_rows)

    def color_model(val):
        if "✅" in str(val): return "color:#00cc66;font-weight:bold"
        if "❌" in str(val): return "color:#ff4444"
        return ""

    st.dataframe(
        df_models.style.map(color_model, subset=["Status"]),
        use_container_width=True, hide_index=True,
    )

    # Accuracy from validation
    val = _get_validation_results()
    if val:
        st.subheader("📊 Walk-Forward Accuracy (last training run)")
        acc_rows = []
        baseline = {"index": 43.2, "equity": 37.5, "futures": 40.3, "crypto": 0.0}
        for ac, r in val.items():
            acc  = r.get("accuracy", 0)
            prev = baseline.get(ac, 0)
            acc_rows.append({
                "Asset":     ac,
                "Accuracy":  f"{acc:.1f}%",
                "vs Baseline":f"{acc-prev:+.1f}%",
                "Folds":     r.get("n_folds", "—"),
                "Bars":      f"{r.get('n_bars', 0):,}",
                "Horizon":   f"{r.get('label_horizon','?')}d",
            })
        df_acc = pd.DataFrame(acc_rows)

        def color_acc(val):
            try:
                v = float(str(val).replace("%","").replace("+",""))
                if v >= 55: return "color:#00cc66;font-weight:bold"
                if v >= 48: return "color:#ffaa00"
                return "color:#ff4444"
            except (ValueError, TypeError):
                return ""

        def color_vs(val):
            if "+" in str(val): return "color:#00cc66"
            if "-" in str(val): return "color:#ff4444"
            return ""

        st.dataframe(
            df_acc.style
                .map(color_acc, subset=["Accuracy"])
                .map(color_vs,  subset=["vs Baseline"]),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No validation results found. Run: `python run_pipeline_v3.py --skip-tuning`")

    st.divider()

    # ── Optuna results ────────────────────────────────────────────────────────
    st.subheader("⚡ Optuna Tuning Results")

    optuna_rows = _get_optuna_results()
    if optuna_rows:
        df_opt = pd.DataFrame(optuna_rows)
        st.dataframe(df_opt, use_container_width=True, hide_index=True)
    else:
        st.info("No Optuna results yet.")

    # Optuna run widget
    try:
        from src.prediction.optuna_tuner import render_optuna_widget
        render_optuna_widget()
    except ImportError:
        st.caption("Copy optuna_tuner.py to src/prediction/ to enable tuning.")

    st.divider()

    # ── API status ────────────────────────────────────────────────────────────
    st.subheader("🔑 API Status")

    try:
        from config.settings import settings
        api_rows = [
            {"API":        "Anthropic Claude",
             "Status":     "✅ Configured" if getattr(settings,"ANTHROPIC_API_KEY","") else "❌ Missing",
             "Usage":      "AI analysis, nerve center, company brief"},
            {"API":        "Alpha Vantage",
             "Status":     "✅ Configured" if getattr(settings,"ALPHA_VANTAGE_KEY","") else "❌ Missing",
             "Usage":      f"Fundamentals | {av['calls_today']}/{av['daily_limit']} calls today"},
            {"API":        "NewsAPI",
             "Status":     "✅ Configured" if getattr(settings,"NEWS_API_KEY","") else "❌ Missing",
             "Usage":      "Nerve Center news feed"},
            {"API":        "Angel One SmartAPI",
             "Status":     "✅ Configured" if getattr(settings,"ANGEL_API_KEY","") else "❌ Missing",
             "Usage":      "Live trading execution"},
            {"API":        "Telegram",
             "Status":     "✅ Configured" if getattr(settings,"TELEGRAM_BOT_TOKEN","") else "❌ Missing",
             "Usage":      "Signal alerts"},
            {"API":        "GNews",
             "Status":     "✅ Configured" if getattr(settings,"GNEWS_API_KEY","") else "❌ Missing",
             "Usage":      "Latest news feed"},
            {"API":        "CoinSwitch",
             "Status":     "✅ Configured" if getattr(settings,"COINSWITCH_API_KEY","") else "❌ Missing",
             "Usage":      "Crypto portfolio"},
        ]
        df_api = pd.DataFrame(api_rows)
        st.dataframe(
            df_api.style.map(color_status, subset=["Status"]),
            use_container_width=True, hide_index=True,
        )
    except Exception as e:
        st.warning(f"Settings load failed: {e}")

    st.divider()

    # ── Cache status ───────────────────────────────────────────────────────────
    st.subheader("💾 Data Cache")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Parquet files", cache["files"])
    c2.metric("Total size",    f"{cache['total_mb']} MB")
    c3.metric("Oldest data",   cache.get("oldest", "—"))
    c4.metric("Newest data",   cache.get("newest", "—"))

    if st.button("🗑️ Clear Stale Cache (>30 days)", key="clear_cache"):
        cleared = 0
        cache_dir = ROOT / "data" / "processed"
        cutoff    = time.time() - 30 * 86400
        for f in cache_dir.glob("*.parquet"):
            if f.stat().st_mtime < cutoff:
                f.unlink()
                cleared += 1
        st.success(f"Cleared {cleared} stale files.")
        st.rerun()

    st.divider()

    # ── Quick actions ──────────────────────────────────────────────────────────
    st.subheader("⚡ Quick Actions")

    qa1, qa2, qa3, qa4 = st.columns(4)
    with qa1:
        if st.button("🏃 Train Index Model", width="stretch"):
            st.info("Run: `python run_pipeline_v3.py --asset index --skip-tuning`")
    with qa2:
        if st.button("🏃 Train All Models", width="stretch"):
            st.info("Run: `python run_pipeline_v3.py --skip-tuning`")
    with qa3:
        if st.button("🔧 Train HMM Regime", width="stretch"):
            with st.spinner("Training HMM..."):
                try:
                    from src.data.manager import DataManager
                    from src.data.models import Interval
                    from src.features.feature_engine import FeatureEngine
                    from src.analysis.regime_detector import RegimeDetector
                    dm = DataManager(); fe = FeatureEngine(); rd = RegimeDetector()
                    df = dm.get_ohlcv("NIFTY50", Interval.D1, days_back=1000)
                    ft = fe.build(df, drop_na=False)
                    res = rd.train(ft, symbol="NIFTY50")
                    if res.get("trained"):
                        st.success(f"HMM trained: {res['n_bars']} bars")
                    else:
                        st.error(f"Failed: {res.get('reason')}")
                except Exception as e:
                    st.error(f"Error: {e}")
    with qa4:
        if st.button("♻️ Clear Module Cache", width="stretch"):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.success("Streamlit cache cleared")
            st.rerun()

    # ── System info ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("💻 System Info")

    import platform
    si1, si2, si3, si4 = st.columns(4)
    si1.metric("Python",   platform.python_version())
    si2.metric("Platform", platform.system())
    si3.metric("ROOT",     str(ROOT.name))

    try:
        import psutil
        mem = psutil.virtual_memory()
        si4.metric("RAM Used", f"{mem.percent:.0f}%",
                   f"{mem.available/1e9:.1f} GB free")
    except ImportError:
        si4.metric("RAM", "psutil not installed")

    # Package versions
    with st.expander("📦 Key Package Versions"):
        packages = ["streamlit", "xgboost", "lightgbm", "optuna",
                    "pandas", "numpy", "scikit-learn", "hmmlearn"]
        rows = []
        for pkg in packages:
            try:
                mod = importlib.import_module(pkg.replace("-","_"))
                ver = getattr(mod, "__version__", "?")
            except ImportError:
                ver = "NOT INSTALLED"
            rows.append({"Package": pkg, "Version": ver,
                         "Status": "✅" if ver != "NOT INSTALLED" else "❌"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)