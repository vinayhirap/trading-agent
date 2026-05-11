"""
Realtime advisor service.

Builds 1-minute rolling candles from cached intraday history plus local ticks,
computes live indicators, applies session-aware intraday logic, and upgrades
index signals into CALL/PUT ideas through the F&O engine when possible.

KEY FIX: Now loads trained ensemble_{index/equity/futures/crypto}.pkl models
and uses them for predictions instead of pure rule-based scoring.
Rule-based scoring is kept as fallback when models are unavailable.
"""
from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

from config.settings import settings
from src.alerts.telegram_sender import make_telegram_sender_from_settings
# intraday_engine removed — graceful stub
class _IntradayStub:
    def enhance_signal(self, symbol, action, confidence, **kwargs):
        from dataclasses import dataclass
        @dataclass
        class _Sig:
            adjusted_confidence: float = confidence
            entry_timing:  str = "ENTER NOW" if action != "HOLD" else "WAIT"
            session_label: str = "Market Hours"
            timing_advice: str = ""
            strategy_fit:  str = ""
            session_notes:    list = None
            session_warnings: list = None
            def __post_init__(self):
                self.session_notes    = self.session_notes    or []
                self.session_warnings = self.session_warnings or []
        return _Sig(adjusted_confidence=confidence)
intraday_engine = _IntradayStub()
from src.data.manager import DataManager
from src.data.models import Interval
from src.features.feature_engine import FeatureEngine
from src.fotrading.fo_engine import fo_engine
from src.fotrading.option_chain import option_chain_fetcher
from src.streaming.price_store import price_store


WATCHLIST = [
    "NIFTY50",
    "BANKNIFTY",
    "NIFTYIT",
    "GOLD",
    "SILVER",
    "CRUDEOIL",
    "BTC",
    "ETH",
]

OPTIONABLE  = {"NIFTY50", "BANKNIFTY"}
CRYPTO_24X7 = {"BTC", "ETH", "SOL", "MATIC", "BNB", "XRP", "ADA", "DOGE", "AVAX", "DOT"}

# Model file mapping: asset_class → pkl filename
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_FILES = {
    "index":   "ensemble_index.pkl",
    "equity":  "ensemble_equity.pkl",
    "futures": "ensemble_futures.pkl",
    "crypto":  "ensemble_crypto.pkl",
}

# Symbol → asset class mapping
SYMBOL_ASSET_CLASS = {
    "NIFTY50":    "index",
    "BANKNIFTY":  "index",
    "NIFTYIT":    "index",
    "SENSEX":     "index",
    "FINNIFTY":   "index",
    "GOLD":       "futures",
    "SILVER":     "futures",
    "CRUDEOIL":   "futures",
    "COPPER":     "futures",
    "NATURALGAS": "futures",
    "BTC":        "crypto",
    "ETH":        "crypto",
    "SOL":        "crypto",
    "BNB":        "crypto",
    "XRP":        "crypto",
}
# Everything else defaults to "equity"


@dataclass
class RealtimeAdvice:
    symbol: str
    action: str
    confidence: float
    recommended_instrument: str
    price: float
    source: str
    updated_at: str
    reason_summary: str
    session_label: str = ""
    entry_timing: str = ""
    regime: str = ""
    instrument_symbol: str = ""
    rsi: float = 0.0
    macd_hist: float = 0.0
    adx: float = 0.0
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    atr_pct: float = 0.0
    stop_loss: float = 0.0
    target: float = 0.0
    macd_signal: str = ""
    entry: float = 0.0
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class RealtimeAdvisorService:
    POLL_SECONDS              = 30.0   # increased: model inference is heavier than rule-based
    BAR_CACHE_TTL_SECONDS     = 35.0
    TELEGRAM_DEBOUNCE_SECONDS = 45.0
    CAPITAL_FOR_FO            = 100_000.0

    def __init__(self):
        self._dm            = DataManager()
        self._feature_engine = FeatureEngine(interval="1m")
        self._intraday      = intraday_engine
        self._fo_fetcher    = option_chain_fetcher
        self._fo_engine     = fo_engine
        self._telegram      = make_telegram_sender_from_settings(settings)
        self._thread        = None
        self._running       = False
        self._lock          = threading.Lock()
        self._advice:       dict[str, RealtimeAdvice] = {}
        self._last_sent:    dict[str, str]   = {}
        self._last_sent_at: dict[str, float] = {}
        self._bar_cache:    dict[str, tuple] = {}
        self._watchlist     = set(WATCHLIST)
        self._chart_context: dict[str, dict] = {}
        self._last_regime:  dict[str, str]   = {}   # P3-A: regime change tracking

        # ── Load trained models once at startup ──────────────────────────────
        self._models: dict[str, object] = {}
        self._load_models()

    def _load_models(self) -> None:
        import joblib
        loaded = []
        for asset_class, fname in MODEL_FILES.items():
            path = MODELS_DIR / fname
            if path.exists():
                try:
                    self._models[asset_class] = joblib.load(path)
                    loaded.append(f"{asset_class}({fname})")
                except Exception as e:
                    logger.warning(f"RealtimeAdvisor: could not load {fname}: {e}")
            else:
                logger.debug(f"RealtimeAdvisor: model not found: {path}")
        if loaded:
            logger.info(f"RealtimeAdvisor: loaded models — {', '.join(loaded)}")
        else:
            logger.warning("RealtimeAdvisor: no trained models found — falling back to rule-based signals.")

    def _get_asset_class(self, symbol: str) -> str:
        return SYMBOL_ASSET_CLASS.get(symbol.upper(), "equity")

    def _predict_with_model(self, symbol: str, features: pd.DataFrame):
        asset_class = self._get_asset_class(symbol)
        bundle = self._models.get(asset_class)
        if bundle is None:
            return None

        try:
            # joblib bundle is a dict — extract the global models
            xgb = bundle.get("global_xgb")
            lgb = bundle.get("global_lgb")
            feat_cols = bundle.get("feature_cols", [])

            latest = features.dropna(how="all").tail(1)
            if latest.empty or not feat_cols:
                return None

            X = latest.reindex(columns=feat_cols, fill_value=0.0)
            X = X.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)

            preds, probas = [], []
            for model in [xgb, lgb]:
                if model is None:
                    continue
                preds.append(model.predict(X)[0])
                p = model.predict_proba(X)[0]
                probas.append(p)

            if not preds:
                return None

            # Ensemble: average probabilities
            import numpy as np
            avg_proba = np.mean(probas, axis=0)
            model_for_classes = xgb if xgb is not None else lgb
            classes = list(model_for_classes.classes_)
            pred_idx  = int(np.argmax(avg_proba))
            pred_label= classes[pred_idx]
            confidence= float(avg_proba[pred_idx])

            label_map = {1: "BUY", 0: "HOLD", -1: "SELL"}
            action    = label_map.get(int(pred_label), "HOLD")
            confidence= max(0.30, min(0.98, confidence))

            return action, confidence

        except Exception as e:
            logger.debug(f"RealtimeAdvisor model predict {symbol}: {e}")
            return None


    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop,
            daemon=True,
            name="RealtimeAdvisor",
        )
        self._thread.start()
        logger.info("RealtimeAdvisor started")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def get(self, symbol: str) -> dict:
        with self._lock:
            advice = self._advice.get(symbol)
        return asdict(advice) if advice else {}

    def get_all(self) -> dict[str, dict]:
        with self._lock:
            return {sym: asdict(item) for sym, item in self._advice.items()}

    def update_chart_context(self, payload: dict) -> dict:
        symbol = str(payload.get("symbol", "")).strip().upper()
        if not symbol:
            return {"ok": False, "error": "Missing symbol"}
        with self._lock:
            self._chart_context[symbol] = {
                "symbol":      symbol,
                "raw_symbol":  payload.get("raw_symbol", ""),
                "timeframe":   payload.get("timeframe", ""),
                "url":         payload.get("url", ""),
                "source":      payload.get("source", "extension"),
                "captured_at": payload.get("captured_at", datetime.now(timezone.utc).isoformat()),
            }
            self._watchlist.add(symbol)
        return {"ok": True, "watching": symbol}

    def get_chart_context(self) -> dict[str, dict]:
        with self._lock:
            return dict(self._chart_context)

    def _loop(self) -> None:
        while self._running:
            started = time.time()
            for symbol in list(self._watchlist):
                try:
                    advice = self._compute_advice(symbol)
                    if advice:
                        with self._lock:
                            self._advice[symbol] = advice
                        self._maybe_send_telegram(advice)
                except Exception as exc:
                    logger.debug(f"RealtimeAdvisor {symbol}: {exc}")
            elapsed = time.time() - started
            time.sleep(max(0.5, self.POLL_SECONDS - elapsed))

    def _compute_advice(self, symbol: str) -> RealtimeAdvice | None:
        candles, source = self._get_live_candles(symbol)
        if candles.empty or len(candles) < 20:
            return None

        featured = self._feature_engine.build(candles, drop_na=False)
        featured  = featured.replace([float("inf"), float("-inf")], pd.NA)
        valid     = featured.dropna(subset=["close"])
        if valid.empty:
            return None

        latest     = valid.iloc[-1]
        close      = float(latest["close"])
        close_s    = valid["close"].astype(float)
        ema_fast   = float(close_s.ewm(span=9,  adjust=False).mean().iloc[-1])
        ema_mid    = float(close_s.ewm(span=21, adjust=False).mean().iloc[-1])
        ema_slow   = float(close_s.ewm(span=50, adjust=False).mean().iloc[-1])
        rsi_14     = self._safe_float(latest.get("rsi_14"), 50.0)
        macd_hist  = self._safe_float(latest.get("macd_hist"), 0.0)
        macd_hist_prev = self._safe_float(valid["macd_hist"].iloc[-2], 0.0) if len(valid) > 1 else 0.0
        adx_val    = self._safe_float(latest.get("adx"), 18.0)
        plus_di    = self._safe_float(latest.get("plus_di"), 0.0)
        minus_di   = self._safe_float(latest.get("minus_di"), 0.0)
        atr_pct    = self._safe_float(latest.get("atr_pct"), 0.01)
        vol_ratio  = self._safe_float(latest.get("vol_ratio"), 1.0)
        atr_14     = self._safe_float(latest.get("atr_14"), close * 0.015)

        # ── Try ML model first ────────────────────────────────────────────────
        model_result = self._predict_with_model(symbol, valid)
        ml_used = model_result is not None

        if ml_used:
            ml_action, ml_conf = model_result
            # Blend with rule-based for regime/session context
            _, rule_score, regime, reasons, warnings = self._evaluate_signal(
                close=close, ema_fast=ema_fast, ema_mid=ema_mid, ema_slow=ema_slow,
                rsi_14=rsi_14, macd_hist=macd_hist, macd_hist_prev=macd_hist_prev,
                adx_val=adx_val, plus_di=plus_di, minus_di=minus_di,
                atr_pct=atr_pct, vol_ratio=vol_ratio,
            )
            # ML confidence is primary; rule score adjusts it slightly (±5%)
            rule_adj   = min(0.05, abs(rule_score) * 0.05)
            rule_agree = (
                (ml_action == "BUY"  and rule_score > 0) or
                (ml_action == "SELL" and rule_score < 0) or
                (ml_action == "HOLD" and abs(rule_score) < 0.18)
            )
            final_conf_raw = ml_conf + (rule_adj if rule_agree else -rule_adj)
            bias = (
                "STRONG BUY" if ml_action == "BUY"  and ml_conf > 0.70 else
                "BUY"        if ml_action == "BUY"  else
                "STRONG SELL"if ml_action == "SELL" and ml_conf > 0.70 else
                "SELL"       if ml_action == "SELL" else
                "HOLD"
            )
            reasons.insert(0, f"XGBoost/LightGBM ensemble: {ml_action} {ml_conf:.0%}")
            source = f"ml_{self._get_asset_class(symbol)}"

        else:
            # Pure rule-based fallback
            bias, rule_score, regime, reasons, warnings = self._evaluate_signal(
                close=close, ema_fast=ema_fast, ema_mid=ema_mid, ema_slow=ema_slow,
                rsi_14=rsi_14, macd_hist=macd_hist, macd_hist_prev=macd_hist_prev,
                adx_val=adx_val, plus_di=plus_di, minus_di=minus_di,
                atr_pct=atr_pct, vol_ratio=vol_ratio,
            )
            final_conf_raw = self._score_to_confidence(rule_score)
            warnings.append("Rule-based signal — run run_pipeline_v3.py to enable ML model")

        # ── Session / timing adjustment ───────────────────────────────────────
        if symbol in CRYPTO_24X7:
            final_conf    = max(0.30, min(0.99, final_conf_raw))
            entry_timing  = "ACTIVE 24x7"
            session_label = "Crypto 24x7"
            timing_advice = "Crypto trades continuously. Act on signal quality."
            strategy_fit  = "GOOD"
        else:
            intraday_signal = self._intraday.enhance_signal(
                symbol=symbol,
                bias=bias,
                confidence=final_conf_raw,
                atr_pct=atr_pct,
            )
            final_conf    = max(0.25, min(0.99, intraday_signal.adjusted_confidence))
            entry_timing  = intraday_signal.entry_timing
            session_label = intraday_signal.session_label
            timing_advice = intraday_signal.timing_advice
            strategy_fit  = intraday_signal.strategy_fit
            reasons.extend(intraday_signal.session_notes[:2])
            warnings.extend(intraday_signal.session_warnings[:2])

        action = self._final_action_from_bias(bias, final_conf, strategy_fit)

        # ── F&O upgrade ───────────────────────────────────────────────────────
        instrument_action  = action
        instrument_symbol  = ""
        stop_loss          = 0.0
        target             = 0.0

        if symbol in OPTIONABLE and action in {"BUY", "SELL"} and final_conf >= 0.55:
            fo_signal = self._build_fo_signal(symbol, bias, regime, final_conf)
            if fo_signal:
                instrument_action = f"BUY {fo_signal.instrument}"
                instrument_symbol = fo_signal.symbol
                stop_loss         = fo_signal.stop_loss_prem
                target            = fo_signal.target_premium
                reasons.extend(fo_signal.reasons[:2])
                warnings.extend(fo_signal.warnings[:2])
            else:
                instrument_action = "BUY CALL" if action == "BUY" else "BUY PUT"
        elif action in {"BUY", "SELL"}:
            instrument_action = action

        # ── ATR-based SL/Target for non-F&O symbols ───────────────────────────
        if stop_loss == 0.0 and action in {"BUY", "SELL"} and atr_14 > 0:
            if action == "BUY":
                stop_loss = round(close - atr_14 * 1.5, 2)
                target    = round(close + atr_14 * 2.0, 2)
            else:
                stop_loss = round(close + atr_14 * 1.5, 2)
                target    = round(close - atr_14 * 2.0, 2)

        if action == "HOLD" and "EXIT" in entry_timing.upper():
            instrument_action = "HOLD"

        reasons  = self._dedupe(reasons)[:5]
        warnings = self._dedupe(warnings)[:3]

        reason_summary = self._build_reason_summary(
            action=instrument_action,
            rsi=rsi_14,
            macd_hist=macd_hist,
            adx=adx_val,
            reasons=reasons,
            timing=timing_advice,
        )

        # Use live price_store price for consistency with header ticker
        live_px = price_store.get(symbol, fallback=False)
        if live_px and live_px > 0:
            close = live_px  # overwrite candle close with live tick # USE_LIVE_PRICE_PATCHED

        _advice = RealtimeAdvice(
            symbol               = symbol,
            action               = action,
            confidence           = round(final_conf, 3),
            recommended_instrument = instrument_action,
            price                = round(close, 2),
            source               = source,
            updated_at           = datetime.now(timezone.utc).isoformat(),
            reason_summary       = reason_summary,
            session_label        = session_label,
            entry_timing         = entry_timing,
            regime               = regime,
            instrument_symbol    = instrument_symbol,
            rsi                  = round(rsi_14, 1),
            macd_hist            = round(macd_hist, 4),
            macd_signal          = "bullish" if macd_hist >= 0 else "bearish",
            adx                  = round(adx_val, 1),
            ema_fast             = round(ema_fast, 2),
            ema_slow             = round(ema_slow, 2),
            atr_pct              = round(atr_pct * 100, 2),
            stop_loss            = round(stop_loss, 2) if stop_loss else 0.0,
            target               = round(target, 2) if target else 0.0,
            entry                = round(close, 2),
            reasons              = reasons,
            warnings             = warnings,
        )

        # ── Record into LearningEngine (P2-B fix) ────────────────────────────
        # Only record actionable signals (not HOLD) and throttle to once per signal
        # change per symbol to avoid flooding the DB every 30s
        try:
            if action != "HOLD":
                prev = getattr(self, "_last_recorded_bias", {})
                prev_bias = prev.get(symbol, {}).get("bias")
                prev_conf = prev.get(symbol, {}).get("conf", 0)
                cur_bias  = _advice.bias if hasattr(_advice, "bias") else action
                # Record if bias changed OR confidence shifted >5%
                if prev_bias != cur_bias or abs(final_conf - prev_conf) > 0.05:
                    from src.analysis.learning_engine_v2 import LearningEngineV2
                    le = LearningEngineV2()
                    le.record(
                        symbol       = symbol,
                        signal       = cur_bias if cur_bias in ("BUY","SELL","STRONG BUY","STRONG SELL","HOLD") else action,
                        confidence   = round(final_conf, 3),
                        asset_class  = self._get_asset_class(symbol),
                        regime       = regime or "UNKNOWN",
                        entry_price  = round(close, 2),
                        horizon_bars = 5,
                    )
                    le.check_and_resolve_all()
                    if not hasattr(self, "_last_recorded_bias"):
                        self._last_recorded_bias = {}
                    self._last_recorded_bias[symbol] = {"bias": cur_bias, "conf": final_conf}
        except Exception as _le_err:
            logger.debug(f"LearningEngine record skipped ({symbol}): {_le_err}")


        # ── P3-A: Regime change Telegram alert ──────────────────────────────
        try:
            if regime and regime != "UNKNOWN":
                prev_regime = self._last_regime.get(symbol)
                if prev_regime and prev_regime != regime:
                    from src.alerts.market_alerts import send_regime_change_alert
                    send_regime_change_alert(symbol, prev_regime, regime, final_conf, rsi_14)
                self._last_regime[symbol] = regime
        except Exception as _rc_err:
            logger.debug("Regime alert err (" + symbol + "): " + str(_rc_err))

        return _advice