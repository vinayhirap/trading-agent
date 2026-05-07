#   trading-agent/src/features/feature_engine.py
import numpy as np
import pandas as pd
from loguru import logger
from src.features.indicators import (
    ema, sma, wma, hma, dema, tema,
    macd, adx, rsi, stochastic, williams_r, cmo,
    atr, bollinger_bands, keltner_channel,
    vwap, vwap_bands, obv, cvd, mfi, cmf,
    supertrend, ichimoku, donchian_channel,
    live_volume_rate, fair_value_gaps,
    market_structure, heikin_ashi,
    elder_ray, pivot_points,
)


class FeatureEngine:
    """
    Converts raw OHLCV → feature matrix for prediction engine.

    Design principles:
    - All features normalised/scaled relative to price (no raw price values)
    - No lookahead bias — every feature at bar t uses only data ≤ t
    - NaN rows at start (warm-up) dropped or kept based on drop_na
    - Returns a copy — never modifies input DataFrame
    """

    def __init__(self, interval: str = "1d"):
        self.interval = interval

    def build(
        self,
        df: pd.DataFrame,
        drop_na: bool = True,
        market_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        if df.empty or len(df) < 30:
            logger.warning("FeatureEngine: DataFrame too short (need ≥30 bars)")
            return df

        out    = df.copy()
        close  = out["close"]
        high   = out["high"]
        low    = out["low"]
        volume = out["volume"]

        try:
            # ── 1. Trend: EMAs + normalised distance ────────────────────────
            out["ema_9"]   = ema(close, 9)
            out["ema_21"]  = ema(close, 21)
            out["ema_50"]  = ema(close, 50)
            out["ema_200"] = ema(close, 200)

            out["ema9_pct"]   = (close - out["ema_9"])   / close
            out["ema21_pct"]  = (close - out["ema_21"])  / close
            out["ema50_pct"]  = (close - out["ema_50"])  / close
            out["ema200_pct"] = (close - out["ema_200"]) / close

            out["ema9_21_xo"]  = out["ema_9"]  - out["ema_21"]
            out["ema21_50_xo"] = out["ema_21"] - out["ema_50"]

            # HMA — faster signal
            out["hma_21"]     = hma(close, 21)
            out["hma21_pct"]  = (close - out["hma_21"]) / close

            # DEMA / TEMA — reduced lag
            out["dema_21"]    = dema(close, 21)
            out["tema_21"]    = tema(close, 21)
            out["dema21_pct"] = (close - out["dema_21"]) / close

            # ── 2. MACD ──────────────────────────────────────────────────────
            macd_df = macd(close)
            out["macd"]          = macd_df["macd"]
            out["macd_signal"]   = macd_df["signal"]
            out["macd_hist"]     = macd_df["histogram"]
            out["macd_hist_chg"] = macd_df["histogram"].diff()

            # ── 3. ADX ───────────────────────────────────────────────────────
            adx_df = adx(out)
            out["adx"]      = adx_df["adx"]
            out["plus_di"]  = adx_df["+di"]
            out["minus_di"] = adx_df["-di"]
            out["di_diff"]  = adx_df["+di"] - adx_df["-di"]

            # ── 4. Supertrend ────────────────────────────────────────────────
            st_df = supertrend(out, period=10, multiplier=3.0)
            out["st_dir"]  = st_df["supertrend_dir"]
            out["st_dist"] = (close - st_df["supertrend"]) / close
            out["st_buy"]  = st_df["supertrend_buy"]
            out["st_sell"] = st_df["supertrend_sell"]

            # ── 5. Ichimoku ──────────────────────────────────────────────────
            ich = ichimoku(out)
            out["ichi_above_cloud"] = ich["above_cloud"]
            out["ichi_cloud_color"] = ich["cloud_color"]
            out["ichi_tk_cross"]    = ich["tk_cross"]
            out["ichi_tenkan_pct"]  = (close - ich["tenkan_sen"]) / close
            out["ichi_kijun_pct"]   = (close - ich["kijun_sen"])  / close

            # ── 6. Donchian channel ──────────────────────────────────────────
            dc = donchian_channel(out, period=20)
            out["dc_pos"]   = dc["dc_pos"]
            out["dc_width"] = dc["dc_width"]

            # ── 7. Momentum oscillators ──────────────────────────────────────
            out["rsi_14"]  = rsi(close, 14)
            out["rsi_7"]   = rsi(close, 7)
            out["rsi_chg"] = out["rsi_14"].diff()

            stoch_df = stochastic(out)
            out["stoch_k"]    = stoch_df["stoch_k"]
            out["stoch_d"]    = stoch_df["stoch_d"]
            out["stoch_diff"] = stoch_df["stoch_k"] - stoch_df["stoch_d"]

            out["williams_r"] = williams_r(out)
            out["cmo_14"]     = cmo(close, 14)

            # ── 8. Elder Ray ─────────────────────────────────────────────────
            er = elder_ray(out)
            out["bull_power"] = er["bull_power"] / close
            out["bear_power"] = er["bear_power"] / close

            # ── 9. Volatility ────────────────────────────────────────────────
            out["atr_14"]    = atr(out, 14)
            out["atr_pct"]   = out["atr_14"] / close
            out["atr_ratio"] = out["atr_14"] / out["atr_14"].rolling(20).mean()

            bb_df = bollinger_bands(close)
            out["bb_pct_b"] = bb_df["bb_pct_b"]
            out["bb_width"] = bb_df["bb_width"]

            kc_df = keltner_channel(out)
            out["squeeze"] = (
                (bb_df["bb_upper"] < kc_df["kc_upper"]) &
                (bb_df["bb_lower"] > kc_df["kc_lower"])
            ).astype(int)

            # ── 10. Volume / order flow ──────────────────────────────────────
            out["vol_sma20"] = sma(volume, 20)
            out["vol_ratio"] = volume / out["vol_sma20"]
            out["vol_chg"]   = volume.pct_change(fill_method=None)

            out["obv"]      = obv(close, volume)
            out["obv_ema"]  = ema(out["obv"], 21)
            out["obv_slope"]= out["obv"] - out["obv_ema"]

            out["mfi_14"] = mfi(out, 14)
            out["cmf_20"] = cmf(out, 20)

            # CVD — buy vs sell pressure
            out["cvd"]       = cvd(out)
            out["cvd_slope"] = out["cvd"] - out["cvd"].shift(5)

            # Live Volume analysis
            lv = live_volume_rate(out, lookback=20)
            out["rvol"]        = lv["rvol"]
            out["vol_thrust"]  = lv["vol_thrust"]
            out["vol_trend"]   = lv["vol_trend"]
            out["session_rvol"]= lv["session_rvol"]

            # VWAP (intraday only)
            if self.interval not in ("1d", "1wk"):
                vb = vwap_bands(out)
                out["vwap_dist"] = vb["vwap_dist"]
                out["vwap_u1_dist"] = (close - vb["vwap_u1"]) / close
                out["vwap_l1_dist"] = (close - vb["vwap_l1"]) / close

            # ── 11. Price action / candle features ───────────────────────────
            out["body_pct"]   = (close - out["open"]).abs() / (high - low + 1e-9)
            out["upper_wick"] = (high - pd.concat([close, out["open"]], axis=1).max(axis=1)) / (high - low + 1e-9)
            out["lower_wick"] = (pd.concat([close, out["open"]], axis=1).min(axis=1) - low) / (high - low + 1e-9)
            out["is_bullish"] = (close > out["open"]).astype(int)

            for n in [1, 3, 5, 10]:
                out[f"ret_{n}d"] = close.pct_change(n, fill_method=None)

            # Heikin-Ashi features
            ha = heikin_ashi(out)
            out["ha_bull"] = ha["ha_bull"]
            out["ha_doji"] = ha["ha_doji"]
            out["ha_body"] = (ha["ha_close"] - ha["ha_open"]).abs() / (ha["ha_high"] - ha["ha_low"] + 1e-9)

            # ── 12. Smart Money Concepts ─────────────────────────────────────
            fvg = fair_value_gaps(out)
            out["fvg_bull"]      = fvg["fvg_bull"]
            out["fvg_bear"]      = fvg["fvg_bear"]
            out["fvg_fill_bull"] = fvg["fvg_fill_bull"]
            out["fvg_fill_bear"] = fvg["fvg_fill_bear"]

            ms = market_structure(out, swing_period=5)
            out["ms_hh"]       = ms["hh"]
            out["ms_ll"]       = ms["ll"]
            out["ms_hl"]       = ms["hl"]
            out["ms_lh"]       = ms["lh"]
            out["ms_bos_bull"] = ms["bos_bull"]
            out["ms_bos_bear"] = ms["bos_bear"]
            out["ms_choch"]    = ms["choch"]

            # ── 13. Pivot Points (daily/swing only) ──────────────────────────
            if self.interval in ("1d", "1h"):
                pp = pivot_points(out, method="classic")
                out["pp_r1_dist"] = (pp["r1"] - close) / close
                out["pp_s1_dist"] = (close - pp["s1"]) / close
                out["pp_dist"]    = (close - pp["pp"])  / close

            # ── 14. Inter-Market (beta / context) ────────────────────────────
            if market_df is not None and not market_df.empty:
                mkt_close = market_df["close"].reindex(out.index).ffill()
                for n in [1, 3, 5, 10]:
                    out[f"mkt_ret_{n}d"] = mkt_close.pct_change(n, fill_method=None)

                ret_5d     = out["ret_5d"]     if "ret_5d"     in out.columns else close.pct_change(5)
                mkt_ret_5d = out["mkt_ret_5d"] if "mkt_ret_5d" in out.columns else mkt_close.pct_change(5)
                out["rel_ret_5d"]    = ret_5d - mkt_ret_5d
                out["mkt_rsi_14"]    = rsi(mkt_close, 14)
                out["mkt_rel_rsi"]   = out["rsi_14"] - out["mkt_rsi_14"]
                out["mkt_correlation"] = (
                    close.pct_change(fill_method=None)
                    .rolling(20)
                    .corr(mkt_close.pct_change(fill_method=None))
                )

            # ── 15. Drop raw EMA columns (price-scale, not normalised) ───────
            out = out.drop(columns=["ema_9","ema_21","ema_50","ema_200",
                                    "hma_21","dema_21","tema_21"], errors="ignore")

        except Exception as e:
            logger.error(f"FeatureEngine error: {e}")
            raise

        out = out.replace([np.inf, -np.inf], np.nan)

        if drop_na:
            before = len(out)
            out    = out.dropna()
            logger.info(f"FeatureEngine: {before} bars → {len(out)} after dropping NaN")
        else:
            logger.info(f"FeatureEngine: {len(out)} bars (NaN rows kept)")

        return out

    def get_feature_names(self) -> list[str]:
        ohlcv = {"open", "high", "low", "close", "volume"}
        dummy = pd.DataFrame({
            "open":   [100.0] * 300,
            "high":   [105.0] * 300,
            "low":    [98.0]  * 300,
            "close":  [102.0] * 300,
            "volume": [1e6]   * 300,
        }, index=pd.date_range("2023-01-01", periods=300, freq="1h", tz="UTC"))
        dummy.index.name = "timestamp"
        dummy_mkt = pd.DataFrame(
            {"close": [100.0 * (1.001 ** i) for i in range(300)]},
            index=dummy.index,
        )
        built = self.build(dummy, drop_na=False, market_df=dummy_mkt)
        return [c for c in built.columns if c not in ohlcv]

    def latest_features(
        self, df: pd.DataFrame, market_df: pd.DataFrame = None
    ) -> pd.Series:
        """Feature vector for most recent bar — live inference."""
        featured = self.build(df, drop_na=False, market_df=market_df)
        return featured.iloc[-1]