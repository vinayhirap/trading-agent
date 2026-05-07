"""
Advanced Feature Engineering — extends base FeatureEngine for better accuracy.
Adds: regime detection, momentum quality, gap analysis, candle patterns, market structure.
"""
import numpy as np
import pandas as pd
from loguru import logger


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    ema20  = close.ewm(span=20,  adjust=False).mean()
    ema50  = close.ewm(span=50,  adjust=False).mean()
    ema100 = close.ewm(span=100, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    df["regime_bull_align"] = (
        (close > ema20) & (ema20 > ema50) & (ema50 > ema100)
    ).astype(int)
    df["regime_bear_align"] = (
        (close < ema20) & (ema20 < ema50) & (ema50 < ema100)
    ).astype(int)
    df["ema20_slope"]  = ema20.diff(5)  / (ema20.shift(5)  + 1e-9)
    df["ema50_slope"]  = ema50.diff(10) / (ema50.shift(10) + 1e-9)
    df["ema200_slope"] = ema200.diff(20)/ (ema200.shift(20)+ 1e-9)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.ewm(span=14, adjust=False).mean()
    atr50 = tr.ewm(span=50, adjust=False).mean()

    df["vol_regime"]      = atr14 / (atr50 + 1e-9)
    df["vol_regime_high"] = (df["vol_regime"] > 1.5).astype(int)
    df["vol_regime_low"]  = (df["vol_regime"] < 0.7).astype(int)

    rets = close.pct_change()
    df["hist_vol_20"]    = rets.rolling(20).std() * np.sqrt(252)
    df["hist_vol_5"]     = rets.rolling(5).std()  * np.sqrt(252)
    df["vol_ratio_5_20"] = df["hist_vol_5"] / (df["hist_vol_20"] + 1e-9)

    range_20             = close.rolling(20).max() - close.rolling(20).min()
    df["range_compression"] = range_20 / (atr14 * 20 + 1e-9)
    return df


def add_momentum_quality(df: pd.DataFrame) -> pd.DataFrame:
    close  = df["close"]
    volume = df["volume"]

    high_52w = close.rolling(min(252, len(close))).max()
    low_52w  = close.rolling(min(252, len(close))).min()
    df["rs_52w"] = (close - low_52w) / (high_52w - low_52w + 1e-9)

    high_1m  = close.rolling(20).max()
    low_1m   = close.rolling(20).min()
    df["rs_1m"] = (close - low_1m) / (high_1m - low_1m + 1e-9)

    for n in [2, 5, 10, 21]:
        df[f"roc_{n}"] = close.pct_change(n)

    df["mom_quality"] = (
        (df["roc_2"]  > 0).astype(int) +
        (df["roc_5"]  > 0).astype(int) +
        (df["roc_10"] > 0).astype(int) +
        (df["roc_21"] > 0).astype(int)
    ) / 4

    vol_ma = volume.rolling(20).mean()
    pchg   = close.pct_change()
    df["bull_volume_score"] = (
        pchg.clip(lower=0) * (volume / (vol_ma + 1e-9))
    ).rolling(5).mean()
    df["bear_volume_score"] = (
        pchg.clip(upper=0).abs() * (volume / (vol_ma + 1e-9))
    ).rolling(5).mean()
    return df


def add_gap_analysis(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    open_ = df["open"]
    prev_close = close.shift()

    df["gap_pct"]   = (open_ - prev_close) / (prev_close + 1e-9)
    df["gap_up"]    = (df["gap_pct"] > 0.005).astype(int)
    df["gap_down"]  = (df["gap_pct"] < -0.005).astype(int)
    df["gap_filled"] = (
        ((df["gap_pct"] > 0) & (df["low"] <= prev_close)) |
        ((df["gap_pct"] < 0) & (df["high"] >= prev_close))
    ).astype(int)
    df["gap_freq"]      = df["gap_pct"].abs().rolling(10).mean()
    df["gap_fill_rate"] = df["gap_filled"].rolling(20).mean()
    return df


def add_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body  = (c - o).abs()
    rng   = h - l
    upper_w = h - pd.concat([c, o], axis=1).max(axis=1)
    lower_w = pd.concat([c, o], axis=1).min(axis=1) - l

    df["body_ratio"]        = body / (rng + 1e-9)
    df["wick_imbalance"]    = (lower_w - upper_w) / (rng + 1e-9)
    df["bull_candle"]       = ((c > o) * df["body_ratio"]).clip(0, 1)
    df["bear_candle"]       = ((c < o) * df["body_ratio"]).clip(0, 1)
    df["doji"]              = (body / (rng + 1e-9) < 0.1).astype(int)

    df["bull_engulf"] = (
        ((c - o) > 0) & ((c.shift() - o.shift()) < 0) &
        (c > o.shift()) & (o < c.shift())
    ).astype(int)
    df["bear_engulf"] = (
        ((c - o) < 0) & ((c.shift() - o.shift()) > 0) &
        (c < o.shift()) & (o > c.shift())
    ).astype(int)
    df["bull_pattern_score"] = df["bull_candle"].rolling(3).mean()
    df["bear_pattern_score"] = df["bear_candle"].rolling(3).mean()
    return df


def add_market_structure(df: pd.DataFrame) -> pd.DataFrame:
    close, high, low = df["close"], df["high"], df["low"]

    pivot_high = (
        (high > high.shift(1)) & (high > high.shift(2)) &
        (high > high.shift(-1).fillna(0)) & (high > high.shift(-2).fillna(0))
    ).astype(int)
    pivot_low = (
        (low < low.shift(1)) & (low < low.shift(2)) &
        (low < low.shift(-1).fillna(float("inf"))) &
        (low < low.shift(-2).fillna(float("inf")))
    ).astype(int)

    ph = high.where(pivot_high == 1).ffill()
    pl = low.where(pivot_low  == 1).ffill()
    df["dist_to_pivot_high"] = (ph - close)  / (close + 1e-9)
    df["dist_to_pivot_low"]  = (close - pl)  / (close + 1e-9)

    df["hh"] = (high > high.rolling(20).max().shift()).astype(int)
    df["ll"] = (low  < low.rolling(20).min().shift()).astype(int)
    df["trend_consistency"] = (
        df["hh"].rolling(10).sum() - df["ll"].rolling(10).sum()
    ) / 10

    res20 = high.rolling(20).max().shift()
    sup20 = low.rolling(20).min().shift()
    df["breakout_up"]      = (close > res20).astype(int)
    df["breakdown"]        = (close < sup20).astype(int)
    df["dist_breakout_up"] = (close - res20)  / (close + 1e-9)
    df["dist_breakdown"]   = (sup20 - close)  / (close + 1e-9)
    return df


def build_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 50:
        return df
    df = df.copy()
    for fn, name in [
        (add_regime_features,   "regime"),
        (add_momentum_quality,  "momentum"),
        (add_gap_analysis,      "gaps"),
        (add_candle_patterns,   "candles"),
        (add_market_structure,  "structure"),
    ]:
        try:
            df = fn(df)
        except Exception as e:
            logger.warning(f"Advanced features [{name}]: {e}")
    df = df.replace([np.inf, -np.inf], np.nan)
    logger.debug(f"Advanced features built: {len(df.columns)} total columns")
    return df
