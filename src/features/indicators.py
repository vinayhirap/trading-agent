# trading-agent/src/features/indicators.py
"""
All technical indicators live here.
Every function takes a DataFrame (OHLCV) and returns a Series or DataFrame.
Pure functions — no side effects, no state.

Pro indicators added:
  - Ichimoku Cloud (trend + support/resistance)
  - VWAP with standard deviation bands (institutional levels)
  - Fair Value Gap / FVG detection (Smart Money Concepts)
  - Supertrend (trend-following with dynamic SL)
  - Heikin-Ashi candles (noise-filtered trend view)
  - CVD — Cumulative Volume Delta (buy vs sell pressure proxy)
  - Market Structure: Higher High / Lower Low / Break of Structure
  - Donchian Channel (breakout levels)
  - Chande Momentum Oscillator
  - Hull Moving Average (fast + smooth)
  - DEMA / TEMA (reduced lag EMAs)
  - Elder Ray Index (bull/bear power)
  - Pivot Points (Classic, Camarilla)
  - Live Volume: real-time volume rate vs average
  - Relative Volume (RVOL) per session
"""
import numpy as np
import pandas as pd
from loguru import logger


# ── Trend ────────────────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()

def wma(series: pd.Series, period: int) -> pd.Series:
    """Weighted Moving Average — more weight to recent bars."""
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def hma(series: pd.Series, period: int) -> pd.Series:
    """
    Hull Moving Average — fast and smooth, minimal lag.
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    half   = max(1, period // 2)
    sqrtn  = max(1, int(np.sqrt(period)))
    raw    = 2 * wma(series, half) - wma(series, period)
    return wma(raw, sqrtn).rename("hma")

def dema(series: pd.Series, period: int) -> pd.Series:
    """Double EMA — reduces EMA lag. DEMA = 2*EMA - EMA(EMA)."""
    e = ema(series, period)
    return (2 * e - ema(e, period)).rename("dema")

def tema(series: pd.Series, period: int) -> pd.Series:
    """Triple EMA — further lag reduction. TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))."""
    e  = ema(series, period)
    e2 = ema(e, period)
    e3 = ema(e2, period)
    return (3 * e - 3 * e2 + e3).rename("tema")

def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    ema_fast    = ema(close, fast)
    ema_slow    = ema(close, slow)
    macd_line   = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram   = macd_line - signal_line
    return pd.DataFrame({
        "macd":      macd_line,
        "signal":    signal_line,
        "histogram": histogram,
    }, index=close.index)

def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index — trend strength (not direction). Bug-fixed: no in-place mutation."""
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm  = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    # Keep only dominant directional move
    plus_dm  = plus_dm.where(plus_dm >= minus_dm, 0.0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)

    atr_val  = tr.ewm(span=period, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(span=period,  adjust=False).mean() / atr_val
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr_val
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val  = dx.ewm(span=period, adjust=False).mean()
    return pd.DataFrame({"adx": adx_val, "+di": plus_di, "-di": minus_di}, index=df.index)

def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Supertrend — dynamic trailing stop + trend direction.
    Returns: supertrend (price level), direction (1=up/-1=down), signal (buy/sell crossover).
    Widely used by pro traders as a trend filter and SL guide.
    """
    atr_v  = atr(df, period)
    hl2    = (df["high"] + df["low"]) / 2
    upper  = hl2 + multiplier * atr_v
    lower  = hl2 - multiplier * atr_v

    close     = df["close"].values
    n         = len(close)
    st        = np.full(n, np.nan)
    direction = np.ones(n)          # 1 = uptrend, -1 = downtrend
    final_up  = lower.values.copy()
    final_dn  = upper.values.copy()

    for i in range(1, n):
        if np.isnan(atr_v.iloc[i]):
            continue

        # Final lower band (support in uptrend)
        final_up[i] = max(lower.iloc[i], final_up[i-1]) if close[i-1] > final_up[i-1] else lower.iloc[i]
        # Final upper band (resistance in downtrend)
        final_dn[i] = min(upper.iloc[i], final_dn[i-1]) if close[i-1] < final_dn[i-1] else upper.iloc[i]

        if direction[i-1] == -1 and close[i] > final_dn[i-1]:
            direction[i] = 1
        elif direction[i-1] == 1 and close[i] < final_up[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]

        st[i] = final_up[i] if direction[i] == 1 else final_dn[i]

    dir_series = pd.Series(direction, index=df.index)
    return pd.DataFrame({
        "supertrend":     pd.Series(st, index=df.index),
        "supertrend_dir": dir_series,
        "supertrend_buy": ((dir_series == 1) & (dir_series.shift() == -1)).astype(int),
        "supertrend_sell":((dir_series == -1) & (dir_series.shift() == 1)).astype(int),
    })

def ichimoku(df: pd.DataFrame,
             tenkan: int = 9, kijun: int = 26,
             senkou_b: int = 52) -> pd.DataFrame:
    """
    Ichimoku Cloud — 5 lines covering trend, momentum, support/resistance.
    Used by institutional traders for multi-timeframe context.
    
    Returns:
      tenkan_sen   : conversion line (fast signal)
      kijun_sen    : base line (slow signal / SL proxy)
      senkou_a     : leading span A (cloud edge)
      senkou_b     : leading span B (cloud edge)
      chikou       : lagging span (confirmation)
      cloud_color  : 1=bullish (green), -1=bearish (red)
      above_cloud  : 1 if price above both cloud spans
    """
    def midpoint(series_high, series_low, period):
        return (series_high.rolling(period).max() + series_low.rolling(period).min()) / 2

    ten  = midpoint(df["high"], df["low"], tenkan)
    kij  = midpoint(df["high"], df["low"], kijun)
    sa   = ((ten + kij) / 2).shift(kijun)
    sb   = midpoint(df["high"], df["low"], senkou_b).shift(kijun)
    chik = df["close"].shift(-kijun)

    close = df["close"]
    cloud_bull = (sa > sb).astype(int)
    above      = ((close > sa) & (close > sb)).astype(int)

    return pd.DataFrame({
        "tenkan_sen":  ten,
        "kijun_sen":   kij,
        "senkou_a":    sa,
        "senkou_b":    sb,
        "chikou":      chik,
        "cloud_color": cloud_bull * 2 - 1,   # +1 bullish, -1 bearish
        "above_cloud": above,
        "tk_cross":    (ten > kij).astype(int),  # 1 = bullish TK cross
    })

def donchian_channel(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Donchian Channel — highest high / lowest low over N bars.
    Classic breakout indicator. Used in turtle trading.
    """
    upper = df["high"].rolling(period).max()
    lower = df["low"].rolling(period).min()
    mid   = (upper + lower) / 2
    return pd.DataFrame({
        "dc_upper": upper,
        "dc_mid":   mid,
        "dc_lower": lower,
        "dc_width": (upper - lower) / mid,
        "dc_pos":   (df["close"] - lower) / (upper - lower).replace(0, np.nan),
    }, index=df.index)


# ── Momentum / Oscillators ───────────────────────────────────────────────────

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).rename("rsi")

def stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    low_min  = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return pd.DataFrame({"stoch_k": k, "stoch_d": d}, index=df.index)

def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_max = df["high"].rolling(period).max()
    low_min  = df["low"].rolling(period).min()
    wr = -100 * (high_max - df["close"]) / (high_max - low_min).replace(0, np.nan)
    return wr.rename("williams_r")

def cmo(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Chande Momentum Oscillator — bounded momentum (-100 to +100).
    Better than ROC for detecting trend exhaustion.
    """
    delta   = close.diff()
    up_sum  = delta.clip(lower=0).rolling(period).sum()
    dn_sum  = (-delta).clip(lower=0).rolling(period).sum()
    return (100 * (up_sum - dn_sum) / (up_sum + dn_sum).replace(0, np.nan)).rename("cmo")

def elder_ray(df: pd.DataFrame, period: int = 13) -> pd.DataFrame:
    """
    Elder Ray Index — Bull Power and Bear Power.
    Bull Power = High - EMA(close, n)  → buying pressure above EMA
    Bear Power = Low  - EMA(close, n)  → selling pressure below EMA
    Used with trend filter: trade bull power dips in uptrend, bear power rallies in downtrend.
    """
    e           = ema(df["close"], period)
    bull_power  = df["high"] - e
    bear_power  = df["low"]  - e
    return pd.DataFrame({
        "bull_power": bull_power,
        "bear_power": bear_power,
    }, index=df.index)


# ── Volatility ───────────────────────────────────────────────────────────────

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean().rename("atr")

def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.DataFrame:
    mid    = sma(close, period)
    std    = close.rolling(period).std()
    upper  = mid + std_dev * std
    lower  = mid - std_dev * std
    pct_b  = (close - lower) / (upper - lower).replace(0, np.nan)
    bwidth = (upper - lower) / mid.replace(0, np.nan)
    return pd.DataFrame({
        "bb_upper": upper,
        "bb_mid":   mid,
        "bb_lower": lower,
        "bb_pct_b": pct_b,
        "bb_width": bwidth,
    }, index=close.index)

def keltner_channel(
    df: pd.DataFrame,
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> pd.DataFrame:
    mid   = ema(df["close"], ema_period)
    atr_v = atr(df, atr_period)
    return pd.DataFrame({
        "kc_upper": mid + multiplier * atr_v,
        "kc_mid":   mid,
        "kc_lower": mid - multiplier * atr_v,
    }, index=df.index)


# ── Volume / Order Flow ───────────────────────────────────────────────────────

def vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP resets each trading day. Requires DatetimeIndex — intraday only."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    df2           = df.copy()
    df2["_tp"]    = typical_price
    df2["_tpv"]   = typical_price * df["volume"]
    df2["_date"]  = df2.index.date
    cum_tpv = df2.groupby("_date")["_tpv"].cumsum()
    cum_vol = df2.groupby("_date")["volume"].cumsum()
    result  = (cum_tpv / cum_vol).rename("vwap")
    return result

def vwap_bands(df: pd.DataFrame, std_dev: float = 1.0) -> pd.DataFrame:
    """
    VWAP + standard deviation bands.
    Institutional traders use these as dynamic support/resistance.
    Price reverting to VWAP = mean reversion opportunity.
    Price holding above 1σ VWAP = strong trend.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3
    df2     = df.copy()
    df2["_tp"]   = typical
    df2["_tpv"]  = typical * df["volume"]
    df2["_tp2v"] = typical**2 * df["volume"]
    df2["_date"] = df2.index.date

    cum_tpv  = df2.groupby("_date")["_tpv"].cumsum()
    cum_vol  = df2.groupby("_date")["volume"].cumsum()
    cum_tp2v = df2.groupby("_date")["_tp2v"].cumsum()

    vwap_val = cum_tpv / cum_vol
    variance = (cum_tp2v / cum_vol) - vwap_val**2
    std      = variance.clip(lower=0).pow(0.5)

    return pd.DataFrame({
        "vwap":      vwap_val,
        "vwap_u1":   vwap_val + std_dev * std,
        "vwap_l1":   vwap_val - std_dev * std,
        "vwap_u2":   vwap_val + 2 * std_dev * std,
        "vwap_l2":   vwap_val - 2 * std_dev * std,
        "vwap_dist": (df["close"] - vwap_val) / vwap_val.replace(0, np.nan),
    }, index=df.index)

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume — cumulative volume based on price direction."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum().rename("obv")

def cvd(df: pd.DataFrame) -> pd.Series:
    """
    Cumulative Volume Delta — proxy for buy vs sell pressure.
    Uses candle body position to estimate aggressive buying/selling.
    Positive CVD diverging from price = hidden buying (bullish).
    Negative CVD diverging from price = hidden selling (bearish).
    
    Formula: delta per bar = volume * ((close-low)-(high-close)) / (high-low)
    This estimates net buying pressure per bar.
    """
    hl      = (df["high"] - df["low"]).replace(0, np.nan)
    buy_vol = df["volume"] * (df["close"] - df["low"])  / hl
    sel_vol = df["volume"] * (df["high"]  - df["close"]) / hl
    delta   = (buy_vol - sel_vol).fillna(0)
    return delta.cumsum().rename("cvd")

def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index — volume-weighted RSI."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    raw_mf  = typical * df["volume"]
    pos_mf  = raw_mf.where(typical > typical.shift(), 0)
    neg_mf  = raw_mf.where(typical < typical.shift(), 0)
    mfr     = pos_mf.rolling(period).sum() / neg_mf.rolling(period).sum().replace(0, np.nan)
    return (100 - 100 / (1 + mfr)).rename("mfi")

def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Chaikin Money Flow — fixed: rename applied to final result."""
    hl  = (df["high"] - df["low"]).replace(0, np.nan)
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl
    result = (clv * df["volume"]).rolling(period).sum() / df["volume"].rolling(period).sum()
    return result.rename("cmf")

def live_volume_rate(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Live Volume Analysis:
      rvol        : relative volume vs rolling average (>1.5 = significant)
      vol_thrust  : volume spike detection (>2σ above mean)
      vol_trend   : volume EMA slope (increasing/decreasing participation)
      session_rvol: relative volume vs same-hour average (intraday only)
    """
    vol      = df["volume"]
    vol_avg  = vol.rolling(lookback).mean()
    vol_std  = vol.rolling(lookback).std()
    rvol     = vol / vol_avg.replace(0, np.nan)
    thrust   = ((vol - vol_avg) / vol_std.replace(0, np.nan)).fillna(0)
    vol_ema  = ema(vol, 10)
    vol_trend = (vol_ema - vol_ema.shift(5)) / vol_ema.shift(5).replace(0, np.nan)

    result = pd.DataFrame({
        "rvol":       rvol,
        "vol_thrust": thrust,
        "vol_trend":  vol_trend,
    }, index=df.index)

    # Session RVOL: compare to same hour (only works with DatetimeIndex)
    try:
        hour_avg = vol.groupby(df.index.hour).transform("mean")
        result["session_rvol"] = vol / hour_avg.replace(0, np.nan)
    except Exception:
        result["session_rvol"] = rvol

    return result


# ── Smart Money Concepts ─────────────────────────────────────────────────────

def fair_value_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fair Value Gap (FVG) / Imbalance detection.
    Smart Money Concept used by institutional traders.
    
    Bullish FVG: candle[i-2].high < candle[i].low  (gap up, unfilled demand zone)
    Bearish FVG: candle[i-2].low  > candle[i].high (gap down, unfilled supply zone)
    
    Returns:
      fvg_bull     : 1 where bullish FVG exists
      fvg_bear     : -1 where bearish FVG exists
      fvg_bull_top : top of bullish FVG (resistance when revisited)
      fvg_bull_bot : bottom of bullish FVG
      fvg_bear_top : top of bearish FVG
      fvg_bear_bot : bottom of bearish FVG
      fvg_fill_bull: 1 when price retraces into a bullish FVG (entry zone)
      fvg_fill_bear: 1 when price retraces into a bearish FVG (entry zone)
    """
    n         = len(df)
    fvg_bull  = np.zeros(n)
    fvg_bear  = np.zeros(n)
    bull_top  = np.full(n, np.nan)
    bull_bot  = np.full(n, np.nan)
    bear_top  = np.full(n, np.nan)
    bear_bot  = np.full(n, np.nan)
    fill_bull = np.zeros(n)
    fill_bear = np.zeros(n)

    high = df["high"].values
    low  = df["low"].values
    close= df["close"].values

    # Active FVG zones (carry forward)
    active_bull: list[tuple[float, float]] = []  # (top, bot)
    active_bear: list[tuple[float, float]] = []  # (top, bot)

    for i in range(2, n):
        # Detect new FVGs
        if high[i-2] < low[i]:           # bullish gap
            fvg_bull[i] = 1
            bull_top[i] = low[i]
            bull_bot[i] = high[i-2]
            active_bull.append((low[i], high[i-2]))

        if low[i-2] > high[i]:           # bearish gap
            fvg_bear[i] = -1
            bear_top[i] = low[i-2]
            bear_bot[i] = high[i]
            active_bear.append((low[i-2], high[i]))

        # Check fills: price retraces into existing FVG
        remaining_bull = []
        for (top, bot) in active_bull:
            if bot <= close[i] <= top:
                fill_bull[i] = 1         # price in bullish FVG = demand zone entry
            elif close[i] < bot:
                pass                     # fully filled/violated — remove
            else:
                remaining_bull.append((top, bot))
        active_bull = remaining_bull

        remaining_bear = []
        for (top, bot) in active_bear:
            if bot <= close[i] <= top:
                fill_bear[i] = 1         # price in bearish FVG = supply zone entry
            elif close[i] > top:
                pass                     # violated — remove
            else:
                remaining_bear.append((top, bot))
        active_bear = remaining_bear

    idx = df.index
    return pd.DataFrame({
        "fvg_bull":      pd.Series(fvg_bull,  index=idx),
        "fvg_bear":      pd.Series(fvg_bear,  index=idx),
        "fvg_bull_top":  pd.Series(bull_top,  index=idx),
        "fvg_bull_bot":  pd.Series(bull_bot,  index=idx),
        "fvg_bear_top":  pd.Series(bear_top,  index=idx),
        "fvg_bear_bot":  pd.Series(bear_bot,  index=idx),
        "fvg_fill_bull": pd.Series(fill_bull, index=idx),
        "fvg_fill_bear": pd.Series(fill_bear, index=idx),
    })

def market_structure(df: pd.DataFrame, swing_period: int = 5) -> pd.DataFrame:
    """
    Market Structure Analysis — Higher High (HH), Lower Low (LL), Break of Structure (BOS).
    Core of Smart Money Concepts. Used to identify trend continuation vs reversal.
    
    Returns:
      swing_high   : 1 at pivot highs
      swing_low    : 1 at pivot lows
      hh           : 1 where new higher high (uptrend confirmation)
      ll           : 1 where new lower low (downtrend confirmation)
      hl           : 1 where higher low (uptrend pullback, buy zone)
      lh           : 1 where lower high (downtrend pullback, sell zone)
      bos_bull     : 1 at bullish break of structure (strong buy signal)
      bos_bear     : 1 at bearish break of structure (strong sell signal)
      choch        : 1 at change of character (potential trend reversal)
    """
    n    = len(df)
    high = df["high"].values
    low  = df["low"].values
    close= df["close"].values
    sp   = swing_period

    sh = np.zeros(n)  # swing highs
    sl = np.zeros(n)  # swing lows

    for i in range(sp, n - sp):
        if high[i] == max(high[i-sp:i+sp+1]):
            sh[i] = 1
        if low[i] == min(low[i-sp:i+sp+1]):
            sl[i] = 1

    # Track HH/LL/HL/LH
    hh = np.zeros(n)
    ll = np.zeros(n)
    hl = np.zeros(n)
    lh = np.zeros(n)

    prev_sh = np.nan
    prev_sl = np.nan

    for i in range(n):
        if sh[i]:
            if not np.isnan(prev_sh):
                if high[i] > prev_sh:
                    hh[i] = 1
                else:
                    lh[i] = 1
            prev_sh = high[i]
        if sl[i]:
            if not np.isnan(prev_sl):
                if low[i] < prev_sl:
                    ll[i] = 1
                else:
                    hl[i] = 1
            prev_sl = low[i]

    # Break of Structure: price closes above last swing high (bull BOS) or below last swing low (bear BOS)
    bos_bull = np.zeros(n)
    bos_bear = np.zeros(n)
    choch    = np.zeros(n)

    last_sh_price = np.nan
    last_sl_price = np.nan
    trend = 0  # 1 = up, -1 = down

    for i in range(1, n):
        if sh[i-1]:
            last_sh_price = high[i-1]
        if sl[i-1]:
            last_sl_price = low[i-1]

        if not np.isnan(last_sh_price) and close[i] > last_sh_price:
            bos_bull[i] = 1
            if trend == -1:
                choch[i] = 1   # was downtrend, now breaking up = Change of Character
            trend = 1

        if not np.isnan(last_sl_price) and close[i] < last_sl_price:
            bos_bear[i] = 1
            if trend == 1:
                choch[i] = 1   # was uptrend, now breaking down = Change of Character
            trend = -1

    idx = df.index
    return pd.DataFrame({
        "swing_high": pd.Series(sh,       index=idx),
        "swing_low":  pd.Series(sl,       index=idx),
        "hh":         pd.Series(hh,       index=idx),
        "ll":         pd.Series(ll,       index=idx),
        "hl":         pd.Series(hl,       index=idx),
        "lh":         pd.Series(lh,       index=idx),
        "bos_bull":   pd.Series(bos_bull, index=idx),
        "bos_bear":   pd.Series(bos_bear, index=idx),
        "choch":      pd.Series(choch,    index=idx),
    })

def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Heikin-Ashi candles — noise-filtered trend view.
    Consecutive same-color HA candles = strong trend.
    HA candle with no lower wick = strong uptrend. No upper wick = strong downtrend.
    """
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_open  = pd.Series(index=df.index, dtype=float)
    ha_open.iloc[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
    ha_high  = pd.concat([df["high"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low   = pd.concat([df["low"],  ha_open, ha_close], axis=1).min(axis=1)
    ha_bull  = (ha_close > ha_open).astype(int)
    ha_doji  = (abs(ha_close - ha_open) < (ha_high - ha_low) * 0.1).astype(int)
    return pd.DataFrame({
        "ha_open":  ha_open,
        "ha_high":  ha_high,
        "ha_low":   ha_low,
        "ha_close": ha_close,
        "ha_bull":  ha_bull,
        "ha_doji":  ha_doji,
    }, index=df.index)


# ── Support / Resistance ──────────────────────────────────────────────────────

def pivot_points(df: pd.DataFrame, method: str = "classic") -> pd.DataFrame:
    """
    Pivot Points — key S/R levels for next session.
    Uses prior bar's H/L/C. Most useful on daily/weekly.
    
    method: 'classic' or 'camarilla'
    Camarilla pivots are tighter, used for intraday mean-reversion.
    """
    h = df["high"].shift(1)
    l = df["low"].shift(1)
    c = df["close"].shift(1)

    if method == "camarilla":
        rng = h - l
        return pd.DataFrame({
            "pp":  (h + l + c) / 3,
            "r4":  c + rng * 1.1 / 2,
            "r3":  c + rng * 1.1 / 4,
            "r2":  c + rng * 1.1 / 6,
            "r1":  c + rng * 1.1 / 12,
            "s1":  c - rng * 1.1 / 12,
            "s2":  c - rng * 1.1 / 6,
            "s3":  c - rng * 1.1 / 4,
            "s4":  c - rng * 1.1 / 2,
        }, index=df.index)
    else:
        pp = (h + l + c) / 3
        return pd.DataFrame({
            "pp": pp,
            "r1": 2 * pp - l,
            "r2": pp + (h - l),
            "r3": h + 2 * (pp - l),
            "s1": 2 * pp - h,
            "s2": pp - (h - l),
            "s3": l - 2 * (h - pp),
        }, index=df.index)