# trading-agent/src/features/asset_features.py
"""
Asset-Class Specific Feature Extensions

Extends FeatureEngine with features that are ONLY meaningful
for specific asset classes. Never modifies FeatureEngine directly.

Design:
  - AssetFeatureEngine wraps FeatureEngine
  - Call build_for_asset(df, symbol) instead of fe.build(df)
  - Returns same structure as FeatureEngine.build() — fully compatible
  - Each asset class adds ~5-15 additional features on top of the base 42

Asset-specific logic:

EQUITY:
  - Sector relative strength (stock vs Nifty50)
  - Earnings proximity flag (avoid holding 2 days before results)
  - Delivery % proxy (high volume + low change = accumulation)
  - Beta to Nifty (rolling 60-day)

INDEX:
  - Global market correlation (S&P500 overnight return)
  - FII/DII proxy (Nifty futures premium)
  - VIX-like measure (Nifty ATR ratio)
  - Options PCR proxy from price momentum

COMMODITY (MCX):
  - USD/INR impact (commodity prices in USD, MCX prices in INR)
  - WTI/Brent overnight for CrudeOil
  - Gold: DXY proxy (inverse of USDINR return)
  - Silver: Gold-Silver ratio
  - Crude: inventory proxy (weekly pattern)
  - Seasonality features (month-of-year encoding)

CRYPTO:
  - BTC dominance proxy (BTC vs ETH correlation)
  - Fear & Greed proxy (rolling volatility rank)
  - Weekend effect (crypto trades 24/7 — weekend = lower liquidity)
  - BTC as benchmark for all other crypto
  - 24h return (crypto moves much more in 24h than equity)
  - Short-term momentum (1-3 bar returns more predictive than 5-10)

FOREX:
  - Interest rate differential proxy (rolling returns)
  - Risk-on/risk-off (USDINR as safe haven indicator)
  - Carry trade signal

Label horizons per asset class (CRITICAL):
  EQUITY:    5 bars (1 week) — institutional holding cycle
  INDEX:     3 bars           — faster mean reversion
  COMMODITY: 7 bars           — supply/demand cycles are slower
  CRYPTO:    2 bars           — high volatility, shorter predictability
  FOREX:     5 bars           — macro-driven, moderate speed
"""
import numpy as np
import pandas as pd
from loguru import logger
from typing import Optional

from src.data.models import AssetClass, ALL_SYMBOLS
from src.features.feature_engine import FeatureEngine


# ── Label horizons per asset class ───────────────────────────────────────────
ASSET_LABEL_HORIZONS = {
    AssetClass.EQUITY:  5,
    AssetClass.INDEX:   3,
    AssetClass.FUTURES: 7,    # commodities
    AssetClass.CRYPTO:  2,
    AssetClass.OPTIONS: 3,
}

# ── Training data requirements per asset class ────────────────────────────────
ASSET_MIN_BARS = {
    AssetClass.EQUITY:  200,
    AssetClass.INDEX:   200,
    AssetClass.FUTURES: 150,
    AssetClass.CRYPTO:  120,   # less history available
    AssetClass.OPTIONS: 100,
}

# ── Feature importance hints per asset class (for model weighting) ────────────
# These are passed to XGBoost as feature_weights
# Higher = model should pay more attention to this feature group
ASSET_FEATURE_WEIGHTS = {
    AssetClass.EQUITY: {
        "momentum":   1.2,   # RSI, MACD matter most for equities
        "trend":      1.1,
        "volume":     1.3,   # delivery volume is KEY for Indian equity
        "volatility": 0.9,
        "crossasset": 0.8,
    },
    AssetClass.INDEX: {
        "momentum":   1.0,
        "trend":      1.2,
        "volume":     0.7,   # index volume less meaningful
        "volatility": 1.1,
        "crossasset": 1.4,   # global correlation matters most for index
    },
    AssetClass.FUTURES: {
        "momentum":   0.9,
        "trend":      1.0,
        "volume":     1.1,
        "volatility": 1.4,   # commodities are volatility-driven
        "crossasset": 1.5,   # USD, oil, gold correlation critical
    },
    AssetClass.CRYPTO: {
        "momentum":   1.5,   # crypto is almost pure momentum
        "trend":      1.3,
        "volume":     1.4,
        "volatility": 1.2,
        "crossasset": 0.8,
    },
}


class AssetFeatureEngine:
    """
    Asset-class aware feature builder.

    Wraps FeatureEngine and adds asset-specific features on top.
    Fully backward compatible — returns same DataFrame structure.

    Usage:
        afe = AssetFeatureEngine()
        df_features = afe.build_for_asset(df, symbol="RELIANCE")
        # or explicitly:
        df_features = afe.build_for_asset(df, asset_class=AssetClass.EQUITY)
    """

    def __init__(self, interval: str = "1d"):
        self._base_engine = FeatureEngine(interval=interval)
        self.interval     = interval

    def build_for_asset(
        self,
        df:          pd.DataFrame,
        symbol:      str = None,
        asset_class: AssetClass = None,
        drop_na:     bool = True,
        dm=None,     # optional DataManager for cross-asset data
    ) -> pd.DataFrame:
        """
        Build full feature set for a specific asset class.

        Priority: symbol → lookup asset class → build accordingly
        Falls back to base FeatureEngine if asset class unknown.
        """
        # Resolve asset class
        ac = asset_class
        if ac is None and symbol:
            ac = self._resolve_asset_class(symbol)
        if ac is None:
            logger.warning(f"Unknown asset class for {symbol} — using base features only")
            return self._base_engine.build(df, drop_na=drop_na)

        # Step 1: base features (42 indicators)
        out = self._base_engine.build(df, drop_na=False)
        if out.empty:
            return out

        # Step 2: asset-specific extensions
        try:
            if ac == AssetClass.EQUITY:
                out = self._add_equity_features(out, symbol, dm)
            elif ac == AssetClass.INDEX:
                out = self._add_index_features(out, symbol, dm)
            elif ac == AssetClass.FUTURES:
                out = self._add_commodity_features(out, symbol, dm)
            elif ac == AssetClass.CRYPTO:
                out = self._add_crypto_features(out, symbol, dm)
            elif ac == AssetClass.OPTIONS:
                out = self._add_index_features(out, symbol, dm)  # treat like index
        except Exception as e:
            logger.warning(f"Asset feature extension failed for {symbol} ({ac}): {e} — using base only")

        # Step 3: universal cross-asset features
        try:
            out = self._add_universal_features(out, symbol, ac)
        except Exception as e:
            logger.warning(f"Universal features failed: {e}")

        # Step 4: asset class label (integer, used as model feature)
        ac_map = {
            AssetClass.EQUITY:  0,
            AssetClass.INDEX:   1,
            AssetClass.FUTURES: 2,
            AssetClass.CRYPTO:  3,
            AssetClass.OPTIONS: 4,
        }
        out["asset_class"] = ac_map.get(ac, 0)

        if drop_na:
            before = len(out)
            out = out.dropna()
            logger.info(
                f"AssetFeatureEngine [{symbol}/{ac.value}]: "
                f"{before} → {len(out)} bars after NaN drop"
            )
        return out

    def get_label_horizon(self, symbol: str = None, asset_class: AssetClass = None) -> int:
        """Return the correct label horizon for this asset class."""
        ac = asset_class or self._resolve_asset_class(symbol)
        return ASSET_LABEL_HORIZONS.get(ac, 5)

    def get_min_bars(self, symbol: str = None, asset_class: AssetClass = None) -> int:
        ac = asset_class or self._resolve_asset_class(symbol)
        return ASSET_MIN_BARS.get(ac, 200)

    # ── Equity-specific features ──────────────────────────────────────────────

    def _add_equity_features(self, df: pd.DataFrame, symbol: str, dm) -> pd.DataFrame:
        """
        Equity-specific: sector relative strength, beta, delivery proxy.
        """
        out = df.copy()

        # ── Relative strength vs Nifty50 (stock alpha) ────────────────────
        nifty_ret = self._fetch_returns("NIFTY50", len(df), dm)
        if nifty_ret is not None and len(nifty_ret) == len(df):
            stock_ret  = df["close"].pct_change()
            out["rel_strength_nifty"] = stock_ret - nifty_ret
            out["rel_strength_5d"]    = out["rel_strength_nifty"].rolling(5).sum()
            out["rel_strength_20d"]   = out["rel_strength_nifty"].rolling(20).sum()

            # Beta to Nifty (rolling 60-day)
            rolling_cov = stock_ret.rolling(60).cov(nifty_ret)
            rolling_var = nifty_ret.rolling(60).var()
            out["beta_nifty"] = (rolling_cov / rolling_var.replace(0, np.nan)).clip(-3, 3)
        else:
            out["rel_strength_nifty"] = 0.0
            out["rel_strength_5d"]    = 0.0
            out["rel_strength_20d"]   = 0.0
            out["beta_nifty"]         = 1.0

        # ── Accumulation/Distribution proxy ──────────────────────────────
        # High volume + small body = institutional accumulation
        body_size   = (df["close"] - df["open"]).abs() / (df["high"] - df["low"] + 1e-9)
        vol_ratio   = df["volume"] / df["volume"].rolling(20).mean()
        out["accum_dist_proxy"] = vol_ratio * (1 - body_size)

        # ── Price vs 52-week high/low ─────────────────────────────────────
        out["pct_from_52w_high"] = df["close"] / df["close"].rolling(252).max() - 1
        out["pct_from_52w_low"]  = df["close"] / df["close"].rolling(252).min() - 1

        # ── Day-of-week (markets behave differently Mon vs Fri) ───────────
        if hasattr(df.index, "dayofweek"):
            out["day_of_week_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 5)
            out["day_of_week_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 5)

        return out

    # ── Index-specific features ───────────────────────────────────────────────

    def _add_index_features(self, df: pd.DataFrame, symbol: str, dm) -> pd.DataFrame:
        """
        Index-specific: global correlation, VIX proxy, overnight gaps.
        """
        out = df.copy()

        # ── S&P 500 overnight return (global risk sentiment) ──────────────
        sp500_ret = self._fetch_returns("^GSPC", len(df), dm)
        if sp500_ret is not None and len(sp500_ret) == len(df):
            out["sp500_ret_1d"]    = sp500_ret
            out["sp500_ret_5d"]    = sp500_ret.rolling(5).sum()
            # Correlation: how much does this index follow S&P?
            nifty_ret = df["close"].pct_change()
            out["sp500_correlation_20d"] = nifty_ret.rolling(20).corr(sp500_ret)
        else:
            out["sp500_ret_1d"]          = 0.0
            out["sp500_ret_5d"]          = 0.0
            out["sp500_correlation_20d"] = 0.5

        # ── VIX-like measure (realised volatility rank) ───────────────────
        daily_ret    = df["close"].pct_change()
        realized_vol = daily_ret.rolling(20).std() * np.sqrt(252)
        vol_52w_high = realized_vol.rolling(252).max()
        vol_52w_low  = realized_vol.rolling(252).min()
        denom        = (vol_52w_high - vol_52w_low).replace(0, np.nan)
        out["vix_proxy_rank"] = ((realized_vol - vol_52w_low) / denom).clip(0, 1)

        # ── Overnight gap (open vs previous close) ────────────────────────
        out["overnight_gap"] = (df["open"] / df["close"].shift(1) - 1)
        out["gap_filled"]    = (
            ((out["overnight_gap"] > 0) & (df["close"] < df["open"])) |
            ((out["overnight_gap"] < 0) & (df["close"] > df["open"]))
        ).astype(int)

        # ── FII activity proxy (index futures premium) ────────────────────
        # When FIIs buy, futures trade at premium to spot
        # Approximate: if close > previous day VWAP, FII buying
        out["fii_proxy"] = (df["close"] > df["close"].shift(1)).astype(int).rolling(5).mean()

        # ── Month-end rebalancing effect (last 3 trading days of month) ───
        if hasattr(df.index, "is_month_end"):
            out["month_end_flag"] = df.index.is_month_end.astype(int)
        else:
            out["month_end_flag"] = 0

        return out

    # ── Commodity-specific features ───────────────────────────────────────────

    def _add_commodity_features(self, df: pd.DataFrame, symbol: str, dm) -> pd.DataFrame:
        """
        Commodity-specific: USD/INR impact, cross-commodity correlation.
        """
        out = df.copy()

        # ── USDINR impact (all commodities priced in USD → MCX in INR) ───
        usdinr_ret = self._fetch_returns("USDINR", len(df), dm)
        if usdinr_ret is not None and len(usdinr_ret) == len(df):
            out["usdinr_ret_1d"] = usdinr_ret
            out["usdinr_ret_5d"] = usdinr_ret.rolling(5).sum()
            # USD strength REDUCES commodity prices in USD terms
            # but INCREASES MCX prices in INR terms — this matters
            out["fx_impact"]     = usdinr_ret  # positive = INR weaker = MCX higher
        else:
            out["usdinr_ret_1d"] = 0.0
            out["usdinr_ret_5d"] = 0.0
            out["fx_impact"]     = 0.0

        # ── Symbol-specific cross-asset features ─────────────────────────
        if symbol == "CRUDEOIL":
            # WTI/Brent proxy: use self price as WTI proxy (MCX tracks it)
            out["crude_ma_ratio"] = df["close"] / df["close"].rolling(50).mean()
            # Weekly inventory cycle: Tuesday release — approximate with day pattern
            if hasattr(df.index, "dayofweek"):
                out["crude_weekday_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
                out["crude_weekday_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

        elif symbol == "GOLD":
            # Gold vs USDINR (inverse relationship)
            if usdinr_ret is not None and len(usdinr_ret) == len(df):
                gold_ret   = df["close"].pct_change()
                out["gold_usd_decorr"] = gold_ret.rolling(20).corr(-usdinr_ret)
            # Gold as safe-haven: spikes when equity falls
            nifty_ret = self._fetch_returns("NIFTY50", len(df), dm)
            if nifty_ret is not None and len(nifty_ret) == len(df):
                gold_ret  = df["close"].pct_change()
                out["gold_nifty_corr_20d"] = gold_ret.rolling(20).corr(nifty_ret)
                # Negative correlation = safe-haven behavior
                out["safe_haven_flag"] = (out["gold_nifty_corr_20d"] < -0.2).astype(int)

        elif symbol == "SILVER":
            # Silver tracks gold but also has industrial demand
            gold_ret = self._fetch_returns("GOLD", len(df), dm)
            if gold_ret is not None and len(gold_ret) == len(df):
                silver_ret = df["close"].pct_change()
                out["gold_silver_corr"] = silver_ret.rolling(20).corr(gold_ret)
                # Gold-Silver ratio proxy (normalised)
                out["gs_ratio_proxy"]   = df["close"].rolling(20).mean() / df["close"].rolling(60).mean()

        # ── Seasonality (month-of-year — commodities have strong seasonality) ──
        if hasattr(df.index, "month"):
            out["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
            out["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

        # ── Contango/backwardation proxy (slope of rolling returns) ──────
        out["term_structure_proxy"] = df["close"].pct_change(30) - df["close"].pct_change(5)

        return out

    # ── Crypto-specific features ──────────────────────────────────────────────

    def _add_crypto_features(self, df: pd.DataFrame, symbol: str, dm) -> pd.DataFrame:
        """
        Crypto-specific: BTC benchmark, fear/greed proxy, 24h dynamics.
        """
        out = df.copy()

        # ── BTC as benchmark for all crypto ───────────────────────────────
        if symbol != "BTC":
            btc_ret = self._fetch_returns("BTC", len(df), dm)
            if btc_ret is not None and len(btc_ret) == len(df):
                coin_ret  = df["close"].pct_change()
                out["btc_ret_1d"]       = btc_ret
                out["btc_correlation"]  = coin_ret.rolling(14).corr(btc_ret)
                # Alpha vs BTC: positive = outperforming BTC
                out["btc_alpha"]        = coin_ret - btc_ret
                out["btc_alpha_7d"]     = out["btc_alpha"].rolling(7).sum()
                # BTC dominance proxy: when BTC rises faster, altcoins lag
                out["btc_dominance_proxy"] = btc_ret.rolling(7).sum() - coin_ret.rolling(7).sum()
            else:
                out["btc_ret_1d"]          = 0.0
                out["btc_correlation"]     = 1.0
                out["btc_alpha"]           = 0.0
                out["btc_alpha_7d"]        = 0.0
                out["btc_dominance_proxy"] = 0.0
        else:
            # For BTC itself: ETH correlation
            eth_ret = self._fetch_returns("ETH", len(df), dm)
            if eth_ret is not None and len(eth_ret) == len(df):
                btc_ret = df["close"].pct_change()
                out["eth_correlation"] = btc_ret.rolling(14).corr(eth_ret)
            else:
                out["eth_correlation"] = 0.8

        # ── Fear & Greed proxy (rolling volatility rank) ──────────────────
        daily_ret    = df["close"].pct_change()
        vol_7d       = daily_ret.rolling(7).std()
        vol_90d_max  = vol_7d.rolling(90).max()
        vol_90d_min  = vol_7d.rolling(90).min()
        denom        = (vol_90d_max - vol_90d_min).replace(0, np.nan)
        # High = fear (high vol), Low = greed (low vol, complacency)
        out["fear_greed_proxy"] = 1 - ((vol_7d - vol_90d_min) / denom).clip(0, 1)

        # ── Weekend effect (lower liquidity Sat/Sun → mean reversion Mon) ─
        if hasattr(df.index, "dayofweek"):
            out["is_monday"]    = (df.index.dayofweek == 0).astype(int)
            out["is_friday"]    = (df.index.dayofweek == 4).astype(int)
            out["day_sin"]      = np.sin(2 * np.pi * df.index.dayofweek / 7)
            out["day_cos"]      = np.cos(2 * np.pi * df.index.dayofweek / 7)

        # ── Shorter-horizon returns (crypto moves fast) ───────────────────
        # Override the standard 5/10-day returns with 1/2/3 day
        out["ret_2d_crypto"] = df["close"].pct_change(2)
        out["ret_3d_crypto"] = df["close"].pct_change(3)

        # ── Extreme return flag (crypto has fat tails) ────────────────────
        ret_std = daily_ret.rolling(30).std()
        out["extreme_move"]  = (daily_ret.abs() > 3 * ret_std).astype(int)
        out["extreme_up"]    = ((daily_ret > 3 * ret_std)).astype(int)
        out["extreme_down"]  = ((daily_ret < -3 * ret_std)).astype(int)

        return out

    # ── Universal features (added to ALL asset classes) ───────────────────────

    def _add_universal_features(
        self, df: pd.DataFrame, symbol: str, ac: AssetClass
    ) -> pd.DataFrame:
        """
        Features that improve prediction for every asset class.
        """
        out = df.copy()

        # ── Trend age (how many bars since last EMA crossover) ────────────
        if "ema9_pct" in out.columns and "ema50_pct" in out.columns:
            trending_up   = (out["ema9_pct"] > 0).astype(int)
            out["trend_age"] = trending_up.groupby(
                (trending_up != trending_up.shift()).cumsum()
            ).cumcount()
            out["trend_age"] = out["trend_age"] / 20   # normalise

        # ── Volatility regime (current vs 1-year) ────────────────────────
        daily_ret    = df["close"].pct_change()
        vol_20d      = daily_ret.rolling(20).std()
        vol_252d_avg = daily_ret.rolling(252).std()
        out["vol_regime"] = (vol_20d / vol_252d_avg.replace(0, np.nan)).clip(0, 3)

        # ── Price momentum quintile (where in distribution is this return) ─
        ret_5d = df["close"].pct_change(5)
        out["momentum_quintile"] = ret_5d.rolling(252).rank(pct=True).fillna(0.5)

        # ── Gap between high and low (intraday range normalised) ──────────
        out["hl_range_pct"]    = (df["high"] - df["low"]) / df["close"]
        out["hl_range_z"]      = (
            (out["hl_range_pct"] - out["hl_range_pct"].rolling(20).mean()) /
            (out["hl_range_pct"].rolling(20).std() + 1e-9)
        ).clip(-3, 3)

        return out

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _resolve_asset_class(self, symbol: str) -> Optional[AssetClass]:
        """Resolve symbol → AssetClass using ALL_SYMBOLS."""
        info = ALL_SYMBOLS.get(symbol)
        if info:
            return info.asset_class
        # Fallback heuristics
        if symbol in ("BTC", "ETH", "SOL", "MATIC", "BNB", "XRP", "ADA", "DOGE"):
            return AssetClass.CRYPTO
        if symbol in ("GOLD", "SILVER", "CRUDEOIL", "COPPER", "NATURALGAS"):
            return AssetClass.FUTURES
        if symbol in ("NIFTY50", "BANKNIFTY", "SENSEX"):
            return AssetClass.INDEX
        if symbol.endswith(("CE", "PE")):
            return AssetClass.OPTIONS
        return AssetClass.EQUITY

    def _fetch_returns(
        self, symbol: str, n_bars: int, dm
    ) -> Optional[pd.Series]:
        """
        Fetch return series for a cross-asset symbol.
        Uses DataManager if available, falls back to yfinance.
        Returns pd.Series of pct_change, aligned to n_bars length.
        """
        try:
            if dm is not None:
                from src.data.models import Interval
                df = dm.get_ohlcv(symbol, Interval.D1, days_back=max(n_bars + 60, 300))
            else:
                # Direct yfinance fallback
                import yfinance as yf
                info   = ALL_SYMBOLS.get(symbol)
                yf_sym = info.symbol if info else f"{symbol}.NS"
                raw    = yf.download(yf_sym, period="2y", interval="1d",
                                     progress=False, auto_adjust=True)
                if raw.empty:
                    return None
                raw.columns = [c.lower() for c in raw.columns]
                df = raw

            if df is None or df.empty:
                return None

            ret = df["close"].pct_change()
            # Preserve the original index so cross-asset rolling correlations align correctly
            aligned = ret.iloc[-n_bars:]
            aligned.index = df.index[-len(aligned):]
            return aligned

        except Exception as e:
            logger.debug(f"_fetch_returns({symbol}): {e}")
            return None