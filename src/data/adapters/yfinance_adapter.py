#   trading-agent/src/data/adapters/yfinance_adapter.py
from datetime import datetime
import pandas as pd
import yfinance as yf
from loguru import logger

from src.data.base_adapter import BaseDataAdapter
from src.data.models import Interval, NSE_SYMBOLS, CRYPTO_SYMBOLS

# yfinance uses its own interval strings — map ours to theirs
INTERVAL_MAP = {
    Interval.M1:  "1m",
    Interval.M5:  "5m",
    Interval.M15: "15m",
    Interval.M30: "30m",
    Interval.H1:  "1h",
    Interval.D1:  "1d",
    Interval.W1:  "1wk",
}

# yfinance limits on intraday history (days back)
INTRADAY_LIMITS = {
    "1m": 7, "5m": 60, "15m": 60, "30m": 60, "1h": 730,
}


class YFinanceAdapter(BaseDataAdapter):
    """
    Free historical data via yfinance.
    Use this for:
    - Development and backtesting
    - NSE equities (suffix .NS), indices (^NSEI)
    - Crypto (BTC-USD, ETH-USD)

    Limitations:
    - 15-minute delay on live prices
    - Intraday data limited (see INTRADAY_LIMITS)
    - Not suitable for live trading
    """

    def __init__(self):
        super().__init__("yfinance")

    def connect(self) -> bool:
        # yfinance is stateless — no auth needed
        self._connected = True
        logger.info("yfinance adapter ready (no auth required)")
        return True

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: Interval,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        symbol: use yfinance format e.g. "RELIANCE.NS", "^NSEI", "BTC-USD"
                OR use our short names e.g. "RELIANCE", "NIFTY50", "BTC"
        """
        # Resolve short names → yfinance tickers
        yf_symbol = self._resolve_symbol(symbol)

        yf_interval = INTERVAL_MAP.get(interval)
        if yf_interval is None:
            raise ValueError(f"Unsupported interval: {interval}")

        logger.info(f"Fetching {yf_symbol} | {yf_interval} | {start.date()} → {end.date()}")

        try:
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval=yf_interval,
                auto_adjust=True,    # adjusts for splits/dividends
                back_adjust=False,
                prepost=False,
            )
        except Exception as e:
            logger.error(f"yfinance fetch failed for {yf_symbol}: {e}")
            return pd.DataFrame()

        if df.empty:
            logger.warning(f"No data returned for {yf_symbol}")
            return df

        # Standardise column names to lowercase
        df.columns = [c.lower() for c in df.columns]

        # Keep only what we need
        keep_cols = [c for c in ["open","high","low","close","volume"] if c in df.columns]
        df = df[keep_cols].copy()

        # Ensure timezone-aware UTC index
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df.index.name = "timestamp"

        return self.validate_dataframe(df, yf_symbol)

    def fetch_latest_price(self, symbol: str) -> float:
        yf_symbol = self._resolve_symbol(symbol)
        try:
            ticker = yf.Ticker(yf_symbol)
            info = ticker.fast_info or {}

            price = None
            for key in ("lastPrice", "last_price", "regularMarketPrice", "previousClose", "previous_close"):
                try:
                    value = info.get(key) if hasattr(info, "get") else getattr(info, key, None)
                except Exception:
                    value = None
                if value is not None:
                    price = value
                    break

            if price is None:
                hist = ticker.history(period="5d", interval="1d", auto_adjust=True)
                if not hist.empty and "Close" in hist.columns:
                    price = hist["Close"].dropna().iloc[-1]

            if price is None:
                raise ValueError("No latest price available from yfinance response")

            logger.debug(f"Latest price {yf_symbol}: {float(price):.2f}")
            return float(price)
        except Exception as e:
            logger.error(f"Could not fetch latest price for {yf_symbol}: {e}")
            return 0.0

    def fetch_multiple(
        self,
        symbols: list[str],
        interval: Interval,
        start: datetime,
        end: datetime,
    ) -> dict[str, pd.DataFrame]:
        """Batch fetch — more efficient than calling fetch_ohlcv in a loop."""
        yf_symbols = [self._resolve_symbol(s) for s in symbols]
        results = {}

        logger.info(f"Batch fetching {len(yf_symbols)} symbols")

        raw = yf.download(
            tickers=yf_symbols,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=INTERVAL_MAP[interval],
            auto_adjust=True,
            group_by="ticker",
            progress=False,
        )

        for orig_sym, yf_sym in zip(symbols, yf_symbols):
            try:
                if len(yf_symbols) == 1:
                    df = raw.copy()
                else:
                    df = raw[yf_sym].copy()

                df.columns = [c.lower() for c in df.columns]
                keep = [c for c in ["open","high","low","close","volume"] if c in df.columns]
                df = df[keep]

                if df.index.tzinfo is None:
                    df.index = df.index.tz_localize("UTC")
                else:
                    df.index = df.index.tz_convert("UTC")

                df.index.name = "timestamp"
                results[orig_sym] = self.validate_dataframe(df, yf_sym)

            except Exception as e:
                logger.error(f"Error processing {yf_sym}: {e}")
                results[orig_sym] = pd.DataFrame()

        return results

    def _resolve_symbol(self, symbol: str) -> str:
        from src.data.models import ALL_SYMBOLS
        symbol_text = str(symbol).strip().upper()
        if symbol_text in ALL_SYMBOLS:
            return ALL_SYMBOLS[symbol_text].symbol

        def normalize(name: str) -> str:
            text = name.upper().strip()
            text = text.replace(".NS", "")
            text = text.replace(".LTD", " LTD")
            text = text.replace(".LIMITED", " LIMITED")
            text = text.replace(".CORP", " CORP")
            text = text.replace("&", " AND ")
            text = text.replace("..", ".")
            text = text.replace("-", " ")
            text = text.replace("/", " ")
            text = text.replace("  ", " ")
            return text.strip()

        lookup_name = normalize(symbol_text)
        for ticker, info in ALL_SYMBOLS.items():
            company_name = normalize(info.name)
            if lookup_name == company_name:
                return info.symbol
            if lookup_name in company_name or company_name in lookup_name:
                return info.symbol

        return symbol
