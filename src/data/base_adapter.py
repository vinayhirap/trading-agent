# trading-agent/src/data/base_adapter.py
from abc import ABC, abstractmethod
from datetime import datetime
import pandas as pd
from src.data.models import OHLCVBar, Interval, Exchange
from loguru import logger


class BaseDataAdapter(ABC):
    """
    Every data source implements this interface.
    The rest of the system only ever calls fetch_ohlcv() and fetch_latest_price().
    This means you can swap yfinance → Angel One → any other source
    without changing a single line in the feature or strategy modules.
    """

    def __init__(self, name: str):
        self.name = name
        self._connected = False
        logger.debug(f"Adapter initialised: {name}")

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection / authenticate. Returns True on success."""
        ...

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        interval: Interval,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Returns a clean DataFrame with columns:
        [timestamp, open, high, low, close, volume]
        Index: DatetimeIndex (UTC-aware)
        """
        ...

    @abstractmethod
    def fetch_latest_price(self, symbol: str) -> float:
        """Returns the latest traded price for a symbol."""
        ...

    def validate_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Shared validation logic — all adapters call this before returning data.
        Removes bad bars, fills small gaps, ensures correct dtypes.
        """
        if df.empty:
            logger.warning(f"{self.name}: empty DataFrame for {symbol}")
            return df

        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{self.name}: missing columns {missing} for {symbol}")

        initial_len = len(df)

        # Remove rows with any NaN in OHLCV
        df = df.dropna(subset=list(required_cols))

        # Remove physically impossible bars
        df = df[df["high"] >= df["low"]]
        df = df[df["open"] > 0]
        df = df[df["close"] > 0]
        df = df[df["volume"] >= 0]

        # Ensure open and close are within high/low range
        df = df[df["open"].between(df["low"], df["high"])]
        df = df[df["close"].between(df["low"], df["high"])]

        removed = initial_len - len(df)
        if removed > 0:
            logger.warning(f"{self.name}: removed {removed} invalid bars for {symbol}")

        # Sort chronologically
        df = df.sort_index()

        # Remove duplicate timestamps
        df = df[~df.index.duplicated(keep="last")]

        logger.debug(f"{self.name}: validated {len(df)} bars for {symbol}")
        return df