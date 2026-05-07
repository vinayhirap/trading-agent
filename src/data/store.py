# trading-agent/src/data/store.py

from datetime import datetime
from pathlib import Path
import pandas as pd
from loguru import logger
from config.settings import settings


class LocalDataStore:
    """
    Stores OHLCV data as Parquet files on disk.

    Structure:
        data/processed/
            equity/
                RELIANCE_1d.parquet
                TCS_5m.parquet
            crypto/
                BTC_1h.parquet
            index/
                NIFTY50_1d.parquet

    Why Parquet (not CSV)?
    - 5-10x smaller file size
    - Much faster read/write
    - Preserves dtypes (no re-parsing dates)
    - Industry standard for time-series data
    """

    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or settings.PROCESSED_DATA_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"DataStore initialised at {self.base_dir}")

    def _get_path(self, symbol: str, interval: str, asset_type: str = "equity") -> Path:
        folder = self.base_dir / asset_type
        folder.mkdir(parents=True, exist_ok=True)
        safe_symbol = symbol.replace(".", "_").replace("^", "IDX_")
        return folder / f"{safe_symbol}_{interval}.parquet"

    def save(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        asset_type: str = "equity",
    ) -> Path:
        if df.empty:
            logger.warning(f"Skipping save — empty DataFrame for {symbol}")
            return None

        path = self._get_path(symbol, interval, asset_type)

        # If file exists, merge new data with existing (no duplicates)
        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
            combined.to_parquet(path, engine="pyarrow", compression="snappy")
            logger.info(f"Updated {path.name}: {len(existing)} → {len(combined)} bars")
        else:
            df.to_parquet(path, engine="pyarrow", compression="snappy")
            logger.info(f"Saved {path.name}: {len(df)} bars")

        return path

    def load(
        self,
        symbol: str,
        interval: str,
        asset_type: str = "equity",
        start: datetime = None,
        end: datetime = None,
    ) -> pd.DataFrame:
        path = self._get_path(symbol, interval, asset_type)

        if not path.exists():
            logger.warning(f"No cached data found: {path.name}")
            return pd.DataFrame()

        df = pd.read_parquet(path)

        if start:
            start_utc = pd.Timestamp(start, tz="UTC") if start.tzinfo is None else pd.Timestamp(start).tz_convert("UTC")
            df = df[df.index >= start_utc]
        if end:
            end_utc = pd.Timestamp(end, tz="UTC") if end.tzinfo is None else pd.Timestamp(end).tz_convert("UTC")
            df = df[df.index <= end_utc]

        logger.debug(f"Loaded {len(df)} bars from {path.name}")
        return df

    def list_available(self) -> list[dict]:
        """Show what data is cached locally."""
        files = []
        for parquet_file in self.base_dir.rglob("*.parquet"):
            try:
                df = pd.read_parquet(parquet_file)
                files.append({
                    "file":   parquet_file.name,
                    "bars":   len(df),
                    "from":   df.index.min().strftime("%Y-%m-%d") if not df.empty else "—",
                    "to":     df.index.max().strftime("%Y-%m-%d") if not df.empty else "—",
                    "size_kb": round(parquet_file.stat().st_size / 1024, 1),
                })
            except Exception as e:
                logger.error(f"Could not read {parquet_file.name}: {e}")
        return files