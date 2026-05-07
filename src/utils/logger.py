# trading-agent/src/utils/logger.py
import sys
from loguru import logger
from pathlib import Path
from config.settings import settings

def setup_logger() -> None:
    """Configure the global logger for the entire system."""
    logger.remove()  # Remove default handler

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # Console output
    logger.add(
        sys.stdout,
        format=log_format,
        level=settings.LOG_LEVEL,
        colorize=True,
    )

    # File output — rotates daily, keeps 30 days
    log_path = settings.LOG_DIR / "trading_agent_{time:YYYY-MM-DD}.log"
    logger.add(
        str(log_path),
        format=log_format,
        level="DEBUG",
        rotation="00:00",
        retention="30 days",
        compression="zip",
    )

    logger.info(f"Logger initialised | env={settings.ENV} | capital=INR{settings.INITIAL_CAPITAL:,.0f}")

# Call this once at startup
setup_logger()