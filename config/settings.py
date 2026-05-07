#trading-agent/config/settings.py

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # =========================
    # 📌 PROJECT
    # =========================
    PROJECT_NAME: str = "TradingAgent"
    ENV: str = "development"  # development | paper | live

    # =========================
    # 💰 CAPITAL & RISK
    # =========================
    INITIAL_CAPITAL: float = 100000.0
    MAX_RISK_PER_TRADE: float = 0.01
    MAX_DAILY_LOSS_PCT: float = 0.02
    MAX_OPEN_POSITIONS: int = 3
    MAX_PORTFOLIO_RISK: float = 0.05

    # =========================
    # 🏦 BROKER APIs
    # =========================
    ZERODHA_API_KEY: str | None = None
    ZERODHA_API_SECRET: str | None = None
    ZERODHA_ACCESS_TOKEN: str | None = None

    ANGEL_API_KEY: str | None = None
    ANGEL_CLIENT_ID: str | None = None
    ANGEL_PASSWORD: str | None = None
    ANGEL_TOTP_SECRET: str | None = None
    ENABLE_ANGEL_STREAMING: bool = False
    ANTHROPIC_API_KEY: str | None = None

    # =========================
    # 🪙 CRYPTO APIs
    # =========================
    COINSWITCH_API_KEY: str | None = None   # ✅ ADDED
    COINSWITCH_API_SECRET: str | None = None  # ✅ ADDED

    # =========================
    # 📊 DATA APIs
    # =========================
    NEWS_API_KEY: str | None = None
    ALPHA_VANTAGE_KEY: str | None = None
    GNEWS_API_KEY: str | None = None

    # =========================
    # 🤖 TELEGRAM
    # =========================
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    # =========================
    # 📂 PATHS
    # =========================
    DATA_DIR: Path = BASE_DIR / "data"
    LOG_DIR: Path = BASE_DIR / "logs"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

    # =========================
    # 📜 LOGGING
    # =========================
    LOG_LEVEL: str = "INFO"

    # =========================
    # ✅ VALIDATIONS
    # =========================
    @field_validator("MAX_RISK_PER_TRADE", "MAX_DAILY_LOSS_PCT", "MAX_PORTFOLIO_RISK")
    @classmethod
    def validate_percentages(cls, v):
        if not 0 < v <= 1:
            raise ValueError("Risk values must be between 0 and 1")
        return v

    @field_validator("ENV")
    @classmethod
    def validate_env(cls, v):
        allowed = {"development", "paper", "live"}
        if v not in allowed:
            raise ValueError(f"ENV must be one of {allowed}")
        return v

    # =========================
    # 🔒 SAFETY CHECKS
    # =========================
    def validate_live_trading(self):
        """Fail fast if live trading without credentials"""
        if self.ENV == "live":

            # Zerodha (primary)
            if not self.ZERODHA_API_KEY or not self.ZERODHA_API_SECRET:
                raise ValueError("Missing Zerodha credentials for live trading")

            # Angel (optional but recommended)
            if self.ANGEL_API_KEY and not self.ANGEL_TOTP_SECRET:
                raise ValueError("Angel TOTP secret required for login")

            # CoinSwitch (if crypto enabled later)
            if self.COINSWITCH_API_KEY and not self.COINSWITCH_API_SECRET:
                raise ValueError("CoinSwitch secret missing")


# =========================
# 🚀 INIT SETTINGS
# =========================
settings = Settings()

# Ensure directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
settings.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Validate live mode
settings.validate_live_trading()
