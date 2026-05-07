# trading-agent/src/streaming/mcx_token_manager.py
"""
Automatic MCX Token Manager

Eliminates manual token updates by:
1. Downloading Angel One ScripMaster on startup
2. Finding near-month FUTCOM contracts for each commodity
3. Auto-rolling to next month when current contract expires
4. Refreshing daily at 8:00 AM IST (before MCX opens at 9:00 AM)
5. Updating angel_one_ticker.py and angel_one_adapter.py live in memory

No manual intervention needed — ever.
Just restart run.py after a rollover and tokens auto-update.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from loguru import logger

ROOT = Path(__file__).resolve().parents[2]

SCRIP_MASTER_URL = (
    "https://margincalculator.angelbroking.com"
    "/OpenAPI_File/files/OpenAPIScripMaster.json"
)

# Commodity config: name → search prefixes + exclusions
COMMODITY_CONFIG = {
    "GOLD": {
        "prefixes":  ["GOLD"],
        "exclude":   ["PETAL", "GUINEA", "MINI", "M26", "M27"],
        "exchange":  "MCX",
        "fallback":  "495213",   # GOLD04DEC26FUT
    },
    "SILVER": {
        "prefixes":  ["SILVER"],
        "exclude":   ["MIC", "MINI", "M26", "M27"],
        "exchange":  "MCX",
        "fallback":  "495214",   # SILVER04DEC26FUT
    },
    "CRUDEOIL": {
        "prefixes":  ["CRUDEOIL"],
        "exclude":   ["MINI", "M26"],
        "exchange":  "MCX",
        "fallback":  "488291",
    },
    "COPPER": {
        "prefixes":  ["COPPER"],
        "exclude":   ["MINI", "M26"],
        "exchange":  "MCX",
        "fallback":  "488791",
    },
    "NATURALGAS": {
        "prefixes":  ["NATURALGAS"],
        "exclude":   ["MINI", "MICRO"],
        "exchange":  "MCX",
        "fallback":  "488505",
    },
    "ZINC": {
        "prefixes":  ["ZINC"],
        "exclude":   ["MINI", "M26"],
        "exchange":  "MCX",
        "fallback":  "510478",
    },
    "ALUMINIUM": {
        "prefixes":  ["ALUMINIUM"],
        "exclude":   ["MINI", "M26"],
        "exchange":  "MCX",
        "fallback":  "488790",
    },
    # Add these 4 entries to COMMODITY_CONFIG dict:
"USDINR": {
    "prefixes":  ["USDINR"],
    "exclude":   [],
    "exchange":  "CDS",
    "scrip_seg": "CDS",
    "inst_type": "FUTCUR",
    "fallback":  "1518",
},
"EURINR": {
    "prefixes":  ["EURINR"],
    "exclude":   [],
    "exchange":  "CDS",
    "scrip_seg": "CDS",
    "inst_type": "FUTCUR",
    "fallback":  "1497",
},
"GBPINR": {
    "prefixes":  ["GBPINR"],
    "exclude":   [],
    "exchange":  "CDS",
    "scrip_seg": "CDS",
    "inst_type": "FUTCUR",
    "fallback":  "1498",
},
"JPYINR": {
    "prefixes":  ["JPYINR"],
    "exclude":   [],
    "exchange":  "CDS",
    "scrip_seg": "CDS",
    "inst_type": "FUTCUR",
    "fallback":  "1510",
},
}

# In-memory token store — updated automatically
_current_tokens: dict[str, dict] = {}
_last_refresh:   Optional[datetime] = None
_lock = threading.Lock()


def get_tokens() -> dict[str, dict]:
    """
    Return current MCX tokens.
    Format: { "GOLD": {"exchange": "MCX", "token": "495213", "symbol": "GOLD04DEC26FUT", "expiry": datetime} }
    """
    with _lock:
        return dict(_current_tokens)


def get_token(commodity: str) -> tuple[str, str] | None:
    """Return (exchange, token) for a commodity, or None if not found."""
    with _lock:
        info = _current_tokens.get(commodity.upper())
        if info:
            return info["exchange"], info["token"]
    return None


def _download_scrip_master() -> list[dict]:
    """Download Angel One ScripMaster JSON."""
    try:
        req = urllib.request.Request(
            SCRIP_MASTER_URL,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8", errors="ignore"))
        logger.debug(f"MCXTokenManager: downloaded {len(data)} instruments")
        return data
    except Exception as e:
        logger.warning(f"MCXTokenManager: ScripMaster download failed — {e}")
        return []


def _find_near_month(
    instruments: list[dict],
    commodity:   str,
    config:      dict,
) -> dict | None:
    """
    Find the nearest-expiry main-contract FUTCOM for a commodity.
    Prefers contracts expiring 3+ days from now to avoid rollover day issues.
    """
    today    = datetime.now()
    rollover = today + timedelta(days=3)   # switch 3 days before expiry

    # Support both MCX commodities and CDS forex futures
    seg      = config.get("scrip_seg", "MCX")
    inst_typ = config.get("inst_type", "FUTCOM")
    mcx_fut  = [
        i for i in instruments
        if i.get("exch_seg") == seg
        and i.get("instrumenttype") == inst_typ
    ]

    candidates = []
    for inst in mcx_fut:
        sym = inst.get("symbol", "").upper()
        name = inst.get("name", "").upper()

        # Must match at least one prefix
        if not any(
            sym.startswith(p.upper()) or name.startswith(p.upper())
            for p in config["prefixes"]
        ):
            continue

        # Must not contain any exclusion strings
        if any(ex.upper() in sym for ex in config["exclude"]):
            continue

        # Parse expiry
        try:
            expiry = datetime.strptime(inst.get("expiry", ""), "%d%b%Y")
        except ValueError:
            try:
                expiry = datetime.strptime(inst.get("expiry", ""), "%d%b%y")
            except ValueError:
                continue

        # Must expire after rollover threshold
        if expiry >= rollover:
            candidates.append({
                "token":    inst["token"],
                "symbol":   inst["symbol"],
                "expiry":   expiry,
                "exchange": config["exchange"],
                "name":     inst.get("name", ""),
            })

    if not candidates:
        return None

    # Pick nearest expiry
    return min(candidates, key=lambda x: x["expiry"])


def refresh_tokens(instruments: list[dict] | None = None) -> dict[str, dict]:
    """
    Refresh all MCX tokens from ScripMaster.
    Downloads fresh ScripMaster if not provided.
    Returns the new token map.
    """
    global _last_refresh

    if instruments is None:
        instruments = _download_scrip_master()

    if not instruments:
        logger.warning("MCXTokenManager: no instruments — keeping existing tokens")
        return get_tokens()

    new_tokens = {}
    for commodity, config in COMMODITY_CONFIG.items():
        result = _find_near_month(instruments, commodity, config)

        if result:
            new_tokens[commodity] = result
            days_left = (result["expiry"] - datetime.now()).days
            logger.info(
                f"MCXTokenManager: {commodity} → "
                f"token={result['token']} "
                f"symbol={result['symbol']} "
                f"expiry={result['expiry'].strftime('%d%b%Y')} "
                f"({days_left}d left)"
            )
        else:
            # Keep existing token if refresh fails
            existing = _current_tokens.get(commodity)
            if existing:
                new_tokens[commodity] = existing
                logger.warning(
                    f"MCXTokenManager: {commodity} — no new contract found, "
                    f"keeping {existing['symbol']}"
                )
            else:
                # Use hardcoded fallback
                fallback_token = config["fallback"]
                new_tokens[commodity] = {
                    "token":    fallback_token,
                    "symbol":   f"{commodity}_FALLBACK",
                    "expiry":   datetime.now() + timedelta(days=30),
                    "exchange": config["exchange"],
                }
                logger.warning(
                    f"MCXTokenManager: {commodity} — using fallback token {fallback_token}"
                )

    with _lock:
        _current_tokens.clear()
        _current_tokens.update(new_tokens)
        _last_refresh = datetime.now()

    # Push updated tokens into angel_one_ticker and angel_one_adapter
    _apply_tokens_to_modules(new_tokens)

    logger.info(
        f"MCXTokenManager: refreshed {len(new_tokens)} tokens at "
        f"{_last_refresh.strftime('%H:%M:%S')}"
    )
    return new_tokens


def _apply_tokens_to_modules(tokens: dict[str, dict]):
    """
    Push new tokens into angel_one_ticker and angel_one_adapter
    live in memory — no file writes, no restart needed.
    """
    try:
        from src.streaming import angel_one_ticker as ticker_mod
        for commodity, info in tokens.items():
            ticker_mod.MCX_TOKENS[commodity] = (info["exchange"], info["token"])
            ticker_mod.TOKEN_TO_SYMBOL[info["token"]] = commodity
        logger.debug("MCXTokenManager: applied to angel_one_ticker")
    except Exception as e:
        logger.debug(f"MCXTokenManager: ticker update skipped — {e}")

    try:
        from src.data.adapters import angel_one_adapter as adapter_mod
        for commodity, info in tokens.items():
            adapter_mod.ANGEL_TOKENS[commodity] = (info["exchange"], info["token"])
        logger.debug("MCXTokenManager: applied to angel_one_adapter")
    except Exception as e:
        logger.debug(f"MCXTokenManager: adapter update skipped — {e}")


def _needs_refresh() -> bool:
    """Check if any token is expiring within 3 days or refresh is overdue."""
    if _last_refresh is None:
        return True

    # Refresh daily
    hours_since = (datetime.now() - _last_refresh).total_seconds() / 3600
    if hours_since >= 24:
        return True

    # Check for imminent expiry
    with _lock:
        for commodity, info in _current_tokens.items():
            days_left = (info["expiry"] - datetime.now()).days
            if days_left <= 3:
                logger.info(
                    f"MCXTokenManager: {commodity} expires in {days_left}d — "
                    f"triggering refresh"
                )
                return True

    return False


def _background_loop():
    """Background thread: check every hour, refresh when needed."""
    # Initial delay — let startup complete first
    time.sleep(10)

    while True:
        try:
            if _needs_refresh():
                # Download once, use for all commodities
                instruments = _download_scrip_master()
                if instruments:
                    refresh_tokens(instruments)
                else:
                    logger.warning("MCXTokenManager: download failed, will retry in 1h")
            else:
                # Log status
                with _lock:
                    for c, info in _current_tokens.items():
                        days = (info["expiry"] - datetime.now()).days
                        if days <= 7:
                            logger.debug(f"MCXTokenManager: {c} expires in {days}d")

        except Exception as e:
            logger.error(f"MCXTokenManager: background error — {e}")

        # Check every hour
        time.sleep(3600)


class MCXTokenManager:
    """
    Manages MCX contract tokens with automatic rollover.

    Usage:
        from src.streaming.mcx_token_manager import mcx_token_manager
        mcx_token_manager.start()   # call once at startup
    """

    def __init__(self):
        self._started = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Initialize tokens and start background refresh thread."""
        if self._started:
            return

        logger.info("MCXTokenManager: starting...")

        # Synchronous first refresh at startup
        try:
            instruments = _download_scrip_master()
            if instruments:
                refresh_tokens(instruments)
            else:
                logger.warning(
                    "MCXTokenManager: startup download failed — "
                    "using fallback tokens"
                )
                self._apply_fallbacks()
        except Exception as e:
            logger.warning(f"MCXTokenManager: startup error — {e}, using fallbacks")
            self._apply_fallbacks()

        # Start background thread for daily refresh + expiry checks
        self._thread = threading.Thread(
            target=_background_loop,
            daemon=True,
            name="MCXTokenManager",
        )
        self._thread.start()
        self._started = True
        logger.info("MCXTokenManager: running — tokens will auto-refresh daily")

    def _apply_fallbacks(self):
        """Apply hardcoded fallback tokens when ScripMaster unavailable."""
        fallbacks = {
            c: {
                "token":    cfg["fallback"],
                "symbol":   f"{c}_FALLBACK",
                "expiry":   datetime.now() + timedelta(days=30),
                "exchange": cfg["exchange"],
            }
            for c, cfg in COMMODITY_CONFIG.items()
        }
        with _lock:
            _current_tokens.update(fallbacks)
        _apply_tokens_to_modules(fallbacks)

    def get_status(self) -> dict:
        """Return current token status for dashboard."""
        with _lock:
            return {
                "last_refresh": _last_refresh.isoformat() if _last_refresh else None,
                "tokens": {
                    c: {
                        "token":     info["token"],
                        "symbol":    info["symbol"],
                        "expiry":    info["expiry"].strftime("%d%b%Y"),
                        "days_left": (info["expiry"] - datetime.now()).days,
                    }
                    for c, info in _current_tokens.items()
                }
            }

    def force_refresh(self) -> dict:
        """Force immediate token refresh — useful for testing."""
        return refresh_tokens()


# Module singleton
mcx_token_manager = MCXTokenManager()