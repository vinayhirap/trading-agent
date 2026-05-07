"""
Groww Portfolio Connector.

Groww does NOT have a public API for portfolio data.
Workaround options:
1. CSV Import: Download portfolio CSV from Groww app → upload here
2. Manual Entry: Enter holdings manually → system tracks P&L
3. Future: Browser extension or account statement parsing

CSV format expected (Groww export):
Symbol, Quantity, Average Price, Current Price, Invested Value, Current Value, P&L

This module handles:
- CSV parsing and portfolio display
- Real-time price updates via yfinance
- P&L calculation
- Persistent JSON storage so you don't re-enter every time
"""
import json
import csv
import io
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from loguru import logger
import yfinance as yf

GROWW_DB = Path("data/groww_portfolio.json")

GROWW_NAME_TO_TICKER = {
    "ADANI ENTERPRISES":       "ADANIENT",
    "ADANI POWER":             "ADANIPOWER",
    "AKASH INFRA-PROJECTS":    "AKASHINFRA",
    "AMBUJA CEMENTS":          "AMBUJACEM",
    "ETERNAL LIMITED":         "ETERNAL",
    "FSN E COMMERCE VENTURES": "FSN ECOMMERCE",
    "GTL INFRA":               "GTLINFRA",
    "HSG & URBAN DEV CORPN":   "HSGURBAN",
    "ICICI PRUDENTIAL AMC":    "ICICIPRULI",
    "IDFC FIRST BANK":         "IDFCFIRSTB",
    "INDIAN RAILWAY FIN CORP": "IRFC",
    "OLA ELECTRIC MOBILITY":   "OLA",
    "WIPRO":                   "WIPRO",
    "YES BANK":                "YESBANK",
}


@dataclass
class GrowwHolding:
    symbol:         str      # NSE symbol e.g. "RELIANCE"
    name:           str
    quantity:       float
    avg_buy_price:  float
    current_price:  float
    invested_value: float
    current_value:  float
    pnl:            float
    pnl_pct:        float
    exchange:       str = "NSE"
    last_updated:   str = ""


@dataclass
class GrowwPortfolio:
    holdings:       list
    total_invested: float
    total_value:    float
    total_pnl:      float
    total_pnl_pct:  float
    fetched_at:     str
    source:         str   # "csv" / "manual" / "empty"


class GrowwConnector:
    """
    Groww portfolio manager with CSV import and manual entry.

    Usage:
        connector = GrowwConnector()

        # From CSV export
        portfolio = connector.load_from_csv(csv_text)

        # Manual add
        connector.add_holding("RELIANCE", 10, avg_buy=2800)

        # Get with live prices
        portfolio = connector.get_portfolio_with_live_prices()
    """

    def __init__(self):
        GROWW_DB.parent.mkdir(parents=True, exist_ok=True)
        self._holdings = self._load_saved()
        logger.info(f"Groww connector | {len(self._holdings)} holdings loaded")

    def _clean_text(self, value: str) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        return text.replace("\ufeff", "").replace("\u200b", "").strip()

    def _is_ticker_like(self, value: str) -> bool:
        value = self._clean_text(value).upper()
        if not value or " " in value:
            return False
        return bool(__import__("re").fullmatch(r"[A-Z0-9&-]{1,15}", value))

    def _normalize_name(self, name: str) -> str:
        name = self._clean_text(name).upper()
        for suffix in [" LTD", " LIMITED", " LTD.", " PRIVATE LIMITED", " PRIVATE", " PVT LTD", " PVT", " CO", " COMPANY", " CORPORATION", " CORPN"]:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        name = name.replace("&", " ").replace(".", " ").replace("-", " ")
        return " ".join(name.split()).strip()

    def _name_to_symbol(self, name: str) -> str:
        if not name:
            return ""
        normalized = self._normalize_name(name)
        if normalized in GROWW_NAME_TO_TICKER:
            return GROWW_NAME_TO_TICKER[normalized]
        return ""

    def _resolve_symbol(self, raw_sym: str, name: str = "") -> str:
        raw_sym = self._clean_text(raw_sym).upper()
        if self._is_ticker_like(raw_sym):
            return raw_sym
        resolved = self._name_to_symbol(raw_sym)
        if resolved:
            return resolved
        resolved = self._name_to_symbol(name)
        if resolved:
            return resolved
        candidate = __import__("re").sub(r"[^A-Z0-9&-]", "", raw_sym)
        if self._is_ticker_like(candidate):
            return candidate
        return raw_sym

    def get_portfolio_with_live_prices(self) -> GrowwPortfolio:
        """Get full portfolio with current market prices from yfinance."""
        if not self._holdings:
            return GrowwPortfolio(
                holdings=[], total_invested=0, total_value=0,
                total_pnl=0, total_pnl_pct=0,
                fetched_at=datetime.now(timezone.utc).isoformat(),
                source="empty",
            )

        # Batch fetch prices
        symbols  = [self._resolve_symbol(h["symbol"], h.get("name", "")) for h in self._holdings]
        prices   = self._fetch_live_prices(symbols)
        holdings = []
        total_inv = 0.0
        total_val = 0.0

        for h in self._holdings:
            sym      = h["symbol"]
            qty      = h["quantity"]
            avg_buy  = h["avg_buy_price"]
            cur_px   = prices.get(sym, avg_buy)
            inv_val  = qty * avg_buy
            cur_val  = qty * cur_px
            pnl      = cur_val - inv_val
            pnl_pct  = (pnl / inv_val * 100) if inv_val > 0 else 0

            total_inv += inv_val
            total_val += cur_val

            holdings.append(GrowwHolding(
                symbol=sym, name=h.get("name", sym),
                quantity=qty, avg_buy_price=round(avg_buy, 2),
                current_price=round(cur_px, 2),
                invested_value=round(inv_val, 2),
                current_value=round(cur_val, 2),
                pnl=round(pnl, 2), pnl_pct=round(pnl_pct, 2),
                last_updated=datetime.now(timezone.utc).isoformat(),
            ))

        total_pnl     = total_val - total_inv
        total_pnl_pct = (total_pnl / total_inv * 100) if total_inv > 0 else 0

        return GrowwPortfolio(
            holdings=holdings,
            total_invested=round(total_inv, 2),
            total_value=round(total_val, 2),
            total_pnl=round(total_pnl, 2),
            total_pnl_pct=round(total_pnl_pct, 2),
            fetched_at=datetime.now(timezone.utc).isoformat(),
            source="live_prices",
        )

    def add_holding(self, symbol: str, quantity: float,
                    avg_buy: float, name: str = "") -> None:
        """Add or update a holding manually."""
        symbol = self._resolve_symbol(symbol, name)
        # Remove existing entry for same symbol
        self._holdings = [h for h in self._holdings if h["symbol"] != symbol]
        self._holdings.append({
            "symbol":        symbol,
            "name":          name or symbol,
            "quantity":      quantity,
            "avg_buy_price": avg_buy,
            "added_at":      datetime.now(timezone.utc).isoformat(),
        })
        self._save()
        logger.info(f"Groww: added {quantity}x {symbol} @ ₹{avg_buy}")

    def remove_holding(self, symbol: str) -> bool:
        """Remove a holding."""
        before = len(self._holdings)
        self._holdings = [h for h in self._holdings if h["symbol"] != symbol.upper()]
        if len(self._holdings) < before:
            self._save()
            return True
        return False

    def load_from_csv(self, csv_text: str | bytes) -> int:
        """
        Parse Groww CSV export and load holdings.
        Returns number of holdings loaded.

        Groww CSV columns:
        Stock Name, NSE/BSE Symbol, Quantity, Average Buy Price
        (may vary by export format)
        """
        if isinstance(csv_text, (bytes, bytearray)):
            for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
                try:
                    csv_text = csv_text.decode(encoding)
                    break
                except Exception:
                    continue
            else:
                csv_text = csv_text.decode("utf-8", errors="replace")

        count = 0
        text_io = io.StringIO(str(csv_text))
        reader = csv.DictReader(text_io)
        for row in reader:
            try:
                sym = self._clean_text(
                    row.get("NSE Symbol") or row.get("Stock Symbol") or
                    row.get("Ticker") or row.get("Symbol") or ""
                ).upper()
                name = self._clean_text(
                    row.get("Stock Name") or row.get("Name") or row.get("Symbol") or ""
                )
                qty = self._parse_number(row.get("Quantity") or row.get("Qty") or row.get("Shares") or 0)
                avg = self._parse_number(
                    row.get("Average Buy Price") or row.get("Avg Buy Price") or
                    row.get("Buy Price") or row.get("Price") or 0
                )
                if not sym and name:
                    sym = self._resolve_symbol(name, name)
                if sym and qty > 0 and avg > 0:
                    self.add_holding(sym, qty, avg, name)
                    count += 1
            except (ValueError, KeyError):
                continue
        logger.info(f"Groww CSV: loaded {count} holdings")
        return count

    def _parse_number(self, value) -> float:
        if value is None:
            return 0.0
        text = str(value).replace("₹", "").replace("Rs.", "")
        text = text.replace(",", "").replace("\u200b", "").strip()
        try:
            return float(text)
        except Exception:
            return 0.0

    def clear_all(self) -> None:
        self._holdings = []
        self._save()

    def get_holdings_count(self) -> int:
        return len(self._holdings)

    def _fetch_live_prices(self, symbols: list) -> dict:
        """Fetch NSE prices for a list of symbols."""
        prices = {}
        for sym in symbols:
            yf_sym = f"{sym}.NS"
            try:
                px = yf.Ticker(yf_sym).fast_info.last_price
                if px and px > 0:
                    prices[sym] = float(px)
            except Exception:
                pass
        return prices

    def _load_saved(self) -> list:
        if GROWW_DB.exists():
            try:
                with open(GROWW_DB) as f:
                    data = json.load(f)
                    return data.get("holdings", [])
            except Exception:
                pass
        return []

    def _save(self) -> None:
        with open(GROWW_DB, "w") as f:
            json.dump({"holdings": self._holdings, "updated": datetime.now(timezone.utc).isoformat()}, f, indent=2)
