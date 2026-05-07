# trading-agent/src/data/models.py
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import pandas as pd


class AssetClass(str, Enum):
    EQUITY = "equity"
    FUTURES = "futures"
    OPTIONS = "options"
    CRYPTO = "crypto"
    INDEX = "index"


class Exchange(str, Enum):
    NSE = "NSE"
    BSE = "BSE"
    MCX = "MCX"
    CRYPTO = "CRYPTO"


class Interval(str, Enum):
    M1  = "1m"
    M5  = "5m"
    M15 = "15m"
    M30 = "30m"
    H1  = "1h"
    D1  = "1d"
    W1  = "1wk"


@dataclass
class OHLCVBar:
    """Single price bar — the atomic unit of market data."""
    symbol:     str
    exchange:   Exchange
    interval:   Interval
    timestamp:  datetime
    open:       float
    high:       float
    low:        float
    close:      float
    volume:     float
    oi:         Optional[float] = None   # open interest (F&O only)

    def is_valid(self) -> bool:
        """Basic sanity checks before this bar enters the system."""
        if self.high < self.low:
            return False
        if self.open <= 0 or self.close <= 0:
            return False
        if self.volume < 0:
            return False
        if not (self.low <= self.open <= self.high):
            return False
        if not (self.low <= self.close <= self.high):
            return False
        return True


@dataclass
class MarketDepth:
    """Level 2 order book snapshot."""
    symbol:    str
    timestamp: datetime
    bids:      list[tuple[float, float]]   # [(price, qty), ...]
    asks:      list[tuple[float, float]]
    ltp:       float                        # last traded price


@dataclass
class SymbolInfo:
    """Static metadata for a tradeable instrument."""
    symbol:      str
    name:        str
    exchange:    Exchange
    asset_class: AssetClass
    lot_size:    int   = 1
    tick_size:   float = 0.05
    margin_pct:  float = 1.0   # 1.0 = full price (equity), < 1.0 = leveraged


# ── Commonly traded symbols (expand as needed) ──────────────────────────────
# ── Add to existing NSE_SYMBOLS dict ────────────────────────────────────────
NSE_SYMBOLS: dict[str, SymbolInfo] = {
    # Indices
    "NIFTY50":    SymbolInfo("^NSEI",      "Nifty 50",          Exchange.NSE, AssetClass.INDEX),
    "BANKNIFTY":  SymbolInfo("^NSEBANK",   "Bank Nifty",        Exchange.NSE, AssetClass.INDEX),
    "SENSEX":     SymbolInfo("^BSESN",     "BSE Sensex",        Exchange.BSE, AssetClass.INDEX),
    "NIFTYMID":   SymbolInfo("^NSEMDCP50", "Nifty Midcap 50",   Exchange.NSE, AssetClass.INDEX),
    "NIFTYIT":    SymbolInfo("^CNXIT",     "Nifty IT",          Exchange.NSE, AssetClass.INDEX),
    "NIFTYPHARMA":SymbolInfo("^CNXPHARMA", "Nifty Pharma",      Exchange.NSE, AssetClass.INDEX),
    "NIFTYAUTO":  SymbolInfo("^CNXAUTO",   "Nifty Auto",        Exchange.NSE, AssetClass.INDEX),
    "NIFTYFMCG":  SymbolInfo("^CNXFMCG",   "Nifty FMCG",       Exchange.NSE, AssetClass.INDEX),
    "NIFTYINFRA": SymbolInfo("^CNXINFRA",  "Nifty Infra",       Exchange.NSE, AssetClass.INDEX),
    "FINNIFTY":   SymbolInfo("^CNXFIN",    "Nifty Fin Service", Exchange.NSE, AssetClass.INDEX),

    # Large cap equities
    "RELIANCE":   SymbolInfo("RELIANCE.NS",  "Reliance Industries", Exchange.NSE, AssetClass.EQUITY),
    "TCS":        SymbolInfo("TCS.NS",       "Tata Consultancy Services", Exchange.NSE, AssetClass.EQUITY),
    "HDFCBANK":   SymbolInfo("HDFCBANK.NS",  "HDFC Bank", Exchange.NSE, AssetClass.EQUITY),
    "INFY":       SymbolInfo("INFY.NS",      "Infosys", Exchange.NSE, AssetClass.EQUITY),
    "ICICIBANK":  SymbolInfo("ICICIBANK.NS", "ICICI Bank",        Exchange.NSE, AssetClass.EQUITY),
    "SBIN":       SymbolInfo("SBIN.NS",      "State Bank",        Exchange.NSE, AssetClass.EQUITY),
    "WIPRO":      SymbolInfo("WIPRO.NS",     "Wipro",            Exchange.NSE, AssetClass.EQUITY),
    "AXISBANK":   SymbolInfo("AXISBANK.NS",  "Axis Bank",        Exchange.NSE, AssetClass.EQUITY),
    "KOTAKBANK":  SymbolInfo("KOTAKBANK.NS", "Kotak Bank",       Exchange.NSE, AssetClass.EQUITY),
    "LT":         SymbolInfo("LT.NS",        "L&T",              Exchange.NSE, AssetClass.EQUITY),
    "BAJFINANCE": SymbolInfo("BAJFINANCE.NS","Bajaj Finance",    Exchange.NSE, AssetClass.EQUITY),
    "MARUTI":     SymbolInfo("MARUTI.NS",    "Maruti Suzuki",    Exchange.NSE, AssetClass.EQUITY),
    "ASIANPAINT": SymbolInfo("ASIANPAINT.NS","Asian Paints",     Exchange.NSE, AssetClass.EQUITY),
    "NTPC":       SymbolInfo("NTPC.NS",      "NTPC",             Exchange.NSE, AssetClass.EQUITY),
    "POWERGRID":  SymbolInfo("POWERGRID.NS", "Power Grid",       Exchange.NSE, AssetClass.EQUITY),
    "ONGC":       SymbolInfo("ONGC.NS",      "ONGC",             Exchange.NSE, AssetClass.EQUITY),
    "COALINDIA":  SymbolInfo("COALINDIA.NS", "Coal India",       Exchange.NSE, AssetClass.EQUITY),
    "ULTRACEMCO": SymbolInfo("ULTRACEMCO.NS","UltraTech Cement", Exchange.NSE, AssetClass.EQUITY),
    "SUNPHARMA":  SymbolInfo("SUNPHARMA.NS", "Sun Pharma",       Exchange.NSE, AssetClass.EQUITY),
    "DRREDDY":    SymbolInfo("DRREDDY.NS",   "Dr Reddy's",       Exchange.NSE, AssetClass.EQUITY),
    "BHARTIARTL": SymbolInfo("BHARTIARTL.NS","Bharti Airtel",    Exchange.NSE, AssetClass.EQUITY),
    "JSWSTEEL":   SymbolInfo("JSWSTEEL.NS",  "JSW Steel",        Exchange.NSE, AssetClass.EQUITY),
    "TATASTEEL":  SymbolInfo("TATASTEEL.NS", "Tata Steel",       Exchange.NSE, AssetClass.EQUITY),
    "HINDALCO":   SymbolInfo("HINDALCO.NS",  "Hindalco",         Exchange.NSE, AssetClass.EQUITY),
    "TATAMOTORS": SymbolInfo("TATAMOTORS.NS","Tata Motors",     Exchange.NSE, AssetClass.EQUITY),
    "M&M":        SymbolInfo("M&M.NS",       "Mahindra",         Exchange.NSE, AssetClass.EQUITY),
    "BAJAJFINSV": SymbolInfo("BAJAJFINSV.NS","Bajaj Finserv",    Exchange.NSE, AssetClass.EQUITY),
}

# ── MCX Commodities ───────────────────────────────────────────────────────────
MCX_SYMBOLS: dict[str, SymbolInfo] = {
    "GOLD":    SymbolInfo("GC=F",   "Gold (MCX)",      Exchange.MCX, AssetClass.FUTURES),
    "SILVER":  SymbolInfo("SI=F",   "Silver (MCX)",    Exchange.MCX, AssetClass.FUTURES),
    "CRUDEOIL":SymbolInfo("CL=F",   "Crude Oil (MCX)", Exchange.MCX, AssetClass.FUTURES),
    "COPPER":  SymbolInfo("HG=F",   "Copper (MCX)",    Exchange.MCX, AssetClass.FUTURES),
    "ZINC":      SymbolInfo("",      "Zinc (MCX)",      Exchange.MCX, AssetClass.FUTURES),
    "NATURALGAS":SymbolInfo("NG=F", "Natural Gas",     Exchange.MCX, AssetClass.FUTURES),
    "ALUMINIUM": SymbolInfo("",      "Aluminium (MCX)", Exchange.MCX, AssetClass.FUTURES),
}

# ── Crypto (extended) ────────────────────────────────────────────────────────
CRYPTO_SYMBOLS:dict[str, SymbolInfo] = {
    "BTC":   SymbolInfo("BTC-USD",   "Bitcoin",  Exchange.CRYPTO, AssetClass.CRYPTO),
    "ETH":   SymbolInfo("ETH-USD",   "Ethereum", Exchange.CRYPTO, AssetClass.CRYPTO),
    "SOL":   SymbolInfo("SOL-USD",   "Solana",   Exchange.CRYPTO, AssetClass.CRYPTO),
    "MATIC": SymbolInfo("MATIC-USD", "Polygon",  Exchange.CRYPTO, AssetClass.CRYPTO),
    "BNB":   SymbolInfo("BNB-USD",   "Binance",  Exchange.CRYPTO, AssetClass.CRYPTO),
    "XRP":   SymbolInfo("XRP-USD",   "Ripple",   Exchange.CRYPTO, AssetClass.CRYPTO),
    "ADA":   SymbolInfo("ADA-USD",   "Cardano",  Exchange.CRYPTO, AssetClass.CRYPTO),
    "DOGE":  SymbolInfo("DOGE-USD",  "Dogecoin", Exchange.CRYPTO, AssetClass.CRYPTO),
    "AVAX":  SymbolInfo("AVAX-USD",  "Avalanche",Exchange.CRYPTO, AssetClass.CRYPTO),
    "DOT":   SymbolInfo("DOT-USD",   "Polkadot", Exchange.CRYPTO, AssetClass.CRYPTO),
}

# ── Forex (future-ready) ─────────────────────────────────────────────────────
FOREX_SYMBOLS: dict[str, SymbolInfo] = {
    "USDINR":  SymbolInfo("USDINR=X",  "USD/INR",  Exchange.NSE, AssetClass.FUTURES),
    "EURINR":  SymbolInfo("EURINR=X",  "EUR/INR",  Exchange.NSE, AssetClass.FUTURES),
    "GBPINR":  SymbolInfo("GBPINR=X",  "GBP/INR",  Exchange.NSE, AssetClass.FUTURES),
    "JPYINR":  SymbolInfo("JPYINR=X",  "JPY/INR",  Exchange.NSE, AssetClass.FUTURES),
    "EURUSD":  SymbolInfo("EURUSD=X",  "EUR/USD",  Exchange.NSE, AssetClass.FUTURES),
    "GBPUSD":  SymbolInfo("GBPUSD=X",  "GBP/USD",  Exchange.NSE, AssetClass.FUTURES),
}

# ── Master universe — everything the system can trade ───────────────────────
ALL_SYMBOLS = {**NSE_SYMBOLS, **MCX_SYMBOLS, **CRYPTO_SYMBOLS, **FOREX_SYMBOLS}
