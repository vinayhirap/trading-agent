# trading-agent/src/data/adapters/alphavantage_adapter.py
"""
Alpha Vantage Adapter — Bloomberg FA/DES/ANR/EQRV Data Layer

API key: 5KAR4G0D7CEI8Z0V
Free tier limits: 25 requests/day, 500/month

All calls are Parquet-cached to data/cache/av_{function}_{symbol}.parquet
Cache TTL: 24h for fundamentals, 6h for price/earnings

Bloomberg function equivalents:
  DES  → get_overview(symbol)           — Company snapshot
  FA   → get_income_statement(symbol)   — P&L 5 years
  FA   → get_balance_sheet(symbol)      — Balance sheet 5 years
  FA   → get_cash_flow(symbol)          — Cash flow 5 years
  EEG  → get_earnings(symbol)           — EPS actual vs estimate
  ANR  → yfinance .recommendations      — Analyst consensus (separate)
  EQRV → get_peer_overview(symbols)     — Batch peer comparison
  ALTD → get_news_sentiment(symbol)     — Alpha Vantage news + sentiment

NSE symbol mapping:
  Most Indian large caps are on Alpha Vantage as BSE listings: 
  e.g. RELIANCE → RELIANCE.BSE, TCS → TCS.BSE
  US stocks work directly: AAPL, TSLA, NVDA
"""
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Union
import requests
import pandas as pd
from loguru import logger

# ── Alpha Vantage config ───────────────────────────────────────────────────────
try:
    from config.settings import settings
    AV_API_KEY = getattr(settings, "ALPHA_VANTAGE_KEY", "54KAAF9J04C3BKHU")
except Exception:
    AV_API_KEY = "54KAAF9J04C3BKHU"
AV_BASE_URL  = "https://www.alphavantage.co/query"
CACHE_DIR    = Path("data/cache/av")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Cache TTL per function type
CACHE_TTL = {
    "OVERVIEW":          timedelta(hours=24),
    "INCOME_STATEMENT":  timedelta(hours=24),
    "BALANCE_SHEET":     timedelta(hours=24),
    "CASH_FLOW":         timedelta(hours=24),
    "EARNINGS":          timedelta(hours=12),
    "NEWS_SENTIMENT":    timedelta(hours=2),
    "TIME_SERIES_DAILY": timedelta(hours=6),
    "GLOBAL_QUOTE":      timedelta(minutes=15),
}

# NSE → Alpha Vantage symbol map for Indian stocks
NSE_TO_AV = {
    # Large caps — BSE listing
    "RELIANCE":    "RELIANCE.BSE",
    "TCS":         "TCS.BSE",
    "HDFCBANK":    "HDFCBANK.BSE",
    "INFY":        "INFY.BSE",
    "ICICIBANK":   "ICICIBANK.BSE",
    "SBIN":        "SBIN.BSE",
    "WIPRO":       "WIPRO.BSE",
    "AXISBANK":    "AXISBANK.BSE",
    "KOTAKBANK":   "KOTAKBANK.BSE",
    "LT":          "LT.BSE",
    "BAJFINANCE":  "BAJFINANCE.BSE",
    "MARUTI":      "MARUTI.BSE",
    "SUNPHARMA":   "SUNPHARMA.BSE",
    "BHARTIARTL":  "BHARTIARTL.BSE",
    "TATASTEEL":   "TATASTEEL.BSE",
    "JSWSTEEL":    "JSWSTEEL.BSE",
    "ONGC":        "ONGC.BSE",
    "NTPC":        "NTPC.BSE",
    "ASIANPAINT":  "ASIANPAINT.BSE",
    "DRREDDY":     "DRREDDY.BSE",
    "BAJAJFINSV":  "BAJAJFINSV.BSE",
    "ULTRACEMCO":  "ULTRACEMCO.BSE",
    "COALINDIA":   "COALINDIA.BSE",
    "POWERGRID":   "POWERGRID.BSE",
    "HINDALCO":    "HINDALCO.BSE",
}

# ── Cache helpers ──────────────────────────────────────────────────────────────

def _cache_key(function: str, symbol: str) -> Path:
    safe = symbol.replace(".", "_").replace("/", "_")
    return CACHE_DIR / f"{function}_{safe}.json"


def _cache_valid(path: Path, function: str) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    ttl   = CACHE_TTL.get(function, timedelta(hours=24))
    return datetime.now() - mtime < ttl


def _save_cache(path: Path, data: dict):
    try:
        path.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logger.debug(f"AV cache save failed: {e}")


def _load_cache(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


# ── Core API caller ────────────────────────────────────────────────────────────

def _call_av(function: str, symbol: str, extra_params: dict = None, force_refresh: bool = False) -> Optional[dict]:
    """
    Single Alpha Vantage API call with caching.
    Returns raw JSON dict or None on failure.
    """
    av_symbol = NSE_TO_AV.get(symbol, symbol)
    cache_path = _cache_key(function, av_symbol)

    if not force_refresh and _cache_valid(cache_path, function):
        data = _load_cache(cache_path)
        if data:
            logger.debug(f"AV cache hit: {function}/{av_symbol}")
            return data

    params = {
        "function": function,
        "symbol":   av_symbol,
        "apikey":   AV_API_KEY,
    }
    if extra_params:
        params.update(extra_params)

    try:
        logger.info(f"AV API call: {function}/{av_symbol}")
        resp = requests.get(AV_BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        # Detect rate limit or bad response
        if "Note" in data:
            logger.warning(f"AV rate limit hit: {data['Note'][:80]}")
            return None
        if "Information" in data:
            logger.warning(f"AV info message: {data['Information'][:80]}")
            return None
        if "Error Message" in data:
            logger.warning(f"AV error: {data['Error Message'][:80]}")
            return None

        _save_cache(cache_path, data)
        time.sleep(0.5)  # be polite to free tier
        return data

    except Exception as e:
        logger.error(f"AV API call failed {function}/{av_symbol}: {e}")
        return None


# ── Bloomberg DES equivalent: Company Overview ─────────────────────────────────

def get_overview(symbol: str) -> dict:
    """
    Bloomberg DES equivalent.
    Returns company passport: sector, market cap, P/E, description, etc.

    Returns flat dict with clean keys, empty dict on failure.
    """
    data = _call_av("OVERVIEW", symbol)
    if not data:
        return {}

    def safe_float(v):
        try:
            return float(v) if v and v not in ("None", "-", "") else None
        except (ValueError, TypeError):
            return None

    def safe_int(v):
        try:
            return int(float(v)) if v and v not in ("None", "-", "") else None
        except (ValueError, TypeError):
            return None

    return {
        "symbol":            data.get("Symbol", symbol),
        "name":              data.get("Name", ""),
        "description":       data.get("Description", ""),
        "exchange":          data.get("Exchange", ""),
        "currency":          data.get("Currency", ""),
        "country":           data.get("Country", ""),
        "sector":            data.get("Sector", ""),
        "industry":          data.get("Industry", ""),
        "address":           data.get("Address", ""),
        "fiscal_year_end":   data.get("FiscalYearEnd", ""),
        "latest_quarter":    data.get("LatestQuarter", ""),
        # Valuation
        "market_cap":        safe_int(data.get("MarketCapitalization")),
        "pe_ratio":          safe_float(data.get("PERatio")),
        "forward_pe":        safe_float(data.get("ForwardPE")),
        "peg_ratio":         safe_float(data.get("PEGRatio")),
        "price_to_book":     safe_float(data.get("PriceToBookRatio")),
        "price_to_sales":    safe_float(data.get("PriceToSalesRatioTTM")),
        "ev_to_revenue":     safe_float(data.get("EVToRevenue")),
        "ev_to_ebitda":      safe_float(data.get("EVToEBITDA")),
        # Per share metrics
        "eps":               safe_float(data.get("EPS")),
        "book_value":        safe_float(data.get("BookValue")),
        "dividend_per_share":safe_float(data.get("DividendPerShare")),
        "dividend_yield":    safe_float(data.get("DividendYield")),
        "revenue_per_share": safe_float(data.get("RevenuePerShareTTM")),
        # Profitability
        "profit_margin":     safe_float(data.get("ProfitMargin")),
        "operating_margin":  safe_float(data.get("OperatingMarginTTM")),
        "return_on_equity":  safe_float(data.get("ReturnOnEquityTTM")),
        "return_on_assets":  safe_float(data.get("ReturnOnAssetsTTM")),
        # Growth
        "revenue_growth_yoy":safe_float(data.get("RevenueGrowthYOY")),
        "earnings_growth_yoy":safe_float(data.get("EarningsGrowthYOY")),
        "quarterly_earnings_growth": safe_float(data.get("QuarterlyEarningsGrowthYOY")),
        "quarterly_revenue_growth":  safe_float(data.get("QuarterlyRevenueGrowthYOY")),
        # Scale
        "revenue_ttm":       safe_int(data.get("RevenueTTM")),
        "gross_profit_ttm":  safe_int(data.get("GrossProfitTTM")),
        "ebitda":            safe_int(data.get("EBITDA")),
        "shares_outstanding":safe_int(data.get("SharesOutstanding")),
        "shares_float":      safe_int(data.get("SharesFloat")),
        # Price levels
        "week_52_high":      safe_float(data.get("52WeekHigh")),
        "week_52_low":       safe_float(data.get("52WeekLow")),
        "moving_avg_50":     safe_float(data.get("50DayMovingAverage")),
        "moving_avg_200":    safe_float(data.get("200DayMovingAverage")),
        "beta":              safe_float(data.get("Beta")),
        # Analyst
        "analyst_target":    safe_float(data.get("AnalystTargetPrice")),
        "analyst_rating_strong_buy": safe_int(data.get("AnalystRatingStrongBuy")),
        "analyst_rating_buy":        safe_int(data.get("AnalystRatingBuy")),
        "analyst_rating_hold":       safe_int(data.get("AnalystRatingHold")),
        "analyst_rating_sell":       safe_int(data.get("AnalystRatingSell")),
        "analyst_rating_strong_sell":safe_int(data.get("AnalystRatingStrongSell")),
    }


# ── Bloomberg FA equivalent: Financial Statements ─────────────────────────────

def get_income_statement(symbol: str, annual: bool = True) -> pd.DataFrame:
    """
    Bloomberg FA equivalent — Income Statement.
    Returns DataFrame with columns: fiscalDateEnding, revenue, grossProfit,
    operatingIncome, netIncome, eps, ebitda
    """
    data = _call_av("INCOME_STATEMENT", symbol)
    if not data:
        return pd.DataFrame()

    key     = "annualReports" if annual else "quarterlyReports"
    reports = data.get(key, [])
    if not reports:
        return pd.DataFrame()

    cols_to_keep = [
        "fiscalDateEnding", "reportedCurrency",
        "totalRevenue", "grossProfit", "operatingIncome",
        "netIncome", "ebitda", "eps", "epsDiluted",
        "researchAndDevelopment", "operatingExpenses",
        "incomeBeforeTax", "incomeTaxExpense",
        "interestExpense", "depreciation",
        "sellingGeneralAndAdministrative",
    ]
    df = pd.DataFrame(reports)
    keep = [c for c in cols_to_keep if c in df.columns]
    df   = df[keep].copy()

    # Convert numeric columns
    for col in df.columns:
        if col not in ("fiscalDateEnding", "reportedCurrency"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["fiscalDateEnding"] = pd.to_datetime(df["fiscalDateEnding"], errors="coerce")
    df = df.sort_values("fiscalDateEnding", ascending=False).reset_index(drop=True)
    return df


def get_balance_sheet(symbol: str, annual: bool = True) -> pd.DataFrame:
    """Bloomberg FA equivalent — Balance Sheet."""
    data = _call_av("BALANCE_SHEET", symbol)
    if not data:
        return pd.DataFrame()

    key     = "annualReports" if annual else "quarterlyReports"
    reports = data.get(key, [])
    if not reports:
        return pd.DataFrame()

    df = pd.DataFrame(reports)
    for col in df.columns:
        if col not in ("fiscalDateEnding", "reportedCurrency"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["fiscalDateEnding"] = pd.to_datetime(df["fiscalDateEnding"], errors="coerce")
    df = df.sort_values("fiscalDateEnding", ascending=False).reset_index(drop=True)
    return df


def get_cash_flow(symbol: str, annual: bool = True) -> pd.DataFrame:
    """Bloomberg FA equivalent — Cash Flow Statement."""
    data = _call_av("CASH_FLOW", symbol)
    if not data:
        return pd.DataFrame()

    key     = "annualReports" if annual else "quarterlyReports"
    reports = data.get(key, [])
    if not reports:
        return pd.DataFrame()

    cols_to_keep = [
        "fiscalDateEnding", "reportedCurrency",
        "operatingCashflow", "capitalExpenditures",
        "freeCashFlow", "cashflowFromInvestment",
        "cashflowFromFinancing", "dividendPayout",
        "netBorrowings", "changeInCash",
        "netIncomeFromContinuingOperations",
        "depreciationDepletionAndAmortization",
    ]
    df = pd.DataFrame(reports)
    keep = [c for c in cols_to_keep if c in df.columns]
    df   = df[keep].copy()

    # Compute free cash flow if not present
    if "freeCashFlow" not in df.columns and "operatingCashflow" in df.columns:
        df["freeCashFlow"] = (
            pd.to_numeric(df["operatingCashflow"], errors="coerce")
            - pd.to_numeric(df.get("capitalExpenditures", 0), errors="coerce")
        )

    for col in df.columns:
        if col not in ("fiscalDateEnding", "reportedCurrency"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["fiscalDateEnding"] = pd.to_datetime(df["fiscalDateEnding"], errors="coerce")
    df = df.sort_values("fiscalDateEnding", ascending=False).reset_index(drop=True)
    return df


# ── Bloomberg EEG equivalent: Earnings History + Estimates ────────────────────

def get_earnings(symbol: str) -> dict:
    """
    Bloomberg EEG equivalent — Earnings estimates vs actuals.
    Returns dict with 'annual' and 'quarterly' DataFrames.
    """
    data = _call_av("EARNINGS", symbol)
    if not data:
        return {"annual": pd.DataFrame(), "quarterly": pd.DataFrame()}

    def parse_earnings(reports: list) -> pd.DataFrame:
        if not reports:
            return pd.DataFrame()
        df = pd.DataFrame(reports)
        for col in ["reportedEPS", "estimatedEPS", "surprise", "surprisePercentage"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "fiscalDateEnding" in df.columns:
            df["fiscalDateEnding"] = pd.to_datetime(df["fiscalDateEnding"], errors="coerce")
        if "reportedDate" in df.columns:
            df["reportedDate"] = pd.to_datetime(df["reportedDate"], errors="coerce")
        return df.sort_values("fiscalDateEnding", ascending=False).reset_index(drop=True)

    return {
        "annual":    parse_earnings(data.get("annualEarnings", [])),
        "quarterly": parse_earnings(data.get("quarterlyEarnings", [])),
    }


# ── Bloomberg ANR equivalent: Analyst Sentiment via AV News ───────────────────

def get_news_sentiment(symbol: str, limit: int = 20) -> pd.DataFrame:
    """
    Alpha Vantage NEWS_SENTIMENT endpoint.
    Returns DataFrame with: title, source, time, overall_sentiment_score,
    overall_sentiment_label, ticker_sentiment_score, ticker_sentiment_label
    """
    av_symbol = NSE_TO_AV.get(symbol, symbol)
    data = _call_av(
        "NEWS_SENTIMENT",
        symbol,
        extra_params={"tickers": av_symbol, "limit": str(limit)},
    )
    if not data or "feed" not in data:
        return pd.DataFrame()

    rows = []
    for item in data["feed"]:
        # Find ticker-specific sentiment if available
        ticker_score = None
        ticker_label = None
        for ts in item.get("ticker_sentiment", []):
            if ts.get("ticker", "").upper() in (av_symbol.upper(), symbol.upper()):
                ticker_score = float(ts.get("ticker_sentiment_score", 0))
                ticker_label = ts.get("ticker_sentiment_label", "")
                break

        rows.append({
            "title":                    item.get("title", ""),
            "url":                      item.get("url", ""),
            "source":                   item.get("source", ""),
            "time_published":           item.get("time_published", ""),
            "summary":                  item.get("summary", "")[:300],
            "overall_sentiment_score":  float(item.get("overall_sentiment_score", 0)),
            "overall_sentiment_label":  item.get("overall_sentiment_label", ""),
            "ticker_sentiment_score":   ticker_score,
            "ticker_sentiment_label":   ticker_label,
        })

    df = pd.DataFrame(rows)
    if not df.empty and "time_published" in df.columns:
        df["time_published"] = pd.to_datetime(df["time_published"], format="%Y%m%dT%H%M%S", errors="coerce")
        df = df.sort_values("time_published", ascending=False).reset_index(drop=True)
    return df


# ── Bloomberg EQRV equivalent: Peer Comparison ────────────────────────────────

def get_peer_overview(symbols: list[str]) -> pd.DataFrame:
    """
    Bloomberg EQRV equivalent — comparative valuation table.
    Batch fetches OVERVIEW for each symbol and returns comparison DataFrame.

    Columns: symbol, name, sector, pe_ratio, forward_pe, price_to_book,
             ev_to_ebitda, profit_margin, return_on_equity, market_cap,
             dividend_yield, week_52_high, week_52_low, beta
    """
    rows = []
    for sym in symbols:
        try:
            ov = get_overview(sym)
            if not ov:
                continue
            rows.append({
                "symbol":          ov.get("symbol", sym),
                "name":            ov.get("name", sym)[:25],
                "sector":          ov.get("sector", ""),
                "market_cap_cr":   round(ov["market_cap"] / 1e7, 0) if ov.get("market_cap") else None,
                "pe_ratio":        ov.get("pe_ratio"),
                "forward_pe":      ov.get("forward_pe"),
                "price_to_book":   ov.get("price_to_book"),
                "ev_to_ebitda":    ov.get("ev_to_ebitda"),
                "profit_margin_%": round(ov["profit_margin"] * 100, 1) if ov.get("profit_margin") else None,
                "roe_%":           round(ov["return_on_equity"] * 100, 1) if ov.get("return_on_equity") else None,
                "dividend_yield_%":round(float(ov["dividend_yield"]) * 100, 2) if ov.get("dividend_yield") else None,
                "beta":            ov.get("beta"),
                "52w_high":        ov.get("week_52_high"),
                "52w_low":         ov.get("week_52_low"),
                "analyst_target":  ov.get("analyst_target"),
            })
            time.sleep(0.3)  # rate limit protection
        except Exception as e:
            logger.warning(f"Peer overview failed for {sym}: {e}")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Add vs-median columns for key metrics
    for metric in ["pe_ratio", "price_to_book", "ev_to_ebitda", "profit_margin_%"]:
        if metric in df.columns:
            median = df[metric].median()
            if pd.notna(median) and median != 0:
                df[f"{metric}_vs_med"] = ((df[metric] - median) / abs(median) * 100).round(1)

    return df


# ── GLOBAL QUOTE: Real-time price from Alpha Vantage ─────────────────────────

def get_global_quote(symbol: str) -> dict:
    """
    Real-time quote via Alpha Vantage GLOBAL_QUOTE.
    Returns: price, change, change_pct, volume, prev_close, open, high, low
    """
    data = _call_av("GLOBAL_QUOTE", symbol)
    if not data or "Global Quote" not in data:
        return {}

    q = data["Global Quote"]

    def sf(k):
        try:
            return float(q.get(k, 0))
        except (ValueError, TypeError):
            return None

    return {
        "price":       sf("05. price"),
        "change":      sf("09. change"),
        "change_pct":  sf("10. change percent"),
        "volume":      sf("06. volume"),
        "prev_close":  sf("08. previous close"),
        "open":        sf("02. open"),
        "high":        sf("03. high"),
        "low":         sf("04. low"),
        "latest_day":  q.get("07. latest trading day", ""),
    }


# ── Convenience: get_company_summary for DES page ─────────────────────────────

def get_company_summary(symbol: str) -> dict:
    """
    One-call convenience for Company Intelligence page.
    Returns overview + latest earnings + analyst consensus.
    """
    overview = get_overview(symbol)
    earnings = get_earnings(symbol)
    news     = get_news_sentiment(symbol, limit=5)

    # Analyst consensus from overview
    total_analysts = sum(filter(None, [
        overview.get("analyst_rating_strong_buy"),
        overview.get("analyst_rating_buy"),
        overview.get("analyst_rating_hold"),
        overview.get("analyst_rating_sell"),
        overview.get("analyst_rating_strong_sell"),
    ]))

    buys  = (overview.get("analyst_rating_strong_buy") or 0) + (overview.get("analyst_rating_buy") or 0)
    holds = overview.get("analyst_rating_hold") or 0
    sells = (overview.get("analyst_rating_sell") or 0) + (overview.get("analyst_rating_strong_sell") or 0)

    if total_analysts > 0:
        buy_pct  = round(buys  / total_analysts * 100)
        hold_pct = round(holds / total_analysts * 100)
        sell_pct = round(sells / total_analysts * 100)
        if buy_pct >= 60:
            consensus = "STRONG BUY"
        elif buy_pct >= 45:
            consensus = "BUY"
        elif sell_pct >= 45:
            consensus = "SELL"
        elif sell_pct >= 30:
            consensus = "HOLD/WEAK"
        else:
            consensus = "HOLD"
    else:
        buy_pct = hold_pct = sell_pct = 0
        consensus = "N/A"

    # Latest earnings surprise
    last_surprise = None
    last_surprise_pct = None
    if not earnings["quarterly"].empty:
        row = earnings["quarterly"].iloc[0]
        last_surprise     = row.get("surprise")
        last_surprise_pct = row.get("surprisePercentage")

    # News sentiment average
    avg_news_score = None
    news_label     = None
    if not news.empty and "overall_sentiment_score" in news.columns:
        avg_news_score = round(float(news["overall_sentiment_score"].mean()), 3)
        if avg_news_score > 0.15:
            news_label = "Bullish"
        elif avg_news_score < -0.15:
            news_label = "Bearish"
        else:
            news_label = "Neutral"

    return {
        "overview":           overview,
        "earnings_annual":    earnings["annual"],
        "earnings_quarterly": earnings["quarterly"],
        "news_recent":        news,
        "analyst_consensus":  consensus,
        "analyst_buy_pct":    buy_pct,
        "analyst_hold_pct":   hold_pct,
        "analyst_sell_pct":   sell_pct,
        "analyst_total":      total_analysts,
        "last_earnings_surprise":     last_surprise,
        "last_earnings_surprise_pct": last_surprise_pct,
        "news_avg_sentiment": avg_news_score,
        "news_label":         news_label,
    }


# ── Rate limit tracker ────────────────────────────────────────────────────────

def get_rate_limit_status() -> dict:
    """Check how many API calls have been made today from cache file count."""
    today = datetime.now().strftime("%Y%m%d")
    tracker_path = CACHE_DIR / f"rate_tracker_{today}.json"

    if tracker_path.exists():
        try:
            data = json.loads(tracker_path.read_text())
            calls = data.get("calls", 0)
        except Exception:
            calls = 0
    else:
        # Count calls from today's cache modifications
        calls = sum(
            1 for f in CACHE_DIR.glob("*.json")
            if datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y%m%d") == today
            and "rate_tracker" not in f.name
        )

    return {
        "calls_today":    calls,
        "calls_remaining": max(0, 25 - calls),
        "daily_limit":    25,
        "pct_used":       round(calls / 25 * 100),
    }