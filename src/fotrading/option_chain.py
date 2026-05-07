# trading-agent/src/fotrading/option_chain.py
"""
Option Chain Fetcher — Angel One SmartAPI

Fetches live option chain data and computes Greeks using Black-Scholes.

Angel One provides:
  - LTP, bid, ask for each strike
  - OI and volume
  - IV (implied volatility) — sometimes, not always reliable

We compute Greeks ourselves using Black-Scholes for reliability.

Cache: option chains are cached for 60 seconds to avoid hammering the API.
"""
import math
from datetime import date, datetime, timedelta, timezone
from typing import Optional
from loguru import logger

from src.fotrading.fo_models import (
    OptionChain, OptionStrike, OptionType, FuturesContract,
    InstrumentType, LOT_SIZES, STRIKE_INTERVALS, Greeks,
)


# ── Risk-free rate (RBI repo rate approximately) ──────────────────────────────
RISK_FREE_RATE = 0.065   # 6.5% — update periodically


class OptionChainFetcher:
    """
    Fetches and parses option chains from Angel One.
    Falls back to synthetic chain (from spot + Black-Scholes) if API fails.
    """

    CACHE_TTL = 60   # seconds

    def __init__(self, smart_api=None):
        self._api   = smart_api   # SmartConnect instance (optional)
        self._cache: dict[str, tuple[datetime, OptionChain]] = {}

    def get_chain(
        self,
        underlying: str,
        expiry:     date = None,
        n_strikes:  int  = 10,    # strikes each side of ATM
    ) -> Optional[OptionChain]:
        """
        Fetch option chain for underlying.
        Returns cached result if fresh enough.
        """
        cache_key = f"{underlying}_{expiry}"
        cached    = self._cache.get(cache_key)
        if cached:
            cached_at, chain = cached
            if (datetime.now(timezone.utc) - cached_at).total_seconds() < self.CACHE_TTL:
                return chain

        chain = self._fetch_from_api(underlying, expiry, n_strikes)
        if chain:
            self._cache[cache_key] = (datetime.now(timezone.utc), chain)
        return chain

    def get_futures(
        self,
        underlying: str,
        expiry:     date = None,
    ) -> Optional[FuturesContract]:
        """Fetch nearest futures contract for underlying."""
        try:
            if self._api and self._is_connected():
                return self._fetch_futures_from_api(underlying, expiry)
        except Exception as e:
            logger.warning(f"Futures API fetch failed: {e}")

        return self._synthetic_futures(underlying, expiry)

    # ── Angel One API fetchers ─────────────────────────────────────────────────

    def _fetch_from_api(
        self,
        underlying: str,
        expiry: date,
        n_strikes: int,
    ) -> Optional[OptionChain]:
        """Fetch from Angel One getCandleData / option chain endpoint."""
        try:
            if self._api and self._is_connected():
                # Angel One option chain API
                expiry_str = expiry.strftime("%d%b%Y").upper() if expiry else self._nearest_expiry_str(underlying)
                resp = self._api.option_chain(
                    exch_seg   = "NFO" if underlying in ("NIFTY50", "BANKNIFTY", "NIFTY") else "NFO",
                    symbol     = self._underlying_to_angel(underlying),
                    expiry_date= expiry_str,
                )
                if resp and resp.get("status") and resp.get("data"):
                    return self._parse_api_chain(underlying, expiry, resp["data"], n_strikes)
        except Exception as e:
            logger.warning(f"Option chain API failed for {underlying}: {e}")

        # Fallback: synthetic chain
        return self._synthetic_chain(underlying, expiry, n_strikes)

    def _fetch_futures_from_api(self, underlying: str, expiry: date) -> Optional[FuturesContract]:
        try:
            from src.streaming.price_store import price_store
            from src.streaming.mcx_token_manager import mcx_token_manager

            sym    = self._underlying_to_angel(underlying)
            result = mcx_token_manager.get_token(underlying)
            if not result:
                return None
            token = result[1]

            # Get LTP for futures
            resp = self._api.ltpData("NFO", sym + "FUT", token)
            if resp and resp.get("status"):
                ltp = float(resp["data"]["ltp"])
                return FuturesContract(
                    symbol      = f"{sym}FUT",
                    token       = token,
                    underlying  = underlying,
                    expiry      = expiry or self._nearest_expiry(underlying),
                    instrument  = InstrumentType.FUTIDX,
                    lot_size    = LOT_SIZES.get(underlying, 1),
                    exchange    = "NFO",
                    ltp         = ltp,
                )
        except Exception as e:
            logger.warning(f"Futures API fetch failed: {e}")
        return None

    def _parse_api_chain(
        self,
        underlying: str,
        expiry: date,
        data: list,
        n_strikes: int,
    ) -> OptionChain:
        """Parse Angel One option chain response."""
        from src.streaming.price_store import price_store
        spot = price_store.get(underlying) or 0

        strikes = []
        lot_sz  = LOT_SIZES.get(underlying, 50)
        exp     = expiry or self._nearest_expiry(underlying)
        dte     = max(1, (exp - date.today()).days)

        for item in data:
            for opt_type in (OptionType.CALL, OptionType.PUT):
                key  = "CE" if opt_type == OptionType.CALL else "PE"
                info = item.get(key, {})
                if not info:
                    continue

                strike_price = float(item.get("strikePrice", 0))
                ltp          = float(info.get("lastPrice", 0))
                iv_raw       = float(info.get("impliedVolatility", 0)) / 100

                greeks = self._compute_greeks(
                    spot         = spot,
                    strike       = strike_price,
                    dte          = dte,
                    iv           = iv_raw if iv_raw > 0 else 0.15,
                    option_type  = opt_type,
                    risk_free    = RISK_FREE_RATE,
                )

                strikes.append(OptionStrike(
                    symbol      = f"{underlying}{exp.strftime('%d%b%Y').upper()}{strike_price:.0f}{key}",
                    token       = str(info.get("token", "")),
                    underlying  = underlying,
                    expiry      = exp,
                    strike      = strike_price,
                    option_type = opt_type,
                    lot_size    = lot_sz,
                    ltp         = ltp,
                    bid         = float(info.get("bidPrice", 0)),
                    ask         = float(info.get("askPrice", 0)),
                    oi          = int(info.get("openInterest", 0)),
                    oi_change   = int(info.get("changeinOpenInterest", 0)),
                    volume      = int(info.get("totalTradedVolume", 0)),
                    iv          = iv_raw,
                    greeks      = greeks,
                ))

        atm = self._nearest_strike(spot, underlying)
        return OptionChain(
            underlying  = underlying,
            expiry      = exp,
            spot_price  = spot,
            atm_strike  = atm,
            strikes     = strikes,
        )

    # ── Synthetic chain (fallback when API unavailable) ───────────────────────

    def _synthetic_chain(
        self,
        underlying: str,
        expiry: date,
        n_strikes: int,
    ) -> OptionChain:
        """
        Build option chain using Black-Scholes with assumed IV.
        Used when Angel One API is unavailable.
        Prices won't be exact but Greeks and structure will be correct.
        """
        from src.streaming.price_store import price_store
        spot = price_store.get(underlying) or self._yfinance_spot(underlying)

        if not spot:
            logger.warning(f"Cannot build synthetic chain for {underlying}: no spot price")
            return None

        exp    = expiry or self._nearest_expiry(underlying)
        dte    = max(1, (exp - date.today()).days)
        lot_sz = LOT_SIZES.get(underlying, 50)
        intv   = STRIKE_INTERVALS.get(underlying, 50)
        atm    = self._nearest_strike(spot, underlying)

        # Historical IV estimates for Indian indices (conservative)
        iv_est = {
            "NIFTY50":   0.14,
            "BANKNIFTY": 0.18,
            "SENSEX":    0.14,
            "GOLD":      0.12,
            "CRUDEOIL":  0.28,
        }.get(underlying, 0.20)

        strikes = []
        for i in range(-n_strikes, n_strikes + 1):
            strike_price = atm + i * intv

            for opt_type in (OptionType.CALL, OptionType.PUT):
                greeks = self._compute_greeks(
                    spot        = spot,
                    strike      = strike_price,
                    dte         = dte,
                    iv          = iv_est,
                    option_type = opt_type,
                    risk_free   = RISK_FREE_RATE,
                )
                ltp = self._bs_price(spot, strike_price, dte, iv_est,
                                     RISK_FREE_RATE, opt_type)
                key = "CE" if opt_type == OptionType.CALL else "PE"
                strikes.append(OptionStrike(
                    symbol      = f"{underlying}{exp.strftime('%d%b%Y').upper()}{strike_price:.0f}{key}",
                    token       = "",
                    underlying  = underlying,
                    expiry      = exp,
                    strike      = strike_price,
                    option_type = opt_type,
                    lot_size    = lot_sz,
                    ltp         = round(ltp, 2),
                    iv          = iv_est,
                    greeks      = greeks,
                ))

        return OptionChain(
            underlying = underlying,
            expiry     = exp,
            spot_price = spot,
            atm_strike = atm,
            strikes    = strikes,
        )

    def _synthetic_futures(self, underlying: str, expiry: date) -> FuturesContract:
        from src.streaming.price_store import price_store
        spot   = price_store.get(underlying) or self._yfinance_spot(underlying) or 0
        exp    = expiry or self._nearest_expiry(underlying)
        dte    = max(1, (exp - date.today()).days)
        # Futures fair value = spot * e^(r*t)
        fair   = spot * math.exp(RISK_FREE_RATE * dte / 365)
        exch   = "MCX" if underlying in ("GOLD","SILVER","CRUDEOIL","COPPER","NATURALGAS") else "NFO"
        inst   = InstrumentType.FUTCOM if exch == "MCX" else InstrumentType.FUTIDX

        return FuturesContract(
            symbol     = f"{underlying}FUT",
            token      = "",
            underlying = underlying,
            expiry     = exp,
            instrument = inst,
            lot_size   = LOT_SIZES.get(underlying, 1),
            exchange   = exch,
            ltp        = round(fair, 2),
            prev_close = spot,
        )

    # ── Black-Scholes ─────────────────────────────────────────────────────────

    def _bs_price(
        self,
        S: float,     # spot
        K: float,     # strike
        dte: int,     # days to expiry
        iv: float,    # implied volatility (decimal)
        r: float,     # risk-free rate (decimal)
        opt_type: OptionType,
    ) -> float:
        """Black-Scholes option price."""
        T = dte / 365.0
        if T <= 0 or S <= 0 or K <= 0 or iv <= 0:
            return max(0.0,
                       S - K if opt_type == OptionType.CALL else K - S)
        d1 = (math.log(S / K) + (r + 0.5 * iv**2) * T) / (iv * math.sqrt(T))
        d2 = d1 - iv * math.sqrt(T)
        if opt_type == OptionType.CALL:
            return S * self._N(d1) - K * math.exp(-r * T) * self._N(d2)
        else:
            return K * math.exp(-r * T) * self._N(-d2) - S * self._N(-d1)

    def _compute_greeks(
        self,
        spot: float,
        strike: float,
        dte: int,
        iv: float,
        option_type: OptionType,
        risk_free: float,
    ) -> Greeks:
        """Compute all 4 Greeks using Black-Scholes."""
        T = max(dte, 1) / 365.0
        if spot <= 0 or strike <= 0 or iv <= 0:
            return Greeks()

        try:
            d1 = (math.log(spot / strike) + (risk_free + 0.5 * iv**2) * T) / (iv * math.sqrt(T))
            d2 = d1 - iv * math.sqrt(T)

            pdf_d1 = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)

            if option_type == OptionType.CALL:
                delta = self._N(d1)
                theta = (
                    -(spot * pdf_d1 * iv) / (2 * math.sqrt(T))
                    - risk_free * strike * math.exp(-risk_free * T) * self._N(d2)
                ) / 365
            else:
                delta = self._N(d1) - 1
                theta = (
                    -(spot * pdf_d1 * iv) / (2 * math.sqrt(T))
                    + risk_free * strike * math.exp(-risk_free * T) * self._N(-d2)
                ) / 365

            gamma = pdf_d1 / (spot * iv * math.sqrt(T))
            vega  = spot * pdf_d1 * math.sqrt(T) / 100   # per 1% IV change

            return Greeks(
                delta  = round(delta, 4),
                gamma  = round(gamma, 6),
                theta  = round(theta, 4),
                vega   = round(vega,  4),
                iv     = round(iv,    4),
                iv_pct = round(iv * 100, 2),
            )
        except Exception:
            return Greeks()

    def _N(self, x: float) -> float:
        """Standard normal CDF."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _nearest_strike(self, spot: float, underlying: str) -> float:
        intv = STRIKE_INTERVALS.get(underlying, 50)
        return round(spot / intv) * intv

    def _nearest_expiry(self, underlying: str) -> date:
        """Find nearest Thursday (weekly expiry for Nifty/BankNifty)."""
        today  = date.today()
        # Find next Thursday
        days_ahead = (3 - today.weekday()) % 7   # 3 = Thursday
        if days_ahead == 0:
            days_ahead = 7
        return today + timedelta(days=days_ahead)

    def _nearest_expiry_str(self, underlying: str) -> str:
        return self._nearest_expiry(underlying).strftime("%d%b%Y").upper()

    def _underlying_to_angel(self, underlying: str) -> str:
        mapping = {
            "NIFTY50":   "NIFTY",
            "BANKNIFTY": "BANKNIFTY",
            "SENSEX":    "SENSEX",
        }
        return mapping.get(underlying, underlying)

    def _yfinance_spot(self, underlying: str) -> float:
        try:
            import yfinance as yf
            from src.data.models import ALL_SYMBOLS
            info   = ALL_SYMBOLS.get(underlying)
            yf_sym = info.symbol if info else f"{underlying}.NS"
            return float(yf.Ticker(yf_sym).fast_info.last_price or 0)
        except Exception:
            return 0.0

    def _is_connected(self) -> bool:
        return self._api is not None


# Module-level singleton
option_chain_fetcher = OptionChainFetcher()