# trading-agent/src/fotrading/fo_engine.py
"""
F&O Signal Engine

Generates actionable F&O trade recommendations by combining:
  1. Underlying bias (from ensemble model / rule-based)
  2. Options chain analysis (IV rank, PCR, max pain)
  3. Greeks filter (avoid high theta decay, prefer liquid strikes)
  4. Regime filter (avoid buying options in low-IV regimes)

Output: FO_Signal with specific strike, entry, SL, target in premium terms.

Strategy logic:
  TRENDING (strong ADX, clear direction):
    → Buy ATM or slightly OTM options in trend direction
    → Avoid writing (unlimited risk for small capital)

  RANGING (low ADX, oscillating):
    → Look for RSI extremes → buy reversal options
    → Tight premium-based SL (option loses value fast)

  VOLATILE (high ATR):
    → Avoid options buying (theta + IV crush risk)
    → Consider futures for directional plays if IV is low

  All F&O:
    → Square off intraday by 3:15 PM (avoid overnight theta)
    → Auto-exit if premium loses 50% (hard rule)
    → Never let options expire worthless (exit by DTE=1)
"""
from dataclasses import dataclass, field
from datetime import date, time, datetime
from typing import Optional
from loguru import logger

from src.fotrading.fo_models import (
    OptionChain, OptionStrike, OptionType, FuturesContract,
    InstrumentType, PositionType, LOT_SIZES, Greeks,
)


@dataclass
class FOSignal:
    """Complete F&O trade recommendation."""
    # What to trade
    underlying:     str
    instrument:     str          # CALL / PUT / FUTURES
    direction:      str          # LONG / SHORT
    strike:         Optional[float]
    expiry:         date
    symbol:         str          # full trading symbol

    # Entry details
    entry_premium:  float        # recommended entry premium
    stop_loss_prem: float        # exit if premium falls to this
    target_premium: float        # exit if premium rises to this
    lot_size:       int
    recommended_lots: int        # how many lots given capital

    # Context
    underlying_bias: str         # BUY / SELL / NEUTRAL from model
    regime:         str          # TRENDING_UP / RANGING / etc.
    confidence:     float        # 0-1
    iv_rank:        float        # 0-100, current IV vs 1-year range
    pcr:            float        # put/call ratio
    max_pain:       float        # max pain strike
    greeks:         Greeks = field(default_factory=Greeks)

    # Risk
    max_loss_per_lot: float = 0.0    # entry_premium * lot_size
    capital_required: float = 0.0    # total capital needed
    risk_reward:      float = 0.0

    # Reasoning
    reasons:        list[str] = field(default_factory=list)
    warnings:       list[str] = field(default_factory=list)

    def __str__(self):
        return (
            f"{self.direction} {self.instrument} | {self.underlying} "
            f"{self.strike} {self.expiry} | "
            f"Entry: ₹{self.entry_premium:.1f} SL: ₹{self.stop_loss_prem:.1f} "
            f"TGT: ₹{self.target_premium:.1f} | "
            f"Conf: {self.confidence:.0%}"
        )


class FOEngine:
    """
    Generates F&O trade signals by fusing technical bias + options analytics.
    """

    # Risk parameters
    MAX_PREMIUM_LOSS_PCT = 0.50     # exit if option loses 50% of entry premium
    MIN_DTE_TO_TRADE     = 2        # don't buy options with < 2 DTE
    MAX_DTE_TO_TRADE     = 30       # don't buy options with > 30 DTE
    MAX_SPREAD_PCT       = 0.05     # skip illiquid strikes (spread > 5%)
    MIN_OI               = 1000     # minimum open interest

    # IV rank thresholds
    IV_RANK_LOW    = 30    # IV rank below this → buy options (cheap)
    IV_RANK_HIGH   = 70    # IV rank above this → avoid buying (expensive)

    def analyse(
        self,
        underlying:   str,
        chain:        OptionChain,
        underlying_bias: str,
        regime:       str,
        confidence:   float,
        available_capital: float,
        futures:      FuturesContract = None,
        iv_history:   list[float] = None,
    ) -> Optional[FOSignal]:
        """
        Main entry point — generates best F&O signal given market context.

        Args:
            underlying:       e.g. "NIFTY50"
            chain:            live OptionChain
            underlying_bias:  "BUY", "SELL", "NEUTRAL" from ensemble/rule model
            regime:           "TRENDING_UP", "TRENDING_DOWN", "RANGING", "VOLATILE"
            confidence:       model confidence 0-1
            available_capital: capital available for this trade
            futures:          FuturesContract (if available)
            iv_history:       list of historical IVs for IV rank calc
        """
        if not chain:
            logger.warning(f"FOEngine: no option chain for {underlying}")
            return None

        # ── Pre-checks ────────────────────────────────────────────────────────
        dte = (chain.expiry - date.today()).days
        if dte < self.MIN_DTE_TO_TRADE:
            logger.info(f"FOEngine: {underlying} expiry too close ({dte} DTE) — skip")
            return None

        # ── Market analytics ──────────────────────────────────────────────────
        pcr       = chain.get_pcr()
        max_pain  = chain.get_max_pain()
        iv_rank   = self._compute_iv_rank(chain, iv_history)

        warnings = []

        # ── Strategy selection by regime ──────────────────────────────────────
        if regime == "VOLATILE":
            # High volatility — options are expensive, theta kills buyers
            if iv_rank > self.IV_RANK_HIGH:
                warnings.append(f"High IV rank ({iv_rank:.0f}) — options expensive, premium decay risk")
                # In volatile + high IV: consider futures instead
                if futures and confidence >= 0.60:
                    return self._futures_signal(
                        underlying, futures, underlying_bias,
                        regime, confidence, available_capital,
                        pcr, max_pain, iv_rank, warnings,
                    )
                return None  # no good trade

        if regime in ("TRENDING_UP", "TRENDING_DOWN"):
            # Clear trend — buy directional options
            return self._directional_options_signal(
                underlying, chain, underlying_bias, regime,
                confidence, available_capital, pcr, max_pain,
                iv_rank, dte, warnings,
            )

        if regime == "RANGING":
            # Range-bound — look for RSI extremes → reversal options
            return self._reversal_options_signal(
                underlying, chain, underlying_bias, regime,
                confidence, available_capital, pcr, max_pain,
                iv_rank, dte, warnings,
            )

        return None

    # ── Strategy implementations ──────────────────────────────────────────────

    def _directional_options_signal(
        self,
        underlying, chain, bias, regime, confidence,
        capital, pcr, max_pain, iv_rank, dte, warnings,
    ) -> Optional[FOSignal]:
        """Buy ATM or slightly OTM call/put in trend direction."""

        # Map bias to option type
        if "BUY" in bias:
            opt_type  = OptionType.CALL
            direction = "LONG"
            reasons   = ["Trend is UP — buying calls to participate"]
        elif "SELL" in bias:
            opt_type  = OptionType.PUT
            direction = "LONG"   # we BUY puts (not sell)
            reasons   = ["Trend is DOWN — buying puts to participate"]
        else:
            return None

        if confidence < 0.50:
            warnings.append(f"Low confidence ({confidence:.0%}) — skipping F&O signal")
            return None

        # Pick strike: ATM for strong trend, 1% OTM for moderate
        if "STRONG" in bias:
            strike_obj = (chain.get_atm_call() if opt_type == OptionType.CALL
                          else chain.get_atm_put())
            reasons.append("Strong signal → ATM option (highest delta)")
        else:
            otm_pct    = 0.005 if dte > 7 else 0.002   # closer strike near expiry
            strike_obj = (chain.get_otm_call(otm_pct) if opt_type == OptionType.CALL
                          else chain.get_otm_put(otm_pct))
            reasons.append(f"Moderate signal → ~{otm_pct*100:.1f}% OTM option")

        if not strike_obj:
            return None

        # Liquidity check
        if not strike_obj.is_liquid:
            warnings.append(f"Strike {strike_obj.strike} has low liquidity — entry may slip")

        # IV rank warning
        if iv_rank > self.IV_RANK_HIGH:
            warnings.append(f"IV rank {iv_rank:.0f} — options expensive, use smaller size")
        elif iv_rank < self.IV_RANK_LOW:
            reasons.append(f"IV rank {iv_rank:.0f} — options cheap, good time to buy")

        # SL: 50% of premium (hard rule for option buyers)
        entry    = strike_obj.ltp or strike_obj.mid_price
        sl       = round(entry * (1 - self.MAX_PREMIUM_LOSS_PCT), 2)
        target   = round(entry * 2.5, 2)    # 150% profit target (1:2.5 R:R)

        return self._build_signal(
            underlying, strike_obj, direction, entry, sl, target,
            regime, confidence, iv_rank, pcr, max_pain,
            capital, reasons, warnings, bias=bias,
        )

    def _reversal_options_signal(
        self,
        underlying, chain, bias, regime, confidence,
        capital, pcr, max_pain, iv_rank, dte, warnings,
    ) -> Optional[FOSignal]:
        """Buy reversal options at RSI extremes in ranging market."""

        if confidence < 0.55:
            return None

        # In ranging: RSI extremes give mean-reversion signals
        if "BUY" in bias:
            # Oversold bounce → buy calls
            opt_type  = OptionType.CALL
            direction = "LONG"
            reasons   = ["Ranging market + oversold → mean-reversion call"]
        elif "SELL" in bias:
            opt_type  = OptionType.PUT
            direction = "LONG"
            reasons   = ["Ranging market + overbought → mean-reversion put"]
        else:
            return None

        # In ranging markets use ATM (delta ~0.5 gives best R:R)
        strike_obj = (chain.get_atm_call() if opt_type == OptionType.CALL
                      else chain.get_atm_put())
        if not strike_obj:
            return None

        entry  = strike_obj.ltp or strike_obj.mid_price
        sl     = round(entry * 0.55, 2)    # tighter SL in ranging (45% loss)
        target = round(entry * 2.0, 2)     # 1:2 R:R

        # PCR context
        if pcr > 1.3:
            reasons.append(f"PCR {pcr:.2f} — bearish sentiment, contrarian buy signal")
        elif pcr < 0.7:
            reasons.append(f"PCR {pcr:.2f} — bullish sentiment, contrarian put signal")

        return self._build_signal(
            underlying, strike_obj, direction, entry, sl, target,
            regime, confidence, iv_rank, pcr, max_pain,
            capital, reasons, warnings, bias=bias,
        )

    # ── Helpers

    def _futures_signal(
        self,
        underlying, futures, bias, regime, confidence,
        capital, pcr, max_pain, iv_rank, warnings,
    ) -> Optional[FOSignal]:
        """Generate a futures directional signal."""

        if "BUY" in bias:
            direction = "LONG"
            reasons   = ["Volatile regime + strong trend → futures long (avoids IV crush)"]
        elif "SELL" in bias:
            direction = "SHORT"
            reasons   = ["Volatile regime + strong downtrend → futures short"]
        else:
            return None

        lot_sz   = futures.lot_size
        margin   = futures.approx_margin
        n_lots   = max(1, int(capital / margin))

        # Futures SL: 0.5% of contract price
        entry = futures.ltp
        if direction == "LONG":
            sl     = round(entry * 0.995, 2)
            target = round(entry * 1.01,  2)
        else:
            sl     = round(entry * 1.005, 2)
            target = round(entry * 0.99,  2)

        rr = abs(target - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0

        return FOSignal(
            underlying       = underlying,
            instrument       = "FUTURES",
            direction        = direction,
            strike           = None,
            expiry           = futures.expiry,
            symbol           = futures.symbol,
            entry_premium    = entry,
            stop_loss_prem   = sl,
            target_premium   = target,
            lot_size         = lot_sz,
            recommended_lots = n_lots,
            underlying_bias  = bias,
            regime           = regime,
            confidence       = confidence,
            iv_rank          = iv_rank,
            pcr              = pcr,
            max_pain         = max_pain,
            max_loss_per_lot = abs(entry - sl) * lot_sz,
            capital_required = margin * n_lots,
            risk_reward      = round(rr, 2),
            reasons          = reasons,
            warnings         = warnings,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_signal(
            self,
            underlying, strike_obj, direction, entry, sl, target,
            regime, confidence, iv_rank, pcr, max_pain,
            capital, reasons, warnings, bias="",
        ) -> FOSignal:
        lot_sz   = strike_obj.lot_size
        cost_lot = entry * lot_sz
        n_lots   = max(1, int(capital * 0.20 / cost_lot))   # risk 20% of capital
        rr       = (target - entry) / (entry - sl) if (entry - sl) > 0 else 0

        instr = ("CALL" if strike_obj.option_type == OptionType.CALL else "PUT")

        return FOSignal(
            underlying       = underlying,
            instrument       = instr,
            direction        = direction,
            strike           = strike_obj.strike,
            expiry           = strike_obj.expiry,
            symbol           = strike_obj.symbol,
            entry_premium    = round(entry, 2),
            stop_loss_prem   = sl,
            target_premium   = target,
            lot_size         = lot_sz,
            recommended_lots = n_lots,
            underlying_bias  = bias or direction,
            regime           = regime,
            confidence       = confidence,
            iv_rank          = iv_rank,
            pcr              = pcr,
            max_pain         = max_pain,
            greeks           = strike_obj.greeks,
            max_loss_per_lot = round((entry - sl) * lot_sz, 2),
            capital_required = round(cost_lot * n_lots, 2),
            risk_reward      = round(rr, 2),
            reasons          = reasons,
            warnings         = warnings,
        )

    def _compute_iv_rank(self, chain: OptionChain, iv_history: list[float] = None) -> float:
        """
        IV rank = (current IV - 52w low) / (52w high - 52w low) * 100
        If no history provided, returns 50 (neutral).
        """
        if not iv_history or len(iv_history) < 10:
            # Estimate from chain ATM IV
            atm_call = chain.get_atm_call()
            if atm_call and atm_call.iv > 0:
                # Without history, rough estimate: <12% = low, >20% = high for Nifty
                iv = atm_call.iv * 100
                if iv < 12:   return 20.0
                if iv > 25:   return 80.0
                return 50.0
            return 50.0

        current_iv = chain.get_atm_call().iv * 100 if chain.get_atm_call() else 15.0
        iv_low     = min(iv_history)
        iv_high    = max(iv_history)
        if iv_high == iv_low:
            return 50.0
        return round((current_iv - iv_low) / (iv_high - iv_low) * 100, 1)


# Module-level singleton
fo_engine = FOEngine()