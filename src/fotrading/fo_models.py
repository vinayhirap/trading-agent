# trading-agent/src/fotrading/fo_models.py
"""
F&O Data Models — Options and Futures

All the data structures needed for F&O trading.
Kept separate from execution so they can be imported anywhere safely.
"""
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional
from enum import Enum


# ── Enums ─────────────────────────────────────────────────────────────────────

class OptionType(str, Enum):
    CALL = "CE"
    PUT  = "PE"

class InstrumentType(str, Enum):
    OPTION  = "OPTIDX"    # Index option
    OPTSTK  = "OPTSTK"    # Stock option
    FUTIDX  = "FUTIDX"    # Index future
    FUTSTK  = "FUTSTK"    # Stock future
    FUTCOM  = "FUTCOM"    # Commodity future (MCX)

class PositionType(str, Enum):
    LONG  = "LONG"
    SHORT = "SHORT"


# ── Lot sizes (NSE-defined, update when NSE revises) ─────────────────────────
LOT_SIZES = {
    # Indices
    "NIFTY":     50,
    "NIFTY50":   50,
    "BANKNIFTY": 15,
    "FINNIFTY":  40,
    "MIDCPNIFTY":75,
    "SENSEX":    10,
    # Large cap stocks (F&O eligible)
    "RELIANCE":  250,
    "TCS":       150,
    "HDFCBANK":  550,
    "INFY":      300,
    "ICICIBANK": 700,
    "SBIN":      1500,
    "WIPRO":     1500,
    "AXISBANK":  625,
    "KOTAKBANK": 400,
    "LT":        300,
    "BAJFINANCE":125,
    "MARUTI":    100,
    "SUNPHARMA": 700,
    "BHARTIARTL":950,
    "TATAMOTORS":1425,
    # MCX commodities (in units)
    "GOLD":      1,        # 1 kg per lot
    "SILVER":    30,       # 30 kg per lot
    "CRUDEOIL":  100,      # 100 barrels per lot
    "COPPER":    2500,     # 2500 kg per lot
    "NATURALGAS":1250,     # 1250 mmBtu per lot
}

# Strike intervals (distance between strikes)
STRIKE_INTERVALS = {
    "NIFTY":     50,
    "NIFTY50":   50,
    "BANKNIFTY": 100,
    "FINNIFTY":  50,
    "SENSEX":    100,
}

# Margin requirements (approximate SPAN + Exposure)
# Actual margin varies — use as minimum check only
MARGIN_APPROX_PCT = {
    "FUTIDX":  0.12,   # ~12% of contract value
    "FUTSTK":  0.15,   # ~15% of contract value
    "FUTCOM":  0.08,   # ~8% for commodities
    "OPTIDX":  1.0,    # options: pay full premium
    "OPTSTK":  1.0,
}


# ── Greeks dataclass ──────────────────────────────────────────────────────────

@dataclass
class Greeks:
    """Option Greeks — sensitivity measures."""
    delta:  float = 0.0    # price change per ₹1 move in underlying
    gamma:  float = 0.0    # delta change per ₹1 move
    theta:  float = 0.0    # time decay per day (negative for buyers)
    vega:   float = 0.0    # price change per 1% IV move
    iv:     float = 0.0    # implied volatility (as decimal, e.g. 0.15 = 15%)
    iv_pct: float = 0.0    # IV as percentage

    @property
    def theta_per_lot(self) -> float:
        return self.theta  # already per lot in Angel One data

    @property
    def is_high_theta(self) -> bool:
        """True if time decay is eating >2% of premium per day."""
        return abs(self.theta) > 0.02


@dataclass
class OptionStrike:
    """A single strike in the option chain."""
    symbol:       str           # e.g. NIFTY24DEC22500CE
    token:        str           # Angel One instrument token
    underlying:   str           # e.g. NIFTY50
    expiry:       date
    strike:       float
    option_type:  OptionType
    lot_size:     int

    # Market data (updated live)
    ltp:          float = 0.0   # last traded price (premium)
    bid:          float = 0.0
    ask:          float = 0.0
    oi:           int   = 0     # open interest
    oi_change:    int   = 0     # OI change vs previous day
    volume:       int   = 0
    iv:           float = 0.0   # implied volatility

    # Greeks
    greeks:       Greeks = field(default_factory=Greeks)

    @property
    def mid_price(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.ltp

    @property
    def spread_pct(self) -> float:
        if self.mid_price > 0:
            return (self.ask - self.bid) / self.mid_price
        return 0.0

    @property
    def cost_per_lot(self) -> float:
        return self.ltp * self.lot_size

    @property
    def days_to_expiry(self) -> int:
        return max(0, (self.expiry - date.today()).days)

    @property
    def is_liquid(self) -> bool:
        """Basic liquidity check — avoid illiquid strikes."""
        return (
            self.volume > 100 and
            self.oi > 1000 and
            self.spread_pct < 0.05  # spread < 5%
        )


@dataclass
class OptionChain:
    """Complete option chain for one underlying + expiry."""
    underlying:   str
    expiry:       date
    spot_price:   float
    atm_strike:   float
    strikes:      list[OptionStrike] = field(default_factory=list)
    fetched_at:   datetime = field(default_factory=datetime.utcnow)

    @property
    def calls(self) -> list[OptionStrike]:
        return [s for s in self.strikes if s.option_type == OptionType.CALL]

    @property
    def puts(self) -> list[OptionStrike]:
        return [s for s in self.strikes if s.option_type == OptionType.PUT]

    def get_atm_call(self) -> Optional[OptionStrike]:
        calls = sorted(self.calls, key=lambda s: abs(s.strike - self.spot_price))
        return calls[0] if calls else None

    def get_atm_put(self) -> Optional[OptionStrike]:
        puts = sorted(self.puts, key=lambda s: abs(s.strike - self.spot_price))
        return puts[0] if puts else None

    def get_otm_call(self, otm_pct: float = 0.01) -> Optional[OptionStrike]:
        """Get OTM call at approximately otm_pct above spot."""
        target = self.spot_price * (1 + otm_pct)
        calls  = [s for s in self.calls if s.strike >= self.spot_price]
        if not calls:
            return None
        return min(calls, key=lambda s: abs(s.strike - target))

    def get_otm_put(self, otm_pct: float = 0.01) -> Optional[OptionStrike]:
        """Get OTM put at approximately otm_pct below spot."""
        target = self.spot_price * (1 - otm_pct)
        puts   = [s for s in self.puts if s.strike <= self.spot_price]
        if not puts:
            return None
        return min(puts, key=lambda s: abs(s.strike - target))

    def get_pcr(self) -> float:
        """Put/Call ratio by OI — sentiment indicator."""
        total_call_oi = sum(s.oi for s in self.calls)
        total_put_oi  = sum(s.oi for s in self.puts)
        if total_call_oi > 0:
            return round(total_put_oi / total_call_oi, 2)
        return 1.0

    def get_max_pain(self) -> float:
        """
        Max pain strike — where option buyers lose the most.
        Market tends to gravitate toward this on expiry day.
        """
        if not self.strikes:
            return self.atm_strike

        strike_values = set(s.strike for s in self.strikes)
        min_loss      = float("inf")
        max_pain_strike = self.atm_strike

        for test_strike in strike_values:
            total_loss = 0
            for s in self.calls:
                # Call writer loss if price > strike
                if test_strike > s.strike:
                    total_loss += (test_strike - s.strike) * s.oi
            for s in self.puts:
                # Put writer loss if price < strike
                if test_strike < s.strike:
                    total_loss += (s.strike - test_strike) * s.oi
            if total_loss < min_loss:
                min_loss         = total_loss
                max_pain_strike  = test_strike

        return max_pain_strike


@dataclass
class FuturesContract:
    """A single futures contract."""
    symbol:       str        # e.g. NIFTYFUT, CRUDEOILFUT
    token:        str
    underlying:   str
    expiry:       date
    instrument:   InstrumentType
    lot_size:     int
    exchange:     str        # NSE, MCX

    # Market data
    ltp:          float = 0.0
    bid:          float = 0.0
    ask:          float = 0.0
    oi:           int   = 0
    volume:       int   = 0
    prev_close:   float = 0.0

    @property
    def change_pct(self) -> float:
        if self.prev_close > 0:
            return (self.ltp - self.prev_close) / self.prev_close * 100
        return 0.0

    @property
    def contract_value(self) -> float:
        return self.ltp * self.lot_size

    @property
    def approx_margin(self) -> float:
        pct = MARGIN_APPROX_PCT.get(self.instrument.value, 0.15)
        return self.contract_value * pct

    @property
    def days_to_expiry(self) -> int:
        return max(0, (self.expiry - date.today()).days)


@dataclass
class FOPosition:
    """An open F&O position."""
    position_id:  str
    symbol:       str
    underlying:   str
    instrument:   InstrumentType
    option_type:  Optional[OptionType]
    position_type: PositionType
    strike:       Optional[float]
    expiry:       date
    lot_size:     int
    quantity_lots: int        # number of lots
    entry_premium: float      # premium paid per unit
    entry_price:  float       # underlying price at entry
    current_premium: float = 0.0
    stop_loss_premium: float = 0.0    # exit if premium drops to this
    target_premium:    float = 0.0    # exit if premium rises to this
    angel_order_id: str = ""
    opened_at:    str = ""
    is_paper:     bool = True

    @property
    def quantity_units(self) -> int:
        return self.quantity_lots * self.lot_size

    @property
    def entry_cost(self) -> float:
        return self.entry_premium * self.quantity_units

    @property
    def current_value(self) -> float:
        return self.current_premium * self.quantity_units

    @property
    def unrealised_pnl(self) -> float:
        if self.position_type == PositionType.LONG:
            return self.current_value - self.entry_cost
        else:
            return self.entry_cost - self.current_value

    @property
    def unrealised_pnl_pct(self) -> float:
        if self.entry_cost > 0:
            return self.unrealised_pnl / self.entry_cost * 100
        return 0.0

    @property
    def days_to_expiry(self) -> int:
        return max(0, (self.expiry - date.today()).days)

    @property
    def is_expiring_soon(self) -> bool:
        return self.days_to_expiry <= 2