# trading-agent/src/alerts/signal_formatter.py
"""
Enhanced signal formatter for Telegram messages.
Adds SL, Target, hold duration, exit conditions, AND Futures/Options suggestions.
"""
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

# ── Asset classification ───────────────────────────────────────────────────────
CRYPTO_SYMS  = {"BTC","ETH","SOL","BNB","XRP","ADA","DOGE","AVAX","DOT","MATIC","LTC"}
INDEX_SYMS   = {"NIFTY50","BANKNIFTY","NIFTYIT","SENSEX","FINNIFTY"}
MCX_SYMS     = {"GOLD","SILVER","CRUDEOIL","COPPER","NATURALGAS","ZINC","ALUMINIUM"}
EQUITY_SYMS  = set()  # everything else

# ── MCX Futures specs ──────────────────────────────────────────────────────────
MCX_FUTURES = {
    "GOLD":       {"lot": 100,   "unit": "g",     "label": "Gold Mini (100g)", "margin_pct": 0.05},
    "SILVER":     {"lot": 30,    "unit": "kg",    "label": "Silver Mini (30kg)","margin_pct": 0.05},
    "CRUDEOIL":   {"lot": 100,   "unit": "bbl",   "label": "Crude Mini (100bbl)","margin_pct": 0.05},
    "COPPER":     {"lot": 2500,  "unit": "kg",    "label": "Copper (2500kg)",  "margin_pct": 0.05},
    "NATURALGAS": {"lot": 1250,  "unit": "mmBtu", "label": "Natural Gas",      "margin_pct": 0.08},
    "ZINC":       {"lot": 5000,  "unit": "kg",    "label": "Zinc (5MT)",       "margin_pct": 0.05},
}

# ── ATR % defaults by asset ────────────────────────────────────────────────────
DEFAULT_ATR = {
    "index":  0.012, "equity": 0.018,
    "futures":0.015, "crypto": 0.025,
}
SL_MULT  = {"index":1.5, "equity":1.5, "futures":2.0, "crypto":2.5}
TGT_MULT = {"index":2.0, "equity":2.0, "futures":2.5, "crypto":3.5}


def _asset_class(symbol: str) -> str:
    s = symbol.upper()
    if s in INDEX_SYMS:  return "index"
    if s in CRYPTO_SYMS: return "crypto"
    if s in MCX_SYMS:    return "futures"
    return "equity"


def _sl_target(symbol: str, price: float, side: str, atr_pct: float = None):
    ac      = _asset_class(symbol)
    atr_pct = atr_pct if (atr_pct and atr_pct > 0) else DEFAULT_ATR[ac]
    sl_d    = price * atr_pct * SL_MULT[ac]
    tgt_d   = price * atr_pct * TGT_MULT[ac]
    if side == "BUY":
        return round(price - sl_d, 2), round(price + tgt_d, 2)
    return round(price + sl_d, 2), round(price - tgt_d, 2)


def _fmt(price: float, symbol: str = "") -> str:
    """Format price for display."""
    if price <= 0:
        return "—"
    if price >= 10_000_000:
        return f"₹{price/100_000:.2f}L"
    if price >= 100_000:
        return f"₹{price/100_000:.2f}L"
    if price >= 1_000:
        return f"₹{price:,.2f}"
    return f"₹{price:.4f}"


def _hold_duration(symbol: str, hold_type: str) -> str:
    if hold_type == "INTRADAY": return "Exit same day before 15:15 IST"
    if hold_type == "POSITIONAL": return "Hold 5-15 trading days"
    ac = _asset_class(symbol)
    if ac == "crypto":   return "4-24 hours (volatile)"
    if ac == "futures":  return "1-3 days (MCX session)"
    return "2-5 trading days"


def _exit_conditions(symbol, side, price, sl, target, regime, hold_type) -> list:
    ac   = _asset_class(symbol)
    cond = []
    cond.append(f"🔴 EXIT if price {'falls' if side=='BUY' else 'rises'} to {_fmt(sl)} (stop loss)")
    cond.append(f"🟢 EXIT if price {'rises' if side=='BUY' else 'falls'} to {_fmt(target)} (target)")
    if ac == "index":
        cond.append("⏰ EXIT by 15:15 IST (avoid last-15-min volatility)")
        cond.append("⏰ EXIT before overnight if intraday — gap risk is real")
    elif ac == "futures":
        cond.append("⏰ MCX Evening session closes 23:30 IST — exit before")
        cond.append("⏰ Exit before expiry week if not rolling the contract")
    elif ac == "crypto":
        cond.append("⏰ Crypto: no session end — use trailing stop after +2%")
    else:
        cond.append("⏰ EXIT by 15:15 IST or carry overnight with SL in broker")
    if hold_type == "INTRADAY":
        cond.append("📅 INTRADAY: must close TODAY")
    elif hold_type == "SWING":
        trail = round(abs(target - sl) * 0.25, 2)
        cond.append(f"📅 SWING: hold 2-5 days, trail SL by {_fmt(trail)} per day")
    if regime == "VOLATILE":
        cond.append("⚡ VOLATILE regime: tighten SL if -1% adverse move")
    return cond


def _futures_section(symbol: str, side: str, price_inr: float, sl: float, target: float) -> str:
    """Build MCX Futures trade suggestion with correct lot value."""
    spec = MCX_FUTURES.get(symbol.upper())
    if not spec:
        return ""

    # Price unit conversions for correct lot value
    sym = symbol.upper()
    if sym == "GOLD":
        # price_inr is per 10g, lot is 100g → divide by 10 to get per-gram price
        price_per_unit = price_inr / 10
        sl_per_unit    = sl / 10
        tgt_per_unit   = target / 10
        display_unit   = "10g"
    elif sym == "SILVER":
        # price_inr is per kg, lot is 30kg
        price_per_unit = price_inr
        sl_per_unit    = sl
        tgt_per_unit   = target
        display_unit   = "kg"
    elif sym == "CRUDEOIL":
        # price_inr is per bbl, lot is 100 bbl
        price_per_unit = price_inr
        sl_per_unit    = sl
        tgt_per_unit   = target
        display_unit   = "bbl"
    else:
        price_per_unit = price_inr
        sl_per_unit    = sl
        tgt_per_unit   = target
        display_unit   = spec["unit"]

    lot_value = price_per_unit * spec["lot"]
    margin    = lot_value * spec["margin_pct"]
    sl_loss   = abs(price_per_unit - sl_per_unit) * spec["lot"]
    tgt_profit= abs(tgt_per_unit - price_per_unit) * spec["lot"]
    direction = "LONG (BUY)" if side == "BUY" else "SHORT (SELL)"

    lines = [
        f"\n━━━ 📦 <b>MCX Futures Trade</b>",
        f"Contract: <code>{spec['label']}</code>",
        f"Direction: <b>{direction}</b>",
        f"Entry:   <code>{_fmt(price_inr)}/{display_unit}</code>",
        f"SL:      <code>{_fmt(sl)}/{display_unit}</code>  (max loss/lot: ₹{sl_loss:,.0f})",
        f"Target:  <code>{_fmt(target)}/{display_unit}</code>  (profit/lot: ₹{tgt_profit:,.0f})",
        f"Lot val: ~<code>₹{lot_value/100000:.2f}L</code>  |  Margin req: ~<code>₹{margin:,.0f}</code>",
    ]
    return "\n".join(lines)


def _options_section(symbol: str, side: str, price: float, sl: float, target: float,
                     expiry: str = "Weekly") -> str:
    """Build NSE Options trade suggestion for index symbols."""
    if symbol.upper() not in INDEX_SYMS:
        return ""

    # Strike selection: ATM ± 1 strike
    strike_gap = 50 if "NIFTY50" in symbol.upper() else 100
    atm = round(price / strike_gap) * strike_gap

    if side == "BUY":
        opt_type  = "CALL"
        strike    = atm  # ATM call
        otm_strike= atm + strike_gap  # slightly OTM
        hedge     = f"Hedge: BUY {atm + 2*strike_gap} CALL (spread to reduce cost)"
    else:
        opt_type  = "PUT"
        strike    = atm  # ATM put
        otm_strike= atm - strike_gap
        hedge     = f"Hedge: BUY {atm - 2*strike_gap} PUT (spread to reduce cost)"

    # Premium estimates (rough: ATM ≈ 0.5-0.8% of spot for weekly)
    premium_est = round(price * 0.006)
    sl_prem     = round(premium_est * 0.40)  # -40% of premium
    tgt_prem    = round(premium_est * 1.00)  # +100% of premium (2x)
    lot_size    = 75 if "NIFTY50" in symbol.upper() else 30  # NIFTY lot=75, BANKNIFTY=30
    cost        = premium_est * lot_size

    lines = [
        f"\n━━━ 🎯 <b>Options Trade ({expiry})</b>",
        f"Strategy: BUY <b>{atm} {opt_type}</b>",
        f"Est. Premium:  <code>₹{premium_est}</code>  (1 lot cost: ₹{cost:,})",
        f"SL on Premium: <code>₹{sl_prem}</code>  (exit if premium falls here)",
        f"Target Premium:<code>₹{tgt_prem}</code>  (2× the entry premium)",
        f"Lot size: {lot_size} | OTM alt: {otm_strike} {opt_type} (cheaper, higher risk)",
        f"<i>{hedge}</i>",
        f"⏰ Exit options by 15:00 IST on expiry day",
    ]
    return "\n".join(lines)


def _crypto_futures_section(symbol: str, side: str, price_inr: float,
                             sl: float, target: float) -> str:
    """Crypto spot trade note (no Indian exchange F&O for crypto)."""
    if symbol.upper() not in CRYPTO_SYMS:
        return ""
    direction = "LONG" if side == "BUY" else "SHORT"
    lines = [
        f"\n━━━ ₿ <b>Crypto Spot Trade</b>",
        f"Direction: <b>{direction}</b> on CoinSwitch / WazirX",
        f"Entry:  <code>{_fmt(price_inr)}</code>",
        f"SL:     <code>{_fmt(sl)}</code>",
        f"Target: <code>{_fmt(target)}</code>",
        f"⚠️ No Indian crypto F&O — spot trade only",
        f"💡 Use stop-limit order on exchange for automatic SL",
    ]
    return "\n".join(lines)


def format_signal_telegram(
    symbol:  str,
    signal:  dict,
    usdinr:  float = 92.46,
    atr_pct: float = None,
) -> str:
    """
    Complete Telegram signal with:
    - Entry, SL, Target, R:R
    - Hold duration + exit conditions
    - Futures section (MCX or NSE F&O)
    - Options section (NSE index only)
    """
    now_ist  = datetime.now(IST).strftime("%d %b %Y %H:%M IST")
    side     = "BUY" if "BUY" in str(signal.get("bias","")).upper() else "SELL"
    price    = float(signal.get("price", 0) or 0)
    conf     = float(signal.get("confidence", 0.5) or 0.5)
    regime   = signal.get("regime", "RANGING") or "RANGING"
    hold_type= signal.get("hold_type", "SWING") or "SWING"
    session  = signal.get("session_label", "") or ""
    timing   = signal.get("entry_timing", "ENTER") or "ENTER"
    reasons  = signal.get("reasons", []) or []
    atr_pct  = atr_pct or float(signal.get("atr_pct", 0) or 0) or None
    ac       = _asset_class(symbol)

    if price <= 0:
        return f"⚠️ Signal for {symbol}: price unavailable"

    sl, target = _sl_target(symbol, price, side, atr_pct)
    risk   = abs(price - sl)
    reward = abs(target - price)
    rr     = reward / risk if risk > 0 else 0
    risk_pct  = risk / price * 100
    gain_pct  = reward / price * 100

    side_icon = "🟢📈" if side == "BUY" else "🔴📉"
    conf_icon = "🔥" if conf >= 0.70 else "✅" if conf >= 0.55 else "⚠️"
    regime_icon = {"TRENDING_UP":"📈","TRENDING_DOWN":"📉","RANGING":"↔️","VOLATILE":"⚡"}.get(regime,"❓")

    exit_conds = _exit_conditions(symbol, side, price, sl, target, regime, hold_type)
    hold_dur   = _hold_duration(symbol, hold_type)

    lines = [
        f"🔔 <b>TRADING SIGNAL — {symbol}</b>",
        f"<i>{now_ist}</i>",
        "",
        f"{side_icon} <b>{side}</b>  {conf_icon} Confidence: <code>{conf:.0%}</code>",
        f"📍 Entry: <code>{_fmt(price)}</code>",
        "",
        f"━━━ 🎯 <b>Trade Levels</b>",
        f"🔴 Stop Loss:    <code>{_fmt(sl)}</code>   ({risk_pct:.1f}% risk)",
        f"🟢 Target:       <code>{_fmt(target)}</code>   ({gain_pct:.1f}% gain)",
        f"📊 Risk:Reward:  <code>1:{rr:.1f}</code>",
        "",
        f"━━━ ⏱ <b>Hold & Timing</b>",
        f"📅 Duration:  {hold_dur}",
        f"⏰ Session:   {session or 'Standard'}",
        f"🚦 Timing:    {timing}",
        "",
        f"━━━ 📋 <b>Exit Conditions</b>",
    ]

    for c in exit_conds[:5]:
        lines.append(c)

    # ── Futures/Options section ───────────────────────────────────────────────
    if ac == "futures":
        fo = _futures_section(symbol, side, price, sl, target)
        if fo:
            lines.append(fo)

    elif ac == "index":
        opt = _options_section(symbol, side, price, sl, target)
        if opt:
            lines.append(opt)
        # Also add futures suggestion for index
        fut_side = "BUY CALL" if side == "BUY" else "BUY PUT"
        nf_lot   = 75 if "NIFTY50" in symbol.upper() else 30
        lines.append(
            f"\n💡 <b>Futures alt:</b> {symbol} Futures | Margin ~₹{price*nf_lot*0.10/1000:.0f}K"
        )

    elif ac == "crypto":
        cryp = _crypto_futures_section(symbol, side, price, sl, target)
        if cryp:
            lines.append(cryp)

    # ── Signal context ────────────────────────────────────────────────────────
    lines += [
        "",
        f"━━━ 🧠 <b>Signal Context</b>",
        f"{regime_icon} Regime: {regime.replace('_',' ')}",
    ]
    for r in reasons[:2]:
        if r:
            lines.append(f"📝 {r[:90]}")

    lines += [
        "",
        "━━━━━━━━━━━━━━━━━━━━",
        "⚠️ <i>Paper trading. Always use SL. Past signals ≠ future results.</i>",
        "🤖 <i>AI Trading Agent</i>",
    ]

    return "\n".join(lines)