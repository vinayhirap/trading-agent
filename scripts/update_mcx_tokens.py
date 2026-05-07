# trading-agent/scripts/update_mcx_tokens.py
"""
MCX Token Updater — run on the last trading day of each month.

MCX contracts roll over monthly. This script:
1. Fetches the ScripMaster from Angel One
2. Finds the near-month token for each MCX commodity
3. Updates angel_one_ticker.py and angel_one_adapter.py automatically

Usage:
  python scripts/update_mcx_tokens.py

Or just update manually in angel_one_ticker.py MCX_TOKENS dict.
"""
import sys, json, re
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Commodities to find near-month tokens for
COMMODITIES = {
    "GOLD":       ["GOLD", "GOLDM"],
    "SILVER":     ["SILVER", "SILVERM"],
    "CRUDEOIL":   ["CRUDEOIL"],
    "COPPER":     ["COPPER"],
    "NATURALGAS": ["NATURALGAS", "NATGAS"],
    "ZINC":       ["ZINC", "ZINCMINI"],
    "ALUMINIUM":  ["ALUMINIUM", "ALUMINIUMM"],
}

def fetch_scrip_master() -> list[dict]:
    """Download Angel One ScripMaster (instrument list)."""
    import urllib.request
    url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    print(f"Downloading ScripMaster from Angel One...")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.loads(r.read().decode())
        print(f"  Downloaded {len(data)} instruments")
        return data
    except Exception as e:
        print(f"  Failed: {e}")
        return []

def find_near_month_tokens(instruments: list[dict]) -> dict[str, str]:
    """Find near-month MCX futures tokens for each commodity."""
    mcx_instruments = [i for i in instruments if i.get("exch_seg") == "MCX"]
    print(f"  MCX instruments: {len(mcx_instruments)}")

    today    = datetime.now()
    result   = {}

    for our_sym, search_names in COMMODITIES.items():
        candidates = []
        for inst in mcx_instruments:
            name   = inst.get("name", "").upper()
            symbol = inst.get("symbol", "").upper()
            itype  = inst.get("instrumenttype", "").upper()

            # Must be a futures contract
            if "FUT" not in itype and "FUT" not in symbol:
                continue

            # Must match one of our search names
            if not any(n in name or n in symbol for n in search_names):
                continue

            # Parse expiry
            expiry_str = inst.get("expiry", "")
            try:
                expiry = datetime.strptime(expiry_str, "%d%b%Y")
                if expiry >= today:   # only future/current contracts
                    candidates.append({
                        "token":  inst.get("token"),
                        "symbol": inst.get("symbol"),
                        "expiry": expiry,
                        "name":   inst.get("name"),
                    })
            except ValueError:
                continue

        if candidates:
            # Pick nearest expiry
            nearest = min(candidates, key=lambda x: x["expiry"])
            result[our_sym] = nearest["token"]
            print(f"  {our_sym}: token={nearest['token']} symbol={nearest['symbol']} expiry={nearest['expiry'].strftime('%d%b%Y')}")
        else:
            print(f"  {our_sym}: NOT FOUND — keeping existing token")

    return result

def update_ticker_file(new_tokens: dict[str, str]):
    """Update MCX_TOKENS in angel_one_ticker.py."""
    ticker_path = ROOT / "src" / "streaming" / "angel_one_ticker.py"
    if not ticker_path.exists():
        print(f"  File not found: {ticker_path}")
        return

    src = ticker_path.read_text(encoding="utf-8")

    for sym, token in new_tokens.items():
        # Replace token in MCX_TOKENS dict
        pattern = rf'("{sym}":\s*\("MCX",\s*")[^"]+(")'
        replacement = rf'\g<1>{token}\g<2>'
        src_new = re.sub(pattern, replacement, src)
        if src_new != src:
            print(f"  Updated {sym} token → {token}")
            src = src_new

    ticker_path.write_text(src, encoding="utf-8")
    print(f"  Saved {ticker_path}")

def main():
    print("=" * 50)
    print("MCX Token Updater")
    print("=" * 50)

    instruments = fetch_scrip_master()
    if not instruments:
        print("Could not fetch ScripMaster — update tokens manually")
        return

    print("\nFinding near-month MCX tokens:")
    new_tokens = find_near_month_tokens(instruments)

    if new_tokens:
        print("\nUpdating angel_one_ticker.py:")
        update_ticker_file(new_tokens)
        print("\nDone! Restart run.py to apply new tokens.")
    else:
        print("\nNo tokens found — check ScripMaster format")

    print("\nCurrent tokens:")
    for sym, tok in new_tokens.items():
        print(f"  {sym}: {tok}")

if __name__ == "__main__":
    main()