import requests
r = requests.get('https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json')
data = r.json()

# Exactly replicate the token_resolver filter for COPPER
candidates = [
    row for row in data
    if row.get("exch_seg") == "MCX"
    and row.get("name", "").upper().startswith("COPPER")
    and row.get("instrumenttype", "") in ("FUTCOM", "FUTCUR", "")
    and "CE" not in row.get("name", "")
    and "PE" not in row.get("name", "")
    and row.get("expiry", "")
]

print(f"Candidates found: {len(candidates)}")
for c in candidates:
    print(c.get("token"), c.get("name"), c.get("expiry"), repr(c.get("instrumenttype")))