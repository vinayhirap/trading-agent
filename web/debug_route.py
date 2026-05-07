"""
DROP THIS FILE in trading-agent/web/ and run:
  python web/debug_route.py

It starts the same FastAPI app but prints the REAL traceback for GET /
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import traceback
from fastapi.testclient import TestClient

# Patch page_view BEFORE importing main so we capture the real error
import web.main as m

original_template_response = m.templates.TemplateResponse

def loud_template_response(name, context, *a, **kw):
    try:
        return original_template_response(name, context, *a, **kw)
    except Exception as e:
        print("\n" + "="*60)
        print(f"TEMPLATE ERROR in {name}:")
        traceback.print_exc()
        print("="*60 + "\n")
        raise

m.templates.TemplateResponse = loud_template_response

# Also patch _template_context to print errors
original_ctx = m._template_context
def loud_ctx(request):
    try:
        return original_ctx(request)
    except Exception as e:
        print("\n" + "="*60)
        print("_template_context() ERROR:")
        traceback.print_exc()
        print("="*60 + "\n")
        raise
m._template_context = loud_ctx

# Also patch _market_status
original_ms = m._market_status
def loud_ms():
    try:
        result = original_ms()
        print(f"[DEBUG] _market_status() returned: {result}")
        return result
    except Exception as e:
        print(f"[DEBUG] _market_status() CRASHED: {e}")
        traceback.print_exc()
        return {
            "nse_open": False, "mcx_open": True,
            "is_holiday": False, "holiday_name": "",
            "secs_to_close": 0, "secs_to_open": 0,
        }
m._market_status = loud_ms

print("Making GET / request...")
client = TestClient(m.app, raise_server_exceptions=False)
resp = client.get("/")
print(f"\nStatus: {resp.status_code}")
if resp.status_code != 200:
    print("Response body (first 2000 chars):")
    print(resp.text[:2000])