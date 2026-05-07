"""
patch_main.py — Safe, idempotent patcher for web/main.py
Run as many times as you want — it checks before applying each change.

Changes applied:
  1. _get_prices() → fully async _get_prices_async()
  2. price_broadcast_loop → uses async version, sleep 0.1s (already done)
  3. startup_event → registers tick callback for instant Angel One push
  4. /ws/prices handler → uses async version
  5. /api/prices endpoint → uses async version
  6. price_store.register_tick_callback wired on startup
  7. /api/mcx-tokens moved before app.include_router (fixes routing bug)
"""
import re
from pathlib import Path

MAIN_PATH = Path("web/main.py")

def read():
    return MAIN_PATH.read_text(encoding="utf-8")

def write(src):
    MAIN_PATH.write_text(src, encoding="utf-8")
    print("  ✅ Written")

def already_has(src, marker):
    return marker in src

def patch_1_async_get_prices(src):
    """Add _get_prices_async() after _get_prices() if not already present."""
    marker = "async def _get_prices_async()"
    if already_has(src, marker):
        print("[1] _get_prices_async already present — skip")
        return src

    new_fn = '''

async def _get_prices_async() -> dict:
    """Async version of _get_prices — no thread blocking."""
    try:
        from src.streaming.price_store import (
            price_store, PRICE_META, ACTIVE_EXPIRY, convert_price
        )
        usdinr = price_store.get("USDINR", fallback=False) or 92.46
        if not (70 < usdinr < 120):
            usdinr = 92.46
        all_raw = price_store.get_all()
        result  = {}
        for sym, raw in all_raw.items():
            if not raw or raw <= 0:
                continue
            meta    = PRICE_META.get(sym, {"convert": "none", "label": ""})
            display = convert_price(sym, raw, usdinr)
            age     = price_store.age_seconds(sym) or 0
            result[sym] = {
                "raw":     round(raw, 4),
                "display": round(display, 2),
                "label":   meta.get("label", ""),
                "convert": meta.get("convert", "none"),
                "age":     round(age, 1),
                "stale":   age > 300,
                "expiry":  ACTIVE_EXPIRY.get(sym, ""),
            }
        return {"type": "prices", "usdinr": usdinr, "prices": result}
    except Exception as e:
        return {"type": "prices", "usdinr": 92.46, "prices": {}, "error": str(e)}
'''
    # Insert after the closing of _get_prices()
    insert_after = 'return {"type": "prices", "usdinr": 92.46, "prices": {}, "error": str(e)}\n\n\ndef _get_signals'
    if insert_after not in src:
        # Try alternate ending pattern
        idx = src.find('def _get_signals()')
        if idx == -1:
            print("[1] Could not find insertion point — skip")
            return src
        src = src[:idx] + new_fn.lstrip('\n') + '\n\n' + src[idx:]
    else:
        src = src.replace(
            'return {"type": "prices", "usdinr": 92.46, "prices": {}, "error": str(e)}\n\n\ndef _get_signals',
            'return {"type": "prices", "usdinr": 92.46, "prices": {}, "error": str(e)}\n' + new_fn + '\n\ndef _get_signals'
        )
    print("[1] ✅ _get_prices_async() added")
    return src


def patch_2_broadcast_loop(src):
    """Make price_broadcast_loop use _get_prices_async."""
    marker = "data = await _get_prices_async()"
    if already_has(src, marker):
        print("[2] broadcast_loop already uses async — skip")
        return src

    old = "data = await asyncio.get_event_loop().run_in_executor(None, _get_prices)\n                await manager.broadcast_prices(data)"
    new = "data = await _get_prices_async()\n                await manager.broadcast_prices(data)"
    if old in src:
        src = src.replace(old, new, 1)
        print("[2] ✅ broadcast_loop uses _get_prices_async()")
    else:
        print("[2] Could not find old broadcast pattern — skip")
    return src


def patch_3_ws_handler(src):
    """Make WS initial send use async version."""
    marker = "data = await _get_prices_async()\n        await websocket.send_text"
    if already_has(src, marker):
        print("[3] WS handler already uses async — skip")
        return src

    old = "data = await asyncio.get_event_loop().run_in_executor(None, _get_prices)\n        await websocket.send_text(json.dumps(data))"
    new = "data = await _get_prices_async()\n        await websocket.send_text(json.dumps(data))"
    if old in src:
        src = src.replace(old, new, 1)
        print("[3] ✅ WS handler uses _get_prices_async()")
    else:
        print("[3] WS handler pattern not found — skip")
    return src


def patch_4_api_prices(src):
    """Make /api/prices use async version."""
    marker = "return await _get_prices_async()"
    if already_has(src, marker):
        print("[4] /api/prices already uses async — skip")
        return src

    old = 'async def api_prices():\n    data = await asyncio.get_event_loop().run_in_executor(None, _get_prices)\n    return data'
    new = 'async def api_prices():\n    return await _get_prices_async()'
    if old in src:
        src = src.replace(old, new, 1)
        print("[4] ✅ /api/prices uses _get_prices_async()")
    else:
        print("[4] /api/prices pattern not found — skip")
    return src


def patch_5_tick_callback(src):
    """Register Angel One tick callback on startup for instant push."""
    marker = "price_store.register_tick_callback"
    if already_has(src, marker):
        print("[5] tick_callback already registered — skip")
        return src

    tick_code = '''
    # Register instant push on Angel One tick (bypasses 0.1s poll loop)
    try:
        from src.streaming.price_store import price_store as _ps
        from src.streaming.price_store import PRICE_META as _PM, convert_price as _cp, ACTIVE_EXPIRY as _AE
        _loop = asyncio.get_event_loop()
        def _on_angel_tick(sym, raw_price):
            try:
                usdinr = _ps.get("USDINR", fallback=False) or 92.46
                if not (70 < usdinr < 120):
                    usdinr = 92.46
                meta    = _PM.get(sym, {"convert": "none", "label": ""})
                display = _cp(sym, raw_price, usdinr)
                payload = {"type": "prices", "usdinr": usdinr, "prices": {
                    sym: {
                        "raw":     round(raw_price, 4),
                        "display": round(display, 2),
                        "label":   meta.get("label", ""),
                        "convert": meta.get("convert", "none"),
                        "age":     0.0,
                        "stale":   False,
                        "expiry":  _AE.get(sym, ""),
                    }
                }}
                if manager.price_clients:
                    _loop.call_soon_threadsafe(
                        lambda p=payload: asyncio.ensure_future(manager.broadcast_prices(p))
                    )
            except Exception:
                pass
        _ps.register_tick_callback(_on_angel_tick)
    except Exception as _e:
        pass  # tick callback is optional — poll loop is the fallback
'''

    old = "@app.on_event(\"startup\")\nasync def startup_event():\n    asyncio.create_task(price_broadcast_loop())\n    asyncio.create_task(signal_broadcast_loop())"
    new = "@app.on_event(\"startup\")\nasync def startup_event():\n    asyncio.create_task(price_broadcast_loop())\n    asyncio.create_task(signal_broadcast_loop())" + tick_code

    if old in src:
        src = src.replace(old, new, 1)
        print("[5] ✅ tick_callback registered on startup")
    else:
        print("[5] startup_event pattern not found — skip")
    return src


def patch_6_mcx_tokens_route(src):
    """Move /api/mcx-tokens before app.include_router to fix routing."""
    marker = "# MCX_TOKENS_ROUTE_FIXED"
    if already_has(src, marker):
        print("[6] mcx-tokens route already fixed — skip")
        return src

    # Remove the misplaced route after include_router
    old_misplaced = '''
@api.get("/mcx-tokens")
async def api_mcx_tokens():
    try:
        from src.streaming.mcx_token_manager import mcx_token_manager
        return {"ok": True, "status": mcx_token_manager.get_status()}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    '''

    new_route = '''
# MCX_TOKENS_ROUTE_FIXED
@api.get("/mcx-tokens")
async def api_mcx_tokens():
    try:
        from src.streaming.mcx_token_manager import mcx_token_manager
        return {"ok": True, "status": mcx_token_manager.get_status()}
    except Exception as e:
        return {"ok": False, "error": str(e)}
'''

    if old_misplaced in src:
        # Remove old misplaced version
        src = src.replace(old_misplaced, "", 1)
        # Add before include_router
        src = src.replace(
            "app.include_router(api)",
            new_route + "\napp.include_router(api)",
            1
        )
        print("[6] ✅ /api/mcx-tokens moved before include_router")
    elif "@api.get(\"/mcx-tokens\")" not in src:
        # Just add it before include_router
        src = src.replace(
            "app.include_router(api)",
            new_route + "\napp.include_router(api)",
            1
        )
        print("[6] ✅ /api/mcx-tokens added before include_router")
    else:
        print("[6] mcx-tokens already in correct position — skip")
    return src


def patch_7_register_tick_callback_in_price_store():
    """Add register_tick_callback to price_store.py if missing."""
    ps_path = Path("src/streaming/price_store.py")
    if not ps_path.exists():
        print("[7] price_store.py not found — skip")
        return

    src = ps_path.read_text(encoding="utf-8")
    marker = "def register_tick_callback"
    if marker in src:
        print("[7] register_tick_callback already in price_store — skip")
        return

    new_methods = '''
    def register_tick_callback(self, cb):
        """Register a callable(symbol, price) fired on every update."""
        if not hasattr(self, "_tick_callbacks"):
            self._tick_callbacks = []
        self._tick_callbacks.append(cb)
'''

    # Add _tick_callbacks firing in update()
    old_update_end = '''            self._history[sym].append((now, price))
            self._tick_count += 1'''
    new_update_end = '''            self._history[sym].append((now, price))
            self._tick_count += 1
        # Fire tick callbacks (used for instant WS push on Angel One ticks)
        if hasattr(self, "_tick_callbacks"):
            for _cb in self._tick_callbacks:
                try:
                    _cb(sym, price)
                except Exception:
                    pass'''

    if old_update_end in src:
        src = src.replace(old_update_end, new_update_end, 1)
        # Add method before the singleton line
        src = src.replace(
            "\n# Module-level singleton",
            new_methods + "\n# Module-level singleton",
            1
        )
        ps_path.write_text(src, encoding="utf-8")
        print("[7] ✅ register_tick_callback added to price_store.py")
    else:
        print("[7] Could not find update() pattern in price_store — skip")


def main():
    print("=" * 55)
    print("main.py patcher — safe, idempotent")
    print("=" * 55)

    if not MAIN_PATH.exists():
        print(f"ERROR: {MAIN_PATH} not found. Run from project root.")
        return

    # Backup
    backup = MAIN_PATH.with_suffix(".py.bak")
    if not backup.exists():
        backup.write_text(read(), encoding="utf-8")
        print(f"Backup saved → {backup}")
    else:
        print(f"Backup already exists → {backup}")

    src = read()

    src = patch_1_async_get_prices(src)
    src = patch_2_broadcast_loop(src)
    src = patch_3_ws_handler(src)
    src = patch_4_api_prices(src)
    src = patch_5_tick_callback(src)
    src = patch_6_mcx_tokens_route(src)

    write(src)

    # price_store.py patch (separate file)
    patch_7_register_tick_callback_in_price_store()

    print()
    print("=" * 55)
    print("All patches applied. Now run:")
    print("  python run.py")
    print()
    print("Expected in startup log:")
    print("  ✅ PriceStore: background refresh started")
    print("  ✅ AngelOneTicker: WebSocket streaming started")
    print("  ✅ RealtimeAdvisor: started")
    print("=" * 55)


if __name__ == "__main__":
    main()