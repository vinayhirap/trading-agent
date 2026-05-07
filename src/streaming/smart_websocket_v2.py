"""Local fallback shim for SmartApi SmartWebSocketV2 when smartapi-python is absent."""

class SmartWebSocketV2:
    def __init__(self, *args, **kwargs):
        raise ImportError(
            "SmartApi.SmartWebSocketV2 not installed. "
            "Install with: pip install smartapi-python"
        )

    def subscribe(self, *args, **kwargs):
        raise RuntimeError("SmartWebSocketV2 unavailable")

    def connect(self, *args, **kwargs):
        raise RuntimeError("SmartWebSocketV2 unavailable")

    def close_connection(self, *args, **kwargs):
        return
