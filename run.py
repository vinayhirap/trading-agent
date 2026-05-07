#!/usr/bin/env python3
"""
AI Trading Agent — FastAPI Dashboard
Run: python run.py
Access: http://localhost:8010
API Docs: http://localhost:8010/docs
"""
import sys, os
from pathlib import Path

# Ensure project root is in path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import uvicorn
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════╗
║  AI Trading Agent — FastAPI v3                       ║
║                                                      ║
║  Dashboard: http://localhost:8010                    ║
║  API Docs:  http://localhost:8010/docs               ║
╚══════════════════════════════════════════════════════╝
    """)
    uvicorn.run("web.main:app", host="0.0.0.0", port=8010,
                reload=False, log_level="info",
                ws_ping_interval=20, ws_ping_timeout=20)
