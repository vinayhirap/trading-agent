"""
Run from trading-agent root:
    python patch_base_nav.py
"""
from pathlib import Path

BASE = Path("web/templates/base.html")
assert BASE.exists(), f"Not found: {BASE}"

OLD = """  <span class="nav-label">Intelligence</span>
  <a href="/news"     class="nav-link {{ 'active' if active_page == 'news' }}"><span class="nav-icon">📰</span>News Feed</a>
  <a href="/ai"       class="nav-link {{ 'active' if active_page == 'ai' }}"><span class="nav-icon">🤖</span>AI Insights</a>
  <a href="/events"   class="nav-link {{ 'active' if active_page == 'events' }}"><span class="nav-icon">🌍</span>Events</a>
  <a href="/timing"   class="nav-link {{ 'active' if active_page == 'timing' }}"><span class="nav-icon">⏰</span>Timing</a>

  <span class="nav-label">Trading</span>
  <a href="/trade"    class="nav-link {{ 'active' if active_page == 'trade' }}"><span class="nav-icon">📟</span>Trade</a>
  <a href="/alerts"   class="nav-link {{ 'active' if active_page == 'alerts' }}"><span class="nav-icon">🔔</span>Alerts</a>
  <a href="/portfolio"class="nav-link {{ 'active' if active_page == 'portfolio' }}"><span class="nav-icon">💼</span>Portfolio</a>
  <a href="/backtest" class="nav-link {{ 'active' if active_page == 'backtest' }}"><span class="nav-icon">📊</span>Backtest</a>

  <span class="nav-label">System</span>
  <a href="/accuracy" class="nav-link {{ 'active' if active_page == 'accuracy' }}"><span class="nav-icon">🎯</span>Accuracy</a>
  <a href="/status"   class="nav-link {{ 'active' if active_page == 'status' }}"><span class="nav-icon">⚙️</span>Status</a>"""

NEW = """  <span class="nav-label">Intelligence</span>
  <a href="/news"         class="nav-link {{ 'active' if active_page == 'news' }}"><span class="nav-icon">📰</span>News Feed</a>
  <a href="/nerve-center" class="nav-link {{ 'active' if active_page == 'nerve_center' }}"><span class="nav-icon">🧠</span>Nerve Center</a>
  <a href="/ai"           class="nav-link {{ 'active' if active_page == 'ai' }}"><span class="nav-icon">🤖</span>AI Insights</a>
  <a href="/events"       class="nav-link {{ 'active' if active_page == 'events' }}"><span class="nav-icon">🌍</span>Events</a>
  <a href="/timing"       class="nav-link {{ 'active' if active_page == 'timing' }}"><span class="nav-icon">⏰</span>Timing</a>

  <span class="nav-label">Analytics</span>
  <a href="/regime"       class="nav-link {{ 'active' if active_page == 'regime' }}"><span class="nav-icon">🔭</span>Regime Detector</a>
  <a href="/fii-dii"      class="nav-link {{ 'active' if active_page == 'fii_dii' }}"><span class="nav-icon">📡</span>FII/DII Tracker</a>
  <a href="/options-oi"   class="nav-link {{ 'active' if active_page == 'options_oi' }}"><span class="nav-icon">📊</span>Options OI</a>

  <span class="nav-label">Trading</span>
  <a href="/trade"        class="nav-link {{ 'active' if active_page == 'trade' }}"><span class="nav-icon">📟</span>Trade</a>
  <a href="/alerts"       class="nav-link {{ 'active' if active_page == 'alerts' }}"><span class="nav-icon">🔔</span>Alerts</a>
  <a href="/portfolio"    class="nav-link {{ 'active' if active_page == 'portfolio' }}"><span class="nav-icon">💼</span>Portfolio</a>
  <a href="/backtest"     class="nav-link {{ 'active' if active_page == 'backtest' }}"><span class="nav-icon">📊</span>Backtest</a>

  <span class="nav-label">System</span>
  <a href="/accuracy"     class="nav-link {{ 'active' if active_page == 'accuracy' }}"><span class="nav-icon">🎯</span>Accuracy</a>
  <a href="/status"       class="nav-link {{ 'active' if active_page == 'status' }}"><span class="nav-icon">⚙️</span>Status</a>
  <a href="/system-health"class="nav-link {{ 'active' if active_page == 'system_health' }}"><span class="nav-icon">🏥</span>System Health</a>"""

content = BASE.read_text(encoding="utf-8")
if OLD not in content:
    print("❌ Pattern not found — base.html may have been modified. Patch manually.")
else:
    BASE.write_text(content.replace(OLD, NEW, 1), encoding="utf-8")
    print("✅ base.html patched — 5 new nav links added.")