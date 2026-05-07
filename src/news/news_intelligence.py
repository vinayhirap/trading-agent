# trading-agent/src/news/news_intelligence.py
"""
News Intelligence Engine — unified parallel news system.

Wraps and extends the existing NewsManager without breaking it.
Adds: parallel fetching, symbol linking, sector sentiment,
      historical event patterns, macro event detection.

Usage:
    from src.news.news_intelligence import news_intelligence

    # Get all news with symbol linkages (parallel fetch)
    articles = news_intelligence.get_all_linked()

    # Get symbol-specific news + sentiment
    result = news_intelligence.get_symbol_news("NIFTY50")
    # result = {"sentiment": {...}, "articles": [...], "macro_events": [...]}

    # Get sector sentiment
    banking_sentiment = news_intelligence.get_sector_sentiment("banking")

    # Get market-wide news summary
    summary = news_intelligence.get_market_summary()
"""
import time
from datetime import datetime, timezone
from loguru import logger

from src.news.news_manager import NewsManager, ScoredNewsItem
from src.news.parallel_fetcher import ParallelFetcher, parallel_fetcher
from src.news.symbol_linker import SymbolLinker, symbol_linker, LinkedArticle


class NewsIntelligence:
    """
    Unified news intelligence engine.

    Architecture:
      1. ParallelFetcher: fetch all feeds concurrently (~2s vs 15s sequential)
      2. NewsManager: score/sentiment each article (existing VADER logic)
      3. SymbolLinker: tag each article with relevant symbols + sectors
      4. Cache the linked articles for 2 min

    Backward compatible: existing code using NewsManager still works.
    This class adds on top without modifying NewsManager.
    """

    CACHE_TTL = 120   # 2 minutes

    def __init__(self):
        self._news_mgr  = NewsManager()
        self._fetcher   = parallel_fetcher
        self._linker    = symbol_linker
        self._cache: dict = {}
        logger.info("NewsIntelligence ready")

    # ── Main public API ───────────────────────────────────────────────────────

    def get_all_linked(self, max_age: float = 120) -> list[LinkedArticle]:
        """
        Fetch all feeds in parallel, score with VADER, link to symbols.
        Returns list of LinkedArticle sorted by relevance.
        Cached for 2 minutes.
        """
        cached = self._get_cache("all_linked")
        if cached is not None:
            return cached

        # Step 1: parallel fetch all feeds
        raw_items = self._fetcher.fetch_all()

        # Step 2: score with VADER (using existing NewsManager logic)
        scored = self._score_batch(raw_items)

        # Step 3: link to symbols
        linked = self._linker.link_batch(scored)

        # Step 4: filter by age and sort by relevance
        fresh  = [a for a in linked if a.age_minutes <= max_age]
        fresh.sort(key=lambda a: (a.relevance, -a.age_minutes), reverse=True)

        self._set_cache("all_linked", fresh)
        logger.info(f"NewsIntelligence: {len(fresh)} linked articles ready")
        return fresh

    def get_symbol_news(
        self,
        symbol: str,
        max_age: float = 120,
        top_n:   int   = 10,
    ) -> dict:
        """
        Get news + sentiment specifically for one symbol.

        Returns:
          {
            "symbol":       str,
            "sentiment":    {"score": float, "label": str, "n": int},
            "articles":     [LinkedArticle, ...],
            "macro_events": [str, ...],  # macro events affecting this symbol
            "fetch_time_s": float,
          }
        """
        cache_key = f"sym_{symbol}_{max_age}"
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached

        t0 = time.time()

        # Use parallel-fetched + linked articles
        all_articles = self.get_all_linked(max_age=max_age)
        articles     = self._linker.get_news_for_symbol(symbol, all_articles,
                                                         max_age=max_age, top_n=top_n)
        sentiment    = self._linker.get_symbol_sentiment_score(symbol, all_articles,
                                                                max_age=max_age)

        # Collect macro events affecting this symbol
        macro_events = []
        for a in articles:
            macro_events.extend(a.macro_events)
        macro_events = list(dict.fromkeys(macro_events))   # deduplicated

        result = {
            "symbol":       symbol,
            "sentiment":    sentiment,
            "articles":     articles,
            "macro_events": macro_events,
            "fetch_time_s": round(time.time() - t0, 2),
        }

        self._set_cache(cache_key, result)
        return result

    def get_sector_sentiment(self, sector: str) -> dict:
        """Get aggregated sentiment for a market sector."""
        articles = self.get_all_linked()
        return self._linker.get_sector_sentiment(sector, articles)

    def get_market_summary(self) -> dict:
        """
        High-level market news summary across all sectors.
        Returns sector sentiment, top macro events, breaking news.
        """
        cached = self._get_cache("market_summary")
        if cached is not None:
            return cached

        articles = self.get_all_linked()

        # Sector sentiments
        sectors = ["banking", "it", "pharma", "energy", "metal", "crypto"]
        sector_sentiment = {
            s: self._linker.get_sector_sentiment(s, articles)
            for s in sectors
        }

        # All macro events (last 60 min, sorted by frequency)
        all_events: dict = {}
        for a in articles:
            if a.age_minutes <= 60:
                for ev in a.macro_events:
                    all_events[ev] = all_events.get(ev, 0) + 1
        top_events = sorted(all_events.items(), key=lambda x: x[1], reverse=True)[:10]

        # Overall market sentiment
        if articles:
            scores = [a.sentiment for a in articles if a.age_minutes <= 60]
            overall = sum(scores) / len(scores) if scores else 0.0
        else:
            overall = 0.0

        overall_label = "BULLISH" if overall > 0.05 else "BEARISH" if overall < -0.05 else "NEUTRAL"

        # Breaking news: highest relevance articles < 30 min old
        breaking = [a for a in articles if a.age_minutes <= 30][:5]

        result = {
            "overall_sentiment": {"score": round(overall, 3), "label": overall_label},
            "sector_sentiment":  sector_sentiment,
            "top_macro_events":  [{"event": e, "count": c} for e, c in top_events],
            "breaking_news":     [
                {
                    "title":    a.title[:80],
                    "symbols":  a.linked_symbols[:5],
                    "sentiment":a.sentiment_label,
                    "age_min":  round(max(0.0, a.age_minutes), 0),
                }
                for a in breaking
            ],
            "total_articles": len(articles),
            "timestamp":      datetime.now(timezone.utc).isoformat(),
        }

        self._set_cache("market_summary", result)
        return result

    def get_market_sentiment(self) -> dict:
        """Get overall market sentiment with additional metadata."""
        summary = self.get_market_summary()
        overall = summary.get("overall_sentiment", {"score": 0.0, "label": "NEUTRAL"})

        # Add additional fields expected by dashboard
        overall["n_articles"] = summary.get("total_articles", 0)
        overall["top"] = [
            {
                "title": news.get("title", ""),
                "source": "Market News",  # Could be enhanced to get actual source
                "sentiment": 0.1 if news.get("sentiment") == "BULLISH" else -0.1 if news.get("sentiment") == "BEARISH" else 0.0,
                "age_mins": news.get("age_min", 0)
            }
            for news in summary.get("breaking_news", [])[:3]
        ]

        return overall

    def get_crude_sentiment(self) -> dict:
        """Backward-compatible crude sentiment (wraps existing NewsManager)."""
        return self._news_mgr.get_crude_sentiment_score()

    def get_crude_news(self, max_age_minutes: float = 120) -> list:
        """Backward-compatible crude news (wraps existing NewsManager)."""
        return self._news_mgr.get_crude_news(max_age_minutes)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _score_batch(self, raw_items: list) -> list:
        """
        Apply VADER sentiment to raw items using NewsManager's logic.
        Returns items with sentiment added (as ScoredNewsItem-compatible dicts).
        """
        scored = []
        for item in raw_items:
            try:
                text = f"{item.title} {item.summary}"
                sentiment, label = self._news_mgr._get_sentiment(text)

                # Add sentiment attributes to raw item
                item.sentiment       = sentiment
                item.sentiment_label = label
                scored.append(item)
            except Exception:
                item.sentiment       = 0.0
                item.sentiment_label = "NEUTRAL"
                scored.append(item)
        return scored

    def _get_cache(self, key: str):
        entry = self._cache.get(key)
        if entry and (time.time() - entry[0]) < self.CACHE_TTL:
            return entry[1]
        return None

    def _set_cache(self, key: str, data):
        self._cache[key] = (time.time(), data)


# Module-level singleton
news_intelligence = NewsIntelligence()