"""
Low-latency latest news service for the dashboard.

Merges fast API feeds (NewsAPI, GNews) with the existing RSS/parallel fetcher,
deduplicates headlines, and returns a newest-first stream suitable for a live
market-news page.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import md5
from typing import Optional

import requests
from loguru import logger

from config.settings import settings
from src.news.parallel_fetcher import parallel_fetcher


@dataclass
class LatestNewsItem:
    title: str
    source: str
    url: str
    published: datetime
    summary: str
    category: str
    region: str

    @property
    def age_minutes(self) -> float:
        return max(0.0, (datetime.now(timezone.utc) - self.published).total_seconds() / 60)


class LatestNewsService:
    CACHE_TTL = 45

    CATEGORY_QUERY = {
        "All": None,
        "India Markets": "india stock market OR nifty OR sensex OR bank nifty",
        "Global Markets": "global markets OR stocks OR bonds OR central bank",
        "Commodities": "crude oil OR gold OR silver OR opec",
        "Crypto": "bitcoin OR ethereum OR crypto market",
        "Geopolitics": "geopolitics OR war OR sanctions OR trade tensions OR elections",
    }

    CATEGORY_RSS = {
        "All": ["india_market", "global", "crude_oil", "crypto"],
        "India Markets": ["india_market"],
        "Global Markets": ["global"],
        "Commodities": ["crude_oil", "gold"],
        "Crypto": ["crypto"],
        "Geopolitics": ["global"],
    }

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "TradingAgent/LatestNews"})
        self._cache: dict[tuple[str, int], tuple[float, list[LatestNewsItem]]] = {}

    def clear_cache(self) -> None:
        self._cache.clear()

    def get_latest(self, category: str = "All", limit: int = 40) -> list[LatestNewsItem]:
        key = (category, limit)
        now = time.time()
        cached = self._cache.get(key)
        if cached and now - cached[0] < self.CACHE_TTL:
            return cached[1]

        items: list[LatestNewsItem] = []
        query = self.CATEGORY_QUERY.get(category)
        rss_categories = self.CATEGORY_RSS.get(category, ["india_market", "global"])

        items.extend(self._fetch_newsapi(query=query, limit=limit))
        items.extend(self._fetch_gnews(query=query, limit=limit))
        items.extend(self._fetch_rss(rss_categories=rss_categories, limit=limit))

        deduped = self._dedupe(items)
        deduped.sort(key=lambda item: item.published, reverse=True)
        result = deduped[:limit]
        self._cache[key] = (now, result)
        logger.info(f"LatestNewsService: {len(result)} items for {category}")
        return result

    def _fetch_newsapi(self, query: Optional[str], limit: int) -> list[LatestNewsItem]:
        api_key = getattr(settings, "NEWS_API_KEY", None)
        if not api_key:
            return []

        params = {
            "apiKey": api_key,
            "language": "en",
            "pageSize": min(limit, 50),
            "sortBy": "publishedAt",
        }
        if query:
            url = "https://newsapi.org/v2/everything"
            params["q"] = query
        else:
            url = "https://newsapi.org/v2/top-headlines"
            params["category"] = "business"
        try:
            resp = self._session.get(url, params=params, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            out = []
            for article in data.get("articles", []):
                published = self._parse_date(article.get("publishedAt"))
                out.append(
                    LatestNewsItem(
                        title=article.get("title", "").strip(),
                        source=(article.get("source") or {}).get("name", "NewsAPI"),
                        url=article.get("url", ""),
                        published=published,
                        summary=(article.get("description") or article.get("content") or "")[:600],
                        category=self._guess_category(article.get("title", ""), article.get("description", "")),
                        region=self._guess_region(article.get("title", ""), article.get("description", "")),
                    )
                )
            return out
        except Exception as exc:
            logger.debug(f"LatestNewsService NewsAPI: {exc}")
            return []

    def _fetch_gnews(self, query: Optional[str], limit: int) -> list[LatestNewsItem]:
        api_key = getattr(settings, "GNEWS_API_KEY", None)
        if not api_key:
            return []

        params = {
            "token": api_key,
            "lang": "en",
            "max": min(limit, 50),
            "sortby": "publishedAt",
        }
        if query:
            url = "https://gnews.io/api/v4/search"
            params["q"] = query
        else:
            url = "https://gnews.io/api/v4/top-headlines"
            params["topic"] = "business"
        try:
            resp = self._session.get(url, params=params, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            out = []
            for article in data.get("articles", []):
                published = self._parse_date(article.get("publishedAt"))
                source_name = (article.get("source") or {}).get("name", "GNews")
                out.append(
                    LatestNewsItem(
                        title=article.get("title", "").strip(),
                        source=source_name,
                        url=article.get("url", ""),
                        published=published,
                        summary=(article.get("description") or article.get("content") or "")[:600],
                        category=self._guess_category(article.get("title", ""), article.get("description", "")),
                        region=self._guess_region(article.get("title", ""), article.get("description", "")),
                    )
                )
            return out
        except Exception as exc:
            logger.debug(f"LatestNewsService GNews: {exc}")
            return []

    def _fetch_rss(self, rss_categories: list[str], limit: int) -> list[LatestNewsItem]:
        try:
            raw = parallel_fetcher.fetch_all(categories=rss_categories)
            out = []
            for item in raw[:limit]:
                out.append(
                    LatestNewsItem(
                        title=item.title.strip(),
                        source=item.source,
                        url=item.url,
                        published=item.published,
                        summary=item.summary[:600],
                        category=self._map_feed_category(item.feed_cat),
                        region="India" if item.feed_cat == "india_market" else "Global",
                    )
                )
            return out
        except Exception as exc:
            logger.debug(f"LatestNewsService RSS: {exc}")
            return []

    @staticmethod
    def _parse_date(value: Optional[str]) -> datetime:
        if not value:
            return datetime.now(timezone.utc)
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S%z"):
            try:
                dt = datetime.strptime(value, fmt)
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return datetime.now(timezone.utc)

    @staticmethod
    def _guess_category(title: str, summary: str) -> str:
        text = f"{title} {summary}".lower()
        if any(word in text for word in ("bitcoin", "ethereum", "crypto", "token", "blockchain")):
            return "Crypto"
        if any(word in text for word in ("gold", "silver", "crude", "oil", "opec", "commodity")):
            return "Commodities"
        if any(word in text for word in ("war", "sanctions", "geopolit", "election", "tariff")):
            return "Geopolitics"
        if any(word in text for word in ("nifty", "sensex", "bank nifty", "india market", "sebi", "fii")):
            return "India Markets"
        return "Global Markets"

    @staticmethod
    def _guess_region(title: str, summary: str) -> str:
        text = f"{title} {summary}".lower()
        if any(word in text for word in ("india", "nifty", "sensex", "sebi", "rbi", "rupee")):
            return "India"
        return "Global"

    @staticmethod
    def _map_feed_category(feed_cat: str) -> str:
        mapping = {
            "india_market": "India Markets",
            "global": "Global Markets",
            "crude_oil": "Commodities",
            "gold": "Commodities",
            "crypto": "Crypto",
            "forex": "Global Markets",
            "gnews": "Global Markets",
        }
        return mapping.get(feed_cat, "Global Markets")

    @staticmethod
    def _dedupe(items: list[LatestNewsItem]) -> list[LatestNewsItem]:
        seen = set()
        out = []
        for item in items:
            title = item.title.strip()
            if not title:
                continue
            key = md5(title.lower()[:80].encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out
