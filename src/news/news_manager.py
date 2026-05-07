"""
Advanced News Manager — extends existing NewsFetcher.
Adds: VADER sentiment, scoring, GNews, deduplication, caching.
Free APIs only. Cache TTL: 3 minutes.
"""
import time
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional
import requests
import feedparser
from loguru import logger

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

SOURCE_SCORES = {
    "reuters":           1.0,
    "bloomberg":         1.0,
    "economic_times":    0.9,
    "moneycontrol":      0.9,
    "business_standard": 0.85,
    "livemint":          0.85,
    "oilprice":          0.85,
    "eia_news":          0.90,
    "cnbc":              0.85,
    "gnews":             0.70,
    "newsapi":           0.65,
}

CRUDE_RSS_FEEDS = {
    "oilprice":       "https://oilprice.com/rss/main",
    "reuters_energy": "https://feeds.reuters.com/reuters/businessNews",
}

INDIA_RSS_FEEDS = {
    "moneycontrol":   "https://www.moneycontrol.com/rss/marketoutlook.xml",
    "economic_times": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "business_std":   "https://www.business-standard.com/rss/markets-106.rss",
    "livemint":       "https://www.livemint.com/rss/markets",
}

CRUDE_KEYWORDS = {
    "bullish": ["supply cut", "opec cut", "production cut", "inventory draw",
                "demand rise", "sanctions", "output cut", "hurricane", "refinery fire"],
    "bearish": ["supply increase", "opec increase", "recession", "demand fall",
                "inventory build", "surplus", "overproduction", "rate hike"],
    "neutral": ["opec", "crude", "oil price", "mcx crude", "wti", "brent",
                "barrel", "petroleum", "eia", "api report"],
}

INDIA_KEYWORDS = {
    "bullish": ["rate cut", "gdp growth", "fii buying", "strong earnings", "stimulus"],
    "bearish": ["rate hike", "inflation", "fii selling", "weak earnings", "recession"],
    "neutral": ["nifty", "sensex", "market", "nse", "bse", "sebi", "reliance"],
}


@dataclass
class ScoredNewsItem:
    title:           str
    source:          str
    url:             str
    published:       datetime
    summary:         str
    sentiment:       float
    sentiment_label: str
    score:           float
    keywords:        list
    asset_type:      str
    age_minutes:     float

    def to_dict(self) -> dict:
        return {
            "title":           self.title,
            "source":          self.source,
            "url":             self.url,
            "published":       self.published.isoformat(),
            "summary":         self.summary[:200],
            "sentiment":       self.sentiment,
            "sentiment_label": self.sentiment_label,
            "score":           self.score,
            "age_mins":        round(self.age_minutes, 1),
            "asset_type":      self.asset_type,
        }


class NewsManager:
    """
    Upgraded news system with sentiment scoring and caching.
    Falls back gracefully when feeds are unavailable.
    """

    CACHE_TTL = 180  # 3 minutes

    def __init__(self):
        self._cache: dict = {}
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 TradingAgent/1.0"
        })
        self._vader = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None
        logger.info(f"NewsManager ready | VADER={'yes' if VADER_AVAILABLE else 'no'}")

    def get_crude_news(self, max_age_minutes: float = 120) -> list:
        return self._get_cached(
            "crude",
            lambda: self._fetch_and_score(
                CRUDE_RSS_FEEDS, CRUDE_KEYWORDS, "crude", max_age_minutes
            )
        )

    def get_india_news(self, max_age_minutes: float = 60) -> list:
        return self._get_cached(
            "india",
            lambda: self._fetch_and_score(
                INDIA_RSS_FEEDS, INDIA_KEYWORDS, "india_market", max_age_minutes
            )
        )

    def get_all_news(self) -> list:
        crude = self.get_crude_news()
        india = self.get_india_news()
        combined = crude + india
        seen = set()
        result = []
        for item in combined:
            h = self._title_hash(item.title)
            if h not in seen:
                seen.add(h)
                result.append(item)
        return sorted(result, key=lambda x: x.score, reverse=True)

    def get_crude_sentiment_score(self) -> dict:
        news = self.get_crude_news(max_age_minutes=60)
        if not news:
            return {"score": 0.0, "label": "NEUTRAL", "n_articles": 0,
                    "bullish_n": 0, "bearish_n": 0, "top": []}

        weighted_scores = []
        for item in news:
            w = 2.0 if item.age_minutes < 30 else 1.5 if item.age_minutes < 60 else 1.0
            weighted_scores.append(item.sentiment * w * item.score)

        agg = sum(weighted_scores) / len(weighted_scores)
        label = "BULLISH" if agg > 0.1 else "BEARISH" if agg < -0.1 else "NEUTRAL"

        return {
            "score":     round(agg, 3),
            "label":     label,
            "n_articles":len(news),
            "bullish_n": sum(1 for n in news if n.sentiment > 0.05),
            "bearish_n": sum(1 for n in news if n.sentiment < -0.05),
            "top":       [n.to_dict() for n in news[:5]],
        }

    def _fetch_and_score(
        self,
        feeds: dict,
        keyword_dict: dict,
        asset_type: str,
        max_age_minutes: float,
    ) -> list:
        raw_items = []
        for source_name, url in feeds.items():
            raw_items.extend(self._parse_rss(url, source_name, asset_type))

        # Filter by age
        fresh = [i for i in raw_items if i.age_minutes <= max_age_minutes]

        # Deduplicate
        seen, dedup = set(), []
        for item in fresh:
            h = self._title_hash(item.title)
            if h not in seen:
                seen.add(h)
                dedup.append(item)

        # Score
        for item in dedup:
            item.score    = self._compute_score(item, keyword_dict)
            item.keywords = self._extract_keywords(
                f"{item.title} {item.summary}", keyword_dict
            )

        dedup.sort(key=lambda x: x.score, reverse=True)
        logger.info(f"{asset_type} news: {len(dedup)} items")
        return dedup[:30]

    def _parse_rss(self, url: str, source: str, asset_type: str) -> list:
        items = []
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:15]:
                try:
                    published = self._parse_date(entry.get("published", ""))
                    age_mins  = (
                        datetime.now(timezone.utc) - published
                    ).total_seconds() / 60
                    title   = entry.get("title", "").strip()
                    summary = entry.get("summary", "")[:400]
                    text    = f"{title} {summary}"
                    sentiment, label = self._get_sentiment(text)
                    items.append(ScoredNewsItem(
                        title=title, source=source,
                        url=entry.get("link", ""),
                        published=published,
                        summary=summary,
                        sentiment=sentiment,
                        sentiment_label=label,
                        score=0.0,
                        keywords=[],
                        asset_type=asset_type,
                        age_minutes=age_mins,
                    ))
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"RSS {source}: {e}")
        return items

    def _compute_score(self, item, keyword_dict: dict) -> float:
        recency       = max(0, 1 - item.age_minutes / 240)
        source_score  = SOURCE_SCORES.get(item.source, 0.5)
        text          = f"{item.title} {item.summary}".lower()
        all_kw        = (
            keyword_dict.get("bullish", []) +
            keyword_dict.get("bearish", []) +
            keyword_dict.get("neutral", [])
        )
        hits          = sum(1 for kw in all_kw if kw in text)
        keyword_score = min(1.0, hits / 3)
        sent_strength = abs(item.sentiment)
        return round(
            0.35 * recency +
            0.25 * source_score +
            0.25 * keyword_score +
            0.15 * sent_strength, 3
        )

    def _get_sentiment(self, text: str) -> tuple:
        if not self._vader:
            return 0.0, "NEUTRAL"
        try:
            s = self._vader.polarity_scores(text)
            c = s["compound"]
            if c > 0.05:  return c, "BULLISH"
            if c < -0.05: return c, "BEARISH"
            return c, "NEUTRAL"
        except Exception:
            return 0.0, "NEUTRAL"

    def _extract_keywords(self, text: str, keyword_dict: dict) -> list:
        tl = text.lower()
        found = []
        for cat, kws in keyword_dict.items():
            for kw in kws:
                if kw in tl:
                    found.append(kw)
        return found[:5]

    def _title_hash(self, title: str) -> str:
        return hashlib.md5(title[:60].lower().encode()).hexdigest()

    def _parse_date(self, date_str: str) -> datetime:
        if not date_str:
            return datetime.now(timezone.utc)
        import email.utils
        try:
            return email.utils.parsedate_to_datetime(date_str).astimezone(timezone.utc)
        except Exception:
            pass
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z"):
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        return datetime.now(timezone.utc)

    def _get_cached(self, key: str, fetcher) -> list:
        now = time.time()
        if key in self._cache:
            ts, data = self._cache[key]
            if now - ts < self.CACHE_TTL:
                return data
        try:
            data = fetcher()
        except Exception as e:
            logger.warning(f"News fetch failed for {key}: {e}")
            data = []
        self._cache[key] = (now, data)
        return data
