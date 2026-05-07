#trading-agent/src/data/news_fetcher.py   

"""
News pipeline for Indian markets.

Sources (in order of speed):
1. NSE official RSS feed          — direct exchange announcements
2. Moneycontrol RSS               — fastest Indian financial news
3. Economic Times Markets RSS     — broad market coverage
4. NewsAPI.org                    — aggregated (free: 100 req/day)

Latency reality check:
- RSS feeds: 1–5 min after publication
- NewsAPI free tier: up to 15 min delay
- For faster news: Refinitiv/Bloomberg terminals cost lakhs/month
- Our edge: we process faster than most retail traders read news
"""
import feedparser
import requests
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger
from config.settings import settings


@dataclass
class NewsItem:
    title:      str
    source:     str
    url:        str
    published:  datetime
    summary:    str = ""
    symbols:    list[str] = field(default_factory=list)
    sentiment:  Optional[float] = None   # -1.0 to +1.0 (filled by NLP in Step 4)

    @property
    def age_minutes(self) -> float:
        now = datetime.now(timezone.utc)
        pub = self.published
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        return (now - pub).total_seconds() / 60


# ── RSS feeds (free, no API key needed) ──────────────────────────────────────
RSS_FEEDS = {
    "nse_announcements": "https://www.nseindia.com/api/rss",  # requires headers
    "moneycontrol":      "https://www.moneycontrol.com/rss/marketoutlook.xml",
    "economic_times":    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "business_standard": "https://www.business-standard.com/rss/markets-106.rss",
    "livemint_markets":  "https://www.livemint.com/rss/markets",
}

# NSE watchlist — news scanner will flag these symbols
WATCHLIST_KEYWORDS = {
    "RELIANCE": ["reliance", "ril", "mukesh ambani"],
    "TCS":      ["tcs", "tata consultancy"],
    "HDFCBANK": ["hdfc bank", "hdfcbank"],
    "INFY":     ["infosys", "infy"],
    "NIFTY50":  ["nifty", "sensex", "market rally", "market crash", "sebi"],
}


class NewsFetcher:
    """
    Fetches and parses news from multiple RSS sources.
    Run every 2–5 minutes during market hours.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (TradingAgent/1.0)"
        })

    def fetch_rss(self, url: str, source_name: str) -> list[NewsItem]:
        """Parse a single RSS feed."""
        items = []
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:20]:   # cap at 20 per feed
                try:
                    published = self._parse_date(entry.get("published", ""))
                    item = NewsItem(
                        title     = entry.get("title", "").strip(),
                        source    = source_name,
                        url       = entry.get("link", ""),
                        published = published,
                        summary   = entry.get("summary", "")[:500],
                        symbols   = self._extract_symbols(
                            entry.get("title", "") + " " + entry.get("summary", "")
                        ),
                    )
                    items.append(item)
                except Exception as e:
                    logger.debug(f"Skipping entry: {e}")
        except Exception as e:
            logger.warning(f"RSS fetch failed [{source_name}]: {e}")
        logger.debug(f"[{source_name}] fetched {len(items)} items")
        return items

    def fetch_all(self, max_age_minutes: float = 60) -> list[NewsItem]:
        """
        Fetch from all RSS sources and return deduplicated, recent items.
        """
        all_items: list[NewsItem] = []
        for name, url in RSS_FEEDS.items():
            items = self.fetch_rss(url, name)
            all_items.extend(items)

        # Filter by age
        fresh = [i for i in all_items if i.age_minutes <= max_age_minutes]

        # Deduplicate by title similarity (basic)
        seen_titles: set[str] = set()
        deduped = []
        for item in fresh:
            key = item.title[:60].lower()
            if key not in seen_titles:
                seen_titles.add(key)
                deduped.append(item)

        # Sort newest first
        deduped.sort(key=lambda x: x.published, reverse=True)
        logger.info(f"News: {len(all_items)} fetched → {len(fresh)} fresh → {len(deduped)} deduped")
        return deduped

    def fetch_newsapi(self, query: str = "NSE India stock market", max_age_hours: int = 6) -> list[NewsItem]:
        """
        NewsAPI.org — free tier: 100 requests/day, ~15min delay.
        Register at https://newsapi.org/ for a free key.
        """
        if not settings.NEWS_API_KEY:
            logger.debug("NEWS_API_KEY not set — skipping NewsAPI")
            return []
        try:
            resp = self.session.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q":        query,
                    "language": "en",
                    "sortBy":   "publishedAt",
                    "pageSize": 20,
                    "apiKey":   settings.NEWS_API_KEY,
                },
                timeout=10,
            )
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
            items = []
            for a in articles:
                published = self._parse_date(a.get("publishedAt", ""))
                item = NewsItem(
                    title    = a.get("title", ""),
                    source   = a.get("source", {}).get("name", "newsapi"),
                    url      = a.get("url", ""),
                    published= published,
                    summary  = (a.get("description") or "")[:500],
                    symbols  = self._extract_symbols(
                        a.get("title","") + " " + (a.get("description") or "")
                    ),
                )
                items.append(item)
            logger.info(f"NewsAPI: {len(items)} articles")
            return items
        except Exception as e:
            logger.warning(f"NewsAPI error: {e}")
            return []

    def _extract_symbols(self, text: str) -> list[str]:
        """Tag news items with relevant symbols from our watchlist."""
        text_lower = text.lower()
        matched = []
        for symbol, keywords in WATCHLIST_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                matched.append(symbol)
        return matched

    def _parse_date(self, date_str: str) -> datetime:
        """Parse various date formats from RSS feeds."""
        if not date_str:
            return datetime.now(timezone.utc)
        import email.utils
        try:
            # RFC 2822 (most RSS feeds)
            parsed = email.utils.parsedate_to_datetime(date_str)
            return parsed.astimezone(timezone.utc)
        except Exception:
            pass
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
            try:
                dt = datetime.strptime(date_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue
        return datetime.now(timezone.utc)