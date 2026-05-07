# trading-agent/src/news/symbol_linker.py
"""
Symbol Linker — maps news articles to specific trading symbols and sectors.

This is the missing link in the current system. Right now news is only
tagged as "crude" or "india_market". With symbol linking:

  "Reliance Q3 profit up 18%" → linked to RELIANCE (equity)
  "OPEC cuts output by 1mbpd" → linked to CRUDEOIL, GOLD (macro)
  "Bitcoin ETF approved"      → linked to BTC, ETH, crypto sector
  "RBI holds rates at 6.5%"   → linked to BANKNIFTY, USDINR, HDFCBANK

Linkage methods (in priority order):
  1. Exact ticker match (RELIANCE, TCS in title)
  2. Company name match ("Tata Consultancy" → TCS)
  3. Macro keyword match ("rate hike" → BANKNIFTY, HDFCBANK, USDINR)
  4. Sector match ("pharma" → SUNPHARMA, DRREDDY)
  5. Global event match ("oil" → CRUDEOIL, ONGC)

Output: each article gets a list of linked symbols + a relevance score (0-1).
Higher score = more directly relevant to that symbol.
"""
from dataclasses import dataclass, field
from loguru import logger


# ── Company name → symbol mapping ─────────────────────────────────────────────
COMPANY_NAME_MAP = {
    # Exact ticker aliases
    "reliance":          ["RELIANCE"],
    "tata consultancy":  ["TCS"],
    " tcs ":             ["TCS"],
    "hdfc bank":         ["HDFCBANK"],
    "hdfc":              ["HDFCBANK"],
    "infosys":           ["INFY"],
    "icici bank":        ["ICICIBANK"],
    "icici":             ["ICICIBANK"],
    "sbi":               ["SBIN"],
    "state bank":        ["SBIN"],
    "wipro":             ["WIPRO"],
    "axis bank":         ["AXISBANK"],
    "kotak":             ["KOTAKBANK"],
    "larsen":            ["LT"],
    "l&t":               ["LT"],
    "bajaj finance":     ["BAJFINANCE"],
    "maruti":            ["MARUTI"],
    "sun pharma":        ["SUNPHARMA"],
    "sunpharma":         ["SUNPHARMA"],
    "bharti airtel":     ["BHARTIARTL"],
    "airtel":            ["BHARTIARTL"],
    "tata motors":       ["TATAMOTORS"],
    "bitcoin":           ["BTC"],
    "btc":               ["BTC"],
    "ethereum":          ["ETH"],
    "eth":               ["ETH"],
    "crude oil":         ["CRUDEOIL"],
    "crude":             ["CRUDEOIL"],
    "wti":               ["CRUDEOIL"],
    "brent":             ["CRUDEOIL"],
    "gold":              ["GOLD"],
    "silver":            ["SILVER"],
    "copper":            ["COPPER"],
    "natural gas":       ["NATURALGAS"],
    "nifty":             ["NIFTY50"],
    "sensex":            ["SENSEX"],
    "bank nifty":        ["BANKNIFTY"],
    "banknifty":         ["BANKNIFTY"],
}

# ── Macro event → affected symbols ────────────────────────────────────────────
MACRO_EVENT_MAP = {
    # Monetary policy
    "rate cut":          {"symbols": ["BANKNIFTY", "HDFCBANK", "ICICIBANK", "SBIN"], "score": 0.7},
    "rate hike":         {"symbols": ["BANKNIFTY", "HDFCBANK", "ICICIBANK", "SBIN", "USDINR"], "score": 0.7},
    "rbi":               {"symbols": ["BANKNIFTY", "HDFCBANK", "USDINR"], "score": 0.6},
    "repo rate":         {"symbols": ["BANKNIFTY", "HDFCBANK", "SBIN"], "score": 0.75},
    "federal reserve":   {"symbols": ["NIFTY50", "USDINR", "GOLD", "BTC"], "score": 0.6},
    "fed rate":          {"symbols": ["NIFTY50", "USDINR", "GOLD", "BTC"], "score": 0.65},
    "inflation":         {"symbols": ["NIFTY50", "GOLD", "CRUDEOIL"], "score": 0.55},

    # Oil & Energy
    "opec":              {"symbols": ["CRUDEOIL", "ONGC"], "score": 0.9},
    "oil production":    {"symbols": ["CRUDEOIL", "ONGC"], "score": 0.85},
    "oil supply":        {"symbols": ["CRUDEOIL"], "score": 0.8},
    "oil demand":        {"symbols": ["CRUDEOIL"], "score": 0.8},
    "iran":              {"symbols": ["CRUDEOIL", "GOLD"], "score": 0.7},
    "russia":            {"symbols": ["CRUDEOIL", "GOLD", "NATURALGAS"], "score": 0.65},
    "ukraine":           {"symbols": ["CRUDEOIL", "GOLD", "NATURALGAS"], "score": 0.65},
    "middle east":       {"symbols": ["CRUDEOIL", "GOLD"], "score": 0.65},
    "sanctions":         {"symbols": ["CRUDEOIL", "GOLD"], "score": 0.7},

    # Gold & Safe haven
    "safe haven":        {"symbols": ["GOLD", "SILVER"], "score": 0.8},
    "dollar index":      {"symbols": ["GOLD", "USDINR", "CRUDEOIL"], "score": 0.7},
    "dxy":               {"symbols": ["GOLD", "USDINR"], "score": 0.7},
    "recession":         {"symbols": ["GOLD", "NIFTY50", "BTC"], "score": 0.65},
    "war":               {"symbols": ["GOLD", "CRUDEOIL", "NIFTY50"], "score": 0.75},
    "conflict":          {"symbols": ["GOLD", "CRUDEOIL"], "score": 0.65},

    # Crypto
    "bitcoin etf":       {"symbols": ["BTC"], "score": 0.95},
    "crypto regulation": {"symbols": ["BTC", "ETH"], "score": 0.8},
    "sec":               {"symbols": ["BTC", "ETH"], "score": 0.6},
    "blockchain":        {"symbols": ["BTC", "ETH"], "score": 0.5},
    "halving":           {"symbols": ["BTC"], "score": 0.9},

    # India macro
    "fii":               {"symbols": ["NIFTY50", "BANKNIFTY"], "score": 0.7},
    "dii":               {"symbols": ["NIFTY50"], "score": 0.6},
    "sebi":              {"symbols": ["NIFTY50"], "score": 0.55},
    "budget":            {"symbols": ["NIFTY50", "BANKNIFTY", "CRUDEOIL"], "score": 0.7},
    "gdp":               {"symbols": ["NIFTY50", "USDINR"], "score": 0.6},
    "rupee":             {"symbols": ["USDINR", "NIFTY50"], "score": 0.75},
    "usdinr":            {"symbols": ["USDINR"], "score": 0.9},
    "earnings":          {"symbols": ["NIFTY50"], "score": 0.5},  # generic earnings
}

# ── Sector → symbols mapping ──────────────────────────────────────────────────
SECTOR_MAP = {
    "banking":     ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "BANKNIFTY"],
    "it":          ["TCS", "INFY", "WIPRO"],
    "technology":  ["TCS", "INFY", "WIPRO"],
    "pharma":      ["SUNPHARMA"],
    "telecom":     ["BHARTIARTL"],
    "auto":        ["MARUTI", "TATAMOTORS"],
    "energy":      ["CRUDEOIL", "NATURALGAS", "ONGC"],
    "metal":       ["GOLD", "SILVER", "COPPER"],
    "crypto":      ["BTC", "ETH"],
}


@dataclass
class LinkedArticle:
    """A news article with its symbol linkages."""
    title:         str
    source:        str
    url:           str
    summary:       str
    sentiment:     float
    sentiment_label: str
    age_minutes:   float
    feed_cat:      str

    # Linkage output
    linked_symbols:  list[str] = field(default_factory=list)
    link_scores:     dict      = field(default_factory=dict)   # symbol → relevance 0-1
    link_reasons:    list[str] = field(default_factory=list)   # why each symbol was linked
    sectors:         list[str] = field(default_factory=list)
    macro_events:    list[str] = field(default_factory=list)
    relevance:       float     = 0.0   # overall relevance score


class SymbolLinker:
    """
    Links news articles to specific trading symbols.

    Usage:
        linker = SymbolLinker()
        articles = linker.link_batch(raw_items, scored_items)
        nifty_news = linker.get_news_for_symbol("NIFTY50", articles)
    """

    def link_article(self, title: str, summary: str, source: str,
                     sentiment: float, sentiment_label: str,
                     age_minutes: float, feed_cat: str,
                     url: str = "") -> LinkedArticle:
        """
        Link a single article to symbols. Returns LinkedArticle with all linkages.
        """
        article = LinkedArticle(
            title           = title,
            source          = source,
            url             = url,
            summary         = summary,
            sentiment       = sentiment,
            sentiment_label = sentiment_label,
            age_minutes     = age_minutes,
            feed_cat        = feed_cat,
        )

        text = f"{title} {summary}".lower()

        # Pass 1: company name / ticker matching (highest specificity)
        self._link_company_names(text, article)

        # Pass 2: macro event matching
        self._link_macro_events(text, article)

        # Pass 3: sector matching
        self._link_sectors(text, article)

        # Deduplicate linked symbols
        article.linked_symbols = list(dict.fromkeys(article.linked_symbols))

        # Overall relevance: mean of symbol-specific scores
        if article.link_scores:
            article.relevance = round(
                sum(article.link_scores.values()) / len(article.link_scores), 3
            )

        return article

    def link_batch(self, items: list) -> list[LinkedArticle]:
        """
        Link a batch of items. items can be RawNewsItem or ScoredNewsItem.
        Returns list of LinkedArticle.
        """
        results = []
        for item in items:
            try:
                title    = getattr(item, "title",    "")
                summary  = getattr(item, "summary",  "")
                source   = getattr(item, "source",   "unknown")
                sent     = getattr(item, "sentiment", 0.0)
                sent_lbl = getattr(item, "sentiment_label", "NEUTRAL")
                age      = getattr(item, "age_minutes", 0.0)
                cat      = getattr(item, "feed_cat",  getattr(item, "asset_type", "general"))
                url      = getattr(item, "url",       "")

                linked = self.link_article(
                    title=title, summary=summary, source=source,
                    sentiment=sent, sentiment_label=sent_lbl,
                    age_minutes=age, feed_cat=cat, url=url,
                )
                results.append(linked)
            except Exception as e:
                logger.debug(f"link_batch item failed: {e}")
        return results

    def get_news_for_symbol(
        self,
        symbol: str,
        articles: list[LinkedArticle],
        max_age: float = 120,
        top_n:   int   = 10,
    ) -> list[LinkedArticle]:
        """
        Filter articles linked to a specific symbol, sorted by relevance.
        """
        relevant = [
            a for a in articles
            if symbol in a.linked_symbols and a.age_minutes <= max_age
        ]
        relevant.sort(key=lambda a: (a.link_scores.get(symbol, 0), -a.age_minutes),
                      reverse=True)
        return relevant[:top_n]

    def get_sector_sentiment(
        self,
        sector: str,
        articles: list[LinkedArticle],
    ) -> dict:
        """
        Aggregate sentiment for all articles touching a sector.
        """
        sector_syms = set(SECTOR_MAP.get(sector.lower(), []))
        relevant    = [
            a for a in articles
            if any(s in sector_syms for s in a.linked_symbols)
        ]
        if not relevant:
            return {"score": 0.0, "label": "NEUTRAL", "n": 0}

        scores = [a.sentiment * a.relevance for a in relevant]
        agg    = sum(scores) / len(scores)
        label  = "BULLISH" if agg > 0.05 else "BEARISH" if agg < -0.05 else "NEUTRAL"
        return {"score": round(agg, 3), "label": label, "n": len(relevant)}

    def get_symbol_sentiment_score(
        self,
        symbol: str,
        articles: list[LinkedArticle],
        max_age: float = 60,
    ) -> dict:
        """
        Get weighted sentiment score for a specific symbol.
        Recency-weighted: newer articles count more.
        """
        relevant = self.get_news_for_symbol(symbol, articles, max_age=max_age)
        if not relevant:
            return {"score": 0.0, "label": "NEUTRAL", "n": 0, "articles": []}

        weighted, weights = 0.0, 0.0
        for a in relevant:
            link_score = a.link_scores.get(symbol, 0.5)
            recency_w  = max(0.1, 1 - a.age_minutes / max_age)
            w          = link_score * recency_w
            weighted  += a.sentiment * w
            weights   += w

        score = weighted / weights if weights > 0 else 0.0
        label = "BULLISH" if score > 0.05 else "BEARISH" if score < -0.05 else "NEUTRAL"

        return {
            "score":    round(score, 3),
            "label":    label,
            "n":        len(relevant),
            "articles": [
                {
                    "title":     a.title[:80],
                    "sentiment": a.sentiment_label,
                    "age_m":     round(a.age_minutes, 0),
                    "relevance": round(a.link_scores.get(symbol, 0), 2),
                }
                for a in relevant[:5]
            ],
        }

    # ── Internal linking passes ───────────────────────────────────────────────

    def _link_company_names(self, text: str, article: LinkedArticle):
        for name, symbols in COMPANY_NAME_MAP.items():
            if name in text:
                for sym in symbols:
                    if sym not in article.link_scores or article.link_scores[sym] < 0.85:
                        article.link_scores[sym] = 0.85
                        if sym not in article.linked_symbols:
                            article.linked_symbols.append(sym)
                            article.link_reasons.append(f"name match: '{name}' → {sym}")

    def _link_macro_events(self, text: str, article: LinkedArticle):
        for event, info in MACRO_EVENT_MAP.items():
            if event in text:
                article.macro_events.append(event)
                for sym in info["symbols"]:
                    score = info["score"]
                    if sym not in article.link_scores or article.link_scores[sym] < score:
                        article.link_scores[sym] = score
                        if sym not in article.linked_symbols:
                            article.linked_symbols.append(sym)
                            article.link_reasons.append(f"macro: '{event}' → {sym}")

    def _link_sectors(self, text: str, article: LinkedArticle):
        for sector, symbols in SECTOR_MAP.items():
            if sector in text:
                article.sectors.append(sector)
                for sym in symbols:
                    score = 0.5   # sector match is weaker than direct name match
                    if sym not in article.link_scores or article.link_scores[sym] < score:
                        article.link_scores[sym] = score
                        if sym not in article.linked_symbols:
                            article.linked_symbols.append(sym)
                            article.link_reasons.append(f"sector: '{sector}' → {sym}")


# Module-level singleton
symbol_linker = SymbolLinker()