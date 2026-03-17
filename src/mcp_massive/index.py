import asyncio
import re
import logging
import ssl
from typing import Any

import certifi
import httpx
import numpy as np
import bm25s
from pydantic import BaseModel, Field
import snowballstemmer
from bm25s.stopwords import STOPWORDS_EN

logger = logging.getLogger(__name__)

_DEFAULT_LLMS_TXT_URL = "https://massive.com/docs/rest/llms.txt"

_snowball = snowballstemmer.stemmer("english")
_STOPWORDS = frozenset(STOPWORDS_EN)


class Endpoint(BaseModel):
    name: str = Field(min_length=1)
    category: str
    url: str
    description: str
    endpoint_pattern: str
    compressed_doc: str
    path_prefix: str


_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Aliases map abbreviations and synonyms to canonical terms that appear
# in the llms.txt endpoint names/descriptions.
# Values can be a single string or a list of strings for ambiguous terms.
ALIASES: dict[str, str | list[str]] = {
    # aggregates / bars / OHLC
    "agg": "aggregate",
    "aggs": "aggregate",
    "candle": "aggregate",
    "candles": "aggregate",
    "candlestick": "aggregate",
    "candlesticks": "aggregate",
    "ohlc": "aggregate",
    "ohlcv": "aggregate",
    "bar": "aggregate",
    "bars": "aggregate",
    "historical": "aggregate",
    "history": "aggregate",
    "vwap": "aggregate",
    # forex / currency
    "fx": "forex",
    "currency": "forex",
    "currencies": "forex",
    # crypto
    "coin": "crypto",
    "coins": "crypto",
    "token": "crypto",
    "tokens": "crypto",
    "bitcoin": "crypto",
    "btc": "crypto",
    "eth": "crypto",
    "ethereum": "crypto",
    # reference data
    "ref": "reference",
    "info": "details",
    "detail": "details",
    "lookup": "tickers",
    "symbol": "ticker",
    "symbols": "ticker",
    # trades / quotes
    "transaction": "trade",
    "transactions": "trade",
    "execution": "trade",
    "executions": "trade",
    "bid": "quote",
    "ask": "quote",
    "nbbo": "quote",
    # snapshots
    "snap": "snapshot",
    "snaps": "snapshot",
    "realtime": "snapshot",
    "real-time": "snapshot",
    "live": "snapshot",
    # financials
    "fundamental": "financial",
    "fundamentals": "financial",
    "income": "financial",
    "balance": "financial",
    "earnings": "financial",
    "revenue": "financial",
    "pe": "financial",
    "eps": "financial",
    "cashflow": "financial",
    # market data concepts
    "price": ["trade", "aggregate", "snapshot"],
    "prices": ["trade", "aggregate", "snapshot"],
    "close": "aggregate",
    "open": "aggregate",
    "high": "aggregate",
    "low": "aggregate",
    "volume": "aggregate",
    "prev": "previous",
    # options
    "option": "options",
    "contract": "options",
    "contracts": "options",
    "chain": "options",
    "greek": "greeks",
    "greeks": "options",
    # greeks / black-scholes
    "delta": ["bs_delta", "options"],
    "gamma": ["bs_gamma", "options"],
    "theta": ["bs_theta", "options"],
    "vega": ["bs_vega", "options"],
    "rho": ["bs_rho", "options"],
    "blackscholes": ["bs_price", "bs_delta"],
    # technical indicators
    "sma": "aggregate",
    "moving": "aggregate",
    "rsi": "aggregate",
    "technical": "aggregate",
    "indicator": "aggregate",
    # corporate actions
    "split": "splits",
    "dividend": "dividends",
    "div": "dividends",
    "divs": "dividends",
    "ipo": "ipos",
    # filings
    "10k": "filings",
    "sec": "filings",
    "filing": "filings",
    # etf
    "etf": "etf",
    # float
    "float": "float",
    # labor / employment
    "unemployment": "labor",
    "jobs": "labor",
    # conversion
    "convert": "conversion",
    # indices
    "index": "indices",
    # gainers / losers
    "gainer": "gainers",
    "loser": "losers",
    "mover": "gainers",
    "movers": "gainers",
    # news / sentiment
    "headline": "news",
    "headlines": "news",
    "article": "news",
    "articles": "news",
    "sentiment": "news",
    # analyst
    "analyst": "analysts",
    "rating": "ratings",
    "upgrade": "ratings",
    "downgrade": "ratings",
    "target": "ratings",
    # futures
    "future": "futures",
    "futs": "futures",
    # short interest
    "short": "short",
    "si": "short",
    # economy
    "yield": "treasury",
    "yields": "treasury",
    "bond": "treasury",
    "bonds": "treasury",
    "cpi": "inflation",
    "rate": "treasury",
    "rates": "treasury",
    # market status
    "holiday": "holidays",
    "hours": "status",
    "schedule": "status",
    "closed": "status",
    "exchange": "exchanges",
}

# Category keywords for weight_mask boosting
_CATEGORY_KEYWORDS: dict[str, set[str]] = {
    "Stocks": {"stock", "stocks", "equity", "equities", "share", "shares"},
    "Crypto": {"crypto", "cryptocurrency", "bitcoin", "btc", "eth", "coin"},
    "Forex": {"forex", "fx", "currency", "currencies"},
    "Options": {"option", "options", "call", "put", "strike", "chain"},
    "Futures": {"future", "futures", "futs"},
    "Indices": {"index", "indices", "benchmark"},
    "Economy": {"economy", "economic", "treasury", "inflation", "yield", "bond"},
}


def _stem(token: str) -> str:
    """Stem using Snowball English stemmer."""
    return _snowball.stemWord(token)


def _tokenize(text: str) -> list[str]:
    """Extract lowercase alphanumeric tokens, apply stopword removal, alias expansion and stemming."""
    raw_tokens = _TOKEN_RE.findall(text.lower())
    result = []
    for tok in raw_tokens:
        if tok in _STOPWORDS:
            continue
        # Alias expansion: add canonical form(s) AND the original stem
        if tok in ALIASES:
            val = ALIASES[tok]
            if isinstance(val, list):
                result.extend(val)
            else:
                result.append(val)
        result.append(_stem(tok))
    return result


def _build_corpus_text(ep: Endpoint) -> str:
    """Build enriched corpus text for an endpoint with field weighting."""
    # Repeat name 3x and category 2x for BM25F-like field weighting
    parts = [ep.name, ep.name, ep.name, ep.category, ep.category, ep.description]
    # Extract camelCase path param tokens: {stocksTicker} -> "stocks", "ticker"
    for param in re.findall(r"\{(\w+)\}", ep.endpoint_pattern):
        parts.extend(re.findall(r"[a-z]+", param))
    # Extract meaningful path segments (skip version prefixes and param placeholders)
    for seg in ep.endpoint_pattern.split("/"):
        if (
            seg
            and not seg.startswith("{")
            and not re.match(r"^v\d", seg)
            and len(seg) > 2
        ):
            parts.append(seg)
    # Extract asset class from doc URL: .../rest/stocks/... -> "stocks"
    url_match = re.search(r"/rest/(\w+)/", ep.url)
    if url_match:
        parts.append(url_match.group(1))
    return " ".join(parts)


def _detect_category(query: str) -> str | None:
    """Detect asset-class category from query keywords. Returns category name or None."""
    query_words = set(_TOKEN_RE.findall(query.lower()))
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if query_words & keywords:
            return category
    return None


class EndpointIndex:
    def __init__(self, endpoints: list[Endpoint]):
        self._endpoints = endpoints
        self._doc_cache: dict[str, str] = {
            ep.url: ep.compressed_doc for ep in endpoints
        }

        # Build category array for weight_mask
        self._categories = [ep.category for ep in endpoints]

        # Build BM25 index
        corpus = [_build_corpus_text(ep) for ep in endpoints]
        tokenized = [_tokenize(doc) for doc in corpus]
        if tokenized:
            self._bm25 = bm25s.BM25()
            self._bm25.index(tokenized)
        else:
            self._bm25 = None

        # Build regex allowlist from path prefixes
        self._prefix_patterns = [
            re.compile("^" + re.escape(ep.path_prefix)) for ep in endpoints
        ]

    def search(self, query: str, top_k: int = 7) -> list[Endpoint]:
        if self._bm25 is None:
            return []
        tokenized_query = _tokenize(query)

        # Build weight_mask for category boosting
        retrieve_kwargs: dict[str, Any] = {
            "k": min(top_k, len(self._endpoints)),
        }
        detected_category = _detect_category(query)
        if detected_category and self._endpoints:
            mask = np.ones(len(self._endpoints), dtype=np.float32)
            for i, cat in enumerate(self._categories):
                if cat == detected_category:
                    mask[i] = 2.0
            retrieve_kwargs["weight_mask"] = mask

        # retrieve expects a list of queries; we pass one and unpack
        results, scores = self._bm25.retrieve(
            [tokenized_query],
            **retrieve_kwargs,
        )
        indices: list[int] = list(results[0])  # first (only) query
        query_scores: list[float] = list(scores[0])
        return [
            self._endpoints[idx]
            for idx, score in zip(indices, query_scores)
            if score > 0
        ]

    def is_path_allowed(self, path: str) -> bool:
        return any(pat.search(path) for pat in self._prefix_patterns)

    def get_doc(self, url: str) -> str | None:
        return self._doc_cache.get(url)


def parse_llms_txt(text: str) -> list[dict]:
    entries = []
    current_category = ""
    for line in text.splitlines():
        line = line.strip()
        # Track category headers
        cat_match = re.match(r"^##\s+(.+)$", line)
        if cat_match:
            current_category = cat_match.group(1).strip()
            continue
        # Parse entry lines: - [Name](url): description
        entry_match = re.match(r"^-\s+\[([^\]]+)\]\(([^)]+)\):\s*(.+)$", line)
        if entry_match:
            entries.append(
                {
                    "name": entry_match.group(1).strip(),
                    "url": entry_match.group(2).strip(),
                    "description": entry_match.group(3).strip(),
                    "category": current_category,
                }
            )
    return entries


def extract_endpoint_pattern(doc_text: str) -> str:
    match = re.search(r"\*\*Endpoint:\*\*\s*`([^`]+)`", doc_text)
    if match:
        return match.group(1).strip()
    return ""


def extract_path_prefix(endpoint_pattern: str) -> str:
    # Remove the HTTP method prefix (e.g., "GET ")
    parts = endpoint_pattern.split(" ", 1)
    path = parts[1] if len(parts) > 1 else parts[0]

    # Find the first `{` param placeholder
    brace_idx = path.find("{")
    if brace_idx == -1:
        # No params — entire path is the prefix
        return path
    # Return everything up to and including the last `/` before the brace
    prefix = path[:brace_idx]
    return prefix


def compress_doc(doc_text: str) -> str:
    lines = doc_text.splitlines()
    result_lines = []
    in_query_params = False

    for line in lines:
        stripped = line.strip()

        # Always include endpoint pattern line
        if "**Endpoint:**" in line:
            result_lines.append(stripped)
            continue

        # Detect section transitions
        if re.match(r"^#{1,4}\s+Query Parameters", stripped, re.IGNORECASE):
            in_query_params = True
            result_lines.append(stripped)
            continue

        if re.match(r"^#{1,4}\s+", stripped):
            # Any other section header ends query params
            in_query_params = False
            continue

        # Include query parameter lines
        if in_query_params and stripped:
            result_lines.append(stripped)
            continue

    return "\n".join(result_lines)


_DOC_FETCH_CONCURRENCY = 20


async def _fetch_doc(
    client: httpx.AsyncClient, url: str, sem: asyncio.Semaphore
) -> tuple[str, str | None]:
    async with sem:
        try:
            resp = await client.get(url, follow_redirects=True)
            resp.raise_for_status()
            return url, resp.text
        except Exception as e:
            logger.warning("Failed to fetch doc %s: %s", url, e)
            return url, None


async def build_index(llms_txt_url: str | None = None) -> EndpointIndex:
    if llms_txt_url is None:
        llms_txt_url = _DEFAULT_LLMS_TXT_URL
    logger.info("Building endpoint index from llms.txt...")

    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    async with httpx.AsyncClient(timeout=30.0, verify=ssl_ctx) as client:
        try:
            resp = await client.get(llms_txt_url, follow_redirects=True)
            resp.raise_for_status()
            llms_text = resp.text
        except Exception as e:
            logger.error("Error fetching llms.txt: %s", e)
            return EndpointIndex([])

        entries = parse_llms_txt(llms_text)
        logger.info("Found %d endpoints in llms.txt", len(entries))

        # Fetch all doc pages with bounded concurrency
        sem = asyncio.Semaphore(_DOC_FETCH_CONCURRENCY)
        results = await asyncio.gather(
            *(_fetch_doc(client, entry["url"], sem) for entry in entries)
        )
        doc_texts: dict[str, str | None] = dict(results)

    # Build endpoints
    endpoints = []
    for entry in entries:
        # Filter deprecated endpoints
        if "(Deprecated)" in entry["name"]:
            logger.info("Skipping deprecated endpoint: %s", entry["name"])
            continue

        doc_text = doc_texts.get(entry["url"])
        if doc_text is None:
            logger.warning("Skipping %s — doc fetch failed", entry["name"])
            continue

        pattern = extract_endpoint_pattern(doc_text)
        if not pattern:
            logger.warning("Skipping %s — no endpoint pattern found", entry["name"])
            continue

        prefix = extract_path_prefix(pattern)
        compressed = compress_doc(doc_text)

        endpoints.append(
            Endpoint(
                name=entry["name"],
                category=entry["category"],
                url=entry["url"],
                description=entry["description"],
                endpoint_pattern=pattern,
                compressed_doc=compressed,
                path_prefix=prefix,
            )
        )

    logger.info("Indexed %d endpoints successfully", len(endpoints))
    return EndpointIndex(endpoints)
