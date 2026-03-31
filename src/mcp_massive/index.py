import os
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

_DEFAULT_LLMS_FULL_TXT_URL = "https://massive.com/docs/rest/llms-full.txt"

_snowball = snowballstemmer.stemmer("english")
_STOPWORDS = frozenset(STOPWORDS_EN)


class Endpoint(BaseModel):
    name: str = Field(min_length=1)
    market: str
    url: str
    description: str
    endpoint_pattern: str
    compressed_doc: str
    path_prefix: str


class QueryParam(BaseModel):
    name: str
    type: str
    required: bool
    description: str


class ResponseAttribute(BaseModel):
    name: str
    type: str
    description: str


class ParsedEndpoint(BaseModel):
    title: str
    path: str
    method: str
    market: str
    description: str
    query_params: list[QueryParam]
    response_attributes: list[ResponseAttribute]
    sample_response: str


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

# Market keywords for weight_mask boosting
_MARKET_KEYWORDS: dict[str, set[str]] = {
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
    # Repeat name 3x and market 2x for BM25F-like field weighting
    parts = [ep.name, ep.name, ep.name, ep.market, ep.market, ep.description]
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


def _detect_market(query: str) -> str | None:
    """Detect asset-class market from query keywords. Returns market name or None."""
    query_words = set(_TOKEN_RE.findall(query.lower()))
    for market, keywords in _MARKET_KEYWORDS.items():
        if query_words & keywords:
            return market
    return None


class EndpointIndex:
    def __init__(self, endpoints: list[Endpoint]):
        self._endpoints = endpoints
        self._doc_cache: dict[str, str] = {
            ep.url: ep.compressed_doc for ep in endpoints
        }

        # Build market array for weight_mask
        self._markets = [ep.market for ep in endpoints]

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

        # Build weight_mask for market boosting
        retrieve_kwargs: dict[str, Any] = {
            "k": min(top_k, len(self._endpoints)),
        }
        detected_market = _detect_market(query)
        if detected_market and self._endpoints:
            mask = np.ones(len(self._endpoints), dtype=np.float32)
            for i, market in enumerate(self._markets):
                if market == detected_market:
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
    current_market = ""
    for line in text.splitlines():
        line = line.strip()
        # Track market headers
        market_match = re.match(r"^##\s+(.+)$", line)
        if market_match:
            current_market = market_match.group(1).strip()
            continue
        # Parse entry lines: - [Name](url): description
        entry_match = re.match(r"^-\s+\[([^\]]+)\]\(([^)]+)\):\s*(.+)$", line)
        if entry_match:
            entries.append(
                {
                    "name": entry_match.group(1).strip(),
                    "url": entry_match.group(2).strip(),
                    "description": entry_match.group(3).strip(),
                    "market": current_market,
                }
            )
    return entries


def _slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


_BULLET_PARAM_RE = re.compile(r"^-\s+\*{0,2}(\w+)\*{0,2}\s*\(([^)]+)\)(?::\s*(.*))?$")


def parse_table_rows(text: str, section_header: str) -> list[dict[str, str]]:
    """Extract rows from a markdown pipe-table under the given ``## header``."""
    pattern = rf"## {re.escape(section_header)}\s*\n"
    m = re.search(pattern, text)
    if not m:
        return []
    table_text = text[m.end() :]
    lines = table_text.split("\n")
    rows: list[dict[str, str]] = []
    headers: list[str] = []

    for line in lines:
        line = line.strip()
        if not line.startswith("|"):
            if headers:
                break
            continue
        cells = [c.strip().strip("`") for c in line.split("|")[1:-1]]
        if not headers:
            headers = [c.lower() for c in cells]
        elif all(set(c.strip()) <= {"-", " "} for c in cells):
            continue  # separator row
        else:
            rows.append(dict(zip(headers, cells)))
    return rows


def _parse_bullet_items(text: str, section_header: str) -> list[dict[str, str]]:
    """Extract bullet-list items from under a ``## heading``."""
    pattern = re.compile(rf"^## {re.escape(section_header)}\s*$", re.MULTILINE)
    m = pattern.search(text)
    if not m:
        return []
    rest = text[m.end() :]
    next_section = re.search(r"^## ", rest, re.MULTILINE)
    if next_section:
        rest = rest[: next_section.start()]

    items: list[dict[str, str]] = []
    for line in rest.strip().splitlines():
        bm = _BULLET_PARAM_RE.match(line.strip())
        if bm:
            items.append(
                {
                    "name": bm.group(1),
                    "type": bm.group(2),
                    "description": (bm.group(3) or "").strip(),
                }
            )
    return items


def _parse_section_params(text: str, section_header: str) -> list[dict[str, str]]:
    """Parse params from a section, trying pipe-table format first then bullets."""
    rows = parse_table_rows(text, section_header)
    if rows:
        return rows
    return _parse_bullet_items(text, section_header)


def parse_query_params(section: str) -> list[QueryParam]:
    """Parse query parameters from a doc section."""
    rows = _parse_section_params(section, "Query Parameters")
    return [
        QueryParam(
            name=row.get("parameter", row.get("name", "")),
            type=row.get("type", "").split(",")[0].strip(),
            required=row.get("required", "").lower() in ("yes", "true")
            or "required" in row.get("type", "").lower(),
            description=row.get("description", ""),
        )
        for row in rows
    ]


def parse_response_attributes(section: str) -> list[ResponseAttribute]:
    """Parse response attributes from a doc section."""
    rows = _parse_section_params(section, "Response Attributes")
    return [
        ResponseAttribute(
            name=row.get("field", row.get("name", "")),
            type=row.get("type", "").split(",")[0].strip(),
            description=row.get("description", ""),
        )
        for row in rows
    ]


_STRUCTURAL_SECTIONS = {"Query Parameters", "Response Attributes", "Sample Response"}


def parse_endpoint_section(section: str) -> ParsedEndpoint | None:
    """Parse a single ``---``-delimited doc section into a :class:`ParsedEndpoint`."""
    ep_match = re.search(r"\*\*Endpoint:\*\*\s*`(\w+)\s+(.+?)`", section)
    if not ep_match:
        return None
    method = ep_match.group(1)
    path = ep_match.group(2)

    # Title and market from headings before the **Endpoint:** line
    before_ep = section[: ep_match.start()]
    h2_matches = re.findall(r"^## (.+)$", before_ep, re.MULTILINE)
    h3_matches = re.findall(r"^### (.+)$", before_ep, re.MULTILINE)
    market_candidates = [
        h.strip() for h in h2_matches if h.strip() not in _STRUCTURAL_SECTIONS
    ]
    market = market_candidates[0] if market_candidates else "Unknown"
    title = (
        h3_matches[-1].strip()
        if h3_matches
        else (market_candidates[-1] if market_candidates else "Unknown")
    )

    # Description: text between **Description:** and next ## heading
    desc_match = re.search(
        r"\*\*Description:\*\*\s*\n(.*?)(?=\n## |\Z)", section, re.DOTALL
    )
    description = desc_match.group(1).strip() if desc_match else ""

    # Sample response
    sample_match = re.search(
        r"## Sample Response\s*\n```json\s*\n(.*?)```", section, re.DOTALL
    )
    sample_response = sample_match.group(1).strip() if sample_match else ""

    return ParsedEndpoint(
        title=title,
        path=path,
        method=method,
        market=market,
        description=description,
        query_params=parse_query_params(section),
        response_attributes=parse_response_attributes(section),
        sample_response=sample_response,
    )


def parse_llms_full_txt(text: str) -> list[ParsedEndpoint]:
    """Parse the combined llms-full.txt into per-endpoint :class:`ParsedEndpoint` objects."""
    sections = [s for s in text.split("\n---\n") if s.strip()]
    entries: list[ParsedEndpoint] = []
    for section in sections:
        ep = parse_endpoint_section(section)
        if ep:
            entries.append(ep)
    return entries


def _compressed_doc(ep: ParsedEndpoint) -> str:
    """Build a compressed doc string from structured endpoint data."""
    lines = [f"**Endpoint:** `{ep.method} {ep.path}`"]
    if ep.query_params:
        lines.append("## Query Parameters")
        for qp in ep.query_params:
            req = "required" if qp.required else "optional"
            lines.append(f"- {qp.name} ({qp.type}, {req}): {qp.description}")
    return "\n".join(lines)


def _path_prefix(path: str) -> str:
    """Return the path up to (but not including) the first ``{`` placeholder."""
    brace_idx = path.find("{")
    if brace_idx == -1:
        return path
    return path[:brace_idx]


async def build_index(llms_txt_url: str | None = None) -> EndpointIndex:
    if llms_txt_url is None:
        llms_txt_url = os.environ.get(
            "MASSIVE_LLMS_TXT_URL", _DEFAULT_LLMS_FULL_TXT_URL
        )
    logger.info("Building endpoint index from %s ...", llms_txt_url)

    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    async with httpx.AsyncClient(timeout=30.0, verify=ssl_ctx) as client:
        try:
            resp = await client.get(llms_txt_url, follow_redirects=True)
            resp.raise_for_status()
            full_text = resp.text
        except Exception as e:
            logger.error("Error fetching %s: %s", llms_txt_url, e)
            return EndpointIndex([])

    parsed = parse_llms_full_txt(full_text)
    logger.info("Found %d endpoints in llms-full.txt", len(parsed))

    # Build endpoints
    endpoints = []
    for ep in parsed:
        # Filter deprecated endpoints
        if "(Deprecated)" in ep.title:
            logger.info("Skipping deprecated endpoint: %s", ep.title)
            continue

        pattern = f"{ep.method} {ep.path}"
        prefix = _path_prefix(ep.path)
        compressed = _compressed_doc(ep)
        cat_slug = _slugify(ep.market) if ep.market else "general"
        name_slug = _slugify(ep.title)
        url = f"https://massive.com/docs/rest/{cat_slug}/{name_slug}"

        endpoints.append(
            Endpoint(
                name=ep.title,
                market=ep.market,
                url=url,
                description=ep.description,
                endpoint_pattern=pattern,
                compressed_doc=compressed,
                path_prefix=prefix,
            )
        )

    logger.info("Indexed %d endpoints successfully", len(endpoints))
    return EndpointIndex(endpoints)
