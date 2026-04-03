import os
import re
import logging
import sqlite3
import ssl

import certifi
import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_DEFAULT_LLMS_FULL_TXT_URL = "https://massive.com/docs/rest/llms-full.txt"


class QueryParam(BaseModel):
    name: str
    type: str
    required: bool
    description: str


class ResponseAttribute(BaseModel):
    name: str
    type: str
    description: str


class Endpoint(BaseModel):
    title: str = Field(min_length=1)
    path: str
    market: str
    description: str
    query_params: list[QueryParam] = []
    response_attributes: list[ResponseAttribute] = []
    sample_response: str = ""
    # Set during indexing
    path_prefix: str = ""

    def format(self, detail: str = "default", counter: int = 1) -> str:
        """Format this endpoint for search results at the given detail level."""
        parts = [f"{counter}. {self.title} [{self.market}]"]
        parts.append(f"   GET {self.path}")
        parts.append(f"   {self.description}")

        if detail in ("more", "verbose") and self.query_params:
            parts.append("   Query Parameters:")
            for qp in self.query_params:
                req = "required" if qp.required else "optional"
                parts.append(f"   - {qp.name} ({qp.type}, {req}): {qp.description}")

        if detail == "verbose":
            if self.response_attributes:
                parts.append("   Response Attributes:")
                for ra in self.response_attributes:
                    parts.append(f"   - {ra.name} ({ra.type}): {ra.description}")
            if self.sample_response:
                parts.append(
                    f"   Sample Response:\n   ```json\n   {self.sample_response}\n   ```"
                )

        return "\n".join(parts)


_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Aliases map abbreviations and synonyms to canonical terms that appear
# in the llms.txt endpoint names/descriptions.
# Values can be a single string or a list of strings for ambiguous terms.
ALIASES: dict[str, str | list[str]] = {
    # aggregates / bars / OHLC
    "agg": ["aggregate", "bars", "ohlc"],
    "aggs": ["aggregate", "bars", "ohlc"],
    "candle": ["aggregate", "bars", "ohlc"],
    "candles": ["aggregate", "bars", "ohlc"],
    "candlestick": ["aggregate", "bars", "ohlc"],
    "candlesticks": ["aggregate", "bars", "ohlc"],
    "ohlc": ["aggregate", "bars"],
    "ohlcv": ["aggregate", "bars", "ohlc"],
    "bar": ["aggregate", "bars", "ohlc"],
    "bars": ["aggregate", "ohlc"],
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


# Market keywords used to boost endpoints whose market matches the query.
_MARKET_KEYWORDS: dict[str, set[str]] = {
    "Stocks": {"stock", "stocks", "equity", "equities", "share", "shares"},
    "Crypto": {"crypto", "cryptocurrency", "bitcoin", "btc", "eth", "coin"},
    "Forex": {"forex", "fx", "currency", "currencies"},
    "Options": {"option", "options", "call", "put", "strike", "chain"},
    "Futures": {"future", "futures", "futs"},
    "Indices": {"index", "indices", "benchmark"},
    "Economy": {"economy", "economic", "treasury", "inflation", "yield", "bond"},
}


def _detect_market(query: str) -> str | None:
    """Detect asset-class market from query keywords. Returns market name or None."""
    query_words = set(_TOKEN_RE.findall(query.lower()))
    for market, keywords in _MARKET_KEYWORDS.items():
        if query_words & keywords:
            return market
    return None


def _expand_query(query: str) -> str:
    """Expand aliases in a search query and format as an FTS5 MATCH expression.

    Tokenizes the query into lowercase alphanumeric tokens, expands each
    through the :data:`ALIASES` table, and joins them with ``OR`` so that
    any matching term contributes to the BM25 score.
    """
    raw_tokens = _TOKEN_RE.findall(query.lower())
    terms: list[str] = []
    for tok in raw_tokens:
        if tok in ALIASES:
            val = ALIASES[tok]
            if isinstance(val, list):
                terms.extend(val)
            else:
                terms.append(val)
        terms.append(tok)
    # Deduplicate preserving insertion order
    seen: set[str] = set()
    unique: list[str] = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    if not unique:
        return ""
    # Quote terms containing underscores so FTS5 treats the sub-tokens as a
    # phrase (e.g. "bs_delta" → porter tokenises to "bs" "delta" adjacently).
    # Plain alphanumeric terms are safe unquoted.
    parts: list[str] = []
    for t in unique:
        if "_" in t:
            parts.append(f'"{t}"')
        else:
            parts.append(t)
    return " OR ".join(parts)


class EndpointIndex:
    """BM25-ranked endpoint search backed by SQLite FTS5.

    Columns and their ``bm25()`` weights:
        title (10.0), market (5.0), description (1.0), path_info (0.5)

    FTS5's built-in porter tokenizer handles stemming at both index and
    query time, so explicit stemming/stopword removal is unnecessary.
    """

    # Column weights passed to bm25(): title, market, description, path_info
    _WEIGHTS = "10.0, 5.0, 1.0, 0.5"

    # Market boost: multiply score by this factor when query mentions a market.
    _MARKET_BOOST = 2.0

    def __init__(self, endpoints: list[Endpoint]):
        self._endpoints = endpoints

        # Build FTS5 index in an in-memory SQLite database
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.execute(
            "CREATE VIRTUAL TABLE ep_fts USING fts5("
            "  title, market, description, path_info,"
            "  tokenize='porter'"
            ")"
        )

        for i, ep in enumerate(endpoints):
            # path_info: camelCase param tokens + meaningful path segments
            path_parts: list[str] = []
            for param in re.findall(r"\{(\w+)\}", ep.path):
                path_parts.extend(re.findall(r"[a-z]+", param))
            for seg in ep.path.split("/"):
                if (
                    seg
                    and not seg.startswith("{")
                    and not re.match(r"^v\d", seg)
                    and len(seg) > 2
                ):
                    path_parts.append(seg)

            self._conn.execute(
                "INSERT INTO ep_fts(rowid, title, market, description, path_info) "
                "VALUES (?, ?, ?, ?, ?)",
                (i, ep.title, ep.market, ep.description, " ".join(path_parts)),
            )

        # Regex allowlist from path prefixes
        self._prefix_patterns = [
            re.compile("^" + re.escape(ep.path_prefix)) for ep in endpoints
        ]

    def search(self, query: str, top_k: int = 7) -> list[Endpoint]:
        if not self._endpoints:
            return []

        fts_query = _expand_query(query)
        if not fts_query:
            return []

        # bm25() returns negative scores (lower = better).  When a market
        # keyword is detected we multiply matching-market scores by the boost
        # factor, making them more negative (= ranked higher).  When no market
        # is detected the CASE always evaluates to 1.0 (no-op).
        detected_market = _detect_market(query) or ""

        try:
            cursor = self._conn.execute(
                f"SELECT rowid FROM ep_fts "
                f"WHERE ep_fts MATCH ? "
                f"ORDER BY bm25(ep_fts, {self._WEIGHTS}) "
                f"  * CASE WHEN market = ? THEN ? ELSE 1.0 END "
                f"LIMIT ?",
                (fts_query, detected_market, self._MARKET_BOOST, top_k),
            )
            return [self._endpoints[row[0]] for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            return []

    def is_path_allowed(self, path: str) -> bool:
        return any(pat.search(path) for pat in self._prefix_patterns)


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


def parse_endpoint_section(section: str) -> Endpoint | None:
    """Parse a single ``---``-delimited doc section into a :class:`Endpoint`."""
    ep_match = re.search(r"\*\*Endpoint:\*\*\s*`(\w+)\s+(.+?)`", section)
    if not ep_match:
        return None
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

    return Endpoint(
        title=title,
        path=path,
        market=market,
        description=description,
        query_params=parse_query_params(section),
        response_attributes=parse_response_attributes(section),
        sample_response=sample_response,
    )


def parse_llms_full_txt(text: str) -> list[Endpoint]:
    """Parse the combined llms-full.txt into per-endpoint :class:`Endpoint` objects."""
    sections = [s for s in text.split("\n---\n") if s.strip()]
    entries: list[Endpoint] = []
    for section in sections:
        ep = parse_endpoint_section(section)
        if ep:
            entries.append(ep)
    return entries


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

    endpoints = parse_llms_full_txt(full_text)
    logger.info("Found %d endpoints in llms-full.txt", len(endpoints))

    # Filter deprecated and set derived fields
    kept: list[Endpoint] = []
    for ep in endpoints:
        if "(Deprecated)" in ep.title:
            logger.info("Skipping deprecated endpoint: %s", ep.title)
            continue

        ep.path_prefix = _path_prefix(ep.path)
        kept.append(ep)

    logger.info("Indexed %d endpoints successfully", len(kept))
    return EndpointIndex(kept)
