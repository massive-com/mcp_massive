import re
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import bm25s

logger = logging.getLogger(__name__)

LLMS_TXT_URL = "https://massive.com/docs/rest/llms.txt"


@dataclass
class Endpoint:
    name: str
    category: str
    url: str
    description: str
    endpoint_pattern: str
    compressed_doc: str
    path_prefix: str


_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Aliases map abbreviations and synonyms to canonical terms that appear
# in the llms.txt endpoint names/descriptions.
ALIASES: dict[str, str] = {
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
    # market data concepts
    "price": "trade",
    "prices": "trade",
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
    "greeks": "options",
    # corporate actions
    "split": "splits",
    "dividend": "dividends",
    "div": "dividends",
    "divs": "dividends",
    "ipo": "ipos",
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

# Suffixes to strip, longest first so "ings" is tried before "ing"
_STEM_SUFFIXES = ("ings", "ing", "tion", "es", "ed", "s")


def _stem(token: str) -> str:
    """Minimal suffix stripping. Only strip if result is 3+ chars."""
    for suffix in _STEM_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def _tokenize(text: str) -> list[str]:
    """Extract lowercase alphanumeric tokens, apply alias expansion and stemming."""
    raw_tokens = _TOKEN_RE.findall(text.lower())
    result = []
    for tok in raw_tokens:
        # Alias expansion: add both the canonical form AND the original stem
        # so that exact matches in the corpus still work.
        if tok in ALIASES:
            result.append(ALIASES[tok])
        result.append(_stem(tok))
    return result


class EndpointIndex:
    def __init__(self, endpoints: list[Endpoint]):
        self._endpoints = endpoints
        self._doc_cache: dict[str, str] = {
            ep.url: ep.compressed_doc for ep in endpoints
        }

        # Build BM25 index
        corpus = [f"{ep.name} {ep.category} {ep.description}" for ep in endpoints]
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

    def search(self, query: str, top_k: int = 5) -> list[Endpoint]:
        if self._bm25 is None:
            return []
        tokenized_query = _tokenize(query)
        # retrieve expects a list of queries; we pass one and unpack
        results, scores = self._bm25.retrieve(
            [tokenized_query], k=min(top_k, len(self._endpoints))
        )
        indices = results[0]  # first (only) query
        query_scores = scores[0]
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


def _fetch_doc(client: httpx.Client, url: str) -> tuple[str, str | None]:
    try:
        resp = client.get(url, follow_redirects=True)
        resp.raise_for_status()
        return url, resp.text
    except Exception as e:
        logger.warning("Failed to fetch doc %s: %s", url, e)
        return url, None


def build_index() -> EndpointIndex:
    print("Building endpoint index from llms.txt...")
    client = httpx.Client(timeout=30.0)

    try:
        resp = client.get(LLMS_TXT_URL, follow_redirects=True)
        resp.raise_for_status()
        llms_text = resp.text
    except Exception as e:
        print(f"Error fetching llms.txt: {e}")
        return EndpointIndex([])

    entries = parse_llms_txt(llms_text)
    print(f"Found {len(entries)} endpoints in llms.txt")

    # Fetch all doc pages concurrently
    doc_texts: dict[str, str | None] = {}
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(_fetch_doc, client, entry["url"]): entry["url"]
            for entry in entries
        }
        for future in as_completed(futures):
            url, text = future.result()
            doc_texts[url] = text

    client.close()

    # Build endpoints
    endpoints = []
    for entry in entries:
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

    print(f"Indexed {len(endpoints)} endpoints successfully")
    return EndpointIndex(endpoints)
