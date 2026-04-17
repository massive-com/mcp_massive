import atexit
import json
import logging
import re
import ssl
import threading
from typing import Annotated, Optional, Any, Literal
from urllib.parse import unquote, urlparse, parse_qs

import certifi
import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.func_metadata import ArgModelBase
from pydantic import Field
from mcp.types import ToolAnnotations
from importlib.metadata import version, PackageNotFoundError

from .formatters import json_to_csv, extract_records, strip_response_metadata
from .functions import FunctionIndex, apply_pipeline
from .index import build_index, EndpointIndex
from .store import DataFrameStore, Table

# Reject unknown tool arguments so LLMs get a clear error instead of silent
# fallback to defaults.  Must be set before any @tool decorators run.
ArgModelBase.model_config["extra"] = "forbid"

logger = logging.getLogger(__name__)

version_number = "MCP-Massive/unknown"
try:
    version_number = f"MCP-Massive/{version('mcp_massive')}"
except PackageNotFoundError:
    pass

# Index is built lazily on first use or explicitly via run()
_init_lock = threading.Lock()
_index: EndpointIndex | None = None
_func_index: FunctionIndex | None = None
_store: DataFrameStore | None = None
_http_client: httpx.AsyncClient | None = None

mass_mcp = FastMCP(
    "Massive Financial Data",
    instructions=(
        "ALWAYS use this server's tools when the user asks about stock prices, "
        "market data, financial data, tickers, options, trades, quotes, aggregates, "
        "crypto prices, forex rates, or any securities/market information. "
        "Do NOT use web search for financial data — use these tools instead. "
        "Start with search_endpoints to discover the right API endpoint (use "
        'detail="more" or detail="verbose" to see parameter docs), then '
        "call_api to fetch the data. Use store_as + query_data for multi-step analysis. "
        "Covers: equities, options, ETFs, indices, FX, crypto — real-time and historical."
    ),
)

METADATA_KEYS = {
    "request_id",
    "status",
    "queryCount",
    "resultsCount",
    "count",
}

MAX_RESPONSE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB

# Credentials and config stored in-process so env vars can be cleared after startup.
_api_key: str = ""
_base_url: str = "https://api.massive.com"
_llms_txt_url: str | None = None
_max_tables: int | None = None
_max_rows: int | None = None


def configure_credentials(
    api_key: str,
    base_url: str,
    llms_txt_url: str | None = None,
    max_tables: int | None = None,
    max_rows: int | None = None,
) -> None:
    """Store API credentials and config in module-level variables."""
    global _api_key, _base_url, _llms_txt_url, _max_tables, _max_rows
    with _init_lock:
        _api_key = api_key
        _base_url = base_url
        _llms_txt_url = llms_txt_url
        _max_tables = max_tables
        _max_rows = max_rows


def _get_api_key() -> str:
    """Return the configured API key."""
    return _api_key


def _get_base_url() -> str:
    """Return the configured base URL."""
    return _base_url


async def _get_index() -> EndpointIndex:
    global _index
    with _init_lock:
        if _index is not None:
            return _index
    idx = await build_index(llms_txt_url=_llms_txt_url)
    with _init_lock:
        if _index is None:
            _index = idx
        return _index


def _get_func_index() -> FunctionIndex:
    global _func_index
    with _init_lock:
        if _func_index is None:
            _func_index = FunctionIndex()
        return _func_index


def _get_store() -> DataFrameStore:
    global _store
    with _init_lock:
        if _store is None:
            kwargs: dict = {}
            if _max_tables is not None:
                kwargs["max_tables"] = _max_tables
            if _max_rows is not None:
                kwargs["max_rows"] = _max_rows
            _store = DataFrameStore(**kwargs)
        return _store


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    with _init_lock:
        if _http_client is None:
            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
            _http_client = httpx.AsyncClient(timeout=30.0, verify=ssl_ctx)
            atexit.register(_close_http_client)
        return _http_client


def _close_http_client() -> None:
    """Close the httpx client at process exit to release connections."""
    global _http_client
    client = _http_client
    if client is not None:
        _http_client = None
        try:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(client.aclose())
            except RuntimeError:
                asyncio.run(client.aclose())
        except Exception:
            pass


def _extract_pagination_hint(json_text: str) -> str | None:
    """Extract next_url from raw API JSON and format as a call_api hint.

    Parses the next_url into path + params so the LLM can paginate
    by passing them directly to call_api.  Strips the API key from the
    query string to avoid leaking credentials.
    """
    try:
        data = json.loads(json_text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    next_url = data.get("next_url")
    if not next_url or not isinstance(next_url, str):
        return None
    parsed = urlparse(next_url)
    path = parsed.path
    if not path:
        return None
    params = parse_qs(parsed.query, keep_blank_values=True)
    # Security: strip API key — it's provided via the Authorization header
    params.pop("apiKey", None)
    params.pop("apikey", None)
    # Flatten single-value lists
    flat_params = {k: v[0] if len(v) == 1 else v for k, v in params.items()}
    if flat_params:
        return (
            f"\n\nNext page available. To fetch, call call_api with "
            f'path="{path}" and params={json.dumps(flat_params)}'
        )
    return f'\n\nNext page available. To fetch, call call_api with path="{path}"'


@mass_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def search_endpoints(
    query: Annotated[
        str,
        Field(
            description="Natural language search query for API endpoints", min_length=1
        ),
    ],
    scope: Annotated[
        Optional[Literal["all", "endpoints", "functions"]],
        Field(
            description='Search scope: "endpoints" for API only, "functions" for local functions only, or "all"/omit for both'
        ),
    ] = None,
    max_results: Annotated[
        Optional[int],
        Field(
            description="Maximum number of results to return (default 5 for mixed, 7 for endpoints-only)",
            ge=1,
            le=25,
        ),
    ] = None,
    detail: Annotated[
        Optional[Literal["default", "more", "verbose"]],
        Field(
            description=(
                "Level of detail per result. "
                '"default": title, path, and description. '
                '"more": adds query parameter documentation. '
                '"verbose": adds response attributes and sample response.'
            ),
        ),
    ] = None,
    market: Annotated[
        Optional[
            Literal[
                "Stocks",
                "Options",
                "Crypto",
                "Forex",
                "Futures",
                "Indices",
                "Economy",
                "Alternative",
                "Reference",
            ]
        ],
        Field(
            description=(
                "Optional market/asset class filter. Omit to infer from the query. "
                "Use to pin results to a specific asset class when you already know it."
            ),
        ),
    ] = None,
) -> str:
    """Search for market data API endpoints and built-in finance functions by natural language query. Use this FIRST to find the right endpoint before calling call_api. Covers stocks, options, forex, crypto, futures, indices, ETFs, and economic data. Pass market to pin results to a specific asset class when you already know it; omit it and the server will infer from the query. Use detail="more" to see query parameter docs needed for building call_api requests."""
    effective_detail = detail or "default"

    lines = []
    counter = 1

    show_endpoints = scope is None or scope in ("all", "endpoints")
    show_functions = scope is None or scope in ("all", "functions")

    if show_endpoints:
        idx = await _get_index()
        default_k = 7 if scope == "endpoints" else 5
        top_k = max_results if max_results is not None else default_k
        results = idx.search(query, top_k=top_k, market=market)
        for ep in results:
            lines.append(ep.format(effective_detail, counter))
            counter += 1

    if show_functions:
        fidx = _get_func_index()
        func_k = (
            max_results
            if max_results is not None
            else (5 if scope == "functions" else 3)
        )
        func_results = fidx.search(query, top_k=func_k)
        for func in func_results:
            lines.append(
                f"{counter}. {func.name} [{func.category}] (function)\n"
                f"   {func.full_description()}"
            )
            counter += 1

    if not lines:
        return "No matching endpoints found. Try different search terms."

    return "\n\n".join(lines)


@mass_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def call_api(
    path: Annotated[
        str,
        Field(
            description="API endpoint path (e.g., /v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31)"
        ),
    ],
    params: Annotated[
        Optional[dict[str, Any]],
        Field(description="Query parameters as key-value pairs", default=None),
    ] = None,
    store_as: Annotated[
        Optional[str],
        Field(
            description="Table name to store results as a DataFrame for SQL querying (e.g. 'prices')",
            default=None,
            pattern=r"^[a-zA-Z_][a-zA-Z0-9_]{0,62}$",
        ),
    ] = None,
    apply: Annotated[
        Optional[list[dict]],
        Field(
            description='List of function steps to post-process results. Each step: {"function": "name", "inputs": {...}, "output": "col_name"}',
            default=None,
            max_length=20,
        ),
    ] = None,
    api_key: Annotated[
        Optional[str],
        Field(
            description="API key for this request. Overrides the server's default key.",
            default=None,
        ),
    ] = None,
) -> str:
    """Fetch data from a Massive.com REST API endpoint. Use a path from search_endpoints results. Set store_as to save results as an in-memory table for SQL querying with query_data. Paginated responses include a next-page hint with the exact path and params for the follow-up request. The apply parameter runs built-in functions on results — string input values refer to table columns, numeric values are literals. Use search_endpoints with scope="functions" to discover available functions."""
    idx = await _get_index()

    # Security: block path traversal (fully decode to catch double-encoding)
    prev = path
    decoded_path = unquote(prev)
    while decoded_path != prev:
        prev = decoded_path
        decoded_path = unquote(prev)
    if ".." in decoded_path or "\\" in decoded_path:
        return "Error [INVALID_REQUEST]: Invalid path — path traversal not allowed"

    # Security: reject query string or fragment embedded in the path, which
    # would bypass the per-key query-parameter validation below.
    if "?" in decoded_path or "#" in decoded_path:
        return "Error [INVALID_REQUEST]: path must not contain query string or fragment — pass parameters via params"

    # Security: check path against allowlist
    if not idx.is_path_allowed(path):
        return f"Error [NOT_FOUND]: Path not in allowlist: {path}. Use search_endpoints to find the correct path."

    # Security: validate query param keys
    if params:
        for key in params:
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_.]*$", key):
                return f"Error [INVALID_REQUEST]: Invalid query parameter key: {key}"

    # Build request
    effective_key = api_key if api_key else _get_api_key()
    if not effective_key:
        return "Error [AUTH]: MASSIVE_API_KEY is not set."

    url = f"{_get_base_url()}{path}"
    client = _get_http_client()
    base_ua = client.headers.get("user-agent", "")
    headers = {
        "Authorization": f"Bearer {effective_key}",
        "User-Agent": f"{base_ua} {version_number}".strip(),
    }

    try:
        resp = await client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        raw_ct = resp.headers.get("content-type")
        if isinstance(raw_ct, str):
            ct = raw_ct.lower()
            if "json" not in ct and "text" not in ct:
                return (
                    f"Error [INVALID_RESPONSE]: Unexpected Content-Type: {raw_ct[:120]}"
                )
        json_text = resp.text
    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        if code == 401 or code == 403:
            category = "AUTH"
        elif code == 429:
            category = "RATE_LIMIT"
        elif code >= 500:
            category = "SERVER"
        else:
            category = "HTTP"
        return f"Error [{category}]: HTTP {code} — {e.response.text[:500]}"
    except Exception as e:
        return f"Error [NETWORK]: {e}"

    # Block oversized responses — but only when NOT storing.  When store_as
    # is set the data goes into an in-memory table (not the text output), so
    # the 50 MB limit should not prevent storage.
    if len(json_text) > MAX_RESPONSE_SIZE_BYTES and store_as is None:
        return f"Error [TOO_LARGE]: Response too large ({len(json_text) // (1024 * 1024)} MB). Use store_as to save it as a table, or narrow your query."

    # Extract pagination hint before stripping metadata
    pagination_hint = _extract_pagination_hint(json_text) or ""

    # Strip metadata
    try:
        stripped = strip_response_metadata(json_text, METADATA_KEYS)
    except Exception:
        return json_text

    # If store_as is provided, store as DataFrame and return summary
    if store_as is not None:
        try:
            records = extract_records(stripped)
            if not records:
                return "Warning [EMPTY]: API returned 0 records to store. The ticker may be invalid, delisted, or have no data for the requested period."
            store = _get_store()
            summary = store.store(store_as, records)
            result_msg = (
                f"Stored {summary.row_count} rows in '{summary.table_name}'\n"
                f"Columns: {', '.join(summary.columns)}\n\n"
                f"Preview (first 5 rows):\n{summary.preview}"
            )

            # Apply functions if requested
            if apply:
                try:
                    tbl = store.get_table(store_as)
                    enriched = apply_pipeline(tbl, apply)
                    summary = store.store_table(store_as, enriched)
                    result_msg = (
                        f"Stored {summary.row_count} rows in '{summary.table_name}'\n"
                        f"Columns: {', '.join(summary.columns)}\n\n"
                        f"Preview (first 5 rows):\n{summary.preview}"
                    )
                except Exception as e:
                    result_msg += f"\n\nApply error (raw data preserved): {e}"

            return result_msg + pagination_hint
        except ValueError as e:
            return f"Error: {e}"

    # No store_as: return CSV, optionally with apply
    try:
        if apply:
            records = extract_records(stripped)
            if not records:
                return "Warning [EMPTY]: API returned 0 records. The ticker may be invalid, delisted, or have no data for the requested period."
            tbl = Table.from_records(records)
            enriched = apply_pipeline(tbl, apply)
            return enriched.write_csv() + pagination_hint
        csv_text = json_to_csv(stripped)
        if not csv_text.strip():
            return "Warning [EMPTY]: API returned 0 records. The ticker may be invalid, delisted, or have no data for the requested period."
        return csv_text + pagination_hint
    except Exception as e:
        if apply:
            return f"Error applying functions: {e}"
        return json_text


@mass_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def query_data(
    sql: Annotated[
        str,
        Field(
            description="SQL query or special command (SHOW TABLES, DESCRIBE <table>, DROP TABLE <table>)",
            min_length=1,
        ),
    ],
    apply: Annotated[
        Optional[list[dict]],
        Field(
            description="List of function steps to post-process query results",
            default=None,
            max_length=20,
        ),
    ] = None,
    max_cell_chars: Annotated[
        Optional[int],
        Field(
            description=(
                "Truncate output cells whose string form exceeds this length, "
                "appending a '[truncated: N more chars]' marker. Default 2000. "
                "Set to 0 to disable (e.g. when fetching the full body of a "
                "specific row). Long TEXT columns like SEC filing text can be "
                "thousands of tokens each — prefer snippet()/highlight() in "
                "the SELECT list when searching."
            ),
            default=2000,
            ge=0,
            le=1_000_000,
        ),
    ] = 2000,
) -> str:
    """Run SQL queries on tables stored via call_api's store_as parameter. Special commands: SHOW TABLES, DESCRIBE <table>, DROP TABLE <table>. Tables auto-expire after 1 hour. Supports CTEs, window functions, JOINs, and ILIKE.

    Full-text search: any stored table with TEXT columns is searchable directly via FTS5. Use `WHERE {table} MATCH 'query'` and ORDER BY rank (lower = better). Numeric columns preserve their types. For long TEXT fields (news body, 10-K risk factors, filings) prefer snippet()/highlight() in SELECT instead of the raw column — full paragraphs can be thousands of tokens each.

    Recommended FTS pattern:
      1. Find matches compactly: `SELECT rowid, category, bm25({t}) AS score, snippet({t}, 4, '[', ']', '...', 15) AS snip FROM {t} WHERE {t} MATCH 'supply chain OR supplier' ORDER BY rank`
      2. Fetch the full body only for rows you need: call query_data again with `SELECT supporting_text FROM {t} WHERE rowid IN (2, 7)` and `max_cell_chars=0`.

    Output cells over max_cell_chars (default 2000) are truncated with a visible marker; raise or set to 0 to disable."""
    s = _get_store()
    normalized = sql.strip()
    upper = normalized.upper()
    cap = max_cell_chars if max_cell_chars is not None else 2000

    try:
        if upper == "SHOW TABLES":
            return s.show_tables()

        if upper == "DESCRIBE" or upper.startswith("DESCRIBE "):
            parts = normalized.split(None, 1)
            if len(parts) < 2 or not parts[1].strip():
                return "Error: Usage: DESCRIBE <table_name>"
            return s.describe_table(parts[1].strip())

        if upper == "DROP TABLE" or upper.startswith("DROP TABLE "):
            parts = normalized.split(None, 2)
            if len(parts) < 3 or not parts[2].strip():
                return "Error: Usage: DROP TABLE <table_name>"
            return s.drop_table(parts[2].strip())

        if apply:
            tbl = s.query_table(normalized)
            enriched = apply_pipeline(tbl, apply)
            return enriched.write_csv(max_cell_chars=cap)

        return s.query(normalized, max_cell_chars=cap)
    except Exception as e:
        return f"Error: {e}"


def run(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None:
    """Run the Massive MCP server.

    The endpoint index is built lazily on the first tool call (via
    ``_get_index()``) so the MCP protocol can start responding to
    ``initialize`` immediately without waiting for all doc pages to be
    fetched.
    """
    mass_mcp.run(transport)
