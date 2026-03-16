import atexit
import json
import logging
import re
import threading
from typing import Annotated, Optional, Any, Literal
from urllib.parse import unquote, urlparse, parse_qs

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
        "Start with search_endpoints to discover the right API endpoint, then "
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


def configure_server(
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    """Configure host/port for HTTP transports (sse, streamable-http)."""
    mass_mcp.settings.host = host
    mass_mcp.settings.port = port


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
            _http_client = httpx.AsyncClient(timeout=30.0)
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
) -> str:
    """Search for financial market data API endpoints by natural language query. Use this FIRST whenever you need stock prices, options data, trades, quotes, aggregates, crypto, forex, or any financial/market data. Returns matching endpoint names, URL patterns, and descriptions. Try keywords like: aggregates, tickers, trades, quotes, snapshots, financials, options, IPO, inflation, market status. Also searches local finance functions (Greeks, returns, technicals) that can be applied to results via the apply parameter. Set scope to "endpoints" for API endpoints only, "functions" for local functions only, or omit/set "all" for both."""
    lines = []
    counter = 1

    show_endpoints = scope is None or scope in ("all", "endpoints")
    show_functions = scope is None or scope in ("all", "functions")

    if show_endpoints:
        idx = await _get_index()
        top_k = 7 if scope == "endpoints" else 5
        results = idx.search(query, top_k=top_k)
        for ep in results:
            lines.append(
                f"{counter}. {ep.name} [{ep.category}]\n"
                f"   {ep.endpoint_pattern}\n"
                f"   {ep.description}\n"
                f"   Docs: {ep.url}"
            )
            counter += 1

    if show_functions:
        fidx = _get_func_index()
        top_k = 5 if scope == "functions" else 3
        func_results = fidx.search(query, top_k=top_k)
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
async def get_endpoint_docs(
    url: Annotated[
        str, Field(description="The docs URL from search_endpoints results")
    ],
) -> str:
    """Get parameter documentation for a financial data API endpoint. Pass the docs URL from search_endpoints results. Returns the endpoint pattern and available query parameters."""
    idx = await _get_index()
    doc = idx.get_doc(url)
    if doc is None:
        return f"Error: URL not found in index: {url}"
    return doc


@mass_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def call_api(
    method: Annotated[
        Literal["GET"], Field(description="HTTP method (only GET is supported)")
    ],
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
    """Fetch financial market data (stock prices, options, trades, quotes, aggregates, crypto, forex). Only GET requests are supported. The path should match an endpoint pattern from search_endpoints (e.g., /v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31). Query parameters are passed as a dictionary via params. If the response is paginated, a "Next page available" hint with the exact path and params for the next call_api request is appended to the output. Optionally set store_as to a table name (e.g., "prices") to save the results as an in-memory table for later SQL querying with query_data, instead of returning CSV. Optionally set apply to a list of function steps to post-process results — each step is {"function": "name", "inputs": {"param": value}, "output": "col_name"}. String input values refer to column names; numeric values are literals. Use search_endpoints with scope="functions" to discover available functions."""
    idx = await _get_index()

    # Security: only GET allowed
    if method.upper() != "GET":
        return f"Error [INVALID_REQUEST]: Only GET method is allowed, got {method}"

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
    headers = {
        "Authorization": f"Bearer {effective_key}",
        "User-Agent": version_number,
    }

    try:
        client = _get_http_client()
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

    if len(json_text) > MAX_RESPONSE_SIZE_BYTES:
        return f"Error [TOO_LARGE]: Response too large ({len(json_text) // (1024 * 1024)} MB). Narrow your query."

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
                return "Error [EMPTY]: No records found in API response to store."
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
                return "Error [EMPTY]: No records found in API response."
            tbl = Table.from_records(records)
            enriched = apply_pipeline(tbl, apply)
            return enriched.write_csv() + pagination_hint
        return json_to_csv(stripped) + pagination_hint
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
) -> str:
    """Analyze financial market data using SQL. Queries DataFrames stored via call_api's store_as parameter. Uses SQLite SQL engine — supports standard SQL including scalar subqueries, CTEs, ILIKE, window functions, and complex expressions. Special commands: 'SHOW TABLES' lists stored tables, 'DESCRIBE <table>' shows table schema, 'DROP TABLE <table>' removes a table. Tables auto-expire after 1 hour. Optionally set apply to a list of function steps to post-process query results — each step is {"function": "name", "inputs": {"param": value}, "output": "col_name"}. Use search_endpoints with scope="functions" to discover available functions."""
    s = _get_store()
    normalized = sql.strip()
    upper = normalized.upper()

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
            return enriched.write_csv()

        return s.query(normalized)
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
