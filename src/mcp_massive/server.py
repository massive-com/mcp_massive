import os
import re
from typing import Optional, Any, Literal

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from importlib.metadata import version, PackageNotFoundError
from dotenv import load_dotenv

from .formatters import json_to_csv, strip_response_metadata
from .index import build_index, EndpointIndex

# Load environment variables from .env file if it exists
load_dotenv()

MASSIVE_API_KEY = os.environ.get("MASSIVE_API_KEY", "")
if not MASSIVE_API_KEY:
    print("Warning: MASSIVE_API_KEY environment variable not set.")
    print(
        "Please set it in your environment or create a .env file with MASSIVE_API_KEY=your_key"
    )

MASSIVE_API_BASE_URL = os.environ.get("MASSIVE_API_BASE_URL", "https://api.massive.com")

version_number = "MCP-Massive/unknown"
try:
    version_number = f"MCP-Massive/{version('mcp_massive')}"
except PackageNotFoundError:
    pass

# Index is built lazily on first use or explicitly via run()
index: EndpointIndex | None = None

poly_mcp = FastMCP("Massive")

METADATA_KEYS = {
    "request_id",
    "status",
    "queryCount",
    "resultsCount",
    "count",
    "next_url",
}


def _get_index() -> EndpointIndex:
    global index
    if index is None:
        index = build_index()
    return index


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def search_endpoints(query: str) -> str:
    """Search for Massive.com API endpoints by natural language query. Returns matching endpoint names, URL patterns, and descriptions. Use this to discover which API endpoints are available for your task. Try keywords like: aggregates, tickers, trades, quotes, snapshots, dividends, splits, financials, options, forex, crypto, futures, news, ratings, earnings, IPO, short interest, treasury, inflation, market status, holidays, exchanges, gainers, losers."""
    idx = _get_index()
    results = idx.search(query, top_k=5)
    if not results:
        return "No matching endpoints found. Try different search terms."

    lines = []
    for i, ep in enumerate(results, 1):
        lines.append(
            f"{i}. {ep.name} [{ep.category}]\n"
            f"   {ep.endpoint_pattern}\n"
            f"   {ep.description}\n"
            f"   Docs: {ep.url}"
        )
    return "\n\n".join(lines)


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def get_endpoint_docs(docs_url: str) -> str:
    """Get parameter documentation for a specific Massive.com API endpoint. Pass the docs URL from search_endpoints results. Returns the endpoint pattern and query parameters."""
    idx = _get_index()
    doc = idx.get_doc(docs_url)
    if doc is None:
        return f"Error: URL not found in index: {docs_url}"
    return doc


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def call_api(
    method: str,
    path: str,
    query_params: Optional[dict[str, Any]] = None,
) -> str:
    """Call a Massive.com REST API endpoint. Only GET requests are supported. The path should match an endpoint pattern from the docs (e.g., /v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31). Query parameters are passed as a dictionary."""
    idx = _get_index()

    # Security: only GET allowed
    if method.upper() != "GET":
        return f"Error: Only GET method is allowed, got {method}"

    # Security: block path traversal
    if ".." in path or "\\" in path:
        return "Error: Invalid path — path traversal not allowed"

    # Security: check path against allowlist
    if not idx.is_path_allowed(path):
        return f"Error: Path not in allowlist: {path}"

    # Security: validate query param keys
    if query_params:
        for key in query_params:
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_.]*$", key):
                return f"Error: Invalid query parameter key: {key}"

    # Build request
    url = f"{MASSIVE_API_BASE_URL}{path}"
    headers = {
        "Authorization": f"Bearer {MASSIVE_API_KEY}",
        "User-Agent": version_number,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, params=query_params, headers=headers)
            resp.raise_for_status()
            json_text = resp.text
    except httpx.HTTPStatusError as e:
        return f"Error: HTTP {e.response.status_code} — {e.response.text[:500]}"
    except Exception as e:
        return f"Error: {e}"

    # Strip metadata and convert to CSV
    try:
        stripped = strip_response_metadata(json_text, METADATA_KEYS)
        return json_to_csv(stripped)
    except Exception:
        # If stripping/CSV fails, return raw JSON
        return json_text


def run(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None:
    """Run the Massive MCP server."""
    # Build index eagerly at startup
    _get_index()
    poly_mcp.run(transport)
