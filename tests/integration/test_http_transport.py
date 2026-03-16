"""Streamable HTTP transport integration tests.

Launches ``mcp_massive`` in streamable-http mode and connects via
``streamable_http_client()``.
"""

import os
import signal
import socket
import subprocess
import sys
import time

import pytest
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from .conftest import get_text


def _free_port() -> int:
    """Find an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def http_server(mock_env):
    """Start mcp_massive in streamable-http mode as a subprocess, return MCP URL."""
    port = _free_port()
    env = {**os.environ, **mock_env, "MCP_TRANSPORT": "streamable-http"}

    # main() triggers lazy import of mcp_massive.server which creates mass_mcp
    # with port 8000.  We override the port *after* main() sets up credentials
    # but before run() launches uvicorn by monkey-patching the settings.
    startup_script = (
        "import mcp_massive;"
        "from mcp_massive.server import mass_mcp;"
        f"mass_mcp.settings.port = {port};"
        "mcp_massive.main()"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", startup_script],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready by polling the MCP endpoint
    import httpx

    mcp_url = f"http://127.0.0.1:{port}/mcp"
    for _ in range(50):
        try:
            httpx.get(mcp_url, timeout=1.0)
            # Any response (even 405) means the server is up
            break
        except (httpx.ConnectError, httpx.ReadError):
            time.sleep(0.2)
    else:
        proc.kill()
        stdout, stderr = proc.communicate(timeout=5)
        raise AssertionError(
            f"HTTP server did not start.\nstdout: {stdout.decode()}\nstderr: {stderr.decode()}"
        )

    yield mcp_url

    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


class TestHttpTransport:
    @pytest.mark.asyncio
    async def test_initialize_and_list_tools(self, http_server):
        async with streamable_http_client(http_server) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                names = {t.name for t in result.tools}
                assert names == {
                    "search_endpoints",
                    "get_endpoint_docs",
                    "call_api",
                    "query_data",
                }

    @pytest.mark.asyncio
    async def test_search_endpoints(self, http_server):
        async with streamable_http_client(http_server) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "search_endpoints", {"query": "stock aggregates"}
                )
                assert "Aggregates" in get_text(result)

    @pytest.mark.asyncio
    async def test_call_api(self, http_server):
        async with streamable_http_client(http_server) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "call_api",
                    {
                        "method": "GET",
                        "path": "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-05",
                    },
                )
                assert not result.isError
                text = get_text(result)
                lines = text.strip().split("\n")
                header_cols = lines[0].split(",")
                for col in ("v", "vw", "o", "c", "h", "l", "t", "n"):
                    assert col in header_cols, f"Missing column {col!r}"
                assert len(lines) == 6  # header + 5 data rows

    @pytest.mark.asyncio
    async def test_store_and_query(self, http_server):
        async with streamable_http_client(http_server) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()

                store_result = await session.call_tool(
                    "call_api",
                    {
                        "method": "GET",
                        "path": "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-05",
                        "store_as": "http_prices",
                    },
                )
                assert not store_result.isError
                assert "Stored 5 rows in 'http_prices'" in get_text(store_result)

                query_result = await session.call_tool(
                    "query_data", {"sql": "SELECT * FROM http_prices"}
                )
                assert not query_result.isError
                query_text = get_text(query_result)
                lines = query_text.strip().split("\n")
                header_cols = lines[0].split(",")
                for col in ("v", "vw", "o", "c"):
                    assert col in header_cols, f"Missing column {col!r}"
                assert len(lines) == 6  # header + 5 rows
