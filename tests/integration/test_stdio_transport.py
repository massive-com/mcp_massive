"""STDIO transport integration tests.

Launches `mcp_massive` as a subprocess via stdio_client() and exercises
the MCP protocol over real STDIO pipes.
"""

import os
import sys

import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from .conftest import get_text


@pytest.fixture
def server_params(mock_env):
    """StdioServerParameters pointing at the mock server."""
    env = {**os.environ, **mock_env}
    return StdioServerParameters(
        command=sys.executable,
        args=["-c", "from mcp_massive import main; main()"],
        env=env,
    )


class TestStdioTransport:
    @pytest.mark.asyncio
    async def test_initialize_and_list_tools(self, server_params):
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                names = {t.name for t in result.tools}
                assert names == {
                    "search_endpoints",
                    "call_api",
                    "query_data",
                }

    @pytest.mark.asyncio
    async def test_search_endpoints(self, server_params):
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "search_endpoints", {"query": "stock aggregates"}
                )
                assert "Aggregates" in get_text(result)

    @pytest.mark.asyncio
    async def test_call_api_csv(self, server_params):
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "call_api",
                    {
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
    async def test_store_and_query_roundtrip(self, server_params):
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Store
                store_result = await session.call_tool(
                    "call_api",
                    {
                        "path": "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-05",
                        "store_as": "prices",
                    },
                )
                assert not store_result.isError
                assert "Stored 5 rows in 'prices'" in get_text(store_result)

                # Query
                query_result = await session.call_tool(
                    "query_data", {"sql": "SELECT * FROM prices"}
                )
                assert not query_result.isError
                query_text = get_text(query_result)
                lines = query_text.strip().split("\n")
                header_cols = lines[0].split(",")
                for col in ("v", "vw", "o", "c"):
                    assert col in header_cols, f"Missing column {col!r}"
                assert len(lines) == 6  # header + 5 rows

    @pytest.mark.asyncio
    async def test_full_workflow(self, server_params):
        """search (with params) → call → store → query"""
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # 1. Search with detail=more to get query param docs inline
                search = await session.call_tool(
                    "search_endpoints",
                    {
                        "query": "stock aggregates bars",
                        "scope": "endpoints",
                        "detail": "more",
                    },
                )
                search_text = get_text(search)
                assert "Aggregates" in search_text
                assert "Query Parameters:" in search_text

                # 2. Call API + store
                call = await session.call_tool(
                    "call_api",
                    {
                        "path": "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-05",
                        "store_as": "workflow_prices",
                    },
                )
                assert "Stored" in get_text(call)

                # 3. Query
                query = await session.call_tool(
                    "query_data",
                    {"sql": "SELECT COUNT(*) as cnt FROM workflow_prices"},
                )
                assert not query.isError
                query_text = get_text(query)
                assert "cnt" in query_text
                # Canned stock aggs response has 5 rows
                assert "5" in query_text
