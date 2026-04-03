"""In-process MCP protocol tests.

These tests connect a real MCP ClientSession to the FastMCP server via
in-memory anyio streams. No subprocess, no HTTP for the MCP layer itself,
but the underlying API calls go to the mock Starlette server.
"""

import pytest


# ---------------------------------------------------------------------------
# Tool discovery
# ---------------------------------------------------------------------------


class TestToolDiscovery:
    @pytest.mark.asyncio
    async def test_list_tools_returns_three_tools(self, mcp_session):
        result = await mcp_session.list_tools()
        names = {t.name for t in result.tools}
        assert names == {
            "search_endpoints",
            "call_api",
            "query_data",
        }

    @pytest.mark.asyncio
    async def test_tools_have_descriptions(self, mcp_session):
        result = await mcp_session.list_tools()
        for tool in result.tools:
            assert tool.description, f"Tool {tool.name} missing description"

    @pytest.mark.asyncio
    async def test_tools_have_input_schemas(self, mcp_session):
        result = await mcp_session.list_tools()
        for tool in result.tools:
            schema = tool.inputSchema
            assert schema is not None, f"Tool {tool.name} missing inputSchema"
            assert "properties" in schema, f"Tool {tool.name} schema missing properties"


# ---------------------------------------------------------------------------
# search_endpoints
# ---------------------------------------------------------------------------


class TestSearchEndpoints:
    @pytest.mark.asyncio
    async def test_stock_aggregates(self, mcp_session):
        result = await mcp_session.call_tool(
            "search_endpoints", {"query": "stock aggregate bars"}
        )
        text = result.content[0].text
        assert "Aggregates" in text
        assert "Bars" in text

    @pytest.mark.asyncio
    async def test_trades(self, mcp_session):
        result = await mcp_session.call_tool("search_endpoints", {"query": "trades"})
        text = result.content[0].text
        assert "Trades" in text
        assert "/v3/trades/" in text

    @pytest.mark.asyncio
    async def test_crypto(self, mcp_session):
        result = await mcp_session.call_tool(
            "search_endpoints", {"query": "crypto aggregates"}
        )
        text = result.content[0].text
        assert "Crypto" in text
        assert "/v2/aggs/ticker/" in text

    @pytest.mark.asyncio
    async def test_forex(self, mcp_session):
        result = await mcp_session.call_tool(
            "search_endpoints", {"query": "forex bars"}
        )
        text = result.content[0].text
        assert "Forex" in text
        assert "/v2/aggs/ticker/" in text

    @pytest.mark.asyncio
    async def test_options(self, mcp_session):
        result = await mcp_session.call_tool(
            "search_endpoints", {"query": "options chain"}
        )
        text = result.content[0].text
        assert "Options" in text
        assert "/v3/snapshot/options/" in text

    @pytest.mark.asyncio
    async def test_no_results(self, mcp_session):
        result = await mcp_session.call_tool(
            "search_endpoints", {"query": "xyznonexistent123"}
        )
        text = result.content[0].text
        assert "No matching" in text

    @pytest.mark.asyncio
    async def test_scope_endpoints_only(self, mcp_session):
        result = await mcp_session.call_tool(
            "search_endpoints", {"query": "aggregates", "scope": "endpoints"}
        )
        text = result.content[0].text
        assert "(function)" not in text

    @pytest.mark.asyncio
    async def test_scope_functions_only(self, mcp_session):
        result = await mcp_session.call_tool(
            "search_endpoints", {"query": "delta", "scope": "functions"}
        )
        text = result.content[0].text
        assert "(function)" in text
        assert "/v2/" not in text

    @pytest.mark.asyncio
    async def test_detail_default(self, mcp_session):
        """Default detail level returns title, path, description but no query params."""
        result = await mcp_session.call_tool(
            "search_endpoints",
            {"query": "stock aggregates bars", "scope": "endpoints", "max_results": 1},
        )
        text = result.content[0].text
        assert not result.isError
        assert "Aggregates" in text or "Bars" in text
        assert "Query Parameters:" not in text

    @pytest.mark.asyncio
    async def test_detail_more(self, mcp_session):
        """detail=more includes query parameter documentation."""
        result = await mcp_session.call_tool(
            "search_endpoints",
            {
                "query": "stock aggregates bars",
                "scope": "endpoints",
                "max_results": 1,
                "detail": "more",
            },
        )
        text = result.content[0].text
        assert not result.isError
        assert "Query Parameters:" in text
        assert "Response Attributes:" not in text

    @pytest.mark.asyncio
    async def test_detail_verbose(self, mcp_session):
        """detail=verbose includes full documentation."""
        result = await mcp_session.call_tool(
            "search_endpoints",
            {
                "query": "stock aggregates bars",
                "scope": "endpoints",
                "max_results": 1,
                "detail": "verbose",
            },
        )
        text = result.content[0].text
        assert not result.isError
        assert "Query Parameters:" in text
        assert "Response Attributes:" in text

    @pytest.mark.asyncio
    async def test_max_results_limits_output(self, mcp_session):
        result = await mcp_session.call_tool(
            "search_endpoints",
            {"query": "aggregates", "scope": "endpoints", "max_results": 1},
        )
        text = result.content[0].text
        assert "1." in text
        assert "2." not in text


# ---------------------------------------------------------------------------
# call_api
# ---------------------------------------------------------------------------


class TestCallApi:
    @pytest.mark.asyncio
    async def test_csv_output(self, mcp_session):
        result = await mcp_session.call_tool(
            "call_api",
            {
                "path": "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-05",
            },
        )
        text = result.content[0].text
        assert not result.isError
        lines = text.strip().split("\n")
        # Verify CSV header contains all expected OHLCV columns
        header_cols = lines[0].split(",")
        for col in ("v", "vw", "o", "c", "h", "l", "t", "n"):
            assert col in header_cols, f"Missing column {col!r} in CSV header"
        # 5 data rows from canned response + 1 header
        assert len(lines) == 6
        # Spot-check known values from canned STOCK_AGGS_RESPONSE
        assert "172.28" in text  # close price from first bar
        assert "170.57" in text  # open price from first bar

    @pytest.mark.asyncio
    async def test_store_as_summary(self, mcp_session):
        result = await mcp_session.call_tool(
            "call_api",
            {
                "path": "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-05",
                "store_as": "prices",
            },
        )
        text = result.content[0].text
        assert not result.isError
        assert "Stored 5 rows in 'prices'" in text
        assert "Preview" in text
        # Verify column list includes expected OHLCV columns
        assert "v, vw, o, c, h, l, t, n" in text

    @pytest.mark.asyncio
    async def test_rejects_path_traversal(self, mcp_session):
        result = await mcp_session.call_tool(
            "call_api",
            {"path": "/v2/aggs/../../etc/passwd"},
        )
        text = result.content[0].text
        assert "Error" in text

    @pytest.mark.asyncio
    async def test_query_params(self, mcp_session):
        result = await mcp_session.call_tool(
            "call_api",
            {
                "path": "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-05",
                "params": {"adjusted": "true", "limit": "5"},
            },
        )
        text = result.content[0].text
        assert not result.isError
        lines = text.strip().split("\n")
        header_cols = lines[0].split(",")
        for col in ("v", "vw", "o", "c", "h", "l", "t", "n"):
            assert col in header_cols, f"Missing column {col!r} in CSV header"
        assert len(lines) == 6  # header + 5 data rows


# ---------------------------------------------------------------------------
# query_data
# ---------------------------------------------------------------------------


class TestQueryData:
    @pytest.mark.asyncio
    async def test_select(self, mcp_session):
        # First store data
        await mcp_session.call_tool(
            "call_api",
            {
                "path": "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-05",
                "store_as": "test_prices",
            },
        )
        result = await mcp_session.call_tool(
            "query_data", {"sql": "SELECT * FROM test_prices"}
        )
        text = result.content[0].text
        assert not result.isError
        lines = text.strip().split("\n")
        header_cols = lines[0].split(",")
        for col in ("v", "vw", "o", "c", "h", "l", "t", "n"):
            assert col in header_cols, f"Missing column {col!r} in query result"
        assert len(lines) == 6  # header + 5 rows
        # Spot-check a known value from the canned response
        assert "172.28" in text

    @pytest.mark.asyncio
    async def test_show_tables(self, mcp_session):
        # Store something first
        await mcp_session.call_tool(
            "call_api",
            {
                "path": "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-05",
                "store_as": "show_test",
            },
        )
        result = await mcp_session.call_tool("query_data", {"sql": "SHOW TABLES"})
        text = result.content[0].text
        assert "show_test" in text

    @pytest.mark.asyncio
    async def test_describe(self, mcp_session):
        await mcp_session.call_tool(
            "call_api",
            {
                "path": "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-05",
                "store_as": "desc_test",
            },
        )
        result = await mcp_session.call_tool(
            "query_data", {"sql": "DESCRIBE desc_test"}
        )
        text = result.content[0].text
        assert "column,dtype" in text

    @pytest.mark.asyncio
    async def test_drop_table(self, mcp_session):
        await mcp_session.call_tool(
            "call_api",
            {
                "path": "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-05",
                "store_as": "drop_test",
            },
        )
        result = await mcp_session.call_tool(
            "query_data", {"sql": "DROP TABLE drop_test"}
        )
        text = result.content[0].text
        assert "dropped" in text
