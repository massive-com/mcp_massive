"""Multi-step LLM workflow tests.

Simulates realistic multi-tool sequences an LLM would perform,
exercising the full search → docs → call → store → query pipeline.
"""

import pytest

from .conftest import get_text


class TestStockAnalysis:
    @pytest.mark.asyncio
    async def test_fetch_bars_store_aggregate_query(self, mcp_session):
        """Fetch bars → store → aggregate SQL query."""
        # Store
        store = await mcp_session.call_tool(
            "call_api",
            {
                "path": "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-05",
                "store_as": "aapl_bars",
            },
        )
        assert not store.isError
        assert "Stored 5 rows in 'aapl_bars'" in store.content[0].text

        # Aggregate query
        query = await mcp_session.call_tool(
            "query_data",
            {
                "sql": "SELECT AVG(c) as avg_close, MAX(h) as max_high, MIN(l) as min_low FROM aapl_bars"
            },
        )
        text = query.content[0].text
        assert not query.isError
        lines = text.strip().split("\n")
        header_cols = lines[0].split(",")
        assert header_cols == ["avg_close", "max_high", "min_low"]
        assert len(lines) == 2  # header + 1 aggregate row
        # Verify computed values from known canned data:
        # closes: 172.28, 181.91, 184.25, 185.56, 185.01 → avg ≈ 181.802
        # highs: 172.94, 182.34, 185.15, 186.10, 186.74 → max = 186.74
        # lows: 170.27, 180.17, 183.43, 183.82, 184.35 → min = 170.27
        assert "186.74" in text
        assert "170.27" in text


class TestMultiAssetComparison:
    @pytest.mark.asyncio
    async def test_store_stocks_and_crypto(self, mcp_session):
        """Store stocks and crypto, then inspect tables."""
        # Store stocks
        stock_result = await mcp_session.call_tool(
            "call_api",
            {
                "path": "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-05",
                "store_as": "stocks",
            },
        )
        assert not stock_result.isError

        # Store crypto
        crypto_result = await mcp_session.call_tool(
            "call_api",
            {
                "path": "/v2/aggs/ticker/X:BTCUSD/range/1/day/2024-01-01/2024-01-03",
                "store_as": "crypto",
            },
        )
        assert not crypto_result.isError

        # SHOW TABLES — verify both tables listed with correct row counts
        tables = await mcp_session.call_tool("query_data", {"sql": "SHOW TABLES"})
        text = tables.content[0].text
        assert "stocks,5," in text  # 5 rows for stock aggs
        assert "crypto,3," in text  # 3 rows for crypto aggs

        # DESCRIBE — verify schema columns and data types
        desc = await mcp_session.call_tool("query_data", {"sql": "DESCRIBE stocks"})
        desc_text = desc.content[0].text
        assert "column,dtype" in desc_text
        assert "Table: stocks (5 rows)" in desc_text


class TestOptionsChainWorkflow:
    @pytest.mark.asyncio
    async def test_search_call_store_query(self, mcp_session):
        """Search options → call → store → query."""
        # Search
        search = await mcp_session.call_tool(
            "search_endpoints",
            {"query": "options chain snapshot", "scope": "endpoints"},
        )
        assert (
            "Options" in search.content[0].text
            or "options" in search.content[0].text.lower()
        )

        # Call
        call = await mcp_session.call_tool(
            "call_api",
            {
                "path": "/v3/snapshot/options/AAPL",
                "store_as": "options",
                "params": {"contract_type": "call", "limit": "10"},
            },
        )
        assert "Stored" in call.content[0].text

        # Query
        query = await mcp_session.call_tool(
            "query_data",
            {"sql": "SELECT * FROM options"},
        )
        assert not query.isError
        text = query.content[0].text
        lines = text.strip().split("\n")
        assert len(lines) == 2  # header + 1 row (canned response has 1 option)
        header_cols = lines[0].split(",")
        # Options chain response is flattened; verify key columns exist
        assert "break_even_price" in header_cols
        assert "implied_volatility" in header_cols
        assert "open_interest" in header_cols
        # Spot-check a known value
        assert "177.5" in text  # break_even_price from canned response


class TestApplyPipeline:
    @pytest.mark.asyncio
    async def test_call_api_with_simple_return(self, mcp_session):
        """call_api with apply=[simple_return] should add a column."""
        result = await mcp_session.call_tool(
            "call_api",
            {
                "path": "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-05",
                "apply": [
                    {
                        "function": "simple_return",
                        "inputs": {"column": "c"},
                        "output": "ret",
                    }
                ],
            },
        )
        text = result.content[0].text
        assert "ret" in text

    @pytest.mark.asyncio
    async def test_call_api_with_sma(self, mcp_session):
        """call_api with apply=[sma] should add a moving average column."""
        result = await mcp_session.call_tool(
            "call_api",
            {
                "path": "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-05",
                "store_as": "sma_test",
                "apply": [
                    {
                        "function": "sma",
                        "inputs": {"column": "c", "window": 3},
                        "output": "sma_3",
                    }
                ],
            },
        )
        text = result.content[0].text
        assert "sma_3" in text
        assert "Stored" in text

    @pytest.mark.asyncio
    async def test_query_with_apply(self, mcp_session):
        """Store data, then query with apply=[log_return]."""
        await mcp_session.call_tool(
            "call_api",
            {
                "path": "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-05",
                "store_as": "apply_query_test",
            },
        )

        result = await mcp_session.call_tool(
            "query_data",
            {
                "sql": "SELECT c FROM apply_query_test",
                "apply": [
                    {
                        "function": "log_return",
                        "inputs": {"column": "c"},
                        "output": "log_ret",
                    }
                ],
            },
        )
        text = result.content[0].text
        assert "log_ret" in text


class TestDropAndReplace:
    @pytest.mark.asyncio
    async def test_store_drop_verify_restore(self, mcp_session):
        """Store → drop → verify empty → re-store."""
        # Store
        await mcp_session.call_tool(
            "call_api",
            {
                "path": "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-05",
                "store_as": "temp",
            },
        )

        # Drop
        drop = await mcp_session.call_tool("query_data", {"sql": "DROP TABLE temp"})
        assert "dropped" in drop.content[0].text

        # Verify empty
        tables = await mcp_session.call_tool("query_data", {"sql": "SHOW TABLES"})
        assert "temp" not in tables.content[0].text

        # Re-store
        restore = await mcp_session.call_tool(
            "call_api",
            {
                "path": "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-05",
                "store_as": "temp",
            },
        )
        assert "Stored" in restore.content[0].text


class TestReferenceLookup:
    @pytest.mark.asyncio
    async def test_tickers_store_and_filter(self, mcp_session):
        """Fetch tickers → store → filter query."""
        store = await mcp_session.call_tool(
            "call_api",
            {
                "path": "/v3/reference/tickers",
                "store_as": "tickers",
            },
        )
        assert not store.isError
        assert "Stored 2 rows in 'tickers'" in store.content[0].text

        query = await mcp_session.call_tool(
            "query_data",
            {"sql": "SELECT * FROM tickers WHERE ticker = 'AAPL'"},
        )
        assert not query.isError
        text = query.content[0].text
        lines = text.strip().split("\n")
        assert len(lines) == 2  # header + 1 filtered row
        assert "AAPL" in text
        assert "Apple Inc." in text
        assert "GOOGL" not in text


class TestForexWorkflow:
    @pytest.mark.asyncio
    async def test_search_call_store_query(self, mcp_session):
        """Search forex → call → store → query."""
        # Search
        search = await mcp_session.call_tool(
            "search_endpoints", {"query": "forex currency aggregates"}
        )
        assert "Forex" in search.content[0].text
        assert "/v2/aggs/ticker/" in search.content[0].text

        # Call
        call = await mcp_session.call_tool(
            "call_api",
            {
                "path": "/v2/aggs/ticker/C:EURUSD/range/1/day/2024-01-01/2024-01-03",
                "store_as": "forex",
            },
        )
        assert not call.isError
        assert "Stored 3 rows in 'forex'" in call.content[0].text

        # Query
        query = await mcp_session.call_tool(
            "query_data",
            {"sql": "SELECT AVG(c) as avg_close FROM forex"},
        )
        assert not query.isError
        text = query.content[0].text
        lines = text.strip().split("\n")
        assert lines[0].strip() == "avg_close"
        assert len(lines) == 2  # header + 1 aggregate row


class TestErrorResponses:
    """Test that error responses from the API are handled correctly."""

    @pytest.mark.asyncio
    async def test_api_500_returns_error(self, mcp_session):
        """A 500 from the API should return a meaningful error message."""
        result = await mcp_session.call_tool(
            "call_api",
            {
                "path": "/v2/aggs/ticker/ERROR500/range/1/day/2024-01-01/2024-01-05",
            },
        )
        text = get_text(result)
        assert "Error" in text
        assert "[SERVER]" in text
        assert "500" in text

    @pytest.mark.asyncio
    async def test_path_not_in_allowlist(self, mcp_session):
        """A path not in the index should return a NOT_FOUND error."""
        result = await mcp_session.call_tool(
            "call_api",
            {
                "path": "/v99/nonexistent/endpoint",
            },
        )
        text = get_text(result)
        assert "Error" in text
        assert "[NOT_FOUND]" in text
        assert "search_endpoints" in text

    @pytest.mark.asyncio
    async def test_query_nonexistent_table(self, mcp_session):
        """Querying a table that doesn't exist should return an error."""
        result = await mcp_session.call_tool(
            "query_data",
            {"sql": "SELECT * FROM nonexistent_table"},
        )
        text = get_text(result)
        assert "Error" in text
        assert "Table not found" in text


class TestPagination:
    """Test that paginated API responses include a pagination hint."""

    @pytest.mark.asyncio
    async def test_paginated_csv_includes_hint(self, mcp_session):
        """CSV output should include pagination hint with path and cursor."""
        result = await mcp_session.call_tool(
            "call_api",
            {
                "path": "/v2/aggs/ticker/PAGINATED/range/1/day/2024-01-01/2024-01-05",
            },
        )
        text = get_text(result)
        # Should contain data
        assert "v,vw,o,c" in text
        # Should contain pagination hint
        assert "Next page available" in text
        assert "cursor" in text
        assert "page2_token" in text
        # API key must be stripped from the hint
        assert "LEAKED_KEY" not in text
        assert "apiKey" not in text

    @pytest.mark.asyncio
    async def test_paginated_store_includes_hint(self, mcp_session):
        """store_as output should include pagination hint."""
        result = await mcp_session.call_tool(
            "call_api",
            {
                "path": "/v2/aggs/ticker/PAGINATED/range/1/day/2024-01-01/2024-01-05",
                "store_as": "paginated_test",
            },
        )
        text = get_text(result)
        assert "Stored" in text
        assert "Next page available" in text
        assert "page2_token" in text
        assert "LEAKED_KEY" not in text

    @pytest.mark.asyncio
    async def test_non_paginated_has_no_hint(self, mcp_session):
        """Non-paginated responses should NOT include pagination hint."""
        result = await mcp_session.call_tool(
            "call_api",
            {
                "path": "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-05",
            },
        )
        text = get_text(result)
        assert "Next page" not in text
