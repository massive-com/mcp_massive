"""Error propagation through the MCP protocol layer.

Verifies that errors from the server (HTTP errors, validation failures,
SQL errors, etc.) are correctly propagated through the MCP protocol
and returned as tool results, not protocol-level exceptions.
"""

import pytest


class TestUnknownTool:
    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self, mcp_session):
        """Calling a nonexistent tool should return an error result."""
        result = await mcp_session.call_tool("nonexistent_tool", {"arg": "value"})
        assert result.isError


class TestMissingArguments:
    @pytest.mark.asyncio
    async def test_search_missing_query(self, mcp_session):
        """search_endpoints with no query should return an error result."""
        result = await mcp_session.call_tool("search_endpoints", {})
        assert result.isError


class TestHttpErrors:
    @pytest.mark.asyncio
    async def test_http_500_propagation(self, mcp_session):
        """HTTP 500 from mock server should propagate as error text."""
        result = await mcp_session.call_tool(
            "call_api",
            {
                "path": "/v2/aggs/ticker/ERROR500/range/1/day/2024-01-01/2024-01-05",
            },
        )
        text = result.content[0].text
        assert "Error" in text
        assert "500" in text


class TestPathValidation:
    @pytest.mark.asyncio
    async def test_path_not_in_allowlist(self, mcp_session):
        result = await mcp_session.call_tool(
            "call_api",
            {"path": "/v1/unknown/endpoint"},
        )
        text = result.content[0].text
        assert "Error" in text
        assert "not in allowlist" in text

    @pytest.mark.asyncio
    async def test_path_traversal_dotdot(self, mcp_session):
        result = await mcp_session.call_tool(
            "call_api",
            {"path": "/v2/aggs/../../etc/passwd"},
        )
        text = result.content[0].text
        assert "Error" in text
        assert "path traversal" in text

    @pytest.mark.asyncio
    async def test_path_traversal_encoded(self, mcp_session):
        result = await mcp_session.call_tool(
            "call_api",
            {"path": "/v2/aggs/ticker/%2e%2e/%2e%2e/etc/passwd"},
        )
        text = result.content[0].text
        assert "Error" in text
        assert "path traversal" in text

    @pytest.mark.asyncio
    async def test_path_traversal_backslash(self, mcp_session):
        result = await mcp_session.call_tool(
            "call_api",
            {"path": "/v2/aggs\\..\\etc\\passwd"},
        )
        text = result.content[0].text
        assert "Error" in text
        assert "path traversal" in text


class TestSqlErrors:
    @pytest.mark.asyncio
    async def test_invalid_sql(self, mcp_session):
        result = await mcp_session.call_tool(
            "query_data", {"sql": "NOT VALID SQL AT ALL"}
        )
        text = result.content[0].text
        assert "Error" in text

    @pytest.mark.asyncio
    async def test_query_nonexistent_table(self, mcp_session):
        result = await mcp_session.call_tool(
            "query_data", {"sql": "SELECT * FROM nonexistent_table"}
        )
        text = result.content[0].text
        assert "Error" in text

    @pytest.mark.asyncio
    async def test_describe_nonexistent_table(self, mcp_session):
        result = await mcp_session.call_tool(
            "query_data", {"sql": "DESCRIBE nonexistent_table"}
        )
        text = result.content[0].text
        assert "Error" in text

    @pytest.mark.asyncio
    async def test_drop_nonexistent_table(self, mcp_session):
        result = await mcp_session.call_tool(
            "query_data", {"sql": "DROP TABLE nonexistent_table"}
        )
        text = result.content[0].text
        assert "Error" in text


class TestInvalidScope:
    @pytest.mark.asyncio
    async def test_invalid_scope(self, mcp_session):
        result = await mcp_session.call_tool(
            "search_endpoints", {"query": "test", "scope": "invalid_scope"}
        )
        text = result.content[0].text
        assert "Error" in text
        assert "scope" in text.lower()


class TestInvalidStoreName:
    @pytest.mark.asyncio
    async def test_invalid_store_as_name(self, mcp_session):
        result = await mcp_session.call_tool(
            "call_api",
            {
                "path": "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-05",
                "store_as": "invalid name with spaces!",
            },
        )
        text = result.content[0].text
        assert "Error" in text


class TestApplyErrors:
    @pytest.mark.asyncio
    async def test_apply_unknown_function_without_store(self, mcp_session):
        """apply with unknown function and no store_as → error in CSV path."""
        result = await mcp_session.call_tool(
            "call_api",
            {
                "path": "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-05",
                "apply": [
                    {"function": "nonexistent_func", "inputs": {}, "output": "x"}
                ],
            },
        )
        text = result.content[0].text
        assert "Error" in text or "error" in text.lower()

    @pytest.mark.asyncio
    async def test_apply_unknown_function_with_store(self, mcp_session):
        """apply with unknown function and store_as → data stored, apply error appended."""
        result = await mcp_session.call_tool(
            "call_api",
            {
                "path": "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-05",
                "store_as": "apply_err_test",
                "apply": [
                    {"function": "nonexistent_func", "inputs": {}, "output": "x"}
                ],
            },
        )
        text = result.content[0].text
        # Data should be stored
        assert "Stored" in text
        # Apply error should be noted
        assert "Apply error" in text
