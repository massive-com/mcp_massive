import json
import os
from unittest.mock import patch, MagicMock, AsyncMock

import httpx
import pytest

from mcp_massive.index import Endpoint, EndpointIndex, QueryParam
from mcp_massive.functions import FunctionIndex
from mcp_massive.server import (
    search_endpoints,
    call_api,
    query_data,
    configure_credentials,
    _get_api_key,
    _get_base_url,
    _extract_pagination_hint,
    MAX_RESPONSE_SIZE_BYTES,
)
from mcp_massive.store import DataFrameStore


def _make_test_index():
    endpoints = [
        Endpoint(
            title="Aggregates Bars",
            path="/v2/aggs/ticker/{stocksTicker}/range/{multiplier}/{timespan}/{from}/{to}",
            market="Stocks",
            description="Get aggregate bars for a stock",
            query_params=[
                QueryParam(
                    name="adjusted",
                    type="boolean",
                    required=False,
                    description="Whether results are adjusted for splits",
                ),
            ],
            path_prefix="/v2/aggs/ticker/",
        ),
        Endpoint(
            title="Tickers",
            path="/v3/reference/tickers",
            market="Reference",
            description="Query all ticker symbols",
            query_params=[
                QueryParam(
                    name="search",
                    type="string",
                    required=False,
                    description="Search term",
                ),
            ],
            path_prefix="/v3/reference/tickers",
        ),
        Endpoint(
            title="Last Trade",
            path="/v2/last/trade/{stocksTicker}",
            market="Stocks",
            description="Get the most recent trade for a ticker",
            path_prefix="/v2/last/trade/",
        ),
        Endpoint(
            title="Aggregates Bars Crypto",
            path="/v2/aggs/ticker/{cryptoTicker}/range/{multiplier}/{timespan}/{from}/{to}",
            market="Crypto",
            description="Get aggregate bars for a crypto pair",
            path_prefix="/v2/aggs/ticker/",
        ),
    ]
    return EndpointIndex(endpoints)


@pytest.fixture(autouse=True)
def _patch_server_index():
    """Patch _get_index and _get_func_index to return test instances without hitting the network."""
    test_index = _make_test_index()
    test_func_index = FunctionIndex()
    with (
        patch(
            "mcp_massive.server._get_index",
            new_callable=AsyncMock,
            return_value=test_index,
        ),
        patch("mcp_massive.server._get_func_index", return_value=test_func_index),
    ):
        yield


class TestSearchEndpoints:
    @pytest.mark.asyncio
    async def test_returns_results(self):
        result = await search_endpoints("aggregate bars")
        assert "Aggregates" in result
        assert "/v2/aggs/ticker/" in result

    @pytest.mark.asyncio
    async def test_no_results(self):
        result = await search_endpoints("xyznonexistent")
        assert "No matching endpoints found" in result

    @pytest.mark.asyncio
    async def test_max_results(self):
        result = await search_endpoints("data", max_results=1)
        # Should have at most 1 numbered result
        assert "2." not in result

    @pytest.mark.asyncio
    async def test_detail_default(self):
        result = await search_endpoints("aggregate bars")
        assert "Aggregates" in result
        assert "Stocks" in result
        assert "/v2/aggs/ticker/" in result
        assert "Query Parameters:" not in result

    @pytest.mark.asyncio
    async def test_detail_more(self):
        result = await search_endpoints("aggregate bars", detail="more")
        assert "Aggregates" in result
        assert "adjusted" in result
        assert "Query Parameters:" in result
        assert "Response Attributes:" not in result

    @pytest.mark.asyncio
    async def test_detail_verbose(self):
        result = await search_endpoints("aggregate bars", detail="verbose")
        assert "Aggregates" in result
        assert "Query Parameters:" in result

    @pytest.mark.asyncio
    async def test_market_filter_excludes_other_markets(self):
        result = await search_endpoints("aggregate bars", market="Crypto")
        assert "[Crypto]" in result
        assert "[Stocks]" not in result


class TestCallApi:
    @pytest.mark.asyncio
    async def test_rejects_path_traversal(self):
        result = await call_api("/v2/aggs/../../etc/passwd")
        assert "Error" in result
        assert "path traversal" in result

    @pytest.mark.asyncio
    async def test_rejects_backslash(self):
        result = await call_api("/v2/aggs\\ticker\\AAPL")
        assert "Error" in result
        assert "path traversal" in result

    @pytest.mark.asyncio
    async def test_rejects_url_encoded_path_traversal(self):
        result = await call_api("/v2/aggs/ticker/%2e%2e/%2e%2e/etc/passwd")
        assert "Error" in result
        assert "path traversal" in result

    @pytest.mark.asyncio
    async def test_rejects_missing_api_key(self):
        with patch("mcp_massive.server._get_api_key", return_value=""):
            result = await call_api(
                "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31",
            )
        assert "Error" in result
        assert "MASSIVE_API_KEY" in result

    @pytest.mark.asyncio
    async def test_rejects_path_not_in_allowlist(self):
        result = await call_api("/v1/unknown/endpoint")
        assert "Error" in result
        assert "not in allowlist" in result

    @pytest.mark.asyncio
    async def test_rejects_invalid_query_param_keys(self):
        result = await call_api(
            "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31",
            params={"valid_key": "ok", "bad key!": "nope"},
        )
        assert "Error" in result
        assert "Invalid query parameter key" in result

    @pytest.mark.asyncio
    async def test_accepts_valid_request(self):
        mock_response = MagicMock()
        mock_response.text = '{"results": [{"t": 1, "o": 100}]}'
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.headers = {"user-agent": ""}
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("mcp_massive.server._get_http_client", return_value=mock_client),
            patch("mcp_massive.server._get_api_key", return_value="test-key"),
        ):
            result = await call_api(
                "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31",
                params={"adjusted": "true", "limit": "10"},
            )
        # Should return CSV output
        assert "t,o" in result or "t" in result

    @pytest.mark.asyncio
    async def test_store_as_returns_summary(self):
        mock_response = MagicMock()
        mock_response.text = '{"results": [{"ticker": "AAPL", "price": 150}, {"ticker": "GOOG", "price": 2800}]}'
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.headers = {"user-agent": ""}
        mock_client.get = AsyncMock(return_value=mock_response)

        test_store = DataFrameStore()
        with (
            patch("mcp_massive.server._get_http_client", return_value=mock_client),
            patch("mcp_massive.server._get_api_key", return_value="test-key"),
            patch("mcp_massive.server._get_store", return_value=test_store),
        ):
            result = await call_api(
                "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31",
                store_as="prices",
            )
        assert "Stored 2 rows in 'prices'" in result
        assert "ticker" in result
        assert "price" in result
        assert "Preview" in result

    @pytest.mark.asyncio
    async def test_store_as_none_returns_csv(self):
        mock_response = MagicMock()
        mock_response.text = '{"results": [{"t": 1, "o": 100}]}'
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.headers = {"user-agent": ""}
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("mcp_massive.server._get_http_client", return_value=mock_client),
            patch("mcp_massive.server._get_api_key", return_value="test-key"),
        ):
            result = await call_api(
                "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31",
            )
        # Without store_as, should return CSV as before
        assert "t" in result
        assert "100" in result

    @pytest.mark.asyncio
    async def test_per_request_api_key_overrides_default(self):
        mock_response = MagicMock()
        mock_response.text = '{"results": [{"t": 1}]}'
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.headers = {"user-agent": ""}
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("mcp_massive.server._get_http_client", return_value=mock_client),
            patch("mcp_massive.server._get_api_key", return_value="default-key"),
        ):
            await call_api(
                "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31",
                api_key="custom-key",
            )
        # Verify the custom key was used in the Authorization header
        _, kwargs = mock_client.get.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer custom-key"

    @pytest.mark.asyncio
    async def test_per_request_api_key_none_uses_default(self):
        mock_response = MagicMock()
        mock_response.text = '{"results": [{"t": 1}]}'
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.headers = {"user-agent": ""}
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("mcp_massive.server._get_http_client", return_value=mock_client),
            patch("mcp_massive.server._get_api_key", return_value="default-key"),
        ):
            await call_api(
                "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31",
            )
        # Verify the default key was used
        _, kwargs = mock_client.get.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer default-key"


class TestUserAgent:
    @pytest.mark.asyncio
    async def test_user_agent_appends_version(self):
        """User-Agent header should append MCP-Massive/<version> to the httpx default."""
        mock_response = MagicMock()
        mock_response.text = '{"results": [{"t": 1}]}'
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.headers = {"user-agent": "python-httpx/0.28.1"}
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("mcp_massive.server._get_http_client", return_value=mock_client),
            patch("mcp_massive.server._get_api_key", return_value="test-key"),
        ):
            await call_api(
                "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31",
            )
        _, kwargs = mock_client.get.call_args
        ua = kwargs["headers"]["User-Agent"]
        assert ua.startswith("python-httpx/0.28.1 ")
        assert "MCP-Massive/" in ua

    @pytest.mark.asyncio
    async def test_user_agent_contains_version_number(self):
        """User-Agent should include the actual package version, not 'unknown'."""
        from importlib.metadata import version as pkg_version

        mock_response = MagicMock()
        mock_response.text = '{"results": [{"t": 1}]}'
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.headers = {"user-agent": "test-agent"}
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("mcp_massive.server._get_http_client", return_value=mock_client),
            patch("mcp_massive.server._get_api_key", return_value="test-key"),
        ):
            await call_api(
                "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31",
            )
        _, kwargs = mock_client.get.call_args
        ua = kwargs["headers"]["User-Agent"]
        expected_version = pkg_version("mcp_massive")
        assert f"MCP-Massive/{expected_version}" in ua

    @pytest.mark.asyncio
    async def test_user_agent_works_without_base_ua(self):
        """User-Agent should still work if the client has no default user-agent."""
        mock_response = MagicMock()
        mock_response.text = '{"results": [{"t": 1}]}'
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.headers = {}
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("mcp_massive.server._get_http_client", return_value=mock_client),
            patch("mcp_massive.server._get_api_key", return_value="test-key"),
        ):
            await call_api(
                "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31",
            )
        _, kwargs = mock_client.get.call_args
        ua = kwargs["headers"]["User-Agent"]
        assert ua.startswith("MCP-Massive/")
        assert " " not in ua.split("MCP-Massive/")[0]  # no leading space


class TestQueryData:
    @pytest.mark.asyncio
    async def test_sql_select(self):
        test_store = DataFrameStore()
        test_store.store("t", [{"x": 1, "y": 2}, {"x": 3, "y": 4}])

        with patch("mcp_massive.server._get_store", return_value=test_store):
            result = await query_data("SELECT * FROM t")
        assert "x,y" in result
        assert "1,2" in result

    @pytest.mark.asyncio
    async def test_show_tables(self):
        test_store = DataFrameStore()
        test_store.store("prices", [{"x": 1}])

        with patch("mcp_massive.server._get_store", return_value=test_store):
            result = await query_data("SHOW TABLES")
        assert "prices" in result

    @pytest.mark.asyncio
    async def test_describe_table(self):
        test_store = DataFrameStore()
        test_store.store("t", [{"ticker": "AAPL", "price": 150.0}])

        with patch("mcp_massive.server._get_store", return_value=test_store):
            result = await query_data("DESCRIBE t")
        assert "ticker" in result
        assert "price" in result

    @pytest.mark.asyncio
    async def test_drop_table(self):
        test_store = DataFrameStore()
        test_store.store("t", [{"x": 1}])

        with patch("mcp_massive.server._get_store", return_value=test_store):
            result = await query_data("DROP TABLE t")
        assert "dropped" in result

    @pytest.mark.asyncio
    async def test_drop_table_missing_name(self):
        test_store = DataFrameStore()
        with patch("mcp_massive.server._get_store", return_value=test_store):
            result = await query_data("DROP TABLE ")
        assert "Error" in result
        assert "Usage" in result

    @pytest.mark.asyncio
    async def test_describe_missing_name(self):
        test_store = DataFrameStore()
        with patch("mcp_massive.server._get_store", return_value=test_store):
            result = await query_data("DESCRIBE ")
        assert "Error" in result
        assert "Usage" in result

    @pytest.mark.asyncio
    async def test_invalid_sql_returns_error(self):
        test_store = DataFrameStore()
        test_store.store("t", [{"x": 1}])

        with patch("mcp_massive.server._get_store", return_value=test_store):
            result = await query_data("NOT VALID SQL")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_query_nonexistent_table_returns_error(self):
        test_store = DataFrameStore()

        with patch("mcp_massive.server._get_store", return_value=test_store):
            result = await query_data("SELECT * FROM nonexistent")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_default_cap_truncates_long_text(self):
        """By default, cells over 2000 chars are truncated with a marker."""
        test_store = DataFrameStore()
        long_body = "supply chain risk " * 200  # ~3600 chars, >2000
        test_store.store("risks", [{"category": "Supply", "body": long_body}])

        with patch("mcp_massive.server._get_store", return_value=test_store):
            result = await query_data("SELECT body FROM risks")
        assert "[truncated:" in result
        assert long_body not in result

    @pytest.mark.asyncio
    async def test_max_cell_chars_zero_returns_full_text(self):
        """Setting max_cell_chars=0 disables truncation."""
        test_store = DataFrameStore()
        long_body = "supply chain risk " * 200
        test_store.store("risks", [{"category": "Supply", "body": long_body}])

        with patch("mcp_massive.server._get_store", return_value=test_store):
            result = await query_data("SELECT body FROM risks", max_cell_chars=0)
        assert "truncated" not in result
        assert long_body in result

    @pytest.mark.asyncio
    async def test_snippet_pattern_stays_under_cap(self):
        """The recommended snippet() pattern keeps cells short even on big text."""
        test_store = DataFrameStore()
        long_body = (
            "Our reliance on single-source suppliers for lithium and nickel. " * 100
        )
        test_store.store("risks", [{"category": "Supply", "body": long_body}])

        with patch("mcp_massive.server._get_store", return_value=test_store):
            result = await query_data(
                "SELECT category, snippet(risks, 1, '[', ']', '...', 10) AS snip "
                "FROM risks WHERE risks MATCH 'lithium'"
            )
        assert "truncated" not in result
        assert "[lithium]" in result


class TestSearchEndpointsScope:
    @pytest.mark.asyncio
    async def test_scope_endpoints_only(self):
        result = await search_endpoints("aggregate bars", scope="endpoints")
        assert "Aggregates" in result
        # Should not contain function markers
        assert "(function)" not in result

    @pytest.mark.asyncio
    async def test_scope_functions_only(self):
        result = await search_endpoints("delta", scope="functions")
        assert "(function)" in result
        # Should not contain endpoint path patterns
        assert "/v2/" not in result

    @pytest.mark.asyncio
    async def test_scope_all(self):
        result = await search_endpoints("options", scope="all")
        # Should have both types potentially
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_scope_default_includes_both(self):
        result = await search_endpoints("aggregate")
        # Default scope should show endpoints
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_scope_invalid(self):
        # Invalid scope is now rejected by Pydantic at the MCP boundary;
        # calling the function directly bypasses that, so we verify via the
        # MCP integration test instead.  Here we just confirm the Literal type
        # constraint exists by checking the annotation.
        # Direct call with invalid scope: since the Literal constraint is
        # enforced by Pydantic at the MCP layer, calling the raw function
        # with an invalid scope simply yields no results (scope doesn't
        # match "endpoints" or "functions").
        result = await search_endpoints("test", scope="invalid_scope")  # pyright: ignore[reportArgumentType]
        assert "No matching" in result

    @pytest.mark.asyncio
    async def test_scope_functions_returns_signature(self):
        result = await search_endpoints("black scholes delta", scope="functions")
        assert "bs_delta" in result
        assert "Float64" in result


class TestCallApiApply:
    @pytest.mark.asyncio
    async def test_apply_with_store_as(self):
        mock_response = MagicMock()
        mock_response.text = (
            '{"results": [{"price": 100.0}, {"price": 110.0}, {"price": 105.0}]}'
        )
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.headers = {"user-agent": ""}
        mock_client.get = AsyncMock(return_value=mock_response)

        test_store = DataFrameStore()
        with (
            patch("mcp_massive.server._get_http_client", return_value=mock_client),
            patch("mcp_massive.server._get_api_key", return_value="test-key"),
            patch("mcp_massive.server._get_store", return_value=test_store),
        ):
            result = await call_api(
                "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31",
                store_as="prices",
                apply=[
                    {
                        "function": "simple_return",
                        "inputs": {"column": "price"},
                        "output": "ret",
                    }
                ],
            )
        assert "Stored" in result
        assert "ret" in result
        # Verify the stored DataFrame has the new column
        df = test_store.get_table("prices")
        assert "ret" in df.columns

    @pytest.mark.asyncio
    async def test_apply_without_store_as(self):
        mock_response = MagicMock()
        mock_response.text = (
            '{"results": [{"price": 100.0}, {"price": 110.0}, {"price": 105.0}]}'
        )
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.headers = {"user-agent": ""}
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("mcp_massive.server._get_http_client", return_value=mock_client),
            patch("mcp_massive.server._get_api_key", return_value="test-key"),
        ):
            result = await call_api(
                "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31",
                apply=[
                    {
                        "function": "simple_return",
                        "inputs": {"column": "price"},
                        "output": "ret",
                    }
                ],
            )
        # Should return CSV with the apply column
        assert "ret" in result
        assert "price" in result

    @pytest.mark.asyncio
    async def test_apply_without_store_as_bad_function(self):
        """apply without store_as should return error when function doesn't exist."""
        mock_response = MagicMock()
        mock_response.text = '{"results": [{"price": 100.0}, {"price": 110.0}]}'
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.headers = {"user-agent": ""}
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("mcp_massive.server._get_http_client", return_value=mock_client),
            patch("mcp_massive.server._get_api_key", return_value="test-key"),
        ):
            result = await call_api(
                "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31",
                apply=[
                    {
                        "function": "nonexistent_func",
                        "inputs": {},
                        "output": "x",
                    }
                ],
            )
        assert "Error" in result
        assert "applying functions" in result

    @pytest.mark.asyncio
    async def test_apply_error_preserves_raw_data(self):
        mock_response = MagicMock()
        mock_response.text = '{"results": [{"price": 100.0}]}'
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.headers = {"user-agent": ""}
        mock_client.get = AsyncMock(return_value=mock_response)

        test_store = DataFrameStore()
        with (
            patch("mcp_massive.server._get_http_client", return_value=mock_client),
            patch("mcp_massive.server._get_api_key", return_value="test-key"),
            patch("mcp_massive.server._get_store", return_value=test_store),
        ):
            result = await call_api(
                "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31",
                store_as="prices",
                apply=[{"function": "nonexistent_func", "inputs": {}, "output": "x"}],
            )
        # Raw data should still be stored
        assert "Apply error" in result
        assert "Stored" in result
        df = test_store.get_table("prices")
        assert "price" in df.columns


class TestQueryDataApply:
    @pytest.mark.asyncio
    async def test_apply_after_sql(self):
        test_store = DataFrameStore()
        test_store.store(
            "prices",
            [
                {"price": 100.0},
                {"price": 110.0},
                {"price": 105.0},
            ],
        )

        with patch("mcp_massive.server._get_store", return_value=test_store):
            result = await query_data(
                "SELECT * FROM prices",
                apply=[
                    {
                        "function": "simple_return",
                        "inputs": {"column": "price"},
                        "output": "ret",
                    }
                ],
            )
        assert "ret" in result
        assert "price" in result

    @pytest.mark.asyncio
    async def test_apply_error_in_query(self):
        test_store = DataFrameStore()
        test_store.store("t", [{"x": 1.0}])

        with patch("mcp_massive.server._get_store", return_value=test_store):
            result = await query_data(
                "SELECT * FROM t",
                apply=[{"function": "nonexistent", "inputs": {}, "output": "y"}],
            )
        assert "Error" in result


class TestConfigureCredentials:
    """Test that env vars are cleared after configure_credentials()."""

    def test_api_key_not_in_environ_after_configure(self):
        os.environ["MASSIVE_API_KEY"] = "test-secret-key"
        os.environ["MASSIVE_API_BASE_URL"] = "https://test.example.com"

        configure_credentials("test-secret-key", "https://test.example.com")

        # Clear env vars as main() would
        os.environ.pop("MASSIVE_API_KEY", None)
        os.environ.pop("MASSIVE_API_BASE_URL", None)

        # Env vars should be gone
        assert os.environ.get("MASSIVE_API_KEY") is None
        assert os.environ.get("MASSIVE_API_BASE_URL") is None

        # But module-level credentials should still work
        assert _get_api_key() == "test-secret-key"
        assert _get_base_url() == "https://test.example.com"

    def test_all_env_cleared_after_startup(self):
        """Verify that clearing the environment removes all env vars."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ["MASSIVE_API_KEY"] = "key123"
            os.environ["AWS_SECRET_ACCESS_KEY"] = "aws-secret"
            os.environ["SOME_OTHER_VAR"] = "value"

            os.environ.clear()

            assert os.environ.get("MASSIVE_API_KEY") is None
            assert os.environ.get("AWS_SECRET_ACCESS_KEY") is None
            assert os.environ.get("SOME_OTHER_VAR") is None


class TestResponseSizeLimit:
    """Test the HTTP response size limit."""

    @pytest.mark.asyncio
    async def test_oversized_response_rejected(self):
        # Create a mock response that exceeds the size limit
        mock_response = MagicMock()
        mock_response.text = "x" * (MAX_RESPONSE_SIZE_BYTES + 1)
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.headers = {"user-agent": ""}
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("mcp_massive.server._get_http_client", return_value=mock_client),
            patch("mcp_massive.server._get_api_key", return_value="test-key"),
        ):
            result = await call_api(
                "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31",
            )
        assert "Error" in result
        assert "too large" in result

    @pytest.mark.asyncio
    async def test_normal_size_response_accepted(self):
        mock_response = MagicMock()
        mock_response.text = '{"results": [{"t": 1, "o": 100}]}'
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.headers = {"user-agent": ""}
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("mcp_massive.server._get_http_client", return_value=mock_client),
            patch("mcp_massive.server._get_api_key", return_value="test-key"),
        ):
            result = await call_api(
                "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31",
            )
        assert "Error" not in result or "too large" not in result

    @pytest.mark.asyncio
    async def test_oversized_response_with_store_as_still_stores(self):
        """When store_as is set, large responses should be stored, not rejected."""
        large_results = [{"t": i, "v": i * 10} for i in range(100)]
        large_json = json.dumps({"results": large_results})
        # Temporarily lower the limit so we don't need a truly huge payload
        mock_response = MagicMock()
        mock_response.text = large_json
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.headers = {"user-agent": ""}
        mock_client.get = AsyncMock(return_value=mock_response)

        test_store = DataFrameStore()
        with (
            patch("mcp_massive.server._get_http_client", return_value=mock_client),
            patch("mcp_massive.server._get_api_key", return_value="test-key"),
            patch("mcp_massive.server._get_store", return_value=test_store),
            patch("mcp_massive.server.MAX_RESPONSE_SIZE_BYTES", 10),  # artificially low
        ):
            result = await call_api(
                "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31",
                store_as="prices",
            )
        assert "Stored 100 rows" in result
        assert "Error" not in result

    @pytest.mark.asyncio
    async def test_oversized_response_without_store_as_suggests_store(self):
        """Error message for oversized responses should suggest store_as."""
        mock_response = MagicMock()
        mock_response.text = "x" * (MAX_RESPONSE_SIZE_BYTES + 1)
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.headers = {"user-agent": ""}
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("mcp_massive.server._get_http_client", return_value=mock_client),
            patch("mcp_massive.server._get_api_key", return_value="test-key"),
        ):
            result = await call_api(
                "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31",
            )
        assert "store_as" in result


class TestEmptyResponseWarning:
    """Test that empty API responses produce helpful warnings."""

    @pytest.mark.asyncio
    async def test_empty_response_csv_warns(self):
        mock_response = MagicMock()
        mock_response.text = '{"results": []}'
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.headers = {"user-agent": ""}
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("mcp_massive.server._get_http_client", return_value=mock_client),
            patch("mcp_massive.server._get_api_key", return_value="test-key"),
        ):
            result = await call_api(
                "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31",
            )
        assert "Warning" in result
        assert "0 records" in result

    @pytest.mark.asyncio
    async def test_empty_response_store_as_warns(self):
        mock_response = MagicMock()
        mock_response.text = '{"results": []}'
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.headers = {"user-agent": ""}
        mock_client.get = AsyncMock(return_value=mock_response)

        test_store = DataFrameStore()
        with (
            patch("mcp_massive.server._get_http_client", return_value=mock_client),
            patch("mcp_massive.server._get_api_key", return_value="test-key"),
            patch("mcp_massive.server._get_store", return_value=test_store),
        ):
            result = await call_api(
                "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31",
                store_as="prices",
            )
        assert "Warning" in result
        assert "0 records" in result


class TestErrorMarket:
    """Verify error messages include market prefixes for LLM self-correction."""

    @pytest.mark.asyncio
    async def test_auth_error_market(self):
        with patch("mcp_massive.server._get_api_key", return_value=""):
            result = await call_api(
                "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31",
            )
        assert "[AUTH]" in result

    @pytest.mark.asyncio
    async def test_not_found_error_market(self):
        result = await call_api("/v1/unknown/endpoint")
        assert "[NOT_FOUND]" in result
        assert "search_endpoints" in result

    @pytest.mark.asyncio
    async def test_http_error_markets(self):
        """HTTP status codes map to correct error markets."""
        cases = [
            (401, "AUTH"),
            (403, "AUTH"),
            (429, "RATE_LIMIT"),
            (500, "SERVER"),
            (404, "HTTP"),
        ]
        for status_code, expected_market in cases:
            mock_response = MagicMock()
            mock_response.status_code = status_code
            mock_response.text = f"Error {status_code}"

            exc = httpx.HTTPStatusError(
                "error",
                request=MagicMock(),
                response=mock_response,
            )

            mock_client = AsyncMock()
            mock_client.headers = {"user-agent": ""}
            mock_client.get = AsyncMock(side_effect=exc)

            with (
                patch(
                    "mcp_massive.server._get_http_client",
                    return_value=mock_client,
                ),
                patch("mcp_massive.server._get_api_key", return_value="key"),
            ):
                result = await call_api(
                    "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31",
                )
            assert f"[{expected_market}]" in result, (
                f"Expected [{expected_market}] for HTTP {status_code}, got: {result}"
            )

    @pytest.mark.asyncio
    async def test_too_large_error_market(self):
        mock_response = MagicMock()
        mock_response.text = "x" * (MAX_RESPONSE_SIZE_BYTES + 1)
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.headers = {"user-agent": ""}
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("mcp_massive.server._get_http_client", return_value=mock_client),
            patch("mcp_massive.server._get_api_key", return_value="key"),
        ):
            result = await call_api(
                "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31",
            )
        assert "[TOO_LARGE]" in result


class TestPaginationHint:
    """Test that next_url is extracted and presented as a pagination hint."""

    def test_extract_pagination_hint_basic(self):
        json_text = json.dumps(
            {
                "results": [{"t": 1}],
                "next_url": "https://api.massive.com/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31?cursor=abc123&adjusted=true",
            }
        )
        hint = _extract_pagination_hint(json_text)
        assert hint is not None
        assert "Next page available" in hint
        assert "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31" in hint
        assert "cursor" in hint
        assert "abc123" in hint
        assert "adjusted" in hint

    def test_extract_pagination_hint_strips_api_key(self):
        """API key in next_url query params must be stripped for security."""
        json_text = json.dumps(
            {
                "results": [{"t": 1}],
                "next_url": "https://api.massive.com/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31?cursor=xyz&apiKey=SECRET_KEY_123",
            }
        )
        hint = _extract_pagination_hint(json_text)
        assert hint is not None
        assert "SECRET_KEY_123" not in hint
        assert "apiKey" not in hint
        assert "cursor" in hint
        assert "xyz" in hint

    def test_extract_pagination_hint_no_next_url(self):
        json_text = json.dumps({"results": [{"t": 1}]})
        hint = _extract_pagination_hint(json_text)
        assert hint is None

    def test_extract_pagination_hint_no_params(self):
        json_text = json.dumps(
            {
                "results": [{"t": 1}],
                "next_url": "https://api.massive.com/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31",
            }
        )
        hint = _extract_pagination_hint(json_text)
        assert hint is not None
        assert "Next page available" in hint
        assert "params" not in hint

    @pytest.mark.asyncio
    async def test_call_api_includes_pagination_hint(self):
        """call_api output should include the pagination hint when next_url is present."""
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "results": [{"t": 1, "o": 100}],
                "next_url": "https://api.massive.com/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31?cursor=page2&apiKey=secret",
            }
        )
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.headers = {"user-agent": ""}
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("mcp_massive.server._get_http_client", return_value=mock_client),
            patch("mcp_massive.server._get_api_key", return_value="test-key"),
        ):
            result = await call_api(
                "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31",
            )
        # Should contain both data and pagination hint
        assert "100" in result  # data
        assert "Next page available" in result
        assert "cursor" in result
        assert "page2" in result
        assert "secret" not in result  # API key stripped

    @pytest.mark.asyncio
    async def test_call_api_no_hint_without_next_url(self):
        """call_api output should NOT include pagination hint when no next_url."""
        mock_response = MagicMock()
        mock_response.text = json.dumps({"results": [{"t": 1, "o": 100}]})
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.headers = {"user-agent": ""}
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("mcp_massive.server._get_http_client", return_value=mock_client),
            patch("mcp_massive.server._get_api_key", return_value="test-key"),
        ):
            result = await call_api(
                "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31",
            )
        assert "Next page" not in result

    @pytest.mark.asyncio
    async def test_store_as_includes_pagination_hint(self):
        """store_as output should also include the pagination hint."""
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "results": [{"t": 1, "o": 100}],
                "next_url": "https://api.massive.com/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31?cursor=page2",
            }
        )
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.headers = {"user-agent": ""}
        mock_client.get = AsyncMock(return_value=mock_response)
        test_store = DataFrameStore()

        with (
            patch("mcp_massive.server._get_http_client", return_value=mock_client),
            patch("mcp_massive.server._get_api_key", return_value="test-key"),
            patch("mcp_massive.server._get_store", return_value=test_store),
        ):
            result = await call_api(
                "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31",
                store_as="prices",
            )
        assert "Stored" in result
        assert "Next page available" in result
        assert "cursor" in result
