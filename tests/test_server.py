from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from mcp_massive.index import Endpoint, EndpointIndex


def _make_test_index():
    endpoints = [
        Endpoint(
            name="Aggregates Bars",
            category="Market Data",
            url="https://massive.com/docs/aggs",
            description="Get aggregate bars for a stock",
            endpoint_pattern="GET /v2/aggs/ticker/{stocksTicker}/range/{multiplier}/{timespan}/{from}/{to}",
            compressed_doc="**Endpoint:** `GET /v2/aggs/ticker/{stocksTicker}/range/...`\n- adjusted (boolean): splits",
            path_prefix="/v2/aggs/ticker/",
        ),
        Endpoint(
            name="Tickers",
            category="Reference Data",
            url="https://massive.com/docs/tickers",
            description="Query all ticker symbols",
            endpoint_pattern="GET /v3/reference/tickers",
            compressed_doc="**Endpoint:** `GET /v3/reference/tickers`\n- search (string): search term",
            path_prefix="/v3/reference/tickers",
        ),
        Endpoint(
            name="Last Trade",
            category="Market Data",
            url="https://massive.com/docs/last-trade",
            description="Get the most recent trade for a ticker",
            endpoint_pattern="GET /v2/last/trade/{stocksTicker}",
            compressed_doc="**Endpoint:** `GET /v2/last/trade/{stocksTicker}`",
            path_prefix="/v2/last/trade/",
        ),
    ]
    return EndpointIndex(endpoints)


@pytest.fixture(autouse=True)
def _patch_server_index():
    """Patch _get_index to return our test index without hitting the network."""
    test_index = _make_test_index()
    with patch("mcp_massive.server._get_index", return_value=test_index):
        yield


class TestSearchEndpoints:
    @pytest.mark.asyncio
    async def test_returns_results(self):
        from mcp_massive.server import search_endpoints

        result = await search_endpoints("aggregate bars")
        assert "Aggregates" in result
        assert "Docs:" in result

    @pytest.mark.asyncio
    async def test_no_results(self):
        from mcp_massive.server import search_endpoints

        result = await search_endpoints("xyznonexistent")
        assert "No matching endpoints found" in result


class TestGetEndpointDocs:
    @pytest.mark.asyncio
    async def test_returns_cached_doc(self):
        from mcp_massive.server import get_endpoint_docs

        result = await get_endpoint_docs("https://massive.com/docs/aggs")
        assert "adjusted" in result

    @pytest.mark.asyncio
    async def test_unknown_url(self):
        from mcp_massive.server import get_endpoint_docs

        result = await get_endpoint_docs("https://massive.com/docs/nonexistent")
        assert "Error" in result


class TestCallApi:
    @pytest.mark.asyncio
    async def test_rejects_non_get(self):
        from mcp_massive.server import call_api

        result = await call_api(
            "POST", "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31"
        )
        assert "Error" in result
        assert "Only GET" in result

    @pytest.mark.asyncio
    async def test_rejects_path_traversal(self):
        from mcp_massive.server import call_api

        result = await call_api("GET", "/v2/aggs/../../etc/passwd")
        assert "Error" in result
        assert "path traversal" in result

    @pytest.mark.asyncio
    async def test_rejects_backslash(self):
        from mcp_massive.server import call_api

        result = await call_api("GET", "/v2/aggs\\ticker\\AAPL")
        assert "Error" in result
        assert "path traversal" in result

    @pytest.mark.asyncio
    async def test_rejects_path_not_in_allowlist(self):
        from mcp_massive.server import call_api

        result = await call_api("GET", "/v1/unknown/endpoint")
        assert "Error" in result
        assert "not in allowlist" in result

    @pytest.mark.asyncio
    async def test_rejects_invalid_query_param_keys(self):
        from mcp_massive.server import call_api

        result = await call_api(
            "GET",
            "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31",
            query_params={"valid_key": "ok", "bad key!": "nope"},
        )
        assert "Error" in result
        assert "Invalid query parameter key" in result

    @pytest.mark.asyncio
    async def test_accepts_valid_request(self):
        from mcp_massive.server import call_api

        mock_response = MagicMock()
        mock_response.text = '{"results": [{"t": 1, "o": 100}]}'
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("mcp_massive.server.httpx.AsyncClient", return_value=mock_client):
            result = await call_api(
                "GET",
                "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31",
                query_params={"adjusted": "true", "limit": "10"},
            )
        # Should return CSV output
        assert "t,o" in result or "t" in result
