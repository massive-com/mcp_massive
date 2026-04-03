from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from urllib.parse import urlparse

from mcp_massive.index import (
    Endpoint,
    EndpointIndex,
    parse_llms_txt,
    parse_llms_full_txt,
    parse_endpoint_section,
    parse_query_params,
    parse_response_attributes,
    parse_table_rows,
    build_index,
    _expand_query,
    _path_prefix,
)
from tests.integration.mock_llms_txt import (
    aggs_section,
    llms_partial_txt,
    llms_txt,
)


class TestParseLlmsTxt:
    def test_parses_entries(self):
        entries = parse_llms_txt(llms_txt())
        assert len(entries) == 4

    def test_entry_fields(self):
        entries = parse_llms_txt(llms_txt())
        first = entries[0]
        assert first["name"] == "Custom Bars (OHLC)"
        parsed_url = urlparse(first["url"])
        assert parsed_url.hostname == "massive.com"
        assert "OHLC" in first["description"]
        assert first["market"] == "Stocks"

    def test_market_tracking(self):
        entries = parse_llms_txt(llms_txt())
        assert entries[0]["market"] == "Stocks"
        assert entries[1]["market"] == "Stocks"
        assert entries[2]["market"] == "Stocks"
        assert entries[3]["market"] == "Options"

    def test_empty_input(self):
        assert parse_llms_txt("") == []

    def test_no_entries(self):
        assert parse_llms_txt("# Just a title\n\nSome text.") == []


class TestParseEndpointSection:
    def test_parses_standard_section(self):
        ep = parse_endpoint_section(aggs_section())
        assert ep is not None
        assert ep.title == "Custom Bars (OHLC)"
        assert "/v2/aggs/ticker/{stocksTicker}" in ep.path
        assert ep.market == "Stocks"

    def test_extracts_description(self):
        ep = parse_endpoint_section(aggs_section())
        assert ep is not None
        assert "OHLC" in ep.description

    def test_extracts_query_params(self):
        ep = parse_endpoint_section(aggs_section())
        assert ep is not None
        assert len(ep.query_params) == 8
        assert ep.query_params[0].name == "stocksTicker"
        assert ep.query_params[0].type == "string"
        assert ep.query_params[0].required is True
        # Optional params too
        param_names = {qp.name for qp in ep.query_params}
        assert "adjusted" in param_names
        assert "sort" in param_names
        assert "limit" in param_names

    def test_extracts_response_attributes(self):
        ep = parse_endpoint_section(aggs_section())
        assert ep is not None
        assert len(ep.response_attributes) > 5
        assert ep.response_attributes[0].name == "ticker"

    def test_extracts_sample_response(self):
        ep = parse_endpoint_section(aggs_section())
        assert ep is not None
        assert '"AAPL"' in ep.sample_response

    def test_returns_none_without_endpoint(self):
        assert parse_endpoint_section("No endpoint here.") is None

    def test_no_params_section(self):
        section = """\
## Reference

### Market Holidays

**Endpoint:** `GET /v1/marketstatus/upcoming`
"""
        ep = parse_endpoint_section(section)
        assert ep is not None
        assert ep.title == "Market Holidays"
        assert ep.market == "Reference"
        assert ep.query_params == []
        assert ep.response_attributes == []


class TestParseQueryParams:
    def test_bullet_format(self):
        section = """\
## Query Parameters

- adjusted (boolean, optional): Whether results are adjusted.
- sort (string, required): Sort order.
"""
        params = parse_query_params(section)
        assert len(params) == 2
        assert params[0].name == "adjusted"
        assert params[0].type == "boolean"
        assert params[0].required is False
        assert params[1].name == "sort"
        assert params[1].required is True

    def test_bold_bullet_format(self):
        section = """\
## Query Parameters

- **adjusted** (boolean): Whether results are adjusted.
"""
        params = parse_query_params(section)
        assert len(params) == 1
        assert params[0].name == "adjusted"

    def test_table_format(self):
        section = """\
## Query Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| adjusted | boolean | No | Whether results are adjusted. |
| sort | string | Yes | Sort order. |
"""
        params = parse_query_params(section)
        assert len(params) == 2
        assert params[0].name == "adjusted"
        assert params[0].required is False
        assert params[1].name == "sort"
        assert params[1].required is True

    def test_no_section(self):
        assert parse_query_params("No params here.") == []


class TestParseResponseAttributes:
    def test_bullet_format(self):
        section = """\
## Response Attributes

- ticker (string): The exchange symbol.
- adjusted (boolean): Whether results are adjusted.
"""
        attrs = parse_response_attributes(section)
        assert len(attrs) == 2
        assert attrs[0].name == "ticker"
        assert attrs[0].type == "string"

    def test_table_format(self):
        section = """\
## Response Attributes

| Field | Type | Description |
|-------|------|-------------|
| ticker | string | The exchange symbol. |
"""
        attrs = parse_response_attributes(section)
        assert len(attrs) == 1
        assert attrs[0].name == "ticker"

    def test_no_section(self):
        assert parse_response_attributes("No attrs here.") == []


class TestParseTableRows:
    def test_basic_table(self):
        text = """\
## Query Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| adjusted | boolean | No | Whether adjusted. |
"""
        rows = parse_table_rows(text, "Query Parameters")
        assert len(rows) == 1
        assert rows[0]["parameter"] == "adjusted"

    def test_no_table(self):
        assert parse_table_rows("No table here.", "Query Parameters") == []


class TestParseLlmsFullTxt:
    def test_parses_entries(self):
        entries = parse_llms_full_txt(llms_partial_txt())
        assert len(entries) == 9

    def test_entry_fields(self):
        entries = parse_llms_full_txt(llms_partial_txt())
        first = entries[0]
        assert first.title == "Custom Bars (OHLC)"
        assert first.market == "Crypto"
        assert "/v2/aggs/ticker/" in first.path

    def test_market_tracking(self):
        entries = parse_llms_full_txt(llms_partial_txt())
        markets = [e.market for e in entries]
        assert "Crypto" in markets
        assert "Forex" in markets
        assert "Options" in markets
        assert "Stocks" in markets

    def test_description_extracted(self):
        entries = parse_llms_full_txt(llms_partial_txt())
        assert "OHLC" in entries[0].description

    def test_empty_input(self):
        assert parse_llms_full_txt("") == []

    def test_no_entries(self):
        assert parse_llms_full_txt("# Just a title\n\nSome text.") == []

    def test_query_params_parsed(self):
        entries = parse_llms_full_txt(llms_partial_txt())
        assert len(entries[0].query_params) > 0
        param_names = {qp.name for qp in entries[0].query_params}
        assert "cryptoTicker" in param_names

    def test_entries_independent(self):
        """Each entry should only contain its own params."""
        entries = parse_llms_full_txt(llms_partial_txt())
        # First entry (Crypto) should not contain Stocks params
        first_param_names = {qp.name for qp in entries[0].query_params}
        assert "stocksTicker" not in first_param_names


class TestPathPrefix:
    def test_path_with_params(self):
        assert (
            _path_prefix("/v2/aggs/ticker/{stocksTicker}/range/{multiplier}")
            == "/v2/aggs/ticker/"
        )

    def test_crypto_path(self):
        assert _path_prefix("/v1/last/crypto/{from}/{to}") == "/v1/last/crypto/"

    def test_no_params(self):
        assert _path_prefix("/v3/reference/tickers") == "/v3/reference/tickers"


class TestExpandQuery:
    def test_basic_terms(self):
        q = _expand_query("Hello World 123")
        assert "hello" in q
        assert "world" in q
        assert "123" in q

    def test_alias_expansion(self):
        q = _expand_query("agg candle fx")
        assert "aggregate" in q  # alias for "agg" and "candle"
        assert "forex" in q  # alias for "fx"

    def test_alias_keeps_original(self):
        q = _expand_query("candle")
        assert "aggregate" in q
        assert "candle" in q

    def test_list_alias_expansion(self):
        """'price' should expand to multiple aliases."""
        q = _expand_query("price")
        assert "trade" in q
        assert "aggregate" in q
        assert "snapshot" in q

    def test_empty_query(self):
        assert _expand_query("") == ""

    def test_deduplicates(self):
        """Repeated tokens should not produce duplicate terms."""
        q = _expand_query("agg agg")
        assert q.count("aggregate") == 1

    def test_underscore_terms_quoted(self):
        """Alias values with underscores (e.g. bs_delta) should be quoted."""
        q = _expand_query("delta")
        assert '"bs_delta"' in q
        assert "options" in q

    def test_or_joined(self):
        """Terms should be joined with OR."""
        q = _expand_query("stock trade")
        assert " OR " in q


class TestEndpointIndex:
    def _make_endpoints(self):
        return [
            Endpoint(
                title="Aggregates (Bars)",
                path="/v2/aggs/ticker/{stocksTicker}/range/{multiplier}/{timespan}/{from}/{to}",
                market="Market Data",
                description="Get aggregate bars for a stock.",
                path_prefix="/v2/aggs/ticker/",
            ),
            Endpoint(
                title="Tickers",
                path="/v3/reference/tickers",
                market="Reference Data",
                description="Query all ticker symbols.",
                path_prefix="/v3/reference/tickers",
            ),
            Endpoint(
                title="Last Trade",
                path="/v2/last/trade/{stocksTicker}",
                market="Market Data",
                description="Get the most recent trade for a ticker.",
                path_prefix="/v2/last/trade/",
            ),
        ]

    def test_search_returns_relevant(self):
        idx = EndpointIndex(self._make_endpoints())
        results = idx.search("aggregate bars stock")
        assert len(results) > 0
        assert results[0].title == "Aggregates (Bars)"

    def test_search_no_results(self):
        idx = EndpointIndex(self._make_endpoints())
        results = idx.search("xyznonexistent")
        assert results == []

    def test_search_top_k(self):
        idx = EndpointIndex(self._make_endpoints())
        results = idx.search("data", top_k=2)
        assert len(results) <= 2

    def test_is_path_allowed_valid(self):
        idx = EndpointIndex(self._make_endpoints())
        assert idx.is_path_allowed(
            "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31"
        )
        assert idx.is_path_allowed("/v3/reference/tickers")
        assert idx.is_path_allowed("/v2/last/trade/MSFT")

    def test_is_path_allowed_invalid(self):
        idx = EndpointIndex(self._make_endpoints())
        assert not idx.is_path_allowed("/v1/unknown/endpoint")
        assert not idx.is_path_allowed("/admin/secret")

    def test_search_alias_agg(self):
        """'agg' should find Aggregates via alias expansion."""
        idx = EndpointIndex(self._make_endpoints())
        results = idx.search("agg")
        assert any("Aggregates" in ep.title for ep in results)

    def test_search_alias_candle(self):
        """'candle' should find Aggregates via alias expansion."""
        idx = EndpointIndex(self._make_endpoints())
        results = idx.search("candle")
        assert any("Aggregates" in ep.title for ep in results)

    def test_search_alias_ohlc(self):
        """'ohlc' should find Aggregates via alias expansion."""
        idx = EndpointIndex(self._make_endpoints())
        results = idx.search("ohlc data")
        assert any("Aggregates" in ep.title for ep in results)

    def test_search_stemmed_plural(self):
        """'tickers' (plural) should still match 'Tickers' endpoint."""
        idx = EndpointIndex(self._make_endpoints())
        results = idx.search("tickers")
        assert any("Tickers" in ep.title for ep in results)

    def test_search_alias_symbol(self):
        """'symbol' should find Tickers via alias expansion."""
        idx = EndpointIndex(self._make_endpoints())
        results = idx.search("symbol lookup")
        assert any("Tickers" in ep.title for ep in results)

    def test_search_alias_transaction(self):
        """'transaction' should find Last Trade via alias expansion."""
        idx = EndpointIndex(self._make_endpoints())
        results = idx.search("last transaction")
        assert any("Trade" in ep.title for ep in results)


class TestCrossAssetClassRanking:
    """Test that asset-class keywords properly disambiguate identical endpoint names."""

    def _make_cross_asset_endpoints(self):
        return [
            Endpoint(
                title="SMA",
                path="/v1/indicators/sma/{stocksTicker}",
                market="Stocks",
                description="Get SMA for a stock ticker.",
                path_prefix="/v1/indicators/sma/",
            ),
            Endpoint(
                title="SMA",
                path="/v1/indicators/sma/{cryptoTicker}",
                market="Crypto",
                description="Get SMA for a crypto ticker.",
                path_prefix="/v1/indicators/sma/",
            ),
            Endpoint(
                title="SMA",
                path="/v1/indicators/sma/{forexTicker}",
                market="Forex",
                description="Get SMA for a forex ticker.",
                path_prefix="/v1/indicators/sma/",
            ),
            Endpoint(
                title="Unified Snapshot",
                path="/v3/snapshot/{stocksTicker}",
                market="Stocks",
                description="Get unified snapshot for a stock ticker.",
                path_prefix="/v3/snapshot/",
            ),
            Endpoint(
                title="Unified Snapshot",
                path="/v3/snapshot/{cryptoTicker}",
                market="Crypto",
                description="Get unified snapshot for a crypto ticker.",
                path_prefix="/v3/snapshot/",
            ),
        ]

    def test_stock_sma_ranks_first(self):
        """'stock SMA' should rank Stocks SMA above Crypto/Forex SMA."""
        idx = EndpointIndex(self._make_cross_asset_endpoints())
        results = idx.search("stock SMA")
        assert len(results) > 0
        assert results[0].market == "Stocks"
        assert results[0].title == "SMA"

    def test_crypto_snapshot_ranks_first(self):
        """'crypto snapshot' should rank Crypto snapshot above Stocks snapshot."""
        idx = EndpointIndex(self._make_cross_asset_endpoints())
        results = idx.search("crypto snapshot")
        # Find the first snapshot result
        snapshot_results = [ep for ep in results if "Snapshot" in ep.title]
        assert len(snapshot_results) > 0
        assert snapshot_results[0].market == "Crypto"

    def test_generic_sma_returns_multiple_markets(self):
        """Generic 'SMA' (no asset class) should return results from multiple markets."""
        idx = EndpointIndex(self._make_cross_asset_endpoints())
        results = idx.search("SMA")
        markets = {ep.market for ep in results if ep.title == "SMA"}
        assert len(markets) >= 2

    def test_stock_candlesticks_ranks_stocks_bars_first(self):
        """'stock candlesticks' should rank Stocks OHLC bars above other markets."""
        endpoints = [
            Endpoint(
                title="Merchant Aggregates",
                path="/consumer-spending/eu/v1/merchant-aggregates",
                market="Alternative",
                description="Aggregated European consumer spending data.",
                path_prefix="/consumer-spending/eu/v1/merchant-aggregates",
            ),
            Endpoint(
                title="Aggregate Bars (OHLC)",
                path="/futures/vX/aggs/{ticker}",
                market="Futures",
                description="Retrieve aggregated OHLC and volume data for futures.",
                path_prefix="/futures/vX/aggs/",
            ),
            Endpoint(
                title="Custom Bars (OHLC)",
                path="/v2/aggs/ticker/{stocksTicker}/range/{multiplier}/{timespan}/{from}/{to}",
                market="Stocks",
                description="Retrieve aggregated OHLC and volume data for a stock.",
                path_prefix="/v2/aggs/ticker/",
            ),
            Endpoint(
                title="Custom Bars (OHLC)",
                path="/v2/aggs/ticker/{forexTicker}/range/{multiplier}/{timespan}/{from}/{to}",
                market="Forex",
                description="Retrieve aggregated OHLC and volume data for forex.",
                path_prefix="/v2/aggs/ticker/",
            ),
        ]
        idx = EndpointIndex(endpoints)
        results = idx.search("stock candlesticks")
        assert len(results) > 0
        assert results[0].market == "Stocks"
        assert "Bars" in results[0].title or "OHLC" in results[0].title


class TestFinanceAliases:
    """Test that finance-related aliases expand correctly via _expand_query."""

    def test_delta_alias(self):
        q = _expand_query("delta")
        assert "bs_delta" in q
        assert "options" in q

    def test_gamma_alias(self):
        q = _expand_query("gamma")
        assert "bs_gamma" in q

    def test_theta_alias(self):
        q = _expand_query("theta")
        assert "bs_theta" in q

    def test_vega_alias(self):
        q = _expand_query("vega")
        assert "bs_vega" in q

    def test_rho_alias(self):
        q = _expand_query("rho")
        assert "bs_rho" in q

    def test_blackscholes_alias(self):
        q = _expand_query("blackscholes")
        assert "bs_price" in q
        assert "bs_delta" in q

    def test_greek_alias(self):
        q = _expand_query("greek")
        assert "greeks" in q

    def test_technical_alias(self):
        q = _expand_query("technical")
        assert "aggregate" in q

    def test_indicator_alias(self):
        q = _expand_query("indicator")
        assert "aggregate" in q

    def test_moving_alias(self):
        q = _expand_query("moving")
        assert "aggregate" in q


class TestDeprecatedFilter:
    """Test that deprecated endpoints are filtered during build."""

    @pytest.mark.asyncio
    @patch("mcp_massive.index.httpx.AsyncClient")
    async def test_deprecated_endpoints_skipped(self, mock_client_class):
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        deprecated_txt = """\
## Stocks

### Aggregates (Bars)

**Endpoint:** `GET /v2/aggs/ticker/{stocksTicker}/range/{multiplier}/{timespan}/{from}/{to}`

**Description:**

Get aggregate bars.

## Query Parameters

- adjusted (boolean, optional): Whether results are adjusted for splits.
---
## Stocks

### Aggregates (Bars) (Deprecated)

**Endpoint:** `GET /v1/aggs/ticker/{stocksTicker}/range/{multiplier}/{timespan}/{from}/{to}`

**Description:**

Old aggregate bars.

## Query Parameters

- adjusted (boolean, optional): Whether results are adjusted for splits.
"""
        response = MagicMock()
        response.text = deprecated_txt
        response.raise_for_status = MagicMock()

        mock_client.get = AsyncMock(return_value=response)

        idx = await build_index()
        # Only the non-deprecated endpoint should be indexed
        results = idx.search("aggregates")
        for ep in results:
            assert "(Deprecated)" not in ep.title


class TestBuildIndex:
    @pytest.mark.asyncio
    @patch("mcp_massive.index.httpx.AsyncClient")
    async def test_build_index_success(self, mock_client_class):
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        response = MagicMock()
        response.text = llms_partial_txt()
        response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=response)

        idx = await build_index()
        assert isinstance(idx, EndpointIndex)

    @pytest.mark.asyncio
    @patch("mcp_massive.index.httpx.AsyncClient")
    async def test_build_index_fetch_failure(self, mock_client_class):
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=Exception("Network error"))

        idx = await build_index()
        # Should return empty index on failure
        assert isinstance(idx, EndpointIndex)
        assert idx.search("anything") == []

    @pytest.mark.asyncio
    @patch("mcp_massive.index.httpx.AsyncClient")
    async def test_build_index_explicit_url(self, mock_client_class):
        """build_index(llms_txt_url=...) should use the provided URL."""
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        response = MagicMock()
        response.text = llms_partial_txt()
        response.raise_for_status = MagicMock()

        custom_url = "https://custom-server.example.com/llms-full.txt"
        mock_client.get = AsyncMock(return_value=response)

        idx = await build_index(llms_txt_url=custom_url)
        assert isinstance(idx, EndpointIndex)
        # Verify the custom URL was the only get call
        first_call_url = mock_client.get.call_args_list[0].args[0]
        assert first_call_url == custom_url
        # Should be exactly one fetch (no individual doc page fetches)
        assert mock_client.get.call_count == 1


class TestLlmsFullTxtE2E:
    """End-to-end test that fetches the real llms-full.txt from massive.com.

    This catches breaking changes in the upstream document format.
    """

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_fetch_and_parse_real_llms_full_txt(self):
        """Fetch the real llms-full.txt and verify it parses into valid endpoints."""
        import httpx
        import ssl
        import certifi

        url = "https://massive.com/docs/rest/llms-full.txt"
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        async with httpx.AsyncClient(timeout=30.0, verify=ssl_ctx) as client:
            resp = await client.get(url, follow_redirects=True)
            resp.raise_for_status()
            text = resp.text

        entries = parse_llms_full_txt(text)

        # Should have a substantial number of endpoints
        assert len(entries) > 40, f"Expected 40+ endpoints, got {len(entries)}"

        # Every entry must have required fields
        for entry in entries:
            assert entry.title, f"Entry missing title: {entry}"
            assert entry.market, f"Entry missing market: {entry}"
            assert entry.path, f"Entry missing path: {entry.title}"

        # Most entries should have an endpoint path
        paths_found = sum(1 for e in entries if e.path)
        assert paths_found > 40, f"Expected 40+ entries with paths, got {paths_found}"

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_build_index_from_real_llms_full_txt(self):
        """Build a full index from the real llms-full.txt and verify search works."""
        idx = await build_index(
            llms_txt_url="https://massive.com/docs/rest/llms-full.txt"
        )

        # Should have indexed many endpoints
        assert len(idx._endpoints) > 40

        # Basic search should return results
        results = idx.search("stock aggregate bars")
        assert len(results) > 0
        assert any("aggregate" in ep.title.lower() for ep in results)

        # Market detection should work
        results = idx.search("crypto snapshot")
        assert len(results) > 0
