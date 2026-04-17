from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from urllib.parse import urlparse

from mcp_massive.index import (
    Endpoint,
    EndpointIndex,
    QueryParam,
    ResponseAttribute,
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

    def test_generic_sma_deduplicates_across_markets(self):
        """Generic 'SMA' (no asset class) should return one SMA result (deduped by title)."""
        idx = EndpointIndex(self._make_cross_asset_endpoints())
        results = idx.search("SMA")
        sma_results = [ep for ep in results if ep.title == "SMA"]
        assert len(sma_results) == 1

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
        # Real endpoint titles use "Bars (OHLC)" naming.  Accept any
        # aggregates-family marker.
        assert any(
            marker in ep.title.lower()
            for ep in results
            for marker in ("aggregate", "bars", "ohlc")
        )

        # Market detection should work
        results = idx.search("crypto snapshot")
        assert len(results) > 0


class TestMarketFilter:
    def _endpoints(self):
        return [
            Endpoint(
                title="SMA",
                path="/v1/indicators/sma/{stocksTicker}",
                market="Stocks",
                description="Simple moving average for a stock ticker.",
                path_prefix="/v1/indicators/sma/",
            ),
            Endpoint(
                title="SMA",
                path="/v1/indicators/sma/{cryptoTicker}",
                market="Crypto",
                description="Simple moving average for a crypto ticker.",
                path_prefix="/v1/indicators/sma/",
            ),
            Endpoint(
                title="Splits",
                path="/stocks/v1/splits",
                market="Stocks",
                description="Stock split history.",
                path_prefix="/stocks/v1/splits",
            ),
            Endpoint(
                title="Unified Snapshot",
                path="/v3/snapshot",
                market="Stocks",
                description="Unified snapshot across assets.",
                path_prefix="/v3/snapshot",
            ),
        ]

    def test_explicit_market_is_strict(self):
        """Explicit market filter excludes all other-market rows."""
        idx = EndpointIndex(self._endpoints())
        results = idx.search("sma", market="Crypto")
        assert all(ep.market == "Crypto" for ep in results)
        assert results[0].title == "SMA"

    def test_explicit_market_with_zero_matches_returns_empty(self):
        """Strict mode: no fallback when explicit filter matches nothing."""
        idx = EndpointIndex(self._endpoints())
        results = idx.search("sma", market="Forex")
        assert results == []

    def test_inferred_market_prefers_matching_market(self):
        """When market is inferred, the matching-market row ranks above
        a same-titled row from a different market."""
        idx = EndpointIndex(self._endpoints())
        results = idx.search("crypto sma")
        assert results[0].market == "Crypto"
        assert results[0].title == "SMA"

    def test_empty_endpoints_returns_empty(self):
        """Regression guard: search over an empty index returns []."""
        idx = EndpointIndex([])
        assert idx.search("anything") == []
        assert idx.search("anything", market="Crypto") == []

    def test_empty_query_returns_empty(self):
        """A query that tokenizes to nothing short-circuits before FTS."""
        idx = EndpointIndex(self._endpoints())
        assert idx.search("") == []
        assert idx.search("!!!") == []  # only punctuation → no tokens


class TestInferredMarketBoost:
    """Inferred-market preference is a *soft* boost, not a filter:

    1. The ``market`` column is indexed, so its literal value ("Stocks",
       "Crypto", …) contributes to BM25 whenever the query contains a
       matching token like "stock" or "crypto".
    2. When a market is also *detected* from the query via
       :func:`_detect_market`, we apply a multiplicative
       :attr:`EndpointIndex._MARKET_BOOST` to all rows whose market
       matches.

    The combination shifts ranking toward the inferred market but does
    not hide strong cross-market matches.
    """

    def _endpoints(self):
        return [
            Endpoint(
                title="Aggregates Stocks",
                path="/v2/aggs/ticker/{stocksTicker}",
                market="Stocks",
                description="Aggregate bars for a stock.",
                path_prefix="/v2/aggs/ticker/",
            ),
            Endpoint(
                title="Aggregates Crypto",
                path="/v2/aggs/ticker/{cryptoTicker}",
                market="Crypto",
                description="Aggregate bars for crypto.",
                path_prefix="/v2/aggs/ticker/",
            ),
        ]

    def test_cross_market_rows_still_surface(self):
        """A generic query without a market keyword returns rows from
        multiple markets — the boost only kicks in when a market is
        detected, so without one both Stocks and Crypto rows appear."""
        idx = EndpointIndex(self._endpoints())
        results = idx.search("aggregates", top_k=5)
        markets = {ep.market for ep in results}
        assert "Stocks" in markets
        assert "Crypto" in markets

    def test_explicit_filter_is_strict(self):
        """Explicit market filters do NOT apply the boost; only matching
        rows are returned even when other markets score higher."""
        idx = EndpointIndex(self._endpoints())
        results = idx.search("aggregates", market="Stocks", top_k=5)
        assert all(ep.market == "Stocks" for ep in results)

    def test_unknown_market_endpoint_still_findable(self):
        """Endpoints whose market matches no detection keywords still
        surface — we never exclude rows based on inferred market."""
        endpoints = [
            Endpoint(
                title="Mystery Endpoint",
                path="/some/new/path",
                market="Partners",
                description="A novel endpoint under a partner namespace.",
                path_prefix="/some/new/path",
            ),
        ]
        idx = EndpointIndex(endpoints)
        results = idx.search("mystery")
        assert len(results) == 1
        assert results[0].title == "Mystery Endpoint"

    def test_inferred_market_outranks_other_markets(self):
        """When the query contains an inferred-market keyword, rows of
        that market rank above other-market rows."""
        endpoints = [
            Endpoint(
                title="Crypto Aggregates",
                path="/v2/aggs/ticker/{cryptoTicker}",
                market="Crypto",
                description="Aggregate bars for crypto.",
                path_prefix="/v2/aggs/ticker/",
            ),
            Endpoint(
                title="Crypto Snapshot",
                path="/v2/snapshot/crypto",
                market="Crypto",
                description="Crypto snapshot.",
                path_prefix="/v2/snapshot/crypto",
            ),
            Endpoint(
                title="Stocks Aggregates",
                path="/v2/aggs/ticker/{stocksTicker}",
                market="Stocks",
                description="Aggregate bars for a stock.",
                path_prefix="/v2/aggs/ticker/",
            ),
        ]
        idx = EndpointIndex(endpoints)
        results = idx.search("crypto", top_k=5)
        crypto_ranks = [i for i, ep in enumerate(results) if ep.market == "Crypto"]
        stocks_ranks = [i for i, ep in enumerate(results) if ep.market == "Stocks"]
        assert crypto_ranks, "expected at least one Crypto result"
        if stocks_ranks:
            assert max(crypto_ranks) < min(stocks_ranks), (
                "Crypto rows should all rank above Stocks rows under "
                "inferred Crypto market: "
                f"{[(ep.title, ep.market) for ep in results]}"
            )

    def test_cross_market_title_match_beats_inferred_boost(self):
        """The boost must not be so large that a weak in-market match
        outranks a strong cross-market title match.

        Real-world case: "stock ratings" infers Stocks but the right
        endpoint is Analyst Ratings under Partners — its title match
        should outweigh the 2x boost any generic Stocks row gets.
        """
        endpoints = [
            Endpoint(
                title="Splits",
                path="/stocks/v1/splits",
                market="Stocks",
                description="Stock split history.",
                path_prefix="/stocks/v1/splits",
            ),
            Endpoint(
                title="Trades",
                path="/v3/trades/{stockTicker}",
                market="Stocks",
                description="Historical trades for a stock.",
                path_prefix="/v3/trades/",
            ),
            Endpoint(
                title="Analyst Ratings",
                path="/benzinga/v1/ratings",
                market="Partners",
                description="Historical analyst ratings and price targets.",
                path_prefix="/benzinga/v1/ratings",
            ),
        ]
        idx = EndpointIndex(endpoints)
        results = idx.search("stock ratings", top_k=3)
        # Partners-market Analyst Ratings should win despite the
        # inferred-Stocks boost applied to the other two rows.
        assert results[0].title == "Analyst Ratings"
        assert results[0].market == "Partners"


class TestAttrsColumn:
    """The ``attrs`` FTS column indexes response-attribute field names
    (e.g. ``debt_to_equity``, ``yield_10_year``) so queries for those
    jargon terms surface the right endpoint without hand-curated
    aliases.  Attrs is weighted well below title/description so it
    only influences ranking for otherwise-weak matches.
    """

    def test_response_attr_names_are_indexed(self):
        """A query matching only a response-attribute field name still
        finds the endpoint that owns that field."""
        endpoints = [
            Endpoint(
                title="Ratios",
                path="/stocks/financials/v1/ratios",
                market="Stocks",
                description="Key financial ratios for a company.",
                response_attributes=[
                    ResponseAttribute(
                        name="results[].debt_to_equity",
                        type="number",
                        description="Debt-to-equity ratio.",
                    ),
                    ResponseAttribute(
                        name="results[].return_on_equity",
                        type="number",
                        description="Return on equity.",
                    ),
                ],
                path_prefix="/stocks/financials/v1/ratios",
            ),
            Endpoint(
                title="Last Trade",
                path="/v2/last/trade/{stocksTicker}",
                market="Stocks",
                description="Most recent stock trade.",
                path_prefix="/v2/last/trade/",
            ),
        ]
        idx = EndpointIndex(endpoints)
        # "debt_to_equity" appears only in the Ratios endpoint's
        # response attrs — it should rank Ratios first.
        results = idx.search("debt to equity", top_k=2)
        assert results[0].title == "Ratios"

    def test_results_prefix_stripped_from_attrs(self):
        """The common ``results[].`` / ``results.`` wrapper is stripped
        so searches don't need to include it."""
        endpoints = [
            Endpoint(
                title="Treasury Yields",
                path="/fed/v1/treasury-yields",
                market="Economy",
                description="U.S. Treasury yield data.",
                response_attributes=[
                    ResponseAttribute(
                        name="results[].yield_10_year",
                        type="number",
                        description="",
                    ),
                    ResponseAttribute(
                        name="results[].yield_2_year",
                        type="number",
                        description="",
                    ),
                ],
                path_prefix="/fed/v1/treasury-yields",
            ),
        ]
        idx = EndpointIndex(endpoints)
        # Plain "yield" should find this endpoint via the stripped attrs.
        results = idx.search("10 year yield")
        assert results and results[0].title == "Treasury Yields"

    def test_query_params_not_indexed_in_attrs(self):
        """Query params are NOT added to ``attrs`` — they're dominated
        by generic filter operators (``ticker``, ``date``, ``limit``,
        ``sort``) that appear on nearly every endpoint."""
        endpoints = [
            Endpoint(
                title="Trades",
                path="/v3/trades/{ticker}",
                market="Stocks",
                description="Historical trades.",
                query_params=[
                    QueryParam(
                        name="limit",
                        type="integer",
                        required=False,
                        description="",
                    ),
                ],
                path_prefix="/v3/trades/",
            ),
            Endpoint(
                title="Quotes",
                path="/v3/quotes/{ticker}",
                market="Stocks",
                description="Historical quotes.",
                query_params=[
                    QueryParam(
                        name="limit",
                        type="integer",
                        required=False,
                        description="",
                    ),
                ],
                path_prefix="/v3/quotes/",
            ),
        ]
        idx = EndpointIndex(endpoints)
        # If "limit" were in attrs, this query would match both
        # endpoints via the attrs column.  We expect no matches.
        results = idx.search("limit")
        assert results == []
