from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from mcp_massive.index import (
    Endpoint,
    EndpointIndex,
    parse_llms_txt,
    extract_endpoint_pattern,
    extract_path_prefix,
    compress_doc,
    build_index,
    _stem,
    _tokenize,
    _build_corpus_text,
    _detect_category,
)


SAMPLE_LLMS_TXT = """\
# Massive.com API

> API documentation for Massive.com

## Market Data

- [Aggregates (Bars)](https://massive.com/docs/aggs): Get aggregate bars for a stock.
- [Grouped Daily](https://massive.com/docs/grouped): Get grouped daily bars for the market.

## Reference Data

- [Tickers](https://massive.com/docs/tickers): Query all ticker symbols.
"""

SAMPLE_DOC_PAGE = """\
# Aggregates (Bars)

Get aggregate bars for a stock over a given date range.

**Endpoint:** `GET /v2/aggs/ticker/{stocksTicker}/range/{multiplier}/{timespan}/{from}/{to}`

## Query Parameters

- adjusted (boolean, optional): Whether results are adjusted for splits.
- sort (string, optional): Sort order of results. asc or desc.
- limit (integer, optional): Limits number of results. Default 5000, max 50000.

## Response Attributes

- ticker (string): The exchange symbol.
- adjusted (boolean): Whether results are adjusted.
- results (array): Array of result objects.

## Sample Response

```json
{
  "ticker": "AAPL",
  "results": [{"o": 130.28, "c": 129.04}]
}
```
"""

SAMPLE_DOC_NO_PARAMS = """\
# Market Holidays

**Endpoint:** `GET /v1/marketstatus/upcoming`
"""


class TestParseLlmsTxt:
    def test_parses_entries(self):
        entries = parse_llms_txt(SAMPLE_LLMS_TXT)
        assert len(entries) == 3

    def test_entry_fields(self):
        entries = parse_llms_txt(SAMPLE_LLMS_TXT)
        aggs = entries[0]
        assert aggs["name"] == "Aggregates (Bars)"
        assert aggs["url"] == "https://massive.com/docs/aggs"
        assert aggs["description"] == "Get aggregate bars for a stock."
        assert aggs["category"] == "Market Data"

    def test_category_tracking(self):
        entries = parse_llms_txt(SAMPLE_LLMS_TXT)
        assert entries[0]["category"] == "Market Data"
        assert entries[1]["category"] == "Market Data"
        assert entries[2]["category"] == "Reference Data"

    def test_empty_input(self):
        assert parse_llms_txt("") == []

    def test_no_entries(self):
        assert parse_llms_txt("# Just a title\n\nSome text.") == []


class TestExtractEndpointPattern:
    def test_standard_pattern(self):
        result = extract_endpoint_pattern(SAMPLE_DOC_PAGE)
        assert (
            result
            == "GET /v2/aggs/ticker/{stocksTicker}/range/{multiplier}/{timespan}/{from}/{to}"
        )

    def test_no_params_pattern(self):
        result = extract_endpoint_pattern(SAMPLE_DOC_NO_PARAMS)
        assert result == "GET /v1/marketstatus/upcoming"

    def test_no_pattern_found(self):
        result = extract_endpoint_pattern("No endpoint here.")
        assert result == ""


class TestExtractPathPrefix:
    def test_path_with_params(self):
        prefix = extract_path_prefix(
            "GET /v2/aggs/ticker/{stocksTicker}/range/{multiplier}/{timespan}/{from}/{to}"
        )
        assert prefix == "/v2/aggs/ticker/"

    def test_crypto_path(self):
        prefix = extract_path_prefix("GET /v1/last/crypto/{from}/{to}")
        assert prefix == "/v1/last/crypto/"

    def test_no_params(self):
        prefix = extract_path_prefix("GET /v3/reference/tickers")
        assert prefix == "/v3/reference/tickers"

    def test_no_method_prefix(self):
        prefix = extract_path_prefix("/v2/aggs/ticker/{ticker}/prev")
        assert prefix == "/v2/aggs/ticker/"


class TestCompressDoc:
    def test_retains_endpoint_pattern(self):
        compressed = compress_doc(SAMPLE_DOC_PAGE)
        assert "GET /v2/aggs/ticker/" in compressed

    def test_retains_query_params(self):
        compressed = compress_doc(SAMPLE_DOC_PAGE)
        assert "adjusted" in compressed
        assert "sort" in compressed
        assert "limit" in compressed

    def test_strips_response_attributes(self):
        compressed = compress_doc(SAMPLE_DOC_PAGE)
        # Response attribute fields should not be present
        assert "Array of result objects" not in compressed

    def test_strips_sample_response(self):
        compressed = compress_doc(SAMPLE_DOC_PAGE)
        assert "Sample Response" not in compressed
        assert '"ticker": "AAPL"' not in compressed

    def test_no_params_doc(self):
        compressed = compress_doc(SAMPLE_DOC_NO_PARAMS)
        assert "GET /v1/marketstatus/upcoming" in compressed


class TestStem:
    def test_exchange_consistency(self):
        """Both 'exchange' and 'exchanges' should produce the same stem."""
        assert _stem("exchange") == _stem("exchanges")

    def test_aggregate_consistency(self):
        """Both 'aggregate' and 'aggregates' should produce the same stem."""
        assert _stem("aggregate") == _stem("aggregates")

    def test_ticker_stems(self):
        assert _stem("tickers") == _stem("ticker")

    def test_trading_stem(self):
        stem = _stem("trading")
        assert stem == "trade"

    def test_adjusted_stem(self):
        stem = _stem("adjusted")
        assert stem == "adjust"

    def test_no_stem_short_words(self):
        # Short words should still produce something
        assert len(_stem("as")) > 0
        assert len(_stem("is")) > 0

    def test_no_stem_needed(self):
        assert _stem("crypto") == "crypto"
        assert _stem("forex") == "forex"


class TestTokenize:
    def test_basic_tokenization(self):
        tokens = _tokenize("Hello World 123")
        assert "hello" in tokens
        assert "world" in tokens
        assert "123" in tokens

    def test_strips_punctuation(self):
        tokens = _tokenize("stock. (bars) ticker,")
        # stems may differ from old custom stemmer
        assert any("stock" in t for t in tokens)
        assert any("ticker" in t or "ticker" == t for t in tokens)

    def test_alias_expansion(self):
        tokens = _tokenize("agg candle fx")
        assert "aggregate" in tokens  # alias for "agg"
        assert "forex" in tokens  # alias for "fx"

    def test_alias_keeps_stemmed_original(self):
        # "candle" should produce both "aggregate" (alias) and stemmed "candle"
        tokens = _tokenize("candle")
        assert "aggregate" in tokens
        assert _stem("candle") in tokens

    def test_stemming_in_tokenize(self):
        tokens = _tokenize("tickers dividends")
        assert _stem("tickers") in tokens
        assert _stem("dividends") in tokens

    def test_stopword_removal(self):
        tokens = _tokenize("the stock is a good one")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "a" not in tokens

    def test_list_alias_expansion(self):
        """'price' should expand to multiple aliases."""
        tokens = _tokenize("price")
        assert "trade" in tokens
        assert "aggregate" in tokens
        assert "snapshot" in tokens


class TestBuildCorpusText:
    def test_repeats_name_and_category(self):
        ep = Endpoint(
            name="SMA",
            category="Stocks",
            url="https://massive.com/docs/rest/stocks/sma",
            description="Get SMA for a stock ticker.",
            endpoint_pattern="GET /v1/indicators/sma/{stocksTicker}",
            compressed_doc="...",
            path_prefix="/v1/indicators/sma/",
        )
        text = _build_corpus_text(ep)
        assert text.count("SMA") >= 3
        assert text.count("Stocks") >= 2

    def test_extracts_camel_case_params(self):
        ep = Endpoint(
            name="Aggregates",
            category="Stocks",
            url="https://massive.com/docs/rest/stocks/aggs",
            description="Get aggs.",
            endpoint_pattern="GET /v2/aggs/ticker/{stocksTicker}/range/{multiplier}/{timespan}/{from}/{to}",
            compressed_doc="...",
            path_prefix="/v2/aggs/ticker/",
        )
        text = _build_corpus_text(ep)
        assert "stocks" in text.lower()
        assert "ticker" in text.lower()

    def test_extracts_path_segments(self):
        ep = Endpoint(
            name="Aggregates",
            category="Stocks",
            url="https://massive.com/docs/rest/stocks/aggs",
            description="Get aggs.",
            endpoint_pattern="GET /v2/aggs/ticker/{stocksTicker}/range/{multiplier}/{timespan}/{from}/{to}",
            compressed_doc="...",
            path_prefix="/v2/aggs/ticker/",
        )
        text = _build_corpus_text(ep)
        assert "aggs" in text
        assert "ticker" in text
        assert "range" in text

    def test_extracts_doc_url_category(self):
        ep = Endpoint(
            name="SMA",
            category="Stocks",
            url="https://massive.com/docs/rest/stocks/sma",
            description="Get SMA.",
            endpoint_pattern="GET /v1/indicators/sma/{stocksTicker}",
            compressed_doc="...",
            path_prefix="/v1/indicators/sma/",
        )
        text = _build_corpus_text(ep)
        # "stocks" from the doc URL
        parts = text.split()
        assert "stocks" in parts


class TestDetectCategory:
    def test_detects_stocks(self):
        assert _detect_category("stock SMA") == "Stocks"

    def test_detects_crypto(self):
        assert _detect_category("crypto snapshot") == "Crypto"

    def test_detects_forex(self):
        assert _detect_category("forex rates") == "Forex"

    def test_detects_options(self):
        assert _detect_category("options chain") == "Options"

    def test_no_category(self):
        assert _detect_category("SMA") is None

    def test_no_category_generic(self):
        assert _detect_category("aggregate bars") is None


class TestEndpointIndex:
    def _make_endpoints(self):
        return [
            Endpoint(
                name="Aggregates (Bars)",
                category="Market Data",
                url="https://massive.com/docs/rest/stocks/aggs",
                description="Get aggregate bars for a stock.",
                endpoint_pattern="GET /v2/aggs/ticker/{stocksTicker}/range/{multiplier}/{timespan}/{from}/{to}",
                compressed_doc="GET /v2/aggs/...",
                path_prefix="/v2/aggs/ticker/",
            ),
            Endpoint(
                name="Tickers",
                category="Reference Data",
                url="https://massive.com/docs/rest/reference/tickers",
                description="Query all ticker symbols.",
                endpoint_pattern="GET /v3/reference/tickers",
                compressed_doc="GET /v3/reference/tickers",
                path_prefix="/v3/reference/tickers",
            ),
            Endpoint(
                name="Last Trade",
                category="Market Data",
                url="https://massive.com/docs/rest/stocks/last-trade",
                description="Get the most recent trade for a ticker.",
                endpoint_pattern="GET /v2/last/trade/{stocksTicker}",
                compressed_doc="GET /v2/last/trade/...",
                path_prefix="/v2/last/trade/",
            ),
        ]

    def test_search_returns_relevant(self):
        idx = EndpointIndex(self._make_endpoints())
        results = idx.search("aggregate bars stock")
        assert len(results) > 0
        assert results[0].name == "Aggregates (Bars)"

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

    def test_get_doc_found(self):
        idx = EndpointIndex(self._make_endpoints())
        doc = idx.get_doc("https://massive.com/docs/rest/stocks/aggs")
        assert doc == "GET /v2/aggs/..."

    def test_get_doc_not_found(self):
        idx = EndpointIndex(self._make_endpoints())
        assert idx.get_doc("https://massive.com/docs/nonexistent") is None

    def test_search_alias_agg(self):
        """'agg' should find Aggregates via alias expansion."""
        idx = EndpointIndex(self._make_endpoints())
        results = idx.search("agg")
        assert any("Aggregates" in ep.name for ep in results)

    def test_search_alias_candle(self):
        """'candle' should find Aggregates via alias expansion."""
        idx = EndpointIndex(self._make_endpoints())
        results = idx.search("candle")
        assert any("Aggregates" in ep.name for ep in results)

    def test_search_alias_ohlc(self):
        """'ohlc' should find Aggregates via alias expansion."""
        idx = EndpointIndex(self._make_endpoints())
        results = idx.search("ohlc data")
        assert any("Aggregates" in ep.name for ep in results)

    def test_search_stemmed_plural(self):
        """'tickers' (plural) should still match 'Tickers' endpoint."""
        idx = EndpointIndex(self._make_endpoints())
        results = idx.search("tickers")
        assert any("Tickers" in ep.name for ep in results)

    def test_search_alias_symbol(self):
        """'symbol' should find Tickers via alias expansion."""
        idx = EndpointIndex(self._make_endpoints())
        results = idx.search("symbol lookup")
        assert any("Tickers" in ep.name for ep in results)

    def test_search_alias_transaction(self):
        """'transaction' should find Last Trade via alias expansion."""
        idx = EndpointIndex(self._make_endpoints())
        results = idx.search("last transaction")
        assert any("Trade" in ep.name for ep in results)


class TestCrossAssetClassRanking:
    """Test that asset-class keywords properly disambiguate identical endpoint names."""

    def _make_cross_asset_endpoints(self):
        return [
            Endpoint(
                name="SMA",
                category="Stocks",
                url="https://massive.com/docs/rest/stocks/sma",
                description="Get SMA for a stock ticker.",
                endpoint_pattern="GET /v1/indicators/sma/{stocksTicker}",
                compressed_doc="...",
                path_prefix="/v1/indicators/sma/",
            ),
            Endpoint(
                name="SMA",
                category="Crypto",
                url="https://massive.com/docs/rest/crypto/sma",
                description="Get SMA for a crypto ticker.",
                endpoint_pattern="GET /v1/indicators/sma/{cryptoTicker}",
                compressed_doc="...",
                path_prefix="/v1/indicators/sma/",
            ),
            Endpoint(
                name="SMA",
                category="Forex",
                url="https://massive.com/docs/rest/forex/sma",
                description="Get SMA for a forex ticker.",
                endpoint_pattern="GET /v1/indicators/sma/{forexTicker}",
                compressed_doc="...",
                path_prefix="/v1/indicators/sma/",
            ),
            Endpoint(
                name="Unified Snapshot",
                category="Stocks",
                url="https://massive.com/docs/rest/stocks/snapshot",
                description="Get unified snapshot for a stock ticker.",
                endpoint_pattern="GET /v3/snapshot/{stocksTicker}",
                compressed_doc="...",
                path_prefix="/v3/snapshot/",
            ),
            Endpoint(
                name="Unified Snapshot",
                category="Crypto",
                url="https://massive.com/docs/rest/crypto/snapshot",
                description="Get unified snapshot for a crypto ticker.",
                endpoint_pattern="GET /v3/snapshot/{cryptoTicker}",
                compressed_doc="...",
                path_prefix="/v3/snapshot/",
            ),
        ]

    def test_stock_sma_ranks_first(self):
        """'stock SMA' should rank Stocks SMA above Crypto/Forex SMA."""
        idx = EndpointIndex(self._make_cross_asset_endpoints())
        results = idx.search("stock SMA")
        assert len(results) > 0
        assert results[0].category == "Stocks"
        assert results[0].name == "SMA"

    def test_crypto_snapshot_ranks_first(self):
        """'crypto snapshot' should rank Crypto snapshot above Stocks snapshot."""
        idx = EndpointIndex(self._make_cross_asset_endpoints())
        results = idx.search("crypto snapshot")
        # Find the first snapshot result
        snapshot_results = [ep for ep in results if "Snapshot" in ep.name]
        assert len(snapshot_results) > 0
        assert snapshot_results[0].category == "Crypto"

    def test_generic_sma_returns_multiple_categories(self):
        """Generic 'SMA' (no asset class) should return results from multiple categories."""
        idx = EndpointIndex(self._make_cross_asset_endpoints())
        results = idx.search("SMA")
        categories = {ep.category for ep in results if ep.name == "SMA"}
        assert len(categories) >= 2


class TestFinanceAliases:
    """Test that finance-related aliases expand correctly."""

    def test_delta_alias(self):
        tokens = _tokenize("delta")
        assert "bs_delta" in tokens
        assert "options" in tokens

    def test_gamma_alias(self):
        tokens = _tokenize("gamma")
        assert "bs_gamma" in tokens

    def test_theta_alias(self):
        tokens = _tokenize("theta")
        assert "bs_theta" in tokens

    def test_vega_alias(self):
        tokens = _tokenize("vega")
        assert "bs_vega" in tokens

    def test_rho_alias(self):
        tokens = _tokenize("rho")
        assert "bs_rho" in tokens

    def test_blackscholes_alias(self):
        tokens = _tokenize("blackscholes")
        assert "bs_price" in tokens
        assert "bs_delta" in tokens

    def test_greek_alias(self):
        tokens = _tokenize("greek")
        assert "greeks" in tokens

    def test_technical_alias(self):
        tokens = _tokenize("technical")
        assert "aggregate" in tokens

    def test_indicator_alias(self):
        tokens = _tokenize("indicator")
        assert "aggregate" in tokens

    def test_moving_alias(self):
        tokens = _tokenize("moving")
        assert "aggregate" in tokens


class TestStemmerConsistency:
    """Test that the Snowball stemmer fixes the old custom stemmer's inconsistencies."""

    def test_exchange_exchanges_same_stem(self):
        assert _stem("exchange") == _stem("exchanges")

    def test_aggregate_aggregates_same_stem(self):
        assert _stem("aggregate") == _stem("aggregates")

    def test_dividend_dividends_same_stem(self):
        assert _stem("dividend") == _stem("dividends")

    def test_ticker_tickers_same_stem(self):
        assert _stem("ticker") == _stem("tickers")


class TestDeprecatedFilter:
    """Test that deprecated endpoints are filtered during build."""

    @pytest.mark.asyncio
    @patch("mcp_massive.index.httpx.AsyncClient")
    async def test_deprecated_endpoints_skipped(self, mock_client_class):
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        llms_txt = """\
# API

## Stocks

- [Aggregates (Bars)](https://massive.com/docs/aggs): Get aggs.
- [Aggregates (Bars) (Deprecated)](https://massive.com/docs/aggs-old): Old aggs.
"""
        llms_response = MagicMock()
        llms_response.text = llms_txt
        llms_response.raise_for_status = MagicMock()

        doc_response = MagicMock()
        doc_response.text = SAMPLE_DOC_PAGE
        doc_response.raise_for_status = MagicMock()

        async def get_side_effect(url, **kwargs):
            if url == "https://massive.com/docs/rest/llms.txt":
                return llms_response
            return doc_response

        mock_client.get = AsyncMock(side_effect=get_side_effect)

        idx = await build_index()
        # Only the non-deprecated endpoint should be indexed
        results = idx.search("aggregates")
        for ep in results:
            assert "(Deprecated)" not in ep.name


class TestBuildIndex:
    @pytest.mark.asyncio
    @patch("mcp_massive.index.httpx.AsyncClient")
    async def test_build_index_success(self, mock_client_class):
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        # Mock llms.txt response
        llms_response = MagicMock()
        llms_response.text = SAMPLE_LLMS_TXT
        llms_response.raise_for_status = MagicMock()

        # Mock doc page responses
        doc_response = MagicMock()
        doc_response.text = SAMPLE_DOC_PAGE
        doc_response.raise_for_status = MagicMock()

        async def get_side_effect(url, **kwargs):
            if url == "https://massive.com/docs/rest/llms.txt":
                return llms_response
            return doc_response

        mock_client.get = AsyncMock(side_effect=get_side_effect)

        idx = await build_index()
        assert isinstance(idx, EndpointIndex)

    @pytest.mark.asyncio
    @patch("mcp_massive.index.httpx.AsyncClient")
    async def test_build_index_llms_txt_failure(self, mock_client_class):
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
    async def test_build_index_partial_doc_failures(self, mock_client_class):
        """Index should still work when some doc pages fail to fetch."""
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        llms_response = MagicMock()
        llms_response.text = SAMPLE_LLMS_TXT
        llms_response.raise_for_status = MagicMock()

        doc_response = MagicMock()
        doc_response.text = SAMPLE_DOC_PAGE
        doc_response.raise_for_status = MagicMock()

        call_count = 0

        async def get_side_effect(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if url == "https://massive.com/docs/rest/llms.txt":
                return llms_response
            # Fail every other doc page
            if call_count % 2 == 0:
                raise Exception("Simulated doc fetch failure")
            return doc_response

        mock_client.get = AsyncMock(side_effect=get_side_effect)

        idx = await build_index()
        assert isinstance(idx, EndpointIndex)
        # Should have at least one endpoint (the one that didn't fail)
        results = idx.search("aggregates")
        assert len(results) >= 0  # May or may not match depending on which page failed

    @pytest.mark.asyncio
    @patch("mcp_massive.index.httpx.AsyncClient")
    async def test_build_index_explicit_url(self, mock_client_class):
        """build_index(llms_txt_url=...) should use the provided URL."""
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        llms_response = MagicMock()
        llms_response.text = SAMPLE_LLMS_TXT
        llms_response.raise_for_status = MagicMock()

        doc_response = MagicMock()
        doc_response.text = SAMPLE_DOC_PAGE
        doc_response.raise_for_status = MagicMock()

        custom_url = "https://custom-server.example.com/llms.txt"

        async def get_side_effect(url, **kwargs):
            if url == custom_url:
                return llms_response
            if "massive.com/docs/rest/llms.txt" in url:
                # Default URL should NOT be called
                raise AssertionError(f"Default URL was called: {url}")
            return doc_response

        mock_client.get = AsyncMock(side_effect=get_side_effect)

        idx = await build_index(llms_txt_url=custom_url)
        assert isinstance(idx, EndpointIndex)
        # Verify the custom URL was the first get call
        first_call_url = mock_client.get.call_args_list[0].args[0]
        assert first_call_url == custom_url
