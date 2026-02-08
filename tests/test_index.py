from unittest.mock import patch, MagicMock

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
    def test_strips_trailing_s(self):
        assert _stem("tickers") == "ticker"
        assert _stem("aggregates") == "aggregat"

    def test_strips_trailing_es(self):
        assert _stem("exchanges") == "exchang"

    def test_strips_trailing_ing(self):
        assert _stem("trading") == "trad"

    def test_strips_trailing_ed(self):
        assert _stem("adjusted") == "adjust"

    def test_no_strip_if_too_short(self):
        # "as" minus "s" = "a" (1 char), shouldn't strip
        assert _stem("as") == "as"
        assert _stem("is") == "is"

    def test_no_strip_if_no_suffix(self):
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
        assert "stock" in tokens
        assert "bar" in tokens
        assert "ticker" in tokens

    def test_alias_expansion(self):
        tokens = _tokenize("agg candle fx")
        assert "aggregate" in tokens  # alias for "agg"
        assert "aggregate" in tokens  # alias for "candle"
        assert "forex" in tokens  # alias for "fx"

    def test_alias_keeps_stemmed_original(self):
        # "candle" should produce both "aggregate" (alias) and "candle" (stem, unchanged)
        tokens = _tokenize("candle")
        assert "aggregate" in tokens
        assert "candle" in tokens

    def test_stemming_in_tokenize(self):
        tokens = _tokenize("tickers dividends")
        assert "ticker" in tokens
        assert "dividend" in tokens


class TestEndpointIndex:
    def _make_endpoints(self):
        return [
            Endpoint(
                name="Aggregates (Bars)",
                category="Market Data",
                url="https://massive.com/docs/aggs",
                description="Get aggregate bars for a stock.",
                endpoint_pattern="GET /v2/aggs/ticker/{stocksTicker}/range/{multiplier}/{timespan}/{from}/{to}",
                compressed_doc="GET /v2/aggs/...",
                path_prefix="/v2/aggs/ticker/",
            ),
            Endpoint(
                name="Tickers",
                category="Reference Data",
                url="https://massive.com/docs/tickers",
                description="Query all ticker symbols.",
                endpoint_pattern="GET /v3/reference/tickers",
                compressed_doc="GET /v3/reference/tickers",
                path_prefix="/v3/reference/tickers",
            ),
            Endpoint(
                name="Last Trade",
                category="Market Data",
                url="https://massive.com/docs/last-trade",
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
        doc = idx.get_doc("https://massive.com/docs/aggs")
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


class TestBuildIndex:
    @patch("mcp_massive.index.httpx.Client")
    def test_build_index_success(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock llms.txt response
        llms_response = MagicMock()
        llms_response.text = SAMPLE_LLMS_TXT
        llms_response.raise_for_status = MagicMock()

        # Mock doc page responses
        doc_response = MagicMock()
        doc_response.text = SAMPLE_DOC_PAGE
        doc_response.raise_for_status = MagicMock()

        mock_client.get.return_value = llms_response

        # Override _fetch_doc behavior via side_effect on client.get
        def get_side_effect(url, **kwargs):
            if url == "https://massive.com/docs/rest/llms.txt":
                return llms_response
            return doc_response

        mock_client.get.side_effect = get_side_effect

        idx = build_index()
        assert isinstance(idx, EndpointIndex)

    @patch("mcp_massive.index.httpx.Client")
    def test_build_index_llms_txt_failure(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.get.side_effect = Exception("Network error")

        idx = build_index()
        # Should return empty index on failure
        assert isinstance(idx, EndpointIndex)
        assert idx.search("anything") == []
