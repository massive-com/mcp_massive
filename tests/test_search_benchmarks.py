"""Comprehensive search-quality benchmarks driven by ``usage.md``.

These tests assert that for the most common natural-language queries the
MCP-server actually fields in production, at least one of the expected
endpoint paths appears in the **default-size** result set returned by
:meth:`EndpointIndex.search`.  Default ``top_k`` is 7 (matching what
``search_endpoints`` returns when ``scope='endpoints'``), which is what
the LLM sees on a vanilla call.

Each row in :data:`BENCHMARKS` is a ``(query, expected_paths)`` pair
where ``expected_paths`` is a list of acceptable endpoint paths.  The
list combines ``usage.md``'s "Must return" and "Acceptable also-rank"
columns — both are correct answers for the user's intent, and the test
passes if any of them surfaces.

A failure here means the LLM driving ``search_endpoints`` would be
likely to either pick a wrong endpoint or waste a turn refining its
query.  Both costs the user tokens and erodes trust in the tool.
"""

from pathlib import Path

import pytest

from mcp_massive.index import EndpointIndex, parse_llms_full_txt, _path_prefix


_FIXTURE = Path(__file__).parent / "fixtures" / "llms-full.txt"


# (query, list-of-acceptable-paths).  Paths use the canonical
# parameterized form as parsed from the docs; matching is by prefix so
# concrete instantiations (e.g. ".../AAPL/range/...") still match.
BENCHMARKS: list[tuple[str, list[str]]] = [
    # ── Aggregates / bars ──────────────────────────────────────────
    (
        "daily candles for AAPL last month",
        [
            "/v2/aggs/ticker/{stocksTicker}/range/",
            "/v2/aggs/ticker/{stocksTicker}/prev",
        ],
    ),
    (
        "OHLC bars for SPY",
        ["/v2/aggs/ticker/{stocksTicker}/range/"],
    ),
    (
        "intraday 5-minute bars for NVDA",
        ["/v2/aggs/ticker/{stocksTicker}/range/"],
    ),
    (
        "stock aggregates AAPL",
        ["/v2/aggs/ticker/{stocksTicker}/range/"],
    ),
    # ── Snapshots / quote-of-now ──────────────────────────────────
    (
        "current price of TSLA",
        [
            "/v2/snapshot/locale/us/markets/stocks/tickers/{stocksTicker}",
            "/v3/snapshot",
        ],
    ),
    (
        "snapshot of all stocks",
        [
            "/v2/snapshot/locale/us/markets/stocks/tickers",
            "/v3/snapshot",
        ],
    ),
    (
        "market overview",
        [
            "/v2/snapshot/locale/us/markets/stocks/tickers",
            "/v3/snapshot",
            "/v3/snapshot/indices",
        ],
    ),
    (
        "today's biggest gainers",
        ["/v2/snapshot/locale/us/markets/stocks/{direction}"],
    ),
    # ── Previous close / open-close ────────────────────────────────
    (
        "yesterday's close for NVDA",
        ["/v2/aggs/ticker/{stocksTicker}/prev"],
    ),
    (
        "MSFT open and close yesterday",
        [
            "/v2/aggs/ticker/{stocksTicker}/prev",
            "/v1/open-close/{stocksTicker}/{date}",
        ],
    ),
    # ── Trades / quotes ────────────────────────────────────────────
    (
        "last trade for MSFT",
        ["/v2/last/trade/{stocksTicker}"],
    ),
    (
        "real-time quote for AMZN",
        [
            "/v3/quotes/{stockTicker}",
            "/v2/snapshot/locale/us/markets/stocks/tickers/{stocksTicker}",
        ],
    ),
    (
        "trade history for GOOG",
        ["/v3/trades/{stockTicker}"],
    ),
    # ── Reference / ticker lookup ──────────────────────────────────
    (
        "company info for META",
        ["/v3/reference/tickers/{ticker}", "/v3/reference/tickers"],
    ),
    (
        "search for ticker symbol",
        ["/v3/reference/tickers", "/v3/reference/tickers/{ticker}"],
    ),
    (
        "ticker details for AAPL",
        ["/v3/reference/tickers/{ticker}"],
    ),
    # ── News / market status ───────────────────────────────────────
    (
        "news about TSLA",
        ["/v2/reference/news", "/benzinga/v2/news"],
    ),
    (
        "is the market open",
        # Either the universal market-status endpoint or the
        # futures-specific equivalent is a correct answer.
        ["/v1/marketstatus/now", "/futures/vX/market-status"],
    ),
    (
        "market hours today",
        [
            "/v1/marketstatus/now",
            "/futures/vX/market-status",
            "/v1/marketstatus/upcoming",
        ],
    ),
    # ── Options ────────────────────────────────────────────────────
    (
        "options chain for AAPL",
        ["/v3/snapshot/options/{underlyingAsset}"],
    ),
    (
        "list option contracts for SPY",
        [
            "/v3/reference/options/contracts",
            "/v3/snapshot/options/{underlyingAsset}",
        ],
    ),
    (
        "specific option contract details",
        [
            "/v3/snapshot/options/{underlyingAsset}/{optionContract}",
            "/v3/reference/options/contracts/{options_ticker}",
        ],
    ),
    (
        "options price history",
        [
            "/v2/aggs/ticker/{optionsTicker}/range/",
            "/v2/aggs/ticker/{optionsTicker}/prev",
        ],
    ),
    # ── Technical indicators ───────────────────────────────────────
    (
        "RSI for AAPL",
        ["/v1/indicators/rsi/{stockTicker}"],
    ),
    (
        "moving average for SPY",
        [
            "/v1/indicators/sma/{stockTicker}",
            "/v1/indicators/ema/{stockTicker}",
            "/v1/indicators/macd/{stockTicker}",
        ],
    ),
    (
        "MACD for QQQ",
        ["/v1/indicators/macd/{stockTicker}"],
    ),
    (
        "exponential moving average for IBM",
        ["/v1/indicators/ema/{stockTicker}"],
    ),
    # ── Forex / Crypto / Indices / Futures ─────────────────────────
    (
        "FX rate EUR/USD history",
        [
            "/v2/aggs/ticker/{forexTicker}/range/",
            "/v2/aggs/ticker/{forexTicker}/prev",
            # Conversion endpoint also returns a current FX rate; if the
            # LLM falls back to it the user still gets a usable answer.
            "/v1/conversion/{from}/{to}",
        ],
    ),
    (
        "BTC daily prices",
        [
            "/v2/aggs/ticker/{cryptoTicker}/range/",
            "/v2/snapshot/locale/global/markets/crypto/tickers/{ticker}",
        ],
    ),
    # Cross-market gainers/losers/movers — the explicit market keyword
    # must beat the Stocks-default mapping for "gainers/losers/movers".
    (
        "crypto gainers",
        ["/v2/snapshot/locale/global/markets/crypto/{direction}"],
    ),
    (
        "forex gainers",
        ["/v2/snapshot/locale/global/markets/forex/{direction}"],
    ),
    (
        "VIX index history",
        ["/v2/aggs/ticker/{indicesTicker}/range/"],
    ),
    (
        "S&P 500 snapshot",
        ["/v3/snapshot/indices", "/v3/snapshot"],
    ),
    (
        "futures snapshot",
        ["/futures/vX/snapshot"],
    ),
    # ── Financials ─────────────────────────────────────────────────
    (
        "AAPL income statement",
        ["/stocks/financials/v1/income-statements"],
    ),
    (
        "AAPL balance sheet",
        ["/stocks/financials/v1/balance-sheets"],
    ),
    (
        "P/E ratio for AAPL",
        ["/stocks/financials/v1/ratios"],
    ),
    (
        "cash flow statement for MSFT",
        ["/stocks/financials/v1/cash-flow-statements"],
    ),
    # ── Corporate actions ──────────────────────────────────────────
    (
        "split history for AAPL",
        ["/stocks/v1/splits"],
    ),
    (
        "dividend history",
        ["/stocks/v1/dividends"],
    ),
    (
        "upcoming IPOs",
        ["/vX/reference/ipos"],
    ),
    # ── Benzinga / partners ────────────────────────────────────────
    (
        "earnings calendar",
        ["/benzinga/v1/earnings"],
    ),
    (
        "analyst ratings for NVDA",
        [
            "/benzinga/v1/ratings",
            "/benzinga/v1/analyst-insights",
            "/benzinga/v1/consensus-ratings/{ticker}",
        ],
    ),
    (
        "analyst price targets",
        [
            "/benzinga/v1/ratings",
            "/benzinga/v1/analyst-insights",
            "/benzinga/v1/consensus-ratings/{ticker}",
        ],
    ),
    (
        "related companies to AAPL",
        ["/v1/related-companies/{ticker}"],
    ),
    # ── Filings ────────────────────────────────────────────────────
    (
        "10-K filings for AAPL",
        ["/stocks/filings/10-K/vX/sections", "/stocks/filings/vX/index"],
    ),
    (
        "8-K filings",
        ["/stocks/filings/8-K/vX/text", "/stocks/filings/vX/index"],
    ),
    (
        "13F filings",
        ["/stocks/filings/vX/13-F"],
    ),
    (
        "insider trading form 4",
        ["/stocks/filings/vX/form-4", "/stocks/filings/vX/form-3"],
    ),
    # ── Economy ────────────────────────────────────────────────────
    (
        "10 year treasury yield",
        ["/fed/v1/treasury-yields"],
    ),
    (
        "inflation rate",
        ["/fed/v1/inflation", "/fed/v1/inflation-expectations"],
    ),
    (
        "unemployment rate",
        ["/fed/v1/labor-market"],
    ),
    # ── ETFs ───────────────────────────────────────────────────────
    (
        "ETF holdings",
        ["/etf-global/v1/constituents", "/etf-global/v1/profiles"],
    ),
    (
        "ETF fund flows",
        ["/etf-global/v1/fund-flows"],
    ),
    # ── Greeks (Black-Scholes finance functions) ───────────────────
    # These are local functions, not API endpoints — they appear in
    # search results when scope is "all" / "functions".  We check via
    # the function index rather than the endpoint index.
]


def _build_index() -> EndpointIndex:
    text = _FIXTURE.read_text()
    eps = parse_llms_full_txt(text)
    kept = []
    for ep in eps:
        if "Deprecated" in ep.title:
            continue
        ep.path_prefix = _path_prefix(ep.path)
        kept.append(ep)
    return EndpointIndex(kept)


@pytest.fixture(scope="module")
def index() -> EndpointIndex:
    return _build_index()


def _format_results(results) -> str:
    return "\n".join(
        f"  #{i + 1} [{ep.market}] {ep.title}  {ep.path}"
        for i, ep in enumerate(results)
    )


@pytest.mark.parametrize(
    "query,expected_paths", BENCHMARKS, ids=lambda v: v if isinstance(v, str) else None
)
def test_default_results_include_expected_endpoint(
    index: EndpointIndex, query: str, expected_paths: list[str]
):
    """At least one expected path must surface in the default top-k set.

    Default ``top_k=7`` matches what ``search_endpoints`` returns for
    ``scope='endpoints'`` — what the LLM sees on a vanilla invocation.
    """
    results = index.search(query, top_k=7)
    result_paths = [ep.path for ep in results]

    hit = any(
        any(rp == ep or rp.startswith(ep) for ep in expected_paths)
        for rp in result_paths
    )

    assert hit, (
        f"\nQuery: {query!r}\n"
        f"Expected one of: {expected_paths}\n"
        f"Got top {len(results)}:\n{_format_results(results)}"
    )


def test_top_result_is_relevant(index: EndpointIndex):
    """Spot-check that the very-top result for a few core queries is
    one of the expected endpoints.  This is a stricter test than
    inclusion-in-top-7 and guards against the #1 result being noise
    even when the right endpoint is in slot 6 or 7."""
    must_be_top: list[tuple[str, list[str]]] = [
        # The most common queries by both volume and distinct users —
        # the LLM is most likely to act on result #1, so for these the
        # #1 must be on-target.
        # Either Custom Bars (multi-day range) or Previous Day Bar
        # (single most-recent day) is a valid answer for a casual
        # "daily candles" query — both return aggregate OHLC data.
        (
            "daily candles for AAPL",
            [
                "/v2/aggs/ticker/{stocksTicker}/range/",
                "/v2/aggs/ticker/{stocksTicker}/prev",
            ],
        ),
        ("RSI for AAPL", ["/v1/indicators/rsi/{stockTicker}"]),
        ("MACD for QQQ", ["/v1/indicators/macd/{stockTicker}"]),
        ("AAPL income statement", ["/stocks/financials/v1/income-statements"]),
        ("AAPL balance sheet", ["/stocks/financials/v1/balance-sheets"]),
        ("options chain for AAPL", ["/v3/snapshot/options/{underlyingAsset}"]),
        ("earnings calendar", ["/benzinga/v1/earnings"]),
        ("analyst ratings for NVDA", ["/benzinga/v1/ratings"]),
        ("related companies to AAPL", ["/v1/related-companies/{ticker}"]),
        ("news about TSLA", ["/v2/reference/news", "/benzinga/v2/news"]),
        (
            "today's biggest gainers",
            ["/v2/snapshot/locale/us/markets/stocks/{direction}"],
        ),
        ("trade history for GOOG", ["/v3/trades/{stockTicker}"]),
        ("last trade for MSFT", ["/v2/last/trade/{stocksTicker}"]),
        ("split history for AAPL", ["/stocks/v1/splits"]),
        ("dividend history", ["/stocks/v1/dividends"]),
    ]
    failures = []
    for query, expected in must_be_top:
        results = index.search(query, top_k=7)
        if not results:
            failures.append(f"{query!r}: no results")
            continue
        top = results[0]
        if not any(top.path == p or top.path.startswith(p) for p in expected):
            failures.append(
                f"{query!r}: top is [{top.market}] {top.title} {top.path}; "
                f"expected one of {expected}"
            )
    assert not failures, "\n".join(failures)
