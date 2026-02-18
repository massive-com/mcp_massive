"""Minimal llms.txt content and doc pages for the mock server.

The mock server serves these at the same URL paths that build_index() expects.
"""


def llms_txt(base_url: str) -> str:
    """Return a minimal llms.txt pointing doc URLs at the mock server."""
    return f"""\
# Massive.com REST API

## Stocks
- [Aggregates (Bars)]({base_url}/docs/rest/stocks/aggregates-bars): Get aggregate bars (OHLCV) for a stock ticker over a given date range.
- [Trades]({base_url}/docs/rest/stocks/trades): Get trades for a stock ticker.
- [Quotes (NBBO)]({base_url}/docs/rest/stocks/quotes): Get NBBO quotes for a stock ticker.
- [Last Trade]({base_url}/docs/rest/stocks/last-trade): Get the most recent trade for a ticker.
- [Snapshot - Ticker]({base_url}/docs/rest/stocks/snapshot-ticker): Get the snapshot of a single stock ticker.

## Options
- [Options Chain]({base_url}/docs/rest/options/options-chain): Get the snapshot of all options contracts for an underlying ticker.

## Crypto
- [Aggregates (Bars)]({base_url}/docs/rest/crypto/aggregates-bars): Get aggregate bars for a cryptocurrency pair.

## Forex
- [Aggregates (Bars)]({base_url}/docs/rest/forex/aggregates-bars): Get aggregate bars for a forex currency pair.

## Reference Data
- [Tickers]({base_url}/docs/rest/reference/tickers): Query all ticker symbols supported by Massive.com.
"""


DOC_PAGES = {
    "/docs/rest/stocks/aggregates-bars": """\
# Aggregates (Bars)

Get aggregate bars for a stock over a given date range in custom timespan.

**Endpoint:** `GET /v2/aggs/ticker/{stocksTicker}/range/{multiplier}/{timespan}/{from}/{to}`

### Query Parameters
- **adjusted** (boolean): Whether results are adjusted for splits. Default true.
- **sort** (string): Sort order. "asc" or "desc". Default "asc".
- **limit** (integer): Limits the number of results. Default 5000. Max 50000.
""",
    "/docs/rest/stocks/trades": """\
# Trades

Get trades for a ticker symbol in a given time range.

**Endpoint:** `GET /v3/trades/{stocksTicker}`

### Query Parameters
- **timestamp** (string): Query by trade timestamp (ns).
- **order** (string): Order results. "asc" or "desc".
- **limit** (integer): Limit results. Default 10. Max 50000.
- **sort** (string): Sort field. Default "timestamp".
""",
    "/docs/rest/stocks/quotes": """\
# Quotes (NBBO)

Get NBBO quotes for a ticker symbol in a given time range.

**Endpoint:** `GET /v3/quotes/{stocksTicker}`

### Query Parameters
- **timestamp** (string): Query by quote timestamp.
- **order** (string): Order results.
- **limit** (integer): Limit results.
- **sort** (string): Sort field.
""",
    "/docs/rest/stocks/last-trade": """\
# Last Trade

Get the most recent trade for a ticker.

**Endpoint:** `GET /v2/last/trade/{stocksTicker}`

### Query Parameters
No query parameters.
""",
    "/docs/rest/stocks/snapshot-ticker": """\
# Snapshot - Ticker

Get the current minute, day, and previous day's aggregate, as well as the last trade and quote for a single traded stock ticker.

**Endpoint:** `GET /v2/snapshot/locale/us/markets/stocks/tickers/{stocksTicker}`

### Query Parameters
No query parameters.
""",
    "/docs/rest/options/options-chain": """\
# Options Chain

Get the snapshot of all options contracts for an underlying ticker.

**Endpoint:** `GET /v3/snapshot/options/{underlyingAsset}`

### Query Parameters
- **strike_price** (number): Query by strike price.
- **expiration_date** (string): Query by expiration date.
- **contract_type** (string): Query by contract type ("call" or "put").
- **order** (string): Order results.
- **limit** (integer): Limit results. Default 10. Max 250.
- **sort** (string): Sort field.
""",
    "/docs/rest/crypto/aggregates-bars": """\
# Aggregates (Bars)

Get aggregate bars for a cryptocurrency pair.

**Endpoint:** `GET /v2/aggs/ticker/{cryptoTicker}/range/{multiplier}/{timespan}/{from}/{to}`

### Query Parameters
- **adjusted** (boolean): Whether results are adjusted. Default true.
- **sort** (string): Sort order.
- **limit** (integer): Limit results.
""",
    "/docs/rest/forex/aggregates-bars": """\
# Aggregates (Bars)

Get aggregate bars for a forex currency pair.

**Endpoint:** `GET /v2/aggs/ticker/{forexTicker}/range/{multiplier}/{timespan}/{from}/{to}`

### Query Parameters
- **adjusted** (boolean): Whether results are adjusted. Default true.
- **sort** (string): Sort order.
- **limit** (integer): Limit results.
""",
    "/docs/rest/reference/tickers": """\
# Tickers

Query all ticker symbols which are supported by Massive.com.

**Endpoint:** `GET /v3/reference/tickers`

### Query Parameters
- **ticker** (string): Specify a ticker symbol.
- **type** (string): Specify the type of the tickers.
- **market** (string): Filter by market type.
- **exchange** (string): Filter by primary exchange.
- **active** (boolean): Filter by active status.
- **order** (string): Order results.
- **limit** (integer): Limit results. Default 100. Max 1000.
- **sort** (string): Sort field.
- **search** (string): Search for terms within the ticker and/or company name.
""",
}
