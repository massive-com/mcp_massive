import re

_DEFAULT_LLMS_FULL_TXT_URL = "https://massive.com/docs/rest/llms-full.txt"

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Aliases map abbreviations and synonyms to canonical terms that appear
# in the llms.txt endpoint names/descriptions.
# Values can be a single string or a list of strings for ambiguous terms.
ALIASES: dict[str, str | list[str]] = {
    # aggregates / bars / OHLC
    "agg": ["aggregate", "bars", "ohlc"],
    "aggs": ["aggregate", "bars", "ohlc"],
    "candle": ["aggregate", "bars", "ohlc"],
    "candles": ["aggregate", "bars", "ohlc"],
    "candlestick": ["aggregate", "bars", "ohlc"],
    "candlesticks": ["aggregate", "bars", "ohlc"],
    "ohlc": ["aggregate", "bars"],
    "ohlcv": ["aggregate", "bars", "ohlc"],
    "bar": ["aggregate", "bars", "ohlc"],
    "bars": ["aggregate", "ohlc"],
    "vwap": "aggregate",
    # forex / currency
    "fx": "forex",
    "currency": "forex",
    "currencies": "forex",
    # crypto
    "coin": "crypto",
    "coins": "crypto",
    "token": "crypto",
    "tokens": "crypto",
    "bitcoin": "crypto",
    "btc": "crypto",
    "eth": "crypto",
    "ethereum": "crypto",
    # reference data
    "ref": "reference",
    "info": "details",
    "detail": "details",
    "lookup": "tickers",
    "symbol": "ticker",
    "symbols": "ticker",
    # trades / quotes
    "transaction": "trade",
    "transactions": "trade",
    "execution": "trade",
    "executions": "trade",
    "bid": "quote",
    "ask": "quote",
    "nbbo": "quote",
    # snapshots
    "snap": "snapshot",
    "snaps": "snapshot",
    "realtime": "snapshot",
    "real-time": "snapshot",
    "live": "snapshot",
    # financials
    "fundamental": "financial",
    "fundamentals": "financial",
    "income": "financial",
    "balance": "financial",
    "earnings": "financial",
    "revenue": "financial",
    "pe": "financial",
    "eps": "financial",
    "cashflow": "financial",
    # market data concepts
    "price": ["trade", "aggregate", "snapshot"],
    "prices": ["trade", "aggregate", "snapshot"],
    "close": "aggregate",
    "open": "aggregate",
    "high": "aggregate",
    "low": "aggregate",
    "volume": "aggregate",
    "prev": "previous",
    # options
    "option": "options",
    "contract": "options",
    "contracts": "options",
    "chain": "options",
    "greek": "greeks",
    "greeks": "options",
    # greeks / black-scholes
    "delta": ["bs_delta", "options"],
    "gamma": ["bs_gamma", "options"],
    "theta": ["bs_theta", "options"],
    "vega": ["bs_vega", "options"],
    "rho": ["bs_rho", "options"],
    "blackscholes": ["bs_price", "bs_delta"],
    # technical indicators
    "sma": "aggregate",
    "moving": "aggregate",
    "rsi": "aggregate",
    "technical": "aggregate",
    "indicator": "aggregate",
    # corporate actions
    "split": "splits",
    "dividend": "dividends",
    "div": "dividends",
    "divs": "dividends",
    "ipo": "ipos",
    # filings
    "10k": "filings",
    "sec": "filings",
    "filing": "filings",
    # etf
    "etf": "etf",
    # float
    "float": "float",
    # labor / employment
    "unemployment": "labor",
    "jobs": "labor",
    # conversion
    "convert": "conversion",
    # indices
    "index": "indices",
    # gainers / losers
    "gainer": "gainers",
    "loser": "losers",
    "mover": "gainers",
    "movers": "gainers",
    # news / sentiment
    "headline": "news",
    "headlines": "news",
    "article": "news",
    "articles": "news",
    "sentiment": "news",
    # analyst
    "analyst": "analysts",
    "rating": "ratings",
    "upgrade": "ratings",
    "downgrade": "ratings",
    "target": "ratings",
    # futures
    "future": "futures",
    "futs": "futures",
    # short interest
    "short": "short",
    "si": "short",
    # economy
    "yield": "treasury",
    "yields": "treasury",
    "bond": "treasury",
    "bonds": "treasury",
    "cpi": "inflation",
    "rate": "treasury",
    "rates": "treasury",
    # market status
    "holiday": "holidays",
    "hours": "status",
    "schedule": "status",
    "closed": "status",
    "exchange": "exchanges",
}

# Market keywords used to boost endpoints whose market matches the query.
_MARKET_KEYWORDS: dict[str, set[str]] = {
    "Stocks": {"stock", "stocks", "equity", "equities", "share", "shares"},
    "Crypto": {"crypto", "cryptocurrency", "bitcoin", "btc", "eth", "coin"},
    "Forex": {"forex", "fx", "currency", "currencies"},
    "Options": {"option", "options", "call", "put", "strike", "chain"},
    "Futures": {"future", "futures", "futs"},
    "Indices": {"index", "indices", "benchmark"},
    "Economy": {"economy", "economic", "treasury", "inflation", "yield", "bond"},
}

_BULLET_PARAM_RE = re.compile(r"^-\s+\*{0,2}(\w+)\*{0,2}\s*\(([^)]+)\)(?::\s*(.*))?$")

_STRUCTURAL_SECTIONS = {"Query Parameters", "Response Attributes", "Sample Response"}
