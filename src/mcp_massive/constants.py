import re

_DEFAULT_LLMS_FULL_TXT_URL = "https://massive.com/docs/rest/llms-full.txt"

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Aliases map abbreviations and synonyms to canonical terms that appear
# in the llms.txt endpoint names/descriptions.
# Values can be a single string or a list of strings for ambiguous terms.
ALIASES: dict[str, str | list[str]] = {
    # aggregates / bars / OHLC.  Custom Bars titles don't literally
    # contain "aggregate", so we cross-expand "aggregate(s)" to "bars"
    # and "ohlc" so queries for "aggregates" still surface them.
    "agg": ["aggregate", "bars", "ohlc"],
    "aggs": ["aggregate", "bars", "ohlc"],
    "aggregate": ["bars", "ohlc"],
    "aggregates": ["bars", "ohlc"],
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
    # "info"/"detail(s)" expand to both "details" (matches Firm Details
    # / Analyst Details titles) and "overview" (matches Ticker Overview
    # / Contract Overview titles).  Whichever scores higher under
    # title-weighted BM25 wins.
    "info": ["details", "overview"],
    "detail": ["details", "overview"],
    "details": "overview",
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
    "high": "aggregate",
    "low": "aggregate",
    "volume": "aggregate",
    "prev": "previous",
    "yesterday": "previous",
    "today": "snapshot",
    # "open" is polysemous — "open price" (aggregate) and "is the
    # market open" (status) are both common.  We map to "status"
    # because "Market Status" needs the help (only one endpoint) while
    # "open price" already gets aggregate from the "price" alias.
    "open": "status",
    # company / lookup
    "company": ["ticker", "details"],
    "companies": ["ticker", "details"],
    # gainers / movers natural language
    "biggest": "gainers",
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
    "vanna": ["bs_vanna", "options"],
    "volga": ["bs_volga", "options"],
    "vomma": ["bs_volga", "options"],
    "charm": ["bs_charm", "options"],
    "veta": ["bs_veta", "options"],
    "color": ["bs_color", "options"],
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
    "edgar": "filings",
    "filing": "filings",
    # ratios (financial)
    "roe": "ratios",
    "roa": "ratios",
    "roic": "ratios",
    # alternative / consumer spending
    "mcc": "merchant",
    "merchants": "merchant",
    # etf
    "etf": "etf",
    # float
    "float": "float",
    # labor / employment
    "unemployment": "labor",
    "jobs": "labor",
    "nonfarm": "labor",
    "payroll": "labor",
    "payrolls": "labor",
    "jobless": "labor",
    "claims": "labor",
    "employment": "labor",
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
    "breakeven": "inflation",
    # "rate" / "rates" can mean interest rate (Treasury) or exchange
    # rate (FX bars).  Expand to both so each market's strongest title
    # match decides — Treasury Yields still wins for "interest rate",
    # Forex Custom Bars surfaces for "FX rate history".
    "rate": ["treasury", "aggregate"],
    "rates": ["treasury", "aggregate"],
    # market status
    "holiday": "holidays",
    "hours": "status",
    "schedule": "status",
    "closed": "status",
    "exchange": "exchanges",
}

# Market keywords used to infer the market filter from a free-text query.
# Only include tokens whose sole/primary meaning is the asset class —
# polysemous words (e.g. "index") pull the filter onto wrong endpoints.
_MARKET_KEYWORDS: dict[str, set[str]] = {
    "Stocks": {
        "stock",
        "stocks",
        "equity",
        "equities",
        "share",
        "shares",
        # Gainers/losers/movers terminology is predominantly used for
        # stocks in casual queries ("biggest gainers", "today's movers").
        # The same endpoint exists for Crypto and Forex but users
        # explicitly add "crypto"/"forex" when they mean those.
        "gainers",
        "losers",
        "movers",
    },
    "Crypto": {"crypto", "cryptocurrency", "bitcoin", "btc", "eth", "coin"},
    "Forex": {"forex", "fx", "currency", "currencies"},
    "Options": {"option", "options", "call", "put", "strike", "chain"},
    "Futures": {
        "future",
        "futures",
        "futs",
        "commodity",
        "commodities",
        "crude",
        "oil",
        "gold",
        "silver",
        "gas",
        "wheat",
        "corn",
        "cme",
        "nymex",
        "cbot",
    },
    # NOTE: "index" deliberately omitted — collides with "consumer price
    # index", "SEC EDGAR index", "filings index", etc. where the user
    # means a table/document.  Users who mean the asset class typically
    # say "indices", "benchmark", or a specific index symbol.
    "Indices": {
        "indices",
        "benchmark",
        "spx",
        "ndx",
        "rut",
        "vix",
        "djia",
        "dow",
        "nikkei",
        "ftse",
    },
    "Economy": {"economy", "economic", "treasury", "inflation", "yield", "bond"},
}

_BULLET_PARAM_RE = re.compile(r"^-\s+\*{0,2}(\w+)\*{0,2}\s*\(([^)]+)\)(?::\s*(.*))?$")

_STRUCTURAL_SECTIONS = {"Query Parameters", "Response Attributes", "Sample Response"}

# Uppercase 2-5 letter token pattern used to detect ticker symbols in
# the original-case query.  Used as a fallback Stocks-market signal when
# no explicit market keyword is found (e.g. "RSI for AAPL").
_UPPER_TICKER_RE = re.compile(r"\b([A-Z]{2,5})\b")

# Acronyms / non-stock-ticker uppercase tokens that must NOT trigger the
# Stocks-market inference fallback.  These are tokens that look like
# tickers but route to other markets (currencies → Forex, BTC → Crypto,
# VIX → Indices) or are technical-analysis / general acronyms.
_TICKER_FALLBACK_EXCLUSIONS: set[str] = {
    # technical indicators
    "RSI",
    "SMA",
    "EMA",
    "MACD",
    "OHLC",
    "OHLCV",
    "VWAP",
    "ATR",
    "ADX",
    "BB",
    # currencies — handled by Forex keywords
    "USD",
    "EUR",
    "GBP",
    "JPY",
    "AUD",
    "CAD",
    "CHF",
    "CNY",
    "HKD",
    "INR",
    "MXN",
    "NZD",
    "SEK",
    "NOK",
    "ZAR",
    "BRL",
    "RUB",
    # crypto — handled by Crypto keywords
    "BTC",
    "ETH",
    "BNB",
    "XRP",
    "SOL",
    "ADA",
    "DOGE",
    "USDT",
    "USDC",
    # indices — handled by Indices keywords
    "VIX",
    "SPX",
    "NDX",
    "DJI",
    "RUT",
    "DJIA",
    # general acronyms / asset-class names
    "FX",
    "ETF",
    "ETFS",
    "IPO",
    "API",
    "URL",
    "JSON",
    "HTTP",
    "ID",
    "AI",
    # geo / regulatory / business
    "USA",
    "UK",
    "EU",
    "GMT",
    "AM",
    "PM",
    "OK",
    "OTC",
    "NYSE",
    "NASDAQ",
    "AMEX",
    "CME",
    "NYMEX",
    "CBOT",
    "CEO",
    "CFO",
    "CTO",
    "COO",
    "FED",
    "SEC",
    "FDA",
    "FBI",
    "GDP",
    "CPI",
    "IRS",
    "P",
    "S",
}
