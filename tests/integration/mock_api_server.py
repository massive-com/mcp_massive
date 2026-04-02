"""Mock Massive.com API + docs server using Starlette + uvicorn.

Serves canned financial data at real API path patterns, plus llms.txt
and doc pages so that build_index() works without the real Massive.com.
"""

import threading

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

from . import responses as R
from .mock_llms_txt import llms_partial_txt


def _aggs_handler(request: Request) -> JSONResponse:
    ticker = request.path_params["ticker"]
    if ticker == "ERROR500":
        return JSONResponse({"error": "Internal server error"}, status_code=500)
    if ticker == "PAGINATED":
        base = str(request.base_url).rstrip("/")
        resp = dict(R.STOCK_AGGS_RESPONSE)
        resp["next_url"] = (
            f"{base}/v2/aggs/ticker/PAGINATED/range/1/day/2024-01-01/2024-01-31"
            f"?cursor=page2_token&adjusted=true&apiKey=LEAKED_KEY"
        )
        return JSONResponse(resp)
    if ticker.startswith("X:"):
        return JSONResponse(R.CRYPTO_AGGS_RESPONSE)
    if ticker.startswith("C:"):
        return JSONResponse(R.FOREX_AGGS_RESPONSE)
    return JSONResponse(R.STOCK_AGGS_RESPONSE)


def _trades_handler(request: Request) -> JSONResponse:
    return JSONResponse(R.TRADES_RESPONSE)


def _quotes_handler(request: Request) -> JSONResponse:
    return JSONResponse(R.QUOTES_RESPONSE)


def _last_trade_handler(request: Request) -> JSONResponse:
    return JSONResponse(R.LAST_TRADE_RESPONSE)


def _snapshot_ticker_handler(request: Request) -> JSONResponse:
    return JSONResponse(R.SNAPSHOT_RESPONSE)


def _options_chain_handler(request: Request) -> JSONResponse:
    return JSONResponse(R.OPTIONS_CHAIN_RESPONSE)


def _tickers_handler(request: Request) -> JSONResponse:
    return JSONResponse(R.TICKERS_RESPONSE)


def _llms_full_txt_handler(request: Request) -> PlainTextResponse:
    return PlainTextResponse(llms_partial_txt())


def create_app() -> Starlette:
    routes = [
        # API routes
        Route(
            "/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}",
            _aggs_handler,
        ),
        Route("/v3/trades/{ticker}", _trades_handler),
        Route("/v3/quotes/{ticker}", _quotes_handler),
        Route("/v2/last/trade/{ticker}", _last_trade_handler),
        Route(
            "/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}",
            _snapshot_ticker_handler,
        ),
        Route("/v3/snapshot/options/{ticker}", _options_chain_handler),
        Route("/v3/reference/tickers", _tickers_handler),
        # llms-full.txt
        Route("/docs/rest/llms-full.txt", _llms_full_txt_handler),
    ]
    return Starlette(routes=routes)


class MockServer:
    """Manages a uvicorn server in a daemon thread."""

    def __init__(self) -> None:
        self._server: uvicorn.Server
        self._thread: threading.Thread
        self.port: int = 0

    def start(self) -> str:
        """Start the mock server and return its base URL."""
        import socket

        # Find a free port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]

        app = create_app()
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=self.port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)

        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

        # Wait for server to be ready
        import time

        for _ in range(50):
            if self._server.started:
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("Mock server did not start in time")

        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        if hasattr(self, "_server"):
            self._server.should_exit = True
        if hasattr(self, "_thread"):
            self._thread.join(timeout=5)
