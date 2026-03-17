import argparse
import os
import sys
from typing import Literal

__all__ = ["main"]


def main() -> None:
    """
    Main CLI entry point for the MCP server.
    Accepts --transport CLI argument (falls back to MCP_TRANSPORT env var, then stdio).

    Heavy dependencies (numpy, bm25s, etc.) are imported lazily
    inside this function so that ``uv run`` can finish installing packages
    and Python can start before the 30-second MCP connection timeout fires.
    """
    from dotenv import load_dotenv

    # Load environment variables from .env file if it exists
    load_dotenv()

    parser = argparse.ArgumentParser(description="Massive MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default=None,
        help="Transport protocol (default: stdio). Overrides MCP_TRANSPORT env var.",
    )
    args = parser.parse_args()

    # CLI arg takes precedence over env var; default to stdio
    if args.transport is not None:
        transport: Literal["stdio", "sse", "streamable-http"] = args.transport
    else:
        supported_transports: dict[str, Literal["stdio", "sse", "streamable-http"]] = {
            "stdio": "stdio",
            "sse": "sse",
            "streamable-http": "streamable-http",
        }
        mcp_transport_str = os.environ.get("MCP_TRANSPORT", "stdio")
        transport = supported_transports.get(mcp_transport_str, "stdio")

    # Check API key and print startup message
    massive_api_key = os.environ.get("MASSIVE_API_KEY", "")
    polygon_api_key = os.environ.get("POLYGON_API_KEY", "")

    # Startup messages go to stderr — stdout is the MCP protocol channel
    # for stdio transport; non-JSON data there corrupts the handshake.
    if massive_api_key:
        print("Starting Massive MCP server with API key configured.", file=sys.stderr)
    elif polygon_api_key:
        print(
            "Warning: POLYGON_API_KEY is deprecated. Please migrate to MASSIVE_API_KEY.",
            file=sys.stderr,
        )
        print(
            "Starting Massive MCP server with API key configured (using deprecated POLYGON_API_KEY).",
            file=sys.stderr,
        )
        massive_api_key = polygon_api_key
    else:
        print("Warning: MASSIVE_API_KEY environment variable not set.", file=sys.stderr)

    base_url = os.environ.get("MASSIVE_API_BASE_URL", "https://api.massive.com").rstrip(
        "/"
    )
    llms_txt_url = os.environ.get("MASSIVE_LLMS_TXT_URL")

    max_tables: int | None = None
    max_rows: int | None = None
    if os.environ.get("MASSIVE_MAX_TABLES"):
        max_tables = int(os.environ["MASSIVE_MAX_TABLES"])
    if os.environ.get("MASSIVE_MAX_ROWS"):
        max_rows = int(os.environ["MASSIVE_MAX_ROWS"])

    # Defer importing server until after env vars are read — this triggers
    # loading numpy, bm25s, and other heavy deps.
    from .server import run, configure_credentials

    configure_credentials(
        massive_api_key,
        base_url,
        llms_txt_url=llms_txt_url,
        max_tables=max_tables,
        max_rows=max_rows,
    )

    # SECURITY: Remove secrets from environment variables so they cannot be
    # exfiltrated via user-supplied code or SQL.  All values the server needs
    # (API key, base URL, etc.) have already been captured into module-level
    # variables above via configure_credentials().
    #
    # We use a keep-list instead of os.environ.clear() because hosted
    # runtimes (FastMCP Cloud, Prefect Horizon, ASGI servers) rely on env
    # vars like PATH, HOME, SSL_CERT_FILE, PYTHONPATH, LANG, etc. for basic
    # operation.  Clearing them breaks SSL connections, subprocess spawning,
    # and the HTTP transport itself.
    _SECRETS_TO_REMOVE = {
        "MASSIVE_API_KEY",
        "POLYGON_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "DATABASE_URL",
        "SECRET_KEY",
    }
    for key in _SECRETS_TO_REMOVE:
        os.environ.pop(key, None)

    run(transport=transport)
