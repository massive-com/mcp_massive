"""
FastMCP web deployment entrypoint.

Exposes the ``mcp`` variable expected by FastMCP cloud / streamable-http
hosting (entrypoint ``app.py:mcp``).

Compatible with Prefect Horizon, FastMCP Cloud, and any ASGI-based
MCP hosting platform.

Environment variables (MASSIVE_API_KEY, etc.) are read at import time so the
hosted runtime can inject them via its secrets / env configuration.
"""

import logging
import os

from dotenv import load_dotenv

load_dotenv()

from mcp_massive.server import (  # noqa: E402
    mass_mcp,
    configure_credentials,
    shutdown_http_client,
)

logger = logging.getLogger(__name__)

# Read configuration from environment
_massive_api_key = os.environ.get("MASSIVE_API_KEY", "")
_polygon_api_key = os.environ.get("POLYGON_API_KEY", "")

if not _massive_api_key and _polygon_api_key:
    _massive_api_key = _polygon_api_key

_base_url = os.environ.get("MASSIVE_API_BASE_URL", "https://api.massive.com").rstrip("/")
_llms_txt_url = os.environ.get("MASSIVE_LLMS_TXT_URL")
_max_tables = int(os.environ["MASSIVE_MAX_TABLES"]) if os.environ.get("MASSIVE_MAX_TABLES") else None
_max_rows = int(os.environ["MASSIVE_MAX_ROWS"]) if os.environ.get("MASSIVE_MAX_ROWS") else None

configure_credentials(
    _massive_api_key,
    _base_url,
    llms_txt_url=_llms_txt_url,
    max_tables=_max_tables,
    max_rows=_max_rows,
)

# SECURITY: Remove secret env vars from the process after capturing them,
# without nuking system vars needed by the hosted runtime.
for _secret_key in ("MASSIVE_API_KEY", "POLYGON_API_KEY"):
    os.environ.pop(_secret_key, None)

logger.info("Massive MCP server configured for web deployment (streamable-http).")

# Expose as ``mcp`` — the name FastMCP cloud expects (entrypoint: app.py:mcp)
mcp = mass_mcp
