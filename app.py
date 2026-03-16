"""
FastMCP web deployment entrypoint.

Exposes the ``mcp`` variable expected by FastMCP cloud / streamable-http
hosting (entrypoint ``app.py:mcp``).

Environment variables (MASSIVE_API_KEY, etc.) are read at import time so the
hosted runtime can inject them via its secrets / env configuration.
"""

import os

from dotenv import load_dotenv

load_dotenv()

from mcp_massive.server import mass_mcp, configure_credentials  # noqa: E402

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

# Expose as ``mcp`` — the name FastMCP cloud expects (entrypoint: app.py:mcp)
mcp = mass_mcp
