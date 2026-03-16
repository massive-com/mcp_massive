"""
Entrypoint for fastmcp.app deployment.

The platform imports the FastMCP object directly (no CLI, no main()).
This module reads credentials from environment variables set in the
platform dashboard and configures the server before exporting `mcp`.
"""

import os

from mcp_massive.server import mass_mcp, configure_credentials

# Read credentials from env vars provided by the fastmcp.app platform.
configure_credentials(
    api_key=os.environ.get("MASSIVE_API_KEY", ""),
    base_url=os.environ.get(
        "MASSIVE_API_BASE_URL", "https://api.massive.com"
    ).rstrip("/"),
    llms_txt_url=os.environ.get("MASSIVE_LLMS_TXT_URL"),
    max_tables=int(os.environ["MASSIVE_MAX_TABLES"])
    if os.environ.get("MASSIVE_MAX_TABLES")
    else None,
    max_rows=int(os.environ["MASSIVE_MAX_ROWS"])
    if os.environ.get("MASSIVE_MAX_ROWS")
    else None,
)

# The fastmcp.app platform looks for the object named in the entrypoint field.
# With entrypoint "app.py:mcp", it imports this variable.
mcp = mass_mcp
