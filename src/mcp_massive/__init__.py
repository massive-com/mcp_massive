import os
from typing import Literal
from dotenv import load_dotenv
from .server import run

# Load environment variables from .env file if it exists
load_dotenv()

__all__ = ["run", "main"]


def main() -> None:
    """
    Main CLI entry point for the MCP server.
    Reads MCP_TRANSPORT environment variable and starts the server.
    """
    # Determine transport from environment variable
    mcp_transport_str = os.environ.get("MCP_TRANSPORT", "stdio")

    # These are currently the only supported transports
    supported_transports: dict[str, Literal["stdio", "sse", "streamable-http"]] = {
        "stdio": "stdio",
        "sse": "sse",
        "streamable-http": "streamable-http",
    }

    transport = supported_transports.get(mcp_transport_str, "stdio")

    # Check API key and print startup message
    massive_api_key = os.environ.get("MASSIVE_API_KEY", "")
    polygon_api_key = os.environ.get("POLYGON_API_KEY", "")

    if massive_api_key:
        print("Starting Massive MCP server with API key configured.")
    elif polygon_api_key:
        print(
            "Warning: POLYGON_API_KEY is deprecated. Please migrate to MASSIVE_API_KEY."
        )
        print(
            "Starting Massive MCP server with API key configured (using deprecated POLYGON_API_KEY)."
        )
        # Set MASSIVE_API_KEY from POLYGON_API_KEY for backward compatibility
        os.environ["MASSIVE_API_KEY"] = polygon_api_key
    else:
        print("Warning: MASSIVE_API_KEY environment variable not set.")

    run(transport=transport)
