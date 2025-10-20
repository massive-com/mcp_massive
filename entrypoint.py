#!/usr/bin/env python
import os
from typing import Literal, Optional, Dict, Any
from mcp_polygon import server


def transport() -> Literal["stdio", "sse", "streamable-http"]:
    """
    Determine the transport type for the MCP server.
    Defaults to 'stdio' if not set in environment variables.
    """
    mcp_transport_str = os.environ.get("MCP_TRANSPORT", "stdio")

    # These are currently the only supported transports
    supported_transports: dict[str, Literal["stdio", "sse", "streamable-http"]] = {
        "stdio": "stdio",
        "sse": "sse",
        "streamable-http": "streamable-http",
    }

    return supported_transports.get(mcp_transport_str, "stdio")


def get_server_settings() -> Dict[str, Any]:
    """
    Get all server settings from environment variables.
    Returns a dictionary with all FastMCP server settings.
    """
    settings = {}
    
    # HTTP settings
    # Host/IP address to bind to
    settings["host"] = os.environ.get("MCP_HOST", "127.0.0.1")
    
    # Port to bind to
    port_str = os.environ.get("MCP_PORT", "8000")
    try:
        settings["port"] = int(port_str)
    except ValueError:
        print(f"Warning: Invalid MCP_PORT value '{port_str}', using default port 8000")
        settings["port"] = 8000
    
    # Server settings
    # Debug mode
    debug_str = os.environ.get("MCP_DEBUG", "false")
    settings["debug"] = debug_str.lower() in ["true", "1", "yes", "on"]
    
    # Log level
    log_level = os.environ.get("MCP_LOG_LEVEL", "INFO").upper()
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if log_level in valid_log_levels:
        settings["log_level"] = log_level
    else:
        print(f"Warning: Invalid MCP_LOG_LEVEL '{log_level}', using default 'INFO'")
        settings["log_level"] = "INFO"
    
    # HTTP path settings
    # Mount path (e.g., "/github", defaults to root path)
    settings["mount_path"] = os.environ.get("MCP_MOUNT_PATH", "/")
    
    # SSE endpoint path
    settings["sse_path"] = os.environ.get("MCP_SSE_PATH", "/sse")
    
    # Message endpoint path (for SSE)
    settings["message_path"] = os.environ.get("MCP_MESSAGE_PATH", "/messages/")
    
    # Streamable HTTP endpoint path
    settings["streamable_http_path"] = os.environ.get("MCP_STREAMABLE_HTTP_PATH", "/mcp")
    
    # Streamable HTTP specific settings
    # JSON response mode (returns JSON instead of JSONRPC)
    json_response_str = os.environ.get("MCP_JSON_RESPONSE", "false")
    settings["json_response"] = json_response_str.lower() in ["true", "1", "yes", "on"]
    
    # Stateless HTTP mode (new transport per request)
    stateless_http_str = os.environ.get("MCP_STATELESS_HTTP", "false")
    settings["stateless_http"] = stateless_http_str.lower() in ["true", "1", "yes", "on"]
    
    return settings


# Ensure the server process doesn't exit immediately when run as an MCP server
def start_server():
    polygon_api_key = os.environ.get("POLYGON_API_KEY", "")
    if not polygon_api_key:
        print("Warning: POLYGON_API_KEY environment variable not set.")
    else:
        print("Starting Polygon MCP server with API key configured.")

    transport_type = transport()
    settings = get_server_settings()
    
    # Log configuration for HTTP-based transports
    if transport_type in ["sse", "streamable-http"]:
        print(f"Server configuration for {transport_type} transport:")
        print(f"  Host: {settings['host']}")
        print(f"  Port: {settings['port']}")
        print(f"  Debug: {settings['debug']}")
        print(f"  Log Level: {settings['log_level']}")
        
        if transport_type == "sse":
            print(f"  Mount Path: {settings['mount_path']}")
            print(f"  SSE Path: {settings['sse_path']}")
            print(f"  Message Path: {settings['message_path']}")
        elif transport_type == "streamable-http":
            print(f"  Streamable HTTP Path: {settings['streamable_http_path']}")
            print(f"  JSON Response Mode: {settings['json_response']}")
            print(f"  Stateless HTTP Mode: {settings['stateless_http']}")
    
    server.run(transport=transport_type, **settings)


if __name__ == "__main__":
    start_server()
