FROM python:3.13-slim

WORKDIR /app

# Install uv for dependency management
RUN pip install uv

COPY . ./

RUN uv pip install --system -e .
RUN chmod +x entrypoint.py

ENV PYTHONPATH=/app/src:$PYTHONPATH

# Polygon MCP Server Configuration
# These environment variables can be overridden at runtime

# Required - Must be provided at runtime
# ENV POLYGON_API_KEY=your_api_key_here

# Transport Configuration
# Options: stdio (default), sse, streamable-http
ENV MCP_TRANSPORT=stdio

# Server Settings
# Enable debug mode (true/false)
ENV MCP_DEBUG=false
# Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
ENV MCP_LOG_LEVEL=INFO

# HTTP Settings (used for SSE and Streamable-HTTP transports)
# Host/IP address to bind to (use 0.0.0.0 for all interfaces)
ENV MCP_HOST=127.0.0.1
# Port to bind to
ENV MCP_PORT=8000

# SSE-Specific Settings (only used when MCP_TRANSPORT=sse)
# Mount path for the application (e.g., "/api", "/github")
ENV MCP_MOUNT_PATH=/
# SSE endpoint path
ENV MCP_SSE_PATH=/sse
# Message endpoint path
ENV MCP_MESSAGE_PATH=/messages/

# Streamable-HTTP-Specific Settings (only used when MCP_TRANSPORT=streamable-http)
# Streamable HTTP endpoint path
ENV MCP_STREAMABLE_HTTP_PATH=/mcp

ENTRYPOINT ["uv", "run", "./entrypoint.py"]
