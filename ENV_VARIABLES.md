# Environment Variables

The Polygon MCP server can be configured using the following environment variables:

## Required

- `POLYGON_API_KEY`: Your Polygon.io API key (required for API access)

## Transport Configuration

- `MCP_TRANSPORT`: Transport protocol to use
  - `stdio` (default) - Standard input/output
  - `sse` - Server-Sent Events
  - `streamable-http` - Streamable HTTP

## Server Settings

- `MCP_DEBUG`: Enable debug mode
  - Values: `true`, `1`, `yes`, `on` (enables debug)
  - Default: `false`

- `MCP_LOG_LEVEL`: Logging level
  - Values: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
  - Default: `INFO`

## HTTP Settings (for SSE and Streamable-HTTP transports)

- `MCP_HOST`: Host/IP address to bind to
  - Default: `127.0.0.1`
  - Example: `0.0.0.0` (bind to all interfaces)

- `MCP_PORT`: Port to bind to
  - Default: `8000`
  - Example: `9000`

## SSE-Specific Settings

These settings only apply when using the SSE transport:

- `MCP_MOUNT_PATH`: Mount path for the application
  - Default: `/`
  - Example: `/github`

- `MCP_SSE_PATH`: SSE endpoint path
  - Default: `/sse`

- `MCP_MESSAGE_PATH`: Message endpoint path
  - Default: `/messages/`

## Streamable-HTTP-Specific Settings

These settings only apply when using the Streamable-HTTP transport:

- `MCP_STREAMABLE_HTTP_PATH`: Streamable HTTP endpoint path
  - Default: `/mcp`

- `MCP_JSON_RESPONSE`: Return plain JSON instead of JSONRPC format
  - Values: `true`, `1`, `yes`, `on` (enables JSON response mode)
  - Default: `false`
  - When enabled, returns raw JSON responses instead of JSONRPC wrapped responses

- `MCP_STATELESS_HTTP`: Use stateless mode (new transport per request)
  - Values: `true`, `1`, `yes`, `on` (enables stateless mode)
  - Default: `false`
  - When enabled, creates a new transport connection for each request instead of maintaining state

## Example Usage

### Running with stdio transport (default)
```bash
export POLYGON_API_KEY=your_api_key_here
uv run entrypoint.py
```

### Running with SSE transport
```bash
export POLYGON_API_KEY=your_api_key_here
export MCP_TRANSPORT=sse
export MCP_HOST=0.0.0.0
export MCP_PORT=9000
export MCP_LOG_LEVEL=DEBUG
uv run entrypoint.py
```

### Running with Streamable-HTTP transport
```bash
export POLYGON_API_KEY=your_api_key_here
export MCP_TRANSPORT=streamable-http
export MCP_HOST=0.0.0.0
export MCP_PORT=3000
export MCP_STREAMABLE_HTTP_PATH=/api/mcp
uv run entrypoint.py
```

### Running with Streamable-HTTP in JSON response mode
```bash
export POLYGON_API_KEY=your_api_key_here
export MCP_TRANSPORT=streamable-http
export MCP_HOST=0.0.0.0
export MCP_PORT=3000
export MCP_JSON_RESPONSE=true
export MCP_STATELESS_HTTP=true
uv run entrypoint.py
```

### Docker Examples

#### Building with custom defaults in Dockerfile
```dockerfile
ENV POLYGON_API_KEY=your_api_key_here
ENV MCP_TRANSPORT=streamable-http
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8080
ENV MCP_LOG_LEVEL=INFO
ENV MCP_DEBUG=false
```

#### Running with docker run
```bash
# Basic run with API key
docker run -e POLYGON_API_KEY=your_api_key_here mcp-polygon

# Run with SSE transport on all interfaces
docker run -e POLYGON_API_KEY=your_api_key_here \
  -e MCP_TRANSPORT=sse \
  -e MCP_HOST=0.0.0.0 \
  -e MCP_PORT=9000 \
  -p 9000:9000 \
  mcp-polygon

# Run with Streamable-HTTP and debug logging
docker run -e POLYGON_API_KEY=your_api_key_here \
  -e MCP_TRANSPORT=streamable-http \
  -e MCP_HOST=0.0.0.0 \
  -e MCP_LOG_LEVEL=DEBUG \
  -e MCP_STREAMABLE_HTTP_PATH=/api/mcp \
  -p 8000:8000 \
  mcp-polygon

# Run with Streamable-HTTP in JSON response and stateless mode
docker run -e POLYGON_API_KEY=your_api_key_here \
  -e MCP_TRANSPORT=streamable-http \
  -e MCP_HOST=0.0.0.0 \
  -e MCP_JSON_RESPONSE=true \
  -e MCP_STATELESS_HTTP=true \
  -p 8000:8000 \
  mcp-polygon
```

#### Using docker-compose
See [docker-compose.yml](docker-compose.yml) for a complete example with all configurable environment variables.

## Notes

- The `host` and `port` settings are only used for HTTP-based transports (SSE and Streamable-HTTP)
- When using `stdio` transport, HTTP settings are ignored
- Path settings are transport-specific and only apply to their respective transports
- Debug mode and log level apply to all transport types
