<a href="https://massive.com">
  <div align="center">
    <picture>
        <source media="(prefers-color-scheme: light)" srcset="assets/logo-massive-lightmode.png">
        <source media="(prefers-color-scheme: dark)" srcset="assets/logo-massive-darkmode.png">
        <img alt="Massive.com logo" src="assets/logo-massive-lightmode.png" height="100">
    </picture>
  </div>
</a>
<br>

> [!IMPORTANT]
> :test_tube: This project is experimental and could be subject to breaking changes.

# Massive.com MCP Server

 [![GitHub release](https://img.shields.io/github/v/release/massive-com/mcp_massive)](https://github.com/massive-com/mcp_massive/releases)

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that provides access to the full [Massive.com](https://massive.com?utm_campaign=mcp&utm_medium=referral&utm_source=github) financial data API through an LLM-friendly interface.

Rather than exposing one tool per endpoint, this server gives the LLM four composable tools — **search**, **docs**, **call**, and **query** — that cover the entire Massive.com API surface. Data can be stored in-memory as DataFrames, queried with SQL, and enriched with built-in financial functions.

## Tools

| Tool | Description |
|---|---|
| `search_endpoints` | Search for API endpoints and built-in functions by natural language query. Returns names, URL patterns, and descriptions. Supports scoping to `endpoints`, `functions`, or `all`. |
| `get_endpoint_docs` | Get parameter documentation for a specific endpoint. Pass the docs URL from `search_endpoints` results. |
| `call_api` | Call any Massive.com REST API endpoint. Supports storing results as an in-memory DataFrame (`store_as`) and applying post-processing functions (`apply`). Paginated responses include a next-page hint. |
| `query_data` | Run SQL against stored DataFrames using SQLite. Supports `SHOW TABLES`, `DESCRIBE <table>`, `DROP TABLE <table>`, CTEs, window functions, and more. Results can also be post-processed with `apply`. |

### Built-in Functions

Functions can be applied to API results or query output via the `apply` parameter on `call_api` and `query_data`. Use `search_endpoints` with `scope="functions"` to discover them.

| Category | Functions |
|---|---|
| **Greeks** | `bs_price`, `bs_delta`, `bs_gamma`, `bs_theta`, `bs_vega`, `bs_rho` — Black-Scholes option pricing and greeks |
| **Returns** | `simple_return`, `log_return`, `cumulative_return`, `sharpe_ratio`, `sortino_ratio` |
| **Technical** | `sma` (simple moving average), `ema` (exponential moving average) |

### Data Coverage

The server dynamically indexes all Massive.com API endpoints at startup from [`llms.txt`](https://massive.com/docs/rest/llms.txt), so it automatically stays in sync with the API. Coverage includes:

- Stock, options, forex, crypto, and futures aggregates
- Real-time and historical trades and quotes
- Market snapshots, gainers/losers
- Ticker details and reference data
- Dividends, splits, IPOs
- Financial fundamentals
- Analyst ratings and news (Benzinga)
- Treasury yields, inflation data
- Market status and holidays

## Installation

### Prerequisites

- Python 3.10+
- A Massive.com API key <br> [![Button]][Link]
- [Astral UV](https://docs.astral.sh/uv/getting-started/installation/) (v0.4.0+)

### Claude Code
First, install [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview)

```bash
npm install -g @anthropic-ai/claude-code
```

Install the MCP server, then register it with Claude Code:

```bash
# Install the server (one-time — downloads dependencies ahead of time)
uv tool install "mcp_massive @ git+https://github.com/massive-com/mcp_massive@v0.8.0"

# Register with Claude Code
claude mcp add massive -e MASSIVE_API_KEY=your_api_key_here -- mcp_massive
```

To upgrade to a new version later:

```bash
uv tool upgrade mcp_massive
```

> [!NOTE]
> **Upgrading from `uvx` or `uv run --with`?** Previous versions recommended `uvx --from ... mcp_massive` or `uv run --with`. These commands download dependencies on every cold start, which can cause the server to exceed Claude's 30-second connection timeout. Switch to `uv tool install` as shown above — it downloads dependencies once and starts instantly after that.

This command will install the MCP server in your current project.
If you want to install it globally, you can run the command with `-s <scope>` flag.
See `claude mcp add --help` for more options.

To start Claude Code, run `claude` in your terminal.
- If this is your first time using, follow the setup prompts to authenticate

You can also run `claude mcp add-from-claude-desktop` if the MCP server is installed already for Claude Desktop.

### Claude Desktop

1. Follow the [Claude Desktop MCP installation instructions](https://modelcontextprotocol.io/quickstart/user) to complete the initial installation and find your configuration file.
1. Install the server:

```bash
uv tool install "mcp_massive @ git+https://github.com/massive-com/mcp_massive@v0.8.0"
```

3. Find the installed binary path:

```bash
# Mac/Linux
which mcp_massive

# Windows
where mcp_massive
```

4. Add the server to your Claude Desktop config. Replace `<path_to_mcp_massive>` with the output from the previous step, and fill in the remaining fields.

<details>
  <summary>claude_desktop_config.json</summary>

```json
{
    "mcpServers": {
        "massive": {
            "command": "<path_to_mcp_massive>",
            "env": {
                "MASSIVE_API_KEY": "<your_api_key_here>",
                "HOME": "<your_home_directory>"
            }
        }
    }
}
```
</details>

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `MASSIVE_API_KEY` | Yes | — | Your Massive.com API key |
| `POLYGON_API_KEY` | No | — | Deprecated alias for `MASSIVE_API_KEY` |
| `MCP_TRANSPORT` | No | `stdio` | Transport protocol: `stdio`, `sse`, or `streamable-http` |
| `MASSIVE_API_BASE_URL` | No | `https://api.massive.com` | Base URL for API requests |
| `MASSIVE_LLMS_TXT_URL` | No | `https://massive.com/docs/rest/llms.txt` | URL for the endpoint index |
| `MASSIVE_MAX_TABLES` | No | `50` | Maximum number of in-memory DataFrames |
| `MASSIVE_MAX_ROWS` | No | `50000` | Maximum rows per stored DataFrame |

### Transport

By default, STDIO transport is used. The transport can be set via the `--transport` CLI argument or the `MCP_TRANSPORT` environment variable (CLI argument takes precedence).

```bash
# CLI argument
MASSIVE_API_KEY=<your_api_key_here> uv run mcp_massive --transport streamable-http

# Environment variable
MCP_TRANSPORT=streamable-http MASSIVE_API_KEY=<your_api_key_here> uv run mcp_massive
```

## Usage Examples

Once integrated, you can prompt Claude to access Massive.com data:

```
Get the latest price for AAPL stock
Show me yesterday's trading volume for MSFT
What were the biggest stock market gainers today?
Get me the latest crypto market data for BTC-USD
Calculate the 20-day SMA for AAPL closing prices over the last 3 months
Compute Black-Scholes delta for these option contracts
```

## Development

### Running Locally

Check to ensure you have the [Prerequisites](#prerequisites) installed.

```bash
# Sync dependencies
uv sync

# Run the server
MASSIVE_API_KEY=your_api_key_here uv run mcp_massive
```

<details>
  <summary>Local Dev Config for claude_desktop_config.json</summary>

Install from your local checkout, then reference the binary directly:

```bash
uv tool install --force /path/to/mcp_massive
```

```json
{
  "mcpServers": {
    "massive": {
      "command": "mcp_massive",
      "env": {
        "MASSIVE_API_KEY": "your_api_key_here"
      }
    }
  }
}
```
</details>

### Debugging

For debugging and testing, we recommend using the [MCP Inspector](https://github.com/modelcontextprotocol/inspector):

```bash
npx @modelcontextprotocol/inspector uv --directory /path/to/mcp_massive run mcp_massive
```

This will launch a browser interface where you can interact with your MCP server directly and see input/output for each tool.

### Code Linting

This project uses [just](https://github.com/casey/just) for common development tasks. To lint your code before submitting a PR:

```bash
just lint
```

This will run `ruff format` and `ruff check --fix` to automatically format your code and fix linting issues.

## Links
- [Massive.com Documentation](https://massive.com/docs?utm_campaign=mcp&utm_medium=referral&utm_source=github)
- [Model Context Protocol](https://modelcontextprotocol.io)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)

## Privacy Policy

This MCP server interacts with Massive.com's API to fetch market data. All data requests are subject to Massive.com's privacy policy and terms of service.

- **Massive.com Privacy Policy**: https://massive.com/legal/privacy
- **Data Handling**: This server does not store or cache any user data. All requests are proxied directly to Massive.com's API.
- **API Key**: Your Massive.com API key is used only for authenticating requests to their API.

## Contributing
If you found a bug or have an idea for a new feature, please first discuss it with us by submitting a new issue.
We will respond to issues within at most 3 weeks.
We're also open to volunteers if you want to submit a PR for any open issues but please discuss it with us beforehand.
PRs that aren't linked to an existing issue or discussed with us ahead of time will generally be declined.

<!----------------------------------------------------------------------------->
[Link]: https://massive.com/?utm_campaign=mcp&utm_medium=referral&utm_source=github 'Massive.com Home Page'
<!---------------------------------[ Buttons ]--------------------------------->
[Button]: https://img.shields.io/badge/Get_One_For_Free-5F5CFF?style=for-the-badge&logoColor=white
