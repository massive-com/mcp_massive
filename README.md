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

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that provides access to [Massive.com](https://massive.com?utm_campaign=mcp&utm_medium=referral&utm_source=github) financial market data API through an LLM-friendly interface.

## Overview

This server exposes all Massive.com API endpoints as MCP tools, providing access to comprehensive financial market data including:

- Stock, options, forex, and crypto aggregates and bars
- Real-time and historical trades and quotes
- Market snapshots
- Ticker details and reference data
- Dividends and splits data
- Financial fundamentals
- Market status and holidays

## Installation

### Prerequisites

- Python 3.10+
- A Massive.com API key <br> [![Button]][Link]
- [Astral UV](https://docs.astral.sh/uv/getting-started/installation/)
  - For existing installs, check that you have a version that supports the `uvx` command.

### Claude Code
First, install [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview)

```bash
npm install -g @anthropic-ai/claude-code
```

Use the following command to add the Massive MCP server to your local environment.
This assumes `uvx` is in your $PATH; if not, then you need to provide the full
path to `uvx`.

```bash
# Claude CLI
claude mcp add massive -e MASSIVE_API_KEY=your_api_key_here -- uvx --from git+https://github.com/massive-com/mcp_massive@v0.6.0 mcp_massive
```

This command will install the MCP server in your current project.
If you want to install it globally, you can run the command with `-s <scope>` flag.
See `claude mcp add --help` for more options.

To start Claude Code, run `claude` in your terminal.
- If this is your first time using, follow the setup prompts to authenticate

You can also run `claude mcp add-from-claude-desktop` if the MCP server is installed already for Claude Desktop.

### Claude Desktop

1. Follow the [Claude Desktop MCP installation instructions](https://modelcontextprotocol.io/quickstart/user) to complete the initial installation and find your configuration file.
1. Use the following example as reference to add Massive's MCP server.
Make sure you complete the various fields.
    1. Path find your path to `uvx`, run `which uvx` in your terminal.
    2. Replace `<your_api_key_here>` with your actual Massive.com API key.
    3. Replace `<your_home_directory>` with your home directory path, e.g., `/home/username` (Mac/Linux) or `C:\Users\username` (Windows).

<details>
  <summary>claude_desktop_config.json</summary>

```json
{
    "mcpServers": {
        "massive": {
            "command": "<path_to_your_uvx_install>/uvx",
            "args": [
                "--from",
                "git+https://github.com/massive-com/mcp_massive@v0.6.0",
                "mcp_massive"
            ],
            "env": {
                "MASSIVE_API_KEY": "<your_api_key_here>",
                "HOME": "<your_home_directory>"
            }
        }
    }
}
```
</details>

## Transport Configuration

By default, STDIO transport is used.

To configure [SSE](https://modelcontextprotocol.io/specification/2024-11-05/basic/transports#http-with-sse) or [Streamable HTTP](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports#streamable-http), set the `MCP_TRANSPORT` environment variable.

Example:

```bash
MCP_TRANSPORT=streamable-http \
MASSIVE_API_KEY=<your_api_key_here> \
uv run entrypoint.py
```

## Usage Examples

Once integrated, you can prompt Claude to access Massive.com data:

```
Get the latest price for AAPL stock
Show me yesterday's trading volume for MSFT
What were the biggest stock market gainers today?
Get me the latest crypto market data for BTC-USD
```

## Available Tools

This MCP server implements all Massive.com API endpoints as tools, including:

- `get_aggs` - Stock aggregates (OHLC) data for a specific ticker
- `list_trades` - Historical trade data
- `get_last_trade` - Latest trade for a symbol
- `list_ticker_news` - Recent news articles for tickers
- `get_snapshot_ticker` - Current market snapshot for a ticker
- `get_market_status` - Current market status and trading hours
- `list_stock_financials` - Fundamental financial data
- And many more...

Each tool follows the Massive.com SDK parameter structure while converting responses to standard JSON that LLMs can easily process.

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

```json

  "mcpServers": {
    "massive": {
      "command": "/your/path/.cargo/bin/uv",
      "args": [
        "run",
        "--with",
        "/your/path/mcp_massive",
        "mcp_massive"
      ],
      "env": {
        "MASSIVE_API_KEY": "your_api_key_here",
        "HOME": "/Users/danny"
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
