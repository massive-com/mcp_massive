"""Shared fixtures for integration tests.

Provides:
- session-scoped mock API server
- per-test global state reset
- in-process MCP client session factory
"""

import asyncio
import os

import anyio
import pytest
import pytest_asyncio
from mcp.client.session import ClientSession
from mcp.shared.message import SessionMessage

from mcp.types import CallToolResult, TextContent

from .mock_api_server import MockServer


def get_text(result: CallToolResult) -> str:
    """Extract text from the first content block, asserting it is TextContent."""
    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


# ---------------------------------------------------------------------------
# Session-scoped mock server
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def mock_server():
    """Start a mock Massive.com API server for the test session."""
    server = MockServer()
    base_url = server.start()
    yield base_url
    server.stop()


@pytest.fixture(scope="session")
def mock_env(mock_server):
    """Return env dict pointing at the mock server."""
    return {
        "MASSIVE_API_KEY": "test-integration-key",
        "MASSIVE_API_BASE_URL": mock_server,
        "MASSIVE_LLMS_TXT_URL": f"{mock_server}/docs/rest/llms-full.txt",
    }


# ---------------------------------------------------------------------------
# Per-test global state reset
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_server_globals(mock_env):
    """Reset mcp_massive.server module-level singletons before each test.

    Also sets env vars so that build_index() and call_api() use the mock server.
    """
    # Set env vars
    old_env = {}
    for key, value in mock_env.items():
        old_env[key] = os.environ.get(key)
        os.environ[key] = value

    import mcp_massive.server as srv

    # Configure module-level credentials and LLMS URL (env vars are no longer
    # read at call time; they are stored in-process before os.environ.clear()).
    srv.configure_credentials(
        mock_env["MASSIVE_API_KEY"],
        mock_env["MASSIVE_API_BASE_URL"],
        llms_txt_url=mock_env["MASSIVE_LLMS_TXT_URL"],
    )

    # Reset singletons
    srv._index = None
    srv._func_index = None
    srv._store = None
    srv._http_client = None

    yield

    # Restore
    srv._index = None
    srv._func_index = None
    srv._store = None
    srv._http_client = None

    for key, value in old_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


# ---------------------------------------------------------------------------
# In-process MCP client session
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def mcp_session():
    """Create an in-process MCP ClientSession connected to the real FastMCP server.

    Uses anyio memory streams — no subprocess, no network for the MCP layer.
    The underlying API calls still go through httpx to the mock server.

    We spawn the server + client setup in a background asyncio.Task so that
    the anyio cancel scopes stay in the same task throughout their lifetime,
    avoiding the "cancel scope in a different task" error during teardown.
    """
    from mcp_massive.server import mass_mcp

    # Two pairs of streams: one for server→client, one for client→server
    server_write, server_write_reader = anyio.create_memory_object_stream[
        SessionMessage
    ](0)
    client_write, client_write_reader = anyio.create_memory_object_stream[
        SessionMessage
    ](0)

    mcp_server = mass_mcp._mcp_server
    init_opts = mcp_server.create_initialization_options()

    session_ready = asyncio.Event()
    session_holder: dict = {}

    async def _run():
        async with anyio.create_task_group() as tg:

            async def _server():
                try:
                    await mcp_server.run(client_write_reader, server_write, init_opts)
                except anyio.ClosedResourceError:
                    pass

            tg.start_soon(_server)
            async with ClientSession(server_write_reader, client_write) as session:
                await session.initialize()
                session_holder["session"] = session
                session_ready.set()
                # Block until streams are closed during teardown
                try:
                    await anyio.sleep_forever()
                except anyio.get_cancelled_exc_class():
                    pass
            tg.cancel_scope.cancel()

    task = asyncio.get_event_loop().create_task(_run())
    await session_ready.wait()

    yield session_holder["session"]

    # Teardown: close streams to unblock the server, then cancel the task
    await client_write.aclose()
    await server_write.aclose()
    await client_write_reader.aclose()
    await server_write_reader.aclose()
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, anyio.ClosedResourceError):
        pass
