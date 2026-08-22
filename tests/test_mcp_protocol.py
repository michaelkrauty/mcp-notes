from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.server import MCPServer
from mcp.shared.subscriptions import ResourceUpdated
from mcp.types import TextContent

from mcp_notes import __version__
from mcp_notes.__main__ import EXPECTED_TOOLS, mcp
from mcp_notes.settings import settings

ROOT = Path(__file__).resolve().parents[1]


async def assert_safe_tool_dispatch(client: Client) -> None:
    result = await client.call_tool("read_note", {"note_id": "not-a-uuid"})

    assert not result.is_error
    assert isinstance(result.content[0], TextContent)
    assert "Invalid UUID" in result.content[0].text


def test_server_uses_public_sdk_v2_metadata() -> None:
    assert isinstance(mcp, MCPServer)
    assert mcp.name == "mcp-notes"
    assert mcp.version == __version__


@pytest.mark.asyncio
async def test_server_supports_modern_and_legacy_protocols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "auto_index", False)

    async with Client(mcp, cache=None) as modern:
        assert modern.protocol_version == "2026-07-28"
        assert modern.server_info is not None
        assert modern.server_info.name == "mcp-notes"
        assert modern.server_info.version == __version__
        assert modern.server_capabilities.resources is not None

        first = await modern.list_tools()
        second = await modern.list_tools()
        tool_names = [tool.name for tool in first.tools]

        assert tool_names == [tool.name for tool in second.tools] == EXPECTED_TOOLS
        assert first.result_type == "complete"
        assert first.ttl_ms == 0
        assert first.cache_scope == "private"
        assert first.meta == {
            "io.modelcontextprotocol/serverInfo": {
                "name": "mcp-notes",
                "version": __version__,
            }
        }

        resources = await modern.list_resources()
        assert {str(resource.uri) for resource in resources.resources} == {
            "notes://broken-links",
            "notes://categories",
            "notes://index",
            "notes://orphans",
            "notes://parse-errors",
            "notes://recent",
            "notes://tags",
        }
        assert resources.result_type == "complete"
        assert resources.ttl_ms == 0
        assert resources.cache_scope == "private"
        await assert_safe_tool_dispatch(modern)

    async with Client(mcp, mode="legacy", cache=None) as legacy:
        assert legacy.protocol_version == "2025-11-25"
        assert legacy.server_info is not None
        assert legacy.server_info.name == "mcp-notes"
        assert legacy.server_info.version == __version__
        assert [tool.name for tool in (await legacy.list_tools()).tools] == EXPECTED_TOOLS
        await assert_safe_tool_dispatch(legacy)


@pytest.mark.asyncio
async def test_note_mutation_publishes_resource_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mcp_notes.tools import notes as notes_tools  # noqa: PLC0415

    service = SimpleNamespace(
        create=AsyncMock(
            return_value=SimpleNamespace(
                success=True,
                to_dict=lambda: {"id": "test-note"},
            )
        )
    )

    async def get_service() -> SimpleNamespace:
        return service

    monkeypatch.setattr(settings, "auto_index", False)
    monkeypatch.setattr(notes_tools, "get_note_service", get_service)

    async with Client(mcp, cache=None) as client:
        create_tool = next(
            tool for tool in (await client.list_tools()).tools if tool.name == "create_note"
        )
        assert "context" not in create_tool.input_schema["properties"]

        async with client.listen(resource_subscriptions=["notes://index"]) as subscription:
            result = await client.call_tool(
                "create_note",
                {"title": "Test", "content": "No note data is written."},
            )
            event = await asyncio.wait_for(anext(subscription), timeout=1)

    assert not result.is_error
    assert event == ResourceUpdated(uri="notes://index")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected_protocol"),
    [("auto", "2026-07-28"), ("legacy", "2025-11-25")],
)
async def test_stdio_entrypoint_supports_modern_and_legacy_wire_protocols(
    mode: str,
    expected_protocol: str,
) -> None:
    params = StdioServerParameters(
        command=str(Path(sys.executable).with_name("mcp-notes")),
        cwd=ROOT,
        env={"NOTES_AUTO_INDEX": "false"},
    )

    async with Client(stdio_client(params), mode=mode, cache=None) as client:
        assert client.protocol_version == expected_protocol
        assert client.server_info is not None
        assert client.server_info.name == "mcp-notes"
        assert client.server_info.version == __version__
        assert [tool.name for tool in (await client.list_tools()).tools] == EXPECTED_TOOLS
        await assert_safe_tool_dispatch(client)
