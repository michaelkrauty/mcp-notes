"""Tests for mcp-notes CLI entry point."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import anyio
import pytest
from mcp import Client


class TestStartup:
    """Tests for startup function."""

    @pytest.mark.asyncio
    async def test_startup_auto_index_enabled(self, tmp_notes_dir, monkeypatch):
        """Startup auto-indexes when enabled."""
        from mcp_notes import __main__ as main_module
        from mcp_notes.settings import settings

        # Enable auto-index
        monkeypatch.setattr(settings, "auto_index", True)
        indexer = SimpleNamespace(
            index_all=AsyncMock(return_value=SimpleNamespace(indexed_notes=2, total_notes=2))
        )
        get_indexer = AsyncMock(return_value=indexer)
        monkeypatch.setattr(main_module, "get_indexer", get_indexer)

        await main_module.startup()

        get_indexer.assert_awaited_once_with()
        indexer.index_all.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_startup_auto_index_disabled(self, tmp_notes_dir, monkeypatch):
        """Startup skips indexing when disabled."""
        from mcp_notes import __main__ as main_module
        from mcp_notes.settings import settings

        # Disable auto-index
        monkeypatch.setattr(settings, "auto_index", False)
        get_indexer = AsyncMock()
        monkeypatch.setattr(main_module, "get_indexer", get_indexer)

        await main_module.startup()

        get_indexer.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_startup_handles_index_error(self, tmp_notes_dir, monkeypatch):
        """Startup handles indexing errors gracefully."""
        from mcp_notes import __main__ as main_module
        from mcp_notes.settings import settings

        # Enable auto-index
        monkeypatch.setattr(settings, "auto_index", True)

        # Mock get_indexer to raise
        async def mock_get_indexer():
            raise RuntimeError("Test error")

        monkeypatch.setattr(main_module, "get_indexer", mock_get_indexer)

        # Should not raise - just logs warning
        await main_module.startup()


class TestMain:
    """Tests for main function."""

    def test_main_runs_startup_and_server(self, tmp_notes_dir, monkeypatch):
        """Main runs startup and MCP server."""
        from mcp_notes import __main__ as main_module
        from mcp_notes.server import mcp

        # Mock mcp.run to prevent blocking
        mock_run = MagicMock()
        monkeypatch.setattr(mcp, "run", mock_run)

        # Run main
        main_module.main()

        # Verify mcp.run was called
        mock_run.assert_called_once()

    def test_main_logs_settings(self, tmp_notes_dir, monkeypatch, caplog):
        """Main logs configuration settings."""
        import logging

        from mcp_notes import __main__ as main_module
        from mcp_notes.server import mcp

        # Mock mcp.run
        monkeypatch.setattr(mcp, "run", MagicMock())

        # Capture logs
        with caplog.at_level(logging.INFO):
            main_module.main()

        # Check that settings were logged
        log_text = caplog.text
        assert "Starting mcp-notes server" in log_text or "mcp-notes" in log_text


@pytest.mark.asyncio
async def test_server_lifespan_keeps_startup_and_cleanup_on_serving_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mcp_notes import __main__ as main_module  # noqa: PLC0415
    from mcp_notes import singletons  # noqa: PLC0415
    from mcp_notes.server import mcp  # noqa: PLC0415

    events: list[tuple[str, asyncio.AbstractEventLoop]] = []

    async def fake_startup() -> None:
        events.append(("startup", asyncio.get_running_loop()))

    async def fake_cleanup() -> None:
        events.append(("cleanup", asyncio.get_running_loop()))

    monkeypatch.setattr(main_module, "startup", fake_startup)
    monkeypatch.setattr(singletons, "cleanup_async_resources", fake_cleanup)

    async with Client(mcp, cache=None):
        events.append(("serving", asyncio.get_running_loop()))

    assert [name for name, _ in events] == ["startup", "serving", "cleanup"]
    assert len({id(loop) for _, loop in events}) == 1


@pytest.mark.asyncio
async def test_server_lifespan_cleans_up_cancelled_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mcp_notes import __main__ as main_module  # noqa: PLC0415
    from mcp_notes import singletons  # noqa: PLC0415
    from mcp_notes.app import lifespan, mcp  # noqa: PLC0415

    startup_started = anyio.Event()
    cleanup_finished = anyio.Event()

    async def cancelled_startup() -> None:
        startup_started.set()
        await anyio.sleep_forever()

    async def fake_cleanup() -> None:
        # This checkpoint is cancelled immediately unless the lifespan shields it.
        await anyio.sleep(0)
        cleanup_finished.set()

    async def run_lifespan() -> None:
        async with lifespan(mcp):
            pytest.fail("cancelled startup must not enter the serving phase")

    monkeypatch.setattr(main_module, "startup", cancelled_startup)
    monkeypatch.setattr(singletons, "cleanup_async_resources", fake_cleanup)

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(run_lifespan)
        await startup_started.wait()
        task_group.cancel_scope.cancel()

    assert cleanup_finished.is_set()


@pytest.mark.asyncio
async def test_cleanup_clears_failed_async_singleton_state() -> None:
    from mcp_notes import singletons  # noqa: PLC0415
    from mcp_notes.indexing.indexer import NoteIndexer  # noqa: PLC0415

    await singletons.cleanup_async_resources()

    async def fail_initialization() -> NoteIndexer:
        raise RuntimeError("initialization failed")

    with pytest.raises(RuntimeError, match="initialization failed"):
        await singletons._indexer.get(fail_initialization)

    await singletons.cleanup_async_resources()

    sentinel = MagicMock(spec=NoteIndexer)
    assert await singletons._indexer.get(lambda: sentinel) is sentinel

    await singletons.cleanup_async_resources()
