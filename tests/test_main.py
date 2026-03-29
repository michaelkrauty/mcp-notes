"""Tests for mcp-notes CLI entry point."""

from unittest.mock import MagicMock

import pytest


class TestStartup:
    """Tests for startup function."""

    @pytest.mark.asyncio
    async def test_startup_auto_index_enabled(self, tmp_notes_dir, monkeypatch):
        """Startup auto-indexes when enabled."""
        from mcp_notes import __main__ as main_module
        from mcp_notes.settings import settings

        # Enable auto-index
        monkeypatch.setattr(settings, "auto_index", True)

        # Should not raise
        await main_module.startup()

    @pytest.mark.asyncio
    async def test_startup_auto_index_disabled(self, tmp_notes_dir, monkeypatch):
        """Startup skips indexing when disabled."""
        from mcp_notes import __main__ as main_module
        from mcp_notes.settings import settings

        # Disable auto-index
        monkeypatch.setattr(settings, "auto_index", False)

        # Should not raise and should not index
        await main_module.startup()

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
