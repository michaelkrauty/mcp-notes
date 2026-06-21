"""Tests for Git versioning."""

from unittest.mock import patch
from uuid import uuid4

from mcp_notes.models import NoteVersion
from mcp_notes.settings import settings
from mcp_notes.storage.git import GitManager


class TestGitManagerInit:
    """Tests for GitManager initialization."""

    def test_default_dir(self, monkeypatch, tmp_path):
        """Uses settings dir by default."""
        from mcp_notes.settings import settings
        monkeypatch.setattr(settings, "dir", tmp_path)

        manager = GitManager()

        assert manager.base_dir == tmp_path

    def test_custom_dir(self, tmp_path):
        """Accepts custom directory."""
        manager = GitManager(notes_dir=tmp_path)

        assert manager.base_dir == tmp_path


class TestGitManagerRepoProperty:
    """Tests for repo property."""

    def test_repo_disabled(self, tmp_path, monkeypatch):
        """Returns None when git disabled."""
        from mcp_notes.settings import settings
        monkeypatch.setattr(settings, "git_enabled", False)

        manager = GitManager(notes_dir=tmp_path)

        assert manager.repo is None

    def test_repo_initializes(self, tmp_path, monkeypatch):
        """Initializes new repo if none exists."""
        from mcp_notes.settings import settings
        monkeypatch.setattr(settings, "git_enabled", True)
        monkeypatch.setattr(settings, "git_user_name", "Test User")
        monkeypatch.setattr(settings, "git_user_email", "test@example.com")

        manager = GitManager(notes_dir=tmp_path)
        repo = manager.repo

        assert repo is not None
        assert (tmp_path / ".git").exists()

    def test_repo_cached(self, tmp_path, monkeypatch):
        """Repo is cached after first access."""
        from mcp_notes.settings import settings
        monkeypatch.setattr(settings, "git_enabled", True)
        monkeypatch.setattr(settings, "git_user_name", "Test")
        monkeypatch.setattr(settings, "git_user_email", "test@test.com")

        manager = GitManager(notes_dir=tmp_path)
        repo1 = manager.repo
        repo2 = manager.repo

        assert repo1 is repo2


class TestGitManagerInitRepo:
    """Tests for _init_repo method."""

    def test_creates_gitignore(self, tmp_path, monkeypatch):
        """Creates .gitignore file."""
        from mcp_notes.settings import settings
        monkeypatch.setattr(settings, "git_enabled", True)
        monkeypatch.setattr(settings, "git_user_name", "Test")
        monkeypatch.setattr(settings, "git_user_email", "test@test.com")

        manager = GitManager(notes_dir=tmp_path)
        manager.repo  # Trigger init

        gitignore = tmp_path / ".gitignore"
        assert gitignore.exists()

        content = gitignore.read_text()
        assert ".index/" in content
        assert "__pycache__/" in content

    def test_initial_commit(self, tmp_path, monkeypatch):
        """Makes initial commit."""
        from mcp_notes.settings import settings
        monkeypatch.setattr(settings, "git_enabled", True)
        monkeypatch.setattr(settings, "git_user_name", "Test")
        monkeypatch.setattr(settings, "git_user_email", "test@test.com")

        manager = GitManager(notes_dir=tmp_path)
        repo = manager.repo

        commits = list(repo.iter_commits())
        assert len(commits) == 1
        assert "Initialize" in commits[0].message


class TestGitManagerCommitCreate:
    """Tests for commit_create method."""

    def test_commit_create_disabled(self, tmp_path, monkeypatch):
        """Returns None when git disabled."""
        from mcp_notes.settings import settings
        monkeypatch.setattr(settings, "git_enabled", False)

        manager = GitManager(notes_dir=tmp_path)
        result = manager.commit_create(uuid4(), "Test Note")

        assert result is None

    def test_commit_create_success(self, tmp_path, monkeypatch):
        """Creates commit for new note."""
        from mcp_notes.settings import settings
        monkeypatch.setattr(settings, "git_enabled", True)
        monkeypatch.setattr(settings, "git_user_name", "Test")
        monkeypatch.setattr(settings, "git_user_email", "test@test.com")

        manager = GitManager(notes_dir=tmp_path)
        manager.repo  # Init repo

        # Create note file
        note_id = uuid4()
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(exist_ok=True)
        (notes_dir / f"{note_id}.md").write_text("content")

        sha = manager.commit_create(note_id, "Test Note")

        assert sha is not None
        assert len(sha) == 40

        # Verify commit message
        commits = list(manager.repo.iter_commits())
        assert any("Create note: Test Note" in c.message for c in commits)


class TestGitManagerCommitUpdate:
    """Tests for commit_update method."""

    def test_commit_update_disabled(self, tmp_path, monkeypatch):
        """Returns None when git disabled."""
        from mcp_notes.settings import settings
        monkeypatch.setattr(settings, "git_enabled", False)

        manager = GitManager(notes_dir=tmp_path)
        result = manager.commit_update(uuid4(), "Test Note")

        assert result is None

    def test_commit_update_success(self, tmp_path, monkeypatch):
        """Creates commit for updated note."""
        from mcp_notes.settings import settings
        monkeypatch.setattr(settings, "git_enabled", True)
        monkeypatch.setattr(settings, "git_user_name", "Test")
        monkeypatch.setattr(settings, "git_user_email", "test@test.com")

        manager = GitManager(notes_dir=tmp_path)
        manager.repo

        # Create and commit initial note
        note_id = uuid4()
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(exist_ok=True)
        note_path = notes_dir / f"{note_id}.md"
        note_path.write_text("original")
        manager.commit_create(note_id, "Test")

        # Update note
        note_path.write_text("updated")
        sha = manager.commit_update(note_id, "Test Note")

        assert sha is not None
        commits = list(manager.repo.iter_commits())
        assert any("Update note: Test Note" in c.message for c in commits)


class TestGitManagerCommitDelete:
    """Tests for commit_delete method."""

    def test_commit_delete_disabled(self, tmp_path, monkeypatch):
        """Returns None when git disabled."""
        from mcp_notes.settings import settings
        monkeypatch.setattr(settings, "git_enabled", False)

        manager = GitManager(notes_dir=tmp_path)
        result = manager.commit_delete(uuid4(), "Test Note")

        assert result is None

    def test_commit_delete_success(self, tmp_path, monkeypatch):
        """Creates commit for deleted note."""
        from mcp_notes.settings import settings
        monkeypatch.setattr(settings, "git_enabled", True)
        monkeypatch.setattr(settings, "git_user_name", "Test")
        monkeypatch.setattr(settings, "git_user_email", "test@test.com")

        manager = GitManager(notes_dir=tmp_path)
        manager.repo

        # Create and commit note
        note_id = uuid4()
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(exist_ok=True)
        note_path = notes_dir / f"{note_id}.md"
        note_path.write_text("content")
        manager.commit_create(note_id, "Test")

        # Delete note file
        note_path.unlink()
        sha = manager.commit_delete(note_id, "Test Note")

        assert sha is not None
        commits = list(manager.repo.iter_commits())
        assert any("Delete note: Test Note" in c.message for c in commits)


class TestGitManagerGetHistory:
    """Tests for get_history method."""

    def test_history_disabled(self, tmp_path, monkeypatch):
        """Returns empty list when git disabled."""
        from mcp_notes.settings import settings
        monkeypatch.setattr(settings, "git_enabled", False)

        manager = GitManager(notes_dir=tmp_path)
        result = manager.get_history(uuid4())

        assert result == []

    def test_history_returns_versions(self, tmp_path, monkeypatch):
        """Returns list of NoteVersion objects."""
        from mcp_notes.settings import settings
        monkeypatch.setattr(settings, "git_enabled", True)
        monkeypatch.setattr(settings, "git_user_name", "Test User")
        monkeypatch.setattr(settings, "git_user_email", "test@test.com")

        manager = GitManager(notes_dir=tmp_path)
        manager.repo

        # Create and modify note
        note_id = uuid4()
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(exist_ok=True)
        note_path = notes_dir / f"{note_id}.md"

        note_path.write_text("version 1")
        manager.commit_create(note_id, "Test Note")

        note_path.write_text("version 2")
        manager.commit_update(note_id, "Test Note")

        history = manager.get_history(note_id, limit=10)

        assert len(history) >= 2
        assert all(isinstance(v, NoteVersion) for v in history)

    def test_history_version_fields(self, tmp_path, monkeypatch):
        """Versions have all required fields."""
        from mcp_notes.settings import settings
        monkeypatch.setattr(settings, "git_enabled", True)
        monkeypatch.setattr(settings, "git_user_name", "Test User")
        monkeypatch.setattr(settings, "git_user_email", "test@test.com")

        manager = GitManager(notes_dir=tmp_path)
        manager.repo

        note_id = uuid4()
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(exist_ok=True)
        (notes_dir / f"{note_id}.md").write_text("content")
        manager.commit_create(note_id, "Test")

        history = manager.get_history(note_id)

        assert len(history) >= 1
        version = history[0]
        assert version.commit_sha is not None
        assert version.timestamp is not None
        assert version.message is not None
        assert version.author is not None

    def test_history_respects_limit(self, tmp_path, monkeypatch):
        """Respects limit parameter."""
        from mcp_notes.settings import settings
        monkeypatch.setattr(settings, "git_enabled", True)
        monkeypatch.setattr(settings, "git_user_name", "Test")
        monkeypatch.setattr(settings, "git_user_email", "test@test.com")

        manager = GitManager(notes_dir=tmp_path)
        manager.repo

        note_id = uuid4()
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(exist_ok=True)
        note_path = notes_dir / f"{note_id}.md"

        # Create multiple versions
        for i in range(5):
            note_path.write_text(f"version {i}")
            if i == 0:
                manager.commit_create(note_id, "Test")
            else:
                manager.commit_update(note_id, "Test")

        history = manager.get_history(note_id, limit=2)

        assert len(history) <= 2


class TestGitManagerGetVersionContent:
    """Tests for get_version_content method."""

    def test_content_disabled(self, tmp_path, monkeypatch):
        """Returns None when git disabled."""
        from mcp_notes.settings import settings
        monkeypatch.setattr(settings, "git_enabled", False)

        manager = GitManager(notes_dir=tmp_path)
        result = manager.get_version_content(uuid4(), "abc123")

        assert result is None

    def test_content_returns_old_version(self, tmp_path, monkeypatch):
        """Returns content from old version."""
        from mcp_notes.settings import settings
        monkeypatch.setattr(settings, "git_enabled", True)
        monkeypatch.setattr(settings, "git_user_name", "Test")
        monkeypatch.setattr(settings, "git_user_email", "test@test.com")

        manager = GitManager(notes_dir=tmp_path)
        manager.repo

        note_id = uuid4()
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(exist_ok=True)
        note_path = notes_dir / f"{note_id}.md"

        # Create initial version
        note_path.write_text("original content")
        sha = manager.commit_create(note_id, "Test")

        # Update to new version
        note_path.write_text("updated content")
        manager.commit_update(note_id, "Test")

        # Get original content
        content = manager.get_version_content(note_id, sha)

        assert content == "original content"

    def test_content_unresolvable_hex_sha_returns_none(self, tmp_path, monkeypatch):
        """A syntactically valid 40-hex SHA that names no object must return None,
        not raise (GitPython raises a bare ValueError for an unresolvable hex
        SHA, which get_version_content must catch like is_note_deleted_at does)."""
        monkeypatch.setattr(settings, "git_enabled", True)
        monkeypatch.setattr(settings, "git_user_name", "Test")
        monkeypatch.setattr(settings, "git_user_email", "test@test.com")

        manager = GitManager(notes_dir=tmp_path)
        note_id = uuid4()
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(exist_ok=True)
        (notes_dir / f"{note_id}.md").write_text("content")
        manager.commit_create(note_id, "Test")

        # 40 valid hex chars that name no object in the repo.
        result = manager.get_version_content(note_id, "de" * 20)
        assert result is None

    def test_content_invalid_sha(self, tmp_path, monkeypatch):
        """Returns None for invalid commit SHA."""
        from mcp_notes.settings import settings
        monkeypatch.setattr(settings, "git_enabled", True)
        monkeypatch.setattr(settings, "git_user_name", "Test")
        monkeypatch.setattr(settings, "git_user_email", "test@test.com")

        manager = GitManager(notes_dir=tmp_path)
        manager.repo

        result = manager.get_version_content(uuid4(), "invalid_sha")

        assert result is None


class TestGitManagerRestoreVersion:
    """Tests for restore_version method."""

    def test_restore_disabled(self, tmp_path, monkeypatch):
        """Returns None when git disabled."""
        from mcp_notes.settings import settings
        monkeypatch.setattr(settings, "git_enabled", False)

        manager = GitManager(notes_dir=tmp_path)
        result = manager.restore_version(uuid4(), "sha", "Title")

        assert result is None

    def test_restore_success(self, tmp_path, monkeypatch):
        """Restores note to previous version."""
        from mcp_notes.settings import settings
        monkeypatch.setattr(settings, "git_enabled", True)
        monkeypatch.setattr(settings, "git_user_name", "Test")
        monkeypatch.setattr(settings, "git_user_email", "test@test.com")

        manager = GitManager(notes_dir=tmp_path)
        manager.repo

        note_id = uuid4()
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(exist_ok=True)
        note_path = notes_dir / f"{note_id}.md"

        # Create initial version
        note_path.write_text("original content")
        original_sha = manager.commit_create(note_id, "Test")

        # Update
        note_path.write_text("new content")
        manager.commit_update(note_id, "Test")

        # Restore
        restore_sha = manager.restore_version(note_id, original_sha, "Test")

        assert restore_sha is not None
        assert note_path.read_text() == "original content"

        # Verify commit message
        commits = list(manager.repo.iter_commits())
        assert any("Restore note" in c.message for c in commits)

    def test_restore_invalid_sha(self, tmp_path, monkeypatch):
        """Returns None for invalid commit SHA."""
        from mcp_notes.settings import settings
        monkeypatch.setattr(settings, "git_enabled", True)
        monkeypatch.setattr(settings, "git_user_name", "Test")
        monkeypatch.setattr(settings, "git_user_email", "test@test.com")

        manager = GitManager(notes_dir=tmp_path)
        manager.repo

        result = manager.restore_version(uuid4(), "invalid", "Title")

        assert result is None


class TestGitManagerErrorHandling:
    """Tests for error handling in git operations."""

    def test_get_history_git_error(self, tmp_path, monkeypatch):
        """Returns empty list when git command fails during history."""
        from git.exc import GitCommandError

        from mcp_notes.settings import settings

        monkeypatch.setattr(settings, "git_enabled", True)
        monkeypatch.setattr(settings, "git_user_name", "Test")
        monkeypatch.setattr(settings, "git_user_email", "test@test.com")

        manager = GitManager(notes_dir=tmp_path)
        repo = manager.repo

        # Mock iter_commits to raise GitCommandError
        def raise_error(*args, **kwargs):
            raise GitCommandError("git log", 1)

        monkeypatch.setattr(repo, 'iter_commits', raise_error)
        result = manager.get_history(uuid4())

        assert result == []

    def test_get_version_content_bad_name(self, tmp_path, monkeypatch):
        """Returns None when BadName error occurs for invalid SHA."""
        from mcp_notes.settings import settings

        monkeypatch.setattr(settings, "git_enabled", True)
        monkeypatch.setattr(settings, "git_user_name", "Test")
        monkeypatch.setattr(settings, "git_user_email", "test@test.com")

        manager = GitManager(notes_dir=tmp_path)
        manager.repo

        # Test with an invalid SHA that will cause BadName exception
        result = manager.get_version_content(uuid4(), "invalid_sha_abc123")
        assert result is None

    def test_get_version_content_key_error(self, tmp_path, monkeypatch):
        """Returns None when KeyError occurs (file not in commit)."""
        from mcp_notes.settings import settings

        monkeypatch.setattr(settings, "git_enabled", True)
        monkeypatch.setattr(settings, "git_user_name", "Test")
        monkeypatch.setattr(settings, "git_user_email", "test@test.com")

        manager = GitManager(notes_dir=tmp_path)
        manager.repo

        note_id = uuid4()
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(exist_ok=True)
        (notes_dir / f"{note_id}.md").write_text("content")
        sha = manager.commit_create(note_id, "Create")

        # Try to get content for a different note_id that doesn't exist
        different_id = uuid4()
        result = manager.get_version_content(different_id, sha)
        assert result is None


class TestGitManagerBytesMessage:
    """Tests for handling bytes commit messages."""

    def test_history_handles_bytes_message(self, tmp_path, monkeypatch):
        """History handles bytes commit message."""
        from mcp_notes.settings import settings

        monkeypatch.setattr(settings, "git_enabled", True)
        monkeypatch.setattr(settings, "git_user_name", "Test")
        monkeypatch.setattr(settings, "git_user_email", "test@test.com")

        manager = GitManager(notes_dir=tmp_path)
        manager.repo

        note_id = uuid4()
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(exist_ok=True)
        (notes_dir / f"{note_id}.md").write_text("content")
        manager.commit_create(note_id, "Test")

        # Get history normally - messages are strings
        history = manager.get_history(note_id)
        assert len(history) >= 1
        assert isinstance(history[0].message, str)


class TestGitCommitErrorHandling:
    """Tests for GitCommandError handling in commit operations (lines 102-104, 129-131, 156-158, 263-265)."""

    def test_commit_create_git_error(self, tmp_path, monkeypatch):
        """commit_create returns None when GitCommandError occurs (lines 102-104)."""
        from git import IndexFile
        from git.exc import GitCommandError

        from mcp_notes.settings import settings

        monkeypatch.setattr(settings, "git_enabled", True)
        monkeypatch.setattr(settings, "git_user_name", "Test")
        monkeypatch.setattr(settings, "git_user_email", "test@test.com")

        manager = GitManager(notes_dir=tmp_path)
        manager.repo  # Initialize repo

        note_id = uuid4()
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(exist_ok=True)
        (notes_dir / f"{note_id}.md").write_text("content")

        # Use patch to mock IndexFile.commit
        with patch.object(IndexFile, 'commit', side_effect=GitCommandError("git commit", 1, "error")):
            result = manager.commit_create(note_id, "Test Note")

        assert result is None

    def test_commit_update_git_error(self, tmp_path, monkeypatch):
        """commit_update returns None when GitCommandError occurs (lines 129-131)."""
        from git import IndexFile
        from git.exc import GitCommandError

        from mcp_notes.settings import settings

        monkeypatch.setattr(settings, "git_enabled", True)
        monkeypatch.setattr(settings, "git_user_name", "Test")
        monkeypatch.setattr(settings, "git_user_email", "test@test.com")

        manager = GitManager(notes_dir=tmp_path)
        manager.repo  # Initialize repo

        note_id = uuid4()
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(exist_ok=True)
        (notes_dir / f"{note_id}.md").write_text("content")

        # First commit successfully
        manager.commit_create(note_id, "Test Note")

        # Update the file
        (notes_dir / f"{note_id}.md").write_text("updated content")

        # Now mock to fail on update
        with patch.object(IndexFile, 'commit', side_effect=GitCommandError("git commit", 1, "error")):
            result = manager.commit_update(note_id, "Test Note")

        assert result is None

    def test_commit_delete_git_error(self, tmp_path, monkeypatch):
        """commit_delete returns None when GitCommandError occurs (lines 156-158)."""
        from git import IndexFile
        from git.exc import GitCommandError

        from mcp_notes.settings import settings

        monkeypatch.setattr(settings, "git_enabled", True)
        monkeypatch.setattr(settings, "git_user_name", "Test")
        monkeypatch.setattr(settings, "git_user_email", "test@test.com")

        manager = GitManager(notes_dir=tmp_path)
        manager.repo  # Initialize repo

        note_id = uuid4()
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(exist_ok=True)
        note_path = notes_dir / f"{note_id}.md"
        note_path.write_text("content")

        # First commit successfully
        manager.commit_create(note_id, "Test Note")

        # Delete the file
        note_path.unlink()

        # Mock to fail on delete commit
        with patch.object(IndexFile, 'commit', side_effect=GitCommandError("git commit", 1, "error")):
            result = manager.commit_delete(note_id, "Test Note")

        assert result is None

    def test_restore_version_git_error(self, tmp_path, monkeypatch):
        """restore_version returns None when GitCommandError occurs (lines 263-265)."""
        from git import IndexFile
        from git.exc import GitCommandError

        from mcp_notes.settings import settings

        monkeypatch.setattr(settings, "git_enabled", True)
        monkeypatch.setattr(settings, "git_user_name", "Test")
        monkeypatch.setattr(settings, "git_user_email", "test@test.com")

        manager = GitManager(notes_dir=tmp_path)
        manager.repo  # Initialize repo

        note_id = uuid4()
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(exist_ok=True)
        note_path = notes_dir / f"{note_id}.md"
        note_path.write_text("original content")

        # Create initial commit
        original_sha = manager.commit_create(note_id, "Test Note")

        # Update the note
        note_path.write_text("updated content")
        manager.commit_update(note_id, "Test Note")

        # Mock to fail on restore commit
        with patch.object(IndexFile, 'commit', side_effect=GitCommandError("git commit", 1, "error")):
            result = manager.restore_version(note_id, original_sha, "Test Note")

        assert result is None


class TestGitHistoryMessageTypes:
    """Tests for message handling in get_history (uses subprocess with text output)."""

    def test_history_returns_string_messages(self, tmp_path, monkeypatch):
        """get_history returns string messages from subprocess output."""
        from mcp_notes.settings import settings

        monkeypatch.setattr(settings, "git_enabled", True)
        monkeypatch.setattr(settings, "git_user_name", "Test")
        monkeypatch.setattr(settings, "git_user_email", "test@test.com")

        manager = GitManager(notes_dir=tmp_path)

        note_id = uuid4()
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(exist_ok=True)
        (notes_dir / f"{note_id}.md").write_text("content")
        manager.commit_create(note_id, "Test Note with Émoji 🎉")

        history = manager.get_history(note_id)

        assert len(history) >= 1
        assert isinstance(history[0].message, str)
        # Message should contain the title (subprocess uses text=True)
        assert "Test Note" in history[0].message


class TestRestoreRepoNone:
    """Test for defensive repo=None check in restore_version (line 252)."""

    def test_restore_version_repo_none_after_content(self, tmp_path, monkeypatch):
        """restore_version returns None if repo becomes None after getting content (line 252)."""
        from mcp_notes.settings import settings

        monkeypatch.setattr(settings, "git_enabled", True)
        monkeypatch.setattr(settings, "git_user_name", "Test")
        monkeypatch.setattr(settings, "git_user_email", "test@test.com")

        manager = GitManager(notes_dir=tmp_path)
        manager.repo  # Initialize repo

        # Create the notes directory
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(exist_ok=True)

        # Side effect that returns content AND disables git
        # This simulates git being disabled between content retrieval and commit
        def get_content_and_disable_git(note_id, commit_sha, path=None):
            monkeypatch.setattr(settings, "git_enabled", False)  # Disable git after content
            return "restored content"

        # Mock get_version_content with side effect that also disables git
        with patch.object(manager, 'get_version_content', side_effect=get_content_and_disable_git):
            result = manager.restore_version(uuid4(), "abc123", "Test Note")

        assert result is None


class TestGitManagerDeletedNoteHistory:
    """Tests for get_history on deleted notes (no path hint, issue #13)."""

    def _make_manager(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "git_enabled", True)
        monkeypatch.setattr(settings, "git_user_name", "Test User")
        monkeypatch.setattr(settings, "git_user_email", "test@test.com")

        manager = GitManager(notes_dir=tmp_path)
        assert manager.repo is not None
        return manager

    def test_history_for_deleted_nested_note(self, tmp_path, monkeypatch):
        """create -> update -> delete still yields history without a path hint."""
        manager = self._make_manager(tmp_path, monkeypatch)

        # Current layout: notes/{category}/{slug}-{uuid}.md
        note_id = uuid4()
        category_dir = tmp_path / "notes" / "work"
        category_dir.mkdir(parents=True)
        note_path = category_dir / f"test-note-{note_id}.md"

        note_path.write_text("version 1")
        manager.commit_create(note_id, "Test Note", path=note_path)

        note_path.write_text("version 2")
        manager.commit_update(note_id, "Test Note", path=note_path)

        note_path.unlink()
        manager.commit_delete(note_id, "Test Note", path=note_path)

        # Deleted note: caller has no path (get_note_path returns None)
        history = manager.get_history(note_id, limit=10)

        assert len(history) == 3
        messages = [v.message for v in history]
        assert any("Create note" in m for m in messages)
        assert any("Update note" in m for m in messages)
        assert any("Delete note" in m for m in messages)

    def test_is_note_deleted_at_distinguishes_delete_commit(self, tmp_path, monkeypatch):
        """is_note_deleted_at flags the delete commit, not the create commit or
        an unknown SHA."""
        manager = self._make_manager(tmp_path, monkeypatch)

        note_id = uuid4()
        category_dir = tmp_path / "notes" / "work"
        category_dir.mkdir(parents=True)
        note_path = category_dir / f"test-note-{note_id}.md"

        note_path.write_text("version 1")
        manager.commit_create(note_id, "Test Note", path=note_path)

        note_path.unlink()
        manager.commit_delete(note_id, "Test Note", path=note_path)

        history = manager.get_history(note_id, limit=10)
        delete_sha = history[0].commit_sha  # newest entry is the deletion
        create_sha = history[-1].commit_sha  # oldest entry is the creation

        assert manager.is_note_deleted_at(note_id, delete_sha) is True
        assert manager.is_note_deleted_at(note_id, create_sha) is False
        # An unknown SHA is not a deletion of this note.
        assert manager.is_note_deleted_at(note_id, "0" * 40) is False
        # The underlying restore still aborts on the delete commit (no content).
        assert manager.get_version_content(note_id, delete_sha) is None
        assert manager.restore_version(note_id, delete_sha, "Test Note") is None

    def test_history_for_unknown_note_returns_empty(self, tmp_path, monkeypatch):
        """A UUID that never existed returns an empty history."""
        manager = self._make_manager(tmp_path, monkeypatch)

        # Commit an unrelated note so history is non-empty
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(exist_ok=True)
        other_id = uuid4()
        (notes_dir / f"other-{other_id}.md").write_text("content")
        manager.commit_create(other_id, "Other", path=notes_dir / f"other-{other_id}.md")

        history = manager.get_history(uuid4())

        assert history == []

    def test_find_last_known_path(self, tmp_path, monkeypatch):
        """_find_last_known_path locates a deleted note's last path."""
        manager = self._make_manager(tmp_path, monkeypatch)

        note_id = uuid4()
        category_dir = tmp_path / "notes" / "personal"
        category_dir.mkdir(parents=True)
        note_path = category_dir / f"my-note-{note_id}.md"

        note_path.write_text("content")
        manager.commit_create(note_id, "My Note", path=note_path)

        note_path.unlink()
        manager.commit_delete(note_id, "My Note", path=note_path)

        found = manager._find_last_known_path(note_id)

        assert found == f"notes/personal/my-note-{note_id}.md"

    def test_find_last_known_path_ignores_uuid_mentions_in_slug(self, tmp_path, monkeypatch):
        """A note whose slug mentions another note's UUID must not match."""
        manager = self._make_manager(tmp_path, monkeypatch)

        deleted_id = uuid4()
        other_id = uuid4()
        notes_dir = tmp_path / "notes" / "work"
        notes_dir.mkdir(parents=True)

        # Unrelated note whose filename contains the deleted note's UUID in
        # its slug portion (canonical UUID is at the end of the filename).
        decoy_path = notes_dir / f"mentions-{deleted_id}-{other_id}.md"
        decoy_path.write_text("decoy")
        manager.commit_create(other_id, "Decoy", path=decoy_path)

        target_path = notes_dir / f"target-{deleted_id}.md"
        target_path.write_text("target")
        manager.commit_create(deleted_id, "Target", path=target_path)

        target_path.unlink()
        manager.commit_delete(deleted_id, "Target", path=target_path)

        found = manager._find_last_known_path(deleted_id)

        assert found == f"notes/work/target-{deleted_id}.md"

        history = manager.get_history(deleted_id)
        messages = [v.message for v in history]
        assert all("Decoy" not in m for m in messages)
        assert any("Create note: Target" in m for m in messages)
