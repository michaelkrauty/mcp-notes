"""Git versioning for notes."""

import logging
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from git import GitCommandError, InvalidGitRepositoryError, Repo
from git.exc import BadName
from git.objects import Blob

from mcp_notes.models import NoteVersion
from mcp_notes.settings import settings

logger = logging.getLogger(__name__)

class GitManager:
    """
    Git-based versioning for notes.

    Auto-commits on note creation, update, and deletion.
    Supports tracking file moves/renames with --follow.
    """

    def __init__(self, notes_dir: Path | None = None):
        """
        Initialize git manager.

        Args:
            notes_dir: Notes base directory (defaults to settings.dir)
        """
        self.base_dir = Path(notes_dir or settings.dir)
        self._repo: Repo | None = None
        # Instance-level lock - allows different GitManager instances to operate
        # on different repos concurrently while protecting each repo from corruption
        self._repo_lock = threading.RLock()

    @property
    def repo(self) -> Repo | None:
        """Get or initialize git repository."""
        if not settings.git_enabled:
            return None

        if self._repo is not None:
            return self._repo

        try:
            self._repo = Repo(self.base_dir)
        except InvalidGitRepositoryError:
            # Initialize new repo
            self._repo = self._init_repo()

        return self._repo

    def _init_repo(self) -> Repo:
        """Initialize a new git repository."""
        self.base_dir.mkdir(parents=True, exist_ok=True)

        with self._repo_lock:
            repo = Repo.init(self.base_dir)

            # Configure git user
            with repo.config_writer() as config:
                config.set_value("user", "name", settings.git_user_name)
                config.set_value("user", "email", settings.git_user_email)

            # Create .gitignore
            gitignore = self.base_dir / ".gitignore"
            gitignore_content = """.index/
.config/
.locks/
*.pyc
__pycache__/
.venv/
"""
            gitignore.write_text(gitignore_content)

            # Initial commit
            repo.index.add([".gitignore"])
            repo.index.commit("Initialize notes repository")

            logger.info(f"Initialized git repository at {self.base_dir}")
            return repo

    def commit_create(self, note_id: UUID, title: str, path: Path | None = None) -> str | None:
        """
        Commit a newly created note.

        Args:
            note_id: Note UUID
            title: Note title
            path: Explicit path to note file (optional, for new nested structure)

        Returns:
            Commit SHA or None if git disabled
        """
        repo = self.repo
        if repo is None:
            return None

        if path is not None:
            rel_path = path.relative_to(self.base_dir)
        else:
            # Fallback for legacy - should not be used in new system
            rel_path = Path("notes") / f"{note_id}.md"

        with self._repo_lock:
            try:
                repo.index.add([str(rel_path)])
                commit = repo.index.commit(f"Create note: {title}")
                logger.info(f"Git commit (create): {commit.hexsha[:8]} - {title}")
                return commit.hexsha
            except GitCommandError as e:
                logger.error(f"Git commit failed: {e}")
                return None

    def commit_update(self, note_id: UUID, title: str, path: Path | None = None) -> str | None:
        """
        Commit an updated note.

        Args:
            note_id: Note UUID
            title: Note title
            path: Explicit path to note file (optional)

        Returns:
            Commit SHA or None if git disabled
        """
        repo = self.repo
        if repo is None:
            return None

        if path is not None:
            rel_path = path.relative_to(self.base_dir)
        else:
            # Fallback for legacy
            rel_path = Path("notes") / f"{note_id}.md"

        with self._repo_lock:
            try:
                repo.index.add([str(rel_path)])
                commit = repo.index.commit(f"Update note: {title}")
                logger.info(f"Git commit (update): {commit.hexsha[:8]} - {title}")
                return commit.hexsha
            except GitCommandError as e:
                logger.error(f"Git commit failed: {e}")
                return None

    def commit_delete(self, note_id: UUID, title: str, path: Path | None = None) -> str | None:
        """
        Commit a deleted note.

        Args:
            note_id: Note UUID
            title: Note title
            path: Explicit path to note file (optional)

        Returns:
            Commit SHA or None if git disabled
        """
        repo = self.repo
        if repo is None:
            return None

        if path is not None:
            rel_path = path.relative_to(self.base_dir)
        else:
            # Fallback for legacy
            rel_path = Path("notes") / f"{note_id}.md"

        with self._repo_lock:
            try:
                repo.index.remove([str(rel_path)])
                commit = repo.index.commit(f"Delete note: {title}")
                logger.info(f"Git commit (delete): {commit.hexsha[:8]} - {title}")
                return commit.hexsha
            except GitCommandError as e:
                logger.error(f"Git commit failed: {e}")
                return None

    def commit_move(
        self,
        old_path: Path,
        new_path: Path,
        title: str,
    ) -> str | None:
        """
        Commit a note move/rename with git mv for history tracking.

        Args:
            old_path: Old path to note file
            new_path: New path to note file
            title: Note title (for commit message)

        Returns:
            Commit SHA or None if git disabled
        """
        repo = self.repo
        if repo is None:
            return None

        try:
            old_rel = old_path.relative_to(self.base_dir)
            new_rel = new_path.relative_to(self.base_dir)
        except ValueError as e:
            logger.error(f"Path not relative to base_dir: {e}")
            return None

        with self._repo_lock:
            try:
                # Use git mv for proper history tracking
                repo.git.mv(str(old_rel), str(new_rel))
                commit = repo.index.commit(f"Move note: {title}")
                logger.info(f"Git commit (move): {commit.hexsha[:8]} - {title}")
                return commit.hexsha
            except GitCommandError as e:
                # Fallback: file was already moved, just add new location
                logger.warning(f"Git mv failed, using add: {e}")
                try:
                    repo.index.add([str(new_rel)])
                    try:
                        repo.index.remove([str(old_rel)])
                    except Exception as e:
                        logger.debug(f"Old file already removed from index: {old_rel} ({e})")
                    commit = repo.index.commit(f"Move note: {title}")
                    return commit.hexsha
                except GitCommandError as e2:
                    logger.error(f"Git commit (move fallback) failed: {e2}")
                    return None

    def get_history(
        self, note_id: UUID, limit: int = 10, path: Path | None = None
    ) -> list[NoteVersion]:
        """
        Get version history for a note.

        Uses git log --follow to properly track history across renames.

        Args:
            note_id: Note UUID
            limit: Max versions to return
            path: Current path to note file (optional)

        Returns:
            List of NoteVersion objects, newest first
        """
        repo = self.repo
        if repo is None:
            return []

        if path is not None:
            try:
                note_path = str(path.relative_to(self.base_dir))
            except ValueError:
                note_path = str(path)
        else:
            # Fallback for legacy
            note_path = f"notes/{note_id}.md"

        try:
            # Use subprocess to call git log with --follow for proper rename tracking
            # gitpython's iter_commits uses rev-list which doesn't support --follow
            result = subprocess.run(
                [
                    "git", "log",
                    "--follow",  # Track file across renames
                    f"--max-count={limit}",
                    "--format=%H|%ct|%s|%an",  # hash|timestamp|subject|author
                    "--",
                    note_path,
                ],
                cwd=str(self.base_dir),
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                logger.debug(f"git log failed: {result.stderr}")
                return []

            versions = []
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue

                parts = line.split("|", 3)
                if len(parts) != 4:
                    logger.debug(f"Unexpected git log output: {line}")
                    continue

                commit_sha, timestamp_str, message, author = parts
                try:
                    timestamp = datetime.fromtimestamp(int(timestamp_str), tz=UTC)
                except (ValueError, OSError) as e:
                    logger.debug(f"Invalid timestamp {timestamp_str}: {e}")
                    continue

                versions.append(
                    NoteVersion(
                        commit_sha=commit_sha,
                        timestamp=timestamp,
                        message=message.strip(),
                        author=author,
                    )
                )

            return versions

        except subprocess.TimeoutExpired:
            logger.warning("git log timed out")
            return []
        except FileNotFoundError:
            logger.error("git command not found")
            return []
        except Exception as e:
            logger.warning(f"Failed to get history: {e}")
            return []

    def get_version_content(
        self,
        note_id: UUID,
        commit_sha: str,
        path: Path | None = None,
    ) -> str | None:
        """
        Get note content at a specific version.

        Args:
            note_id: Note UUID
            commit_sha: Git commit SHA
            path: Current path hint (will search tree if not exact match)

        Returns:
            Note content at that version, or None if not found
        """
        repo = self.repo
        if repo is None:
            return None

        try:
            commit = repo.commit(commit_sha)
        except (BadName, GitCommandError) as e:
            logger.warning(f"Failed to get commit: {e}")
            return None

        # Try current path first
        if path is not None:
            try:
                note_path = str(path.relative_to(self.base_dir))
                blob = commit.tree / note_path
                content: str = blob.data_stream.read().decode("utf-8")
                return content
            except (ValueError, KeyError):
                pass

        # Search for file by UUID pattern in commit tree
        uuid_str = str(note_id)
        for item in commit.tree.traverse():
            if isinstance(item, Blob):
                item_path = str(item.path)
                if uuid_str in item_path and item_path.endswith(".md"):
                    try:
                        content = item.data_stream.read().decode("utf-8")
                        return content
                    except Exception as e:
                        logger.debug(f"Failed to read content from {item_path}: {e}")

        # Fallback: legacy path
        note_path = f"notes/{note_id}.md"
        try:
            blob = commit.tree / note_path
            content = blob.data_stream.read().decode("utf-8")
            return content
        except (KeyError, GitCommandError) as e:
            logger.warning(f"Failed to get version content: {e}")
            return None

    def restore_version(
        self,
        note_id: UUID,
        commit_sha: str,
        title: str,
        current_path: Path | None = None,
    ) -> str | None:
        """
        Restore a note to a previous version.

        Creates a new commit with the restored content.

        Args:
            note_id: Note UUID
            commit_sha: Commit to restore from
            title: Note title (for commit message)
            current_path: Current path to note file

        Returns:
            New commit SHA or None if failed
        """
        content = self.get_version_content(note_id, commit_sha, current_path)
        if content is None:
            return None

        # Determine path to write to
        if current_path is not None:
            note_path = current_path
        else:
            note_path = self.base_dir / "notes" / f"{note_id}.md"

        # Ensure parent directory exists
        note_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the old content
        note_path.write_text(content, encoding="utf-8")

        # Commit the restoration
        repo = self.repo
        if repo is None:
            return None

        rel_path = note_path.relative_to(self.base_dir)

        with self._repo_lock:
            try:
                repo.index.add([str(rel_path)])
                commit = repo.index.commit(
                    f"Restore note: {title}\n\nRestored from commit {commit_sha[:8]}"
                )
                logger.info(f"Git commit (restore): {commit.hexsha[:8]} - {title}")
                return commit.hexsha
            except GitCommandError as e:
                logger.error(f"Git commit failed: {e}")
                return None
