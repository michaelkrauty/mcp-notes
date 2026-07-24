"""Git versioning for notes."""

import logging
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from git import GitCommandError, InvalidGitRepositoryError, Repo
from git.exc import BadName
from git.objects import Blob, Commit

from mcp_notes.models import NoteVersion
from mcp_notes.settings import settings
from mcp_notes.storage.slugify import extract_uuid_from_filename

logger = logging.getLogger(__name__)

# Lock directory, relative to the notes root. Named here because both the
# .gitignore written for a new repository and the per-checkout exclude added to
# an adopted one must agree on it.
LOCKS_EXCLUDE = ".locks/"


def _exclude_patterns(content: bytes) -> list[bytes]:
    """Split an exclude file into patterns the way git does.

    Git ends a pattern at a newline and strips at most one carriage return
    immediately before it. A lone carriage return is an ordinary character, so
    ``bytes.splitlines()`` is wrong here: it would treat one as a separator and
    find a rule git does not see.
    """
    return [line[:-1] if line.endswith(b"\r") else line for line in content.split(b"\n")]


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
            self._ensure_locks_excluded(self._repo)
        except InvalidGitRepositoryError:
            # Initialize new repo
            self._repo = self._init_repo()

        return self._repo

    def _ensure_locks_excluded(self, repo: Repo) -> None:
        """Keep the lock directory out of git for a repository we did not create.

        The `.gitignore` written by `_init_repo` covers repositories this class
        created, but a notes directory that was already a git repository never
        gets it. Per-note lock files are persistent runtime state, so without an
        exclusion they show up as untracked clutter and `git add -A` can commit
        them. Worse, a later checkout or a sync tool could then replace the
        inode of a lock somebody is holding, which is the failure the lock files
        are persistent to avoid in the first place.

        Writes to git's `info/exclude` rather than `.gitignore`: it is not
        itself version-controlled and never touches a file the user maintains.
        The path comes from `common_dir`, not `git_dir`, because git reads
        `info/exclude` from the common directory; a linked worktree's own
        `.git/worktrees/<name>/info/exclude` is not consulted, so the rule
        would have no effect there. That also means the rule is shared with
        every worktree of the repository.

        Idempotent, and best-effort: a bare repository has no working tree to
        exclude anything from, and failing to write the rule is not a reason to
        refuse to store notes. Read and written as bytes, since an adopted
        repository's exclude file need not be valid UTF-8 and mangling it would
        be worse than skipping the rule.
        """
        if repo.bare:
            return
        rule = LOCKS_EXCLUDE.encode()
        try:
            exclude_path = Path(repo.common_dir) / "info" / "exclude"
            prefix = b""
            if exclude_path.exists():
                existing = exclude_path.read_bytes()
                # Compared verbatim: leading whitespace is significant to git,
                # so " .locks/" is a different rule and must not satisfy this.
                # Split the way git does rather than with splitlines(), which
                # also breaks on a lone carriage return; git ends a pattern at a
                # newline and strips at most one carriage return before it, so
                # "scratch/\r.locks/" is one ineffective pattern to git and must
                # not read as an existing rule here.
                if rule in _exclude_patterns(existing):
                    return
                if existing and not existing.endswith(b"\n"):
                    prefix = b"\n"
            else:
                exclude_path.parent.mkdir(parents=True, exist_ok=True)
            with exclude_path.open("ab") as f:
                f.write(prefix + rule + b"\n")
            logger.debug(f"Excluded {LOCKS_EXCLUDE} in {exclude_path}")
        except OSError as e:
            logger.warning(f"Could not exclude {LOCKS_EXCLUDE} from git: {e}")

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

        with self._repo_lock:
            # Remove the path git ACTUALLY tracks for this note. The caller's
            # path is only a hint: a concurrent rename can leave the filesystem
            # store and git pointing at different paths (the store moves the file
            # before the git move commits, possibly in another process), and
            # git-rm of an untracked path fails silently, leaving the note's blob
            # committed. The note's UUID is part of its filename, so resolve the
            # tracked path by UUID and trust the hint only when it is tracked.
            tracked_paths = {str(entry_path) for entry_path, _stage in repo.index.entries}
            hint = str(path.relative_to(self.base_dir)) if path is not None else None
            if hint is not None and hint in tracked_paths:
                rel_path = hint
            else:
                # Match on the UUID parsed from each filename, not a substring of
                # the whole path: a different note's slug or category could
                # otherwise contain this UUID and be deleted by mistake.
                rel_path = next(
                    (
                        p
                        for p in tracked_paths
                        if extract_uuid_from_filename(p.rsplit("/", 1)[-1]) == note_id
                    ),
                    hint or str(Path("notes") / f"{note_id}.md"),
                )
            try:
                repo.index.remove([rel_path])
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

    @staticmethod
    def _find_note_path_in_tree(commit: Commit, note_id: UUID) -> str | None:
        """
        Tree-wide UUID search: find a note's path within a commit tree.

        Matches markdown files whose filename ends with the note UUID (the
        canonical "{slug}-{uuid}.md" / legacy "{uuid}.md" layouts), so notes
        that merely mention another note's UUID in their slug don't match.

        Args:
            commit: Commit whose tree to search
            note_id: Note UUID

        Returns:
            Repo-relative path string, or None if not found
        """
        for item in commit.tree.traverse():
            if isinstance(item, Blob):
                item_path = str(item.path)
                filename = item_path.rsplit("/", 1)[-1]
                if extract_uuid_from_filename(filename) == note_id:
                    return item_path
        return None

    def _find_last_known_path(self, note_id: UUID) -> str | None:
        """
        Find the most recent path a note existed at by walking history.

        Used when the note no longer exists on disk (e.g. deleted notes) so
        its history can still be located by the path it last had.

        Args:
            note_id: Note UUID

        Returns:
            Repo-relative path string, or None if the note never existed
        """
        repo = self.repo
        if repo is None:
            return None

        try:
            for commit in repo.iter_commits():
                item_path = self._find_note_path_in_tree(commit, note_id)
                if item_path is not None:
                    return item_path
        except Exception as e:
            logger.warning(f"Failed to search history for note {note_id}: {e}")
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

        note_path = self._resolve_history_path(note_id, path)
        if note_path is None:
            return []

        return self._git_log_versions(note_path, limit)

    def _resolve_history_path(self, note_id: UUID, path: Path | None) -> str | None:
        """
        Resolve the repo-relative path to use for history lookups.

        Args:
            note_id: Note UUID
            path: Current path to note file (optional)

        Returns:
            Repo-relative path string, or None if no path could be found
        """
        if path is not None:
            try:
                return str(path.relative_to(self.base_dir))
            except ValueError:
                return str(path)

        # Path unknown (e.g. deleted note): search history for the path
        # the note last existed at so its commits stay discoverable.
        return self._find_last_known_path(note_id)

    def _git_log_versions(self, note_path: str, limit: int) -> list[NoteVersion]:
        """
        Run git log --follow on a path and parse the output into versions.

        Args:
            note_path: Repo-relative path to the note file
            limit: Max versions to return

        Returns:
            List of NoteVersion objects, newest first
        """
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
        except (BadName, GitCommandError, ValueError) as e:
            # GitPython raises a bare ValueError for a syntactically valid hex
            # SHA that resolves to no object, in addition to BadName for a
            # malformed ref (mirrors is_note_deleted_at's handling).
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
        found_path = self._find_note_path_in_tree(commit, note_id)
        if found_path is not None:
            try:
                blob = commit.tree / found_path
                content = blob.data_stream.read().decode("utf-8")
                return content
            except Exception as e:
                logger.debug(f"Failed to read content from {found_path}: {e}")

        # Fallback: legacy path
        note_path = f"notes/{note_id}.md"
        try:
            blob = commit.tree / note_path
            content = blob.data_stream.read().decode("utf-8")
            return content
        except (KeyError, GitCommandError) as e:
            logger.warning(f"Failed to get version content: {e}")
            return None

    def is_note_deleted_at(self, note_id: UUID, commit_sha: str) -> bool:
        """Whether ``commit_sha`` is the commit that deleted this note.

        True when the commit resolves, the note's blob is absent from its tree,
        but is present in its first parent's tree. This is message-independent:
        it does not rely on the "Delete note: ..." commit subject, which a user
        could reproduce in an ordinary note title.

        Lets the tool layer tell "you asked to restore the deletion commit
        itself" (a clear user error) apart from a genuinely unknown version.

        Args:
            note_id: Note UUID
            commit_sha: Git commit SHA advertised by get_history

        Returns:
            True if the commit removed the note, False otherwise
        """
        repo = self.repo
        if repo is None:
            return False

        # GitPython resolves commits lazily, so a bad SHA only raises when the
        # tree/parents are first accessed: force that inside the try.
        try:
            commit = repo.commit(commit_sha)
            present_now = self._find_note_path_in_tree(commit, note_id) is not None
            parents = list(commit.parents)
        except (BadName, GitCommandError, ValueError) as e:
            logger.debug(f"Cannot resolve commit {commit_sha}: {e}")
            return False

        # Blob present in this commit -> it did not delete the note.
        if present_now:
            return False
        # Absent here but present in the first parent -> this commit removed it.
        if not parents:
            return False
        return self._find_note_path_in_tree(parents[0], note_id) is not None

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
            # Note unknown to the store (e.g. deleted): restore to the path
            # it last existed at so the layout stays consistent, falling back
            # to the legacy flat path for notes that predate the nested layout.
            last_known = self._find_last_known_path(note_id)
            if last_known is not None:
                note_path = self.base_dir / last_known
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
