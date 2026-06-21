"""Filesystem operations for notes with nested folder structure."""

import fcntl
import logging
import os
import shutil
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from mcp_notes.constants import normalize_tag
from mcp_notes.models import Note, NoteSummary
from mcp_notes.settings import settings
from mcp_notes.storage.parser import (
    ParsedNote,
    extract_inline_links,
    parse_note,
    serialize_note,
)
from mcp_notes.storage.slugify import (
    build_filename,
    slugify_category_path,
)
from mcp_notes.storage.uuid_index import UUIDIndex

logger = logging.getLogger(__name__)


def _atomic_write(path: Path, content: str, encoding: str = "utf-8") -> None:
    """
    Atomically write content to a file using temp file + rename.

    This prevents data corruption from concurrent writes or crashes mid-write.
    The rename operation is atomic on POSIX systems.
    """
    # Create temp file in same directory (ensures same filesystem for atomic rename)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp", text=True)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())  # Ensure data is written to disk
        # Atomic rename (POSIX guarantees atomicity for same-filesystem rename)
        os.replace(tmp_path, path)
    except Exception as e:
        # Log the actual error before cleanup and re-raise
        logger.warning(f"Atomic write failed for {path}: {e}")
        # Clean up temp file on error
        try:
            os.unlink(tmp_path)
        except OSError as cleanup_err:
            logger.debug(f"Failed to clean up temp file {tmp_path}: {cleanup_err}")
        raise


@contextmanager
def _file_lock(path: Path, timeout: float = 10.0):
    """
    Acquire an exclusive file lock for safe read-modify-write operations.

    Uses POSIX flock() which is released automatically when the process exits
    or the file descriptor is closed.

    Args:
        path: Path to file to lock
        timeout: Max seconds to wait for lock (raises TimeoutError if exceeded)

    Yields:
        None (lock is held while in context)
    """
    import time

    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break  # Lock acquired
            except BlockingIOError:
                if time.monotonic() - start > timeout:
                    raise TimeoutError(
                        f"Could not acquire lock on {path} within {timeout}s"
                    ) from None
                time.sleep(0.05)  # Brief sleep before retry
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
        # Clean up lock file (best effort)
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass


class NoteNotFoundError(Exception):
    """Raised when a note is not found."""

    pass


class PathTraversalError(Exception):
    """Raised when a path traversal attempt is detected."""

    pass


def _safe_rglob(directory: Path, pattern: str) -> Iterator[Path]:
    """
    Safely iterate over files, skipping symlinks and paths outside container.

    Security: Blocks symlink-based path traversal attacks by:
    1. Checking is_symlink() BEFORE following/resolving paths
    2. Verifying resolved paths stay within the container directory

    Args:
        directory: Base directory to search
        pattern: Glob pattern (e.g., "*.md")

    Yields:
        Safe, validated paths only (no symlinks, within container)
    """
    if not directory.exists():
        return

    container_resolved = directory.resolve()

    for path in directory.rglob(pattern):
        # Security: Skip symlinks entirely (prevents symlink-based escapes)
        if path.is_symlink():
            logger.debug(f"Skipping symlink: {path}")
            continue

        # Security: Verify resolved path stays within container
        try:
            resolved = path.resolve()
            if not resolved.is_relative_to(container_resolved):
                logger.warning(f"Skipping path outside container: {path}")
                continue
        except OSError:
            logger.debug(f"Skipping unresolvable path: {path}")
            continue

        yield path


class NoteTooLargeError(Exception):
    """Raised when a note exceeds the maximum allowed size."""

    def __init__(self, path: Path | str, size_kb: int, max_kb: int):
        self.path = path
        self.size_kb = size_kb
        self.max_kb = max_kb
        super().__init__(
            f"Note too large: {path} is {size_kb} KB (max {max_kb} KB). "
            f"Increase NOTES_MAX_NOTE_SIZE_KB to allow larger notes."
        )


class NoteParseError:
    """Details about a failed note parse."""

    def __init__(self, path: Path, error: Exception):
        self.path = path
        self.error = error
        self.error_type = type(error).__name__
        self.message = str(error)

    def __repr__(self) -> str:
        return f"NoteParseError(path={self.path}, error={self.error_type}: {self.message})"


class NoteStore:
    """
    Filesystem-based note storage with nested folder structure.

    Notes are stored as: $NOTES_DIR/notes/{category_path}/{slug}-{uuid}.md
    Category is derived from folder path, not frontmatter.
    """

    def __init__(self, notes_dir: Path | None = None):
        """
        Initialize note store.

        Args:
            notes_dir: Notes directory (defaults to settings.dir)
        """
        self.base_dir = Path(notes_dir or settings.dir)
        self.notes_dir = self.base_dir / "notes"
        self._locks_dir = self.base_dir / ".locks"
        # Track parsing failures for visibility (protected by lock)
        self._last_parse_errors: list[NoteParseError] = []
        self._parse_errors_lock = threading.Lock()
        # UUID index for fast lookups
        self._uuid_index: UUIDIndex | None = None

    @property
    def uuid_index(self) -> UUIDIndex:
        """Get or create UUID index."""
        if self._uuid_index is None:
            self._uuid_index = UUIDIndex(self.base_dir)
            self._uuid_index.ensure_loaded()
        return self._uuid_index

    def ensure_directories(self) -> None:
        """Ensure required directories exist and index is initialized."""
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        (self.base_dir / ".index").mkdir(exist_ok=True)
        self._locks_dir.mkdir(exist_ok=True)
        # Ensure UUID index is loaded/rebuilt
        self.uuid_index.ensure_loaded()
        # Clean up stale lock files on startup (lazy cleanup)
        self._cleanup_stale_note_locks()

    def _get_note_lock_path(self, note_id: UUID) -> Path:
        """
        Get lock file path for a note based on its UUID (stable identifier).

        Using UUID-based locks (instead of file-path-based locks) prevents race
        conditions where the file path changes between reading the UUID index
        and acquiring the lock.
        """
        self._locks_dir.mkdir(exist_ok=True)
        return self._locks_dir / f"{note_id}.lock"

    def _cleanup_stale_note_locks(self) -> None:
        """
        Clean up stale lock files on startup.

        Lock files older than 1 hour are considered abandoned (from crashed processes).
        """
        import time

        if not self._locks_dir.exists():
            return

        stale_threshold = 3600  # 1 hour
        now = time.time()

        for lock_file in self._locks_dir.glob("*.lock"):
            try:
                mtime = lock_file.stat().st_mtime
                if now - mtime > stale_threshold:
                    lock_file.unlink()
                    logger.debug(f"Cleaned up stale lock file: {lock_file}")
            except OSError:
                pass  # Ignore errors during cleanup

    def _note_path(self, note_id: UUID) -> Path:
        """
        Get path for a note file using UUID index.

        Raises:
            NoteNotFoundError: If note not in index
            PathTraversalError: If path is a symlink or escapes notes_dir
        """
        path = self.uuid_index.get_path(note_id)
        if path is None:
            raise NoteNotFoundError(f"Note not found in index: {note_id}")

        # Security: Reject symlinks BEFORE resolution (prevents symlink-based escapes)
        if path.is_symlink():
            logger.warning(f"Symlink detected in notes directory: {path}")
            raise PathTraversalError(f"Symlinks not allowed: {path}")

        # Defense-in-depth: verify resolved path is within notes_dir
        try:
            resolved = path.resolve(strict=False)
        except OSError as e:
            raise PathTraversalError(f"Cannot resolve path {path}: {e}") from e

        if not resolved.is_relative_to(self.notes_dir.resolve()):
            raise PathTraversalError(
                f"Path traversal detected: {path} escapes {self.notes_dir}"
            )
        return path

    def _build_note_path(
        self,
        note_id: UUID,
        title: str,
        category: str | None,
    ) -> Path:
        """
        Build full path for a note from its components.

        Args:
            note_id: Note UUID
            title: Note title (for slug generation)
            category: Category path or None for root

        Returns:
            Full path like notes/work/projects/my-note-{uuid}.md
        """
        filename = build_filename(title, note_id)

        if category:
            # Slugify category path segments
            category_path = slugify_category_path(category)
            return self.notes_dir / category_path / filename
        else:
            # Root level note
            return self.notes_dir / filename

    def _category_from_path(self, path: Path) -> str | None:
        """
        Derive category from file path.

        Args:
            path: Full path to note file

        Returns:
            Category string or None if in root
        """
        try:
            rel_path = path.relative_to(self.notes_dir)
        except ValueError:
            return None

        # Category is the directory part
        parent = rel_path.parent
        if parent == Path("."):
            return None
        return str(parent)

    def _cleanup_empty_folders(self, folder: Path) -> None:
        """
        Remove empty category folders up to notes_dir.

        Args:
            folder: Starting folder to check
        """
        current = folder
        while current != self.notes_dir and current.is_relative_to(self.notes_dir):
            try:
                if current.exists() and current.is_dir():
                    # Check if empty
                    if not any(current.iterdir()):
                        current.rmdir()
                        logger.debug(f"Removed empty folder: {current}")
                    else:
                        break  # Not empty, stop
            except OSError:
                break  # Permission error or similar, stop

            current = current.parent

    def exists(self, note_id: UUID) -> bool:
        """Check if a note exists."""
        return self.uuid_index.exists(note_id)

    def create(
        self,
        title: str,
        content: str,
        tags: list[str] | None = None,
        category: str | None = None,
        note_id: UUID | None = None,
    ) -> Note:
        """
        Create a new note.

        Args:
            title: Note title
            content: Note body content
            tags: Optional tags
            category: Optional category (determines folder location)
            note_id: Optional UUID (generated if not provided)

        Returns:
            Created Note
        """
        self.ensure_directories()

        note_id = note_id or uuid4()
        now = datetime.now(UTC)

        # Normalize tags through the single source of truth, deduplicating
        # canonical collisions ("My Tag" and "my-tag" both map to "my-tag") so
        # the stored/returned tags match a subsequent read (which also dedups).
        if tags:
            tags = list(dict.fromkeys(n for n in (normalize_tag(t) for t in tags) if n))

        # Extract inline links from content
        inline_links = extract_inline_links(content)

        # Serialize WITHOUT category in frontmatter
        file_content = serialize_note(
            note_id=note_id,
            title=title,
            body=content,
            tags=tags,
            category=None,  # Category is derived from path, not stored
            links=inline_links if inline_links else None,
            created=now,
            modified=now,
        )

        # Build path based on category
        path = self._build_note_path(note_id, title, category)

        # Ensure category directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        _atomic_write(path, file_content)

        # Update UUID index
        self.uuid_index.add(note_id, path)
        self.uuid_index.save()

        logger.info(f"Created note: {note_id} - {title} at {path}")

        return Note(
            id=note_id,
            title=title,
            content=file_content,
            tags=tags or [],
            category=category,
            links=inline_links,
            created=now,
            modified=now,
        )

    def read(self, note_id: UUID) -> Note:
        """
        Read a note by ID.

        Args:
            note_id: Note UUID

        Returns:
            Note with category derived from path

        Raises:
            NoteNotFoundError: If note doesn't exist
            NoteTooLargeError: If note exceeds max_note_size_kb
        """
        path = self._note_path(note_id)
        if not path.exists():
            # Path in index but file missing - thread-safe rebuild
            self.uuid_index.rebuild_if_path_missing(note_id)
            raise NoteNotFoundError(f"Note not found: {note_id}")

        # Check file size before reading into memory
        size_kb = path.stat().st_size // 1024
        if size_kb > settings.max_note_size_kb:
            raise NoteTooLargeError(path, size_kb, settings.max_note_size_kb)

        content = path.read_text(encoding="utf-8")
        parsed = parse_note(content)

        # Derive category from path
        category = self._category_from_path(path)

        return Note(
            id=parsed.id,
            title=parsed.title,
            content=parsed.content,
            tags=parsed.tags,
            category=category,  # From path, not frontmatter
            links=parsed.links,
            created=parsed.created,
            modified=parsed.modified,
        )

    def update(
        self,
        note_id: UUID,
        title: str | None = None,
        content: str | None = None,
        tags: list[str] | None = None,
        category: str | None = None,
    ) -> Note:
        """
        Update an existing note.

        If title or category changes, the file will be moved to a new location.
        Uses UUID-based locking to prevent race conditions from concurrent updates.

        Args:
            note_id: Note UUID
            title: New title (optional, triggers rename if changed)
            content: New body content (optional)
            tags: New tags (optional, pass empty list to clear)
            category: New category (optional, triggers move if changed)

        Returns:
            Updated Note

        Raises:
            NoteNotFoundError: If note doesn't exist
            TimeoutError: If unable to acquire file lock
        """
        # Lock on UUID (stable) BEFORE reading path (unstable) to prevent race condition
        # where another thread moves the note between reading the path and acquiring the lock
        note_lock_path = self._get_note_lock_path(note_id)
        with _file_lock(note_lock_path):
            # NOW read path from UUID index (inside lock, so path is stable)
            old_path = self._note_path(note_id)
            if not old_path.exists():
                # Path in index but file missing - thread-safe rebuild
                self.uuid_index.rebuild_if_path_missing(note_id)
                raise NoteNotFoundError(f"Note not found: {note_id}")
            existing_content = old_path.read_text(encoding="utf-8")
            parsed = parse_note(existing_content)
            old_category = self._category_from_path(old_path)

            # Determine new values
            new_title = title if title is not None else parsed.title
            new_body = content if content is not None else parsed.body

            if tags is not None:
                new_tags = list(
                    dict.fromkeys(n for n in (normalize_tag(t) for t in tags) if n)
                )
            else:
                new_tags = parsed.tags

            # Handle category change (None means keep existing, "" means move to root)
            if category is not None:
                new_category = category if category else None
            else:
                new_category = old_category

            # Extract inline links from the body. When the body content is not
            # being replaced (content is None), also preserve any frontmatter
            # `links` that were not authored inline -- e.g. on a hand-edited or
            # imported note. The read side treats frontmatter links as live
            # edges, so rebuilding from the body alone would silently drop them
            # on an unrelated tags/title/category update.
            inline_links = extract_inline_links(new_body)
            if content is None:
                inline_links = list(dict.fromkeys(inline_links + parsed.links))

            # Update modified timestamp
            now = datetime.now(UTC)

            # Serialize WITHOUT category
            file_content = serialize_note(
                note_id=note_id,
                title=new_title,
                body=new_body,
                tags=new_tags if new_tags else None,
                category=None,  # Category derived from path
                links=inline_links if inline_links else None,
                created=parsed.created,
                modified=now,
            )

            # Check if we need to move the file
            new_path = self._build_note_path(note_id, new_title, new_category)
            needs_move = old_path != new_path

            if needs_move:
                # Ensure new directory exists
                new_path.parent.mkdir(parents=True, exist_ok=True)

                # Write to new location
                _atomic_write(new_path, file_content)

                # Delete old file
                old_path.unlink()

                # Cleanup empty folders
                self._cleanup_empty_folders(old_path.parent)

                # Update index
                self.uuid_index.add(note_id, new_path)
                self.uuid_index.save()

                logger.info(f"Moved note: {note_id} from {old_path} to {new_path}")
            else:
                # Update in place
                _atomic_write(old_path, file_content)
                logger.info(f"Updated note: {note_id} - {new_title}")

            return Note(
                id=note_id,
                title=new_title,
                content=file_content,
                tags=new_tags,
                category=new_category,
                links=inline_links,
                created=parsed.created,
                modified=now,
            )

    def delete(self, note_id: UUID) -> bool:
        """
        Delete a note.

        Uses UUID-based locking to prevent race conditions with concurrent updates.

        Args:
            note_id: Note UUID

        Returns:
            True if deleted

        Raises:
            NoteNotFoundError: If note doesn't exist
        """
        # Lock on UUID (stable) BEFORE reading path (unstable) to prevent race condition
        note_lock_path = self._get_note_lock_path(note_id)
        with _file_lock(note_lock_path):
            path = self._note_path(note_id)
            if not path.exists():
                # Path in index but file missing - thread-safe rebuild
                self.uuid_index.rebuild_if_path_missing(note_id)
                raise NoteNotFoundError(f"Note not found: {note_id}")

            parent = path.parent

            # Delete file
            path.unlink()

            # Remove from index
            self.uuid_index.remove(note_id)
            self.uuid_index.save()

            # Cleanup empty folders
            self._cleanup_empty_folders(parent)

            logger.info(f"Deleted note: {note_id}")
            return True

    def list_all(self) -> list[NoteSummary]:
        """
        List all notes as summaries.

        Parse errors are tracked and can be retrieved via get_parse_errors().
        Symlinks are skipped for security (prevents path traversal attacks).

        Returns:
            List of NoteSummary objects
        """
        if not self.notes_dir.exists():
            with self._parse_errors_lock:
                self._last_parse_errors = []
            return []

        summaries = []
        parse_errors: list[NoteParseError] = []

        # Security: Use _safe_rglob to skip symlinks and escaped paths
        for path in _safe_rglob(self.notes_dir, "*.md"):
            try:
                content = path.read_text(encoding="utf-8")
                parsed = parse_note(content)
                category = self._category_from_path(path)
                summaries.append(self._to_summary(parsed, category))
            except Exception as e:
                parse_error = NoteParseError(path, e)
                parse_errors.append(parse_error)
                logger.warning(f"Failed to parse {path}: {e}")

        with self._parse_errors_lock:
            self._last_parse_errors = parse_errors
        if parse_errors:
            logger.error(
                f"Failed to parse {len(parse_errors)} note(s). "
                f"Use get_parse_errors() for details."
            )

        return summaries

    def iter_all(self) -> Iterator[tuple[ParsedNote, str | None]]:
        """
        Iterate over all notes as (ParsedNote, category) tuples.

        Note: This method also updates _last_parse_errors. After iteration
        completes, call get_parse_errors() for any failures.
        Symlinks are skipped for security (prevents path traversal attacks).

        Yields:
            Tuple of (ParsedNote, category) for each note file
        """
        if not self.notes_dir.exists():
            with self._parse_errors_lock:
                self._last_parse_errors = []
            return

        parse_errors: list[NoteParseError] = []

        # Security: Use _safe_rglob to skip symlinks and escaped paths
        for path in _safe_rglob(self.notes_dir, "*.md"):
            try:
                content = path.read_text(encoding="utf-8")
                parsed = parse_note(content)
                category = self._category_from_path(path)
                yield (parsed, category)
            except Exception as e:
                parse_error = NoteParseError(path, e)
                parse_errors.append(parse_error)
                logger.warning(f"Failed to parse {path}: {e}")

        with self._parse_errors_lock:
            self._last_parse_errors = parse_errors
        if parse_errors:
            logger.error(
                f"Failed to parse {len(parse_errors)} note(s). "
                f"Use get_parse_errors() for details."
            )

    def get_parse_errors(self) -> list[NoteParseError]:
        """
        Get parse errors from the last list_all() or iter_all() operation.

        Returns:
            List of NoteParseError objects with details about failures
        """
        with self._parse_errors_lock:
            return self._last_parse_errors.copy()

    def has_parse_errors(self) -> bool:
        """Check if there were parse errors in the last listing operation."""
        with self._parse_errors_lock:
            return len(self._last_parse_errors) > 0

    def _to_summary(self, parsed: ParsedNote, category: str | None) -> NoteSummary:
        """Convert ParsedNote to NoteSummary with category from path."""
        excerpt = parsed.body[:settings.excerpt_length]
        if len(parsed.body) > settings.excerpt_length:
            # Try to break at word boundary
            last_space = excerpt.rfind(" ")
            if last_space > settings.excerpt_length * 0.7:
                excerpt = excerpt[:last_space] + "..."
            else:
                excerpt += "..."

        return NoteSummary(
            id=parsed.id,
            title=parsed.title,
            tags=parsed.tags,
            category=category,  # From path, not frontmatter
            created=parsed.created,
            modified=parsed.modified,
            excerpt=excerpt,
        )

    def get_summary(self, note_id: UUID) -> NoteSummary:
        """Get summary for a single note."""
        path = self._note_path(note_id)
        if not path.exists():
            # Path in index but file missing (external delete, or a git
            # checkout/pull that moved or removed it). Rebuild the index, then
            # re-resolve: a moved/renamed file is repointed to its new path and
            # read normally, while a genuinely deleted one raises
            # NoteNotFoundError (from _note_path or the re-check below). Without
            # this guard a dangling outgoing link raises a bare FileNotFoundError
            # that crashes the whole get_note_links view instead of being
            # reported as broken.
            self.uuid_index.rebuild_if_path_missing(note_id)
            path = self._note_path(note_id)
            if not path.exists():
                raise NoteNotFoundError(f"Note not found: {note_id}")
        content = path.read_text(encoding="utf-8")
        parsed = parse_note(content)
        category = self._category_from_path(path)
        return self._to_summary(parsed, category)

    def count(self) -> int:
        """Count total notes (excluding symlinks for security)."""
        if not self.notes_dir.exists():
            return 0
        # Security: Use _safe_rglob to skip symlinks
        return sum(1 for _ in _safe_rglob(self.notes_dir, "*.md"))

    def get_note_path(self, note_id: UUID) -> Path | None:
        """
        Get the filesystem path for a note.

        Useful for git operations that need the actual path.

        Args:
            note_id: Note UUID

        Returns:
            Path to note file, or None if not found
        """
        return self.uuid_index.get_path(note_id)

    def move_note(
        self,
        note_id: UUID,
        new_category: str | None,
        new_title: str | None = None,
    ) -> tuple[Path, Path]:
        """
        Move a note to a new category and/or rename it.

        This is a low-level operation for git integration.
        Prefer using update() which handles all aspects.

        Args:
            note_id: Note UUID
            new_category: New category (None for root)
            new_title: New title for slug (None to keep current)

        Returns:
            Tuple of (old_path, new_path)

        Raises:
            NoteNotFoundError: If note doesn't exist
        """
        old_path = self._note_path(note_id)
        if not old_path.exists():
            raise NoteNotFoundError(f"Note not found: {note_id}")

        # Get current title if not provided
        if new_title is None:
            content = old_path.read_text(encoding="utf-8")
            parsed = parse_note(content)
            new_title = parsed.title

        new_path = self._build_note_path(note_id, new_title, new_category)

        if old_path == new_path:
            return (old_path, new_path)

        # Ensure directory exists
        new_path.parent.mkdir(parents=True, exist_ok=True)

        # Move file
        shutil.move(str(old_path), str(new_path))

        # Update index
        self.uuid_index.add(note_id, new_path)
        self.uuid_index.save()

        # Cleanup empty folders
        self._cleanup_empty_folders(old_path.parent)

        return (old_path, new_path)
