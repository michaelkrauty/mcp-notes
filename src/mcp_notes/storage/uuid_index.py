"""UUID-to-path index for fast note lookups."""

import json
import logging
import threading
from pathlib import Path
from uuid import UUID

from mcp_notes.storage.slugify import extract_uuid_from_filename

logger = logging.getLogger(__name__)

# Index filename stored in .index directory
INDEX_FILENAME = "uuid_paths.json"


class UUIDIndex:
    """
    Maintains UUID-to-path mapping for fast link resolution.

    The index is stored as JSON in .index/uuid_paths.json and maps
    note UUIDs to their relative paths within the notes directory.
    """

    def __init__(self, base_dir: Path):
        """
        Initialize UUID index.

        Args:
            base_dir: Base notes directory (contains notes/ and .index/)
        """
        self.base_dir = base_dir
        self.notes_dir = base_dir / "notes"
        self.index_dir = base_dir / ".index"
        self.index_path = self.index_dir / INDEX_FILENAME

        # In-memory cache: UUID string -> relative path string
        self._index: dict[str, str] = {}
        self._dirty = False
        # RLock for thread-safe index operations (RLock allows nested calls like rebuild->save)
        self._lock = threading.RLock()

    def load(self) -> None:
        """Load index from disk (thread-safe)."""
        with self._lock:
            if not self.index_path.exists():
                self._index = {}
                return

            try:
                with open(self.index_path, encoding="utf-8") as f:
                    data = json.load(f)
                    self._index = data.get("uuid_paths", {})
                logger.debug(f"Loaded UUID index with {len(self._index)} entries")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load UUID index: {e}")
                self._index = {}

    def save(self) -> None:
        """Persist index to disk (thread-safe)."""
        with self._lock:
            if not self._dirty:
                return

            self.index_dir.mkdir(parents=True, exist_ok=True)

            try:
                with open(self.index_path, "w", encoding="utf-8") as f:
                    json.dump({"uuid_paths": self._index}, f, indent=2)
                self._dirty = False
                logger.debug(f"Saved UUID index with {len(self._index)} entries")
            except OSError as e:
                logger.error(f"Failed to save UUID index: {e}")

    def rebuild(self) -> int:
        """
        Scan filesystem and rebuild entire index (thread-safe).

        Security: Symlinks are explicitly skipped to prevent path traversal attacks.

        Returns:
            Number of notes indexed
        """
        with self._lock:
            self._index = {}

            if not self.notes_dir.exists():
                self._dirty = True
                self.save()
                return 0

            count = 0
            notes_dir_resolved = self.notes_dir.resolve()

            for md_file in self.notes_dir.rglob("*.md"):
                # Security: Skip symlinks entirely (prevents symlink-based escapes)
                if md_file.is_symlink():
                    logger.debug(f"Skipping symlink during index rebuild: {md_file}")
                    continue

                # Security: Verify path is within notes_dir
                try:
                    if not md_file.resolve().is_relative_to(notes_dir_resolved):
                        logger.warning(f"Skipping path outside notes_dir: {md_file}")
                        continue
                except OSError:
                    logger.debug(f"Skipping unresolvable path: {md_file}")
                    continue

                note_id = extract_uuid_from_filename(md_file.name)
                if note_id:
                    rel_path = md_file.relative_to(self.notes_dir)
                    self._index[str(note_id)] = str(rel_path)
                    count += 1

            self._dirty = True
            self.save()
            logger.info(f"Rebuilt UUID index: {count} notes indexed")
            return count

    def rebuild_if_path_missing(self, note_id: UUID) -> bool:
        """
        Thread-safe rebuild if the given note's path is missing from filesystem.

        This method prevents race conditions where multiple threads simultaneously
        detect a missing file and trigger concurrent rebuilds. Uses RLock to ensure
        only one rebuild occurs.

        Args:
            note_id: UUID of note to check

        Returns:
            True if rebuild was performed, False if not needed
        """
        with self._lock:
            path = self.get_path(note_id)
            if path is None:
                # Not in index - no rebuild needed
                return False
            if path.exists():
                # File exists - no rebuild needed
                return False
            # File missing - rebuild (RLock allows nested call to rebuild())
            self.rebuild()
            return True

    def get_path(self, note_id: UUID) -> Path | None:
        """
        Look up path by UUID.

        Args:
            note_id: Note UUID

        Returns:
            Full path to note file, or None if not found
        """
        rel_path = self._index.get(str(note_id))
        if rel_path:
            return self.notes_dir / rel_path
        return None

    def get_relative_path(self, note_id: UUID) -> str | None:
        """
        Look up relative path by UUID.

        Args:
            note_id: Note UUID

        Returns:
            Relative path from notes_dir, or None if not found
        """
        return self._index.get(str(note_id))

    def get_uuid(self, path: Path) -> UUID | None:
        """
        Reverse lookup: get UUID from path.

        Args:
            path: Full or relative path to note file

        Returns:
            Note UUID, or None if not found
        """
        # Normalize to relative path
        try:
            if path.is_absolute():
                rel_path = str(path.relative_to(self.notes_dir))
            else:
                rel_path = str(path)
        except ValueError:
            return None

        # Search for matching path
        for uuid_str, stored_path in self._index.items():
            if stored_path == rel_path:
                try:
                    return UUID(uuid_str)
                except ValueError:
                    return None

        return None

    def add(self, note_id: UUID, path: Path) -> None:
        """
        Add or update UUID-to-path mapping (thread-safe).

        Args:
            note_id: Note UUID
            path: Full path to note file
        """
        try:
            rel_path = path.relative_to(self.notes_dir)
        except ValueError:
            logger.error(f"Path {path} is not within notes_dir {self.notes_dir}")
            return

        with self._lock:
            self._index[str(note_id)] = str(rel_path)
            self._dirty = True

    def remove(self, note_id: UUID) -> None:
        """
        Remove mapping for UUID (thread-safe).

        Args:
            note_id: Note UUID to remove
        """
        with self._lock:
            uuid_str = str(note_id)
            if uuid_str in self._index:
                del self._index[uuid_str]
                self._dirty = True

    def exists(self, note_id: UUID) -> bool:
        """
        Check if UUID exists in index.

        Args:
            note_id: Note UUID

        Returns:
            True if UUID is in index
        """
        return str(note_id) in self._index

    def count(self) -> int:
        """Return number of indexed notes."""
        return len(self._index)

    def all_uuids(self) -> list[UUID]:
        """Return all indexed UUIDs."""
        result = []
        for uuid_str in self._index:
            try:
                result.append(UUID(uuid_str))
            except ValueError:
                continue
        return result

    def ensure_loaded(self) -> None:
        """Ensure index is loaded, rebuilding if empty or missing (thread-safe)."""
        with self._lock:
            if not self._index:
                self.load()
                if not self._index and self.notes_dir.exists():
                    # Index file missing/empty but notes exist - rebuild
                    self.rebuild()
