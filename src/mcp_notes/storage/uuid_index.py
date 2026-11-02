"""UUID-to-path index for fast note lookups."""

import json
import logging
import os
import threading
from collections.abc import Iterable
from pathlib import Path
from uuid import UUID

from mcp_notes.storage.fileops import atomic_write, file_lock
from mcp_notes.storage.slugify import extract_uuid_from_filename

logger = logging.getLogger(__name__)

# Index filename stored in .index directory
INDEX_FILENAME = "uuid_paths.json"

# Identity of one version of the index file: (device, inode, mtime_ns, ctime_ns, size)
_Signature = tuple[int, int, int, int, int]

# One change to the map: (UUID string, relative path), or (UUID string, None) to remove
_Change = tuple[str, str | None]


def _signature(st: os.stat_result) -> _Signature:
    return (st.st_dev, st.st_ino, st.st_mtime_ns, st.st_ctime_ns, st.st_size)


def _apply_changes(mapping: dict[str, str], changes: Iterable[_Change]) -> bool:
    """Apply changes to mapping in order; return whether anything differed."""
    changed = False
    for uuid_str, rel_path in changes:
        if rel_path is None:
            if uuid_str in mapping:
                del mapping[uuid_str]
                changed = True
        elif mapping.get(uuid_str) != rel_path:
            mapping[uuid_str] = rel_path
            changed = True
    return changed


class UUIDIndex:
    """
    Maintains UUID-to-path mapping for fast link resolution.

    The index is stored as JSON in .index/uuid_paths.json and maps
    note UUIDs to their relative paths within the notes directory.

    Several processes may share one index file. Every change (add, remove,
    rebuild) is a single transaction under an exclusive flock on
    .index/uuid_paths.json.lock: re-read the map on disk, apply only this
    change, write it atomically (temp file in the same directory, fsync,
    os.replace, fsync the directory), and adopt the result in memory. A
    process therefore never overwrites another's changes with a stale view,
    and readers never see a partially written file.

    Lookups are served from memory. A lookup that misses re-reads the file
    first if it has changed since this instance last loaded or wrote it,
    judged by stat alone, so notes created by another process are found
    without a filesystem scan. Hits are not re-checked: a path another
    process moved is caught by rebuild_if_path_missing when the stale path
    turns out not to exist.
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

        # In-memory cache: UUID string -> relative path string. Replaced
        # wholesale, never mutated in place, so lock-free readers always see a
        # complete map.
        self._index: dict[str, str] = {}
        # Changes applied in memory whose write failed; replayed by the next
        # transaction or save()
        self._pending: list[_Change] = []
        # Index file as of this instance's last load or write; None if unknown
        self._signature: _Signature | None = None
        # RLock for in-process safety (allows nested calls like rebuild_if_path_missing->rebuild).
        # Cross-process exclusion is the file lock taken by each transaction.
        self._lock = threading.RLock()

    def _read_file(self) -> tuple[dict[str, str], _Signature] | None:
        """
        Read the index file and the signature of the version read.

        Returns:
            (map, signature), or None if the file is missing or unreadable
        """
        try:
            with open(self.index_path, encoding="utf-8") as f:
                # fstat the descriptor actually read: a replace after open
                # cannot pair this content with a newer file's signature
                signature = _signature(os.fstat(f.fileno()))
                data = json.load(f)
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as e:
            logger.warning(f"Failed to load UUID index: {e}")
            return None
        mapping = data.get("uuid_paths", {}) if isinstance(data, dict) else None
        if not isinstance(mapping, dict):
            logger.warning(f"Failed to load UUID index: unexpected structure in {self.index_path}")
            return None
        return mapping, signature

    def _write(self, mapping: dict[str, str]) -> None:
        """Atomically replace the index file with mapping. Caller holds the file lock."""
        atomic_write(self.index_path, json.dumps({"uuid_paths": mapping}, indent=2), sync_dir=True)
        self._signature = _signature(os.stat(self.index_path))
        logger.debug(f"Saved UUID index with {len(mapping)} entries")

    def _commit(self, changes: list[_Change]) -> None:
        """
        Persist changes in one transaction under the cross-process file lock.

        The map is re-read from disk while the lock is held, so changes other
        processes committed since this instance last looked are kept. If the
        file is missing or unreadable, the in-memory map stands in for it. If
        locking or writing fails, the changes are still applied in memory and
        kept pending for the next transaction or save().
        """
        with self._lock:
            changes = [*self._pending, *changes]
            try:
                with file_lock(self.index_path):
                    loaded = self._read_file()
                    if loaded is None:
                        # Rewrite even if nothing changed, to restore a valid file
                        mapping = dict(self._index)
                        _apply_changes(mapping, changes)
                        self._write(mapping)
                    else:
                        mapping, signature = loaded
                        if _apply_changes(mapping, changes):
                            self._write(mapping)
                        else:
                            self._signature = signature
            except OSError as e:
                logger.error(f"Failed to save UUID index: {e}")
                mapping = dict(self._index)
                _apply_changes(mapping, changes)
                self._pending = changes
            else:
                self._pending = []
            self._index = mapping

    def _reload_if_changed(self) -> bool:
        """
        Re-read the index file if it changed since this instance last loaded or wrote it.

        Costs one stat when nothing changed. A missing or unreadable file
        leaves the in-memory map as it is, and an unreadable version is not
        read again until the file changes once more.

        Returns:
            True if a newer map was loaded
        """
        with self._lock:
            try:
                current = _signature(os.stat(self.index_path))
            except OSError:
                return False
            if current == self._signature:
                return False
            loaded = self._read_file()
            if loaded is None:
                self._signature = current
                return False
            mapping, self._signature = loaded
            _apply_changes(mapping, self._pending)
            self._index = mapping
            return True

    def load(self) -> None:
        """Load index from disk (thread-safe); empty if the file is missing or unreadable."""
        with self._lock:
            loaded = self._read_file()
            mapping, self._signature = loaded if loaded else ({}, None)
            _apply_changes(mapping, self._pending)
            self._index = mapping
            logger.debug(f"Loaded UUID index with {len(mapping)} entries")

    def save(self) -> None:
        """
        Retry persisting changes whose earlier write failed (thread-safe).

        add(), remove() and rebuild() persist themselves, so this does nothing
        unless one of those writes failed.
        """
        with self._lock:
            if self._pending:
                self._commit([])

    def _scan(self) -> tuple[dict[str, str], int]:
        """Map every note file under notes_dir by the UUID in its filename."""
        mapping: dict[str, str] = {}
        count = 0
        if not self.notes_dir.exists():
            return mapping, count

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
                mapping[str(note_id)] = str(rel_path)
                count += 1
        return mapping, count

    def rebuild(self) -> int:
        """
        Scan filesystem and rebuild entire index (thread- and process-safe).

        The filesystem is the source of truth: the scan replaces the whole map
        and any pending changes. It runs under the same file lock as every
        other change, so a change another process commits cannot fall between
        the scan and the write and be lost. If the lock cannot be taken, the
        scan still replaces the in-memory map but is not written.

        Security: Symlinks are explicitly skipped to prevent path traversal attacks.

        Returns:
            Number of notes indexed
        """
        with self._lock:
            locked = False
            try:
                with file_lock(self.index_path):
                    locked = True
                    mapping, count = self._scan()
                    try:
                        self._write(mapping)
                    except OSError as e:
                        logger.error(f"Failed to save UUID index: {e}")
            except OSError as e:
                if locked:
                    raise
                logger.error(f"Failed to lock UUID index for rebuild: {e}")
                mapping, count = self._scan()
            self._pending = []
            self._index = mapping
            logger.info(f"Rebuilt UUID index: {count} notes indexed")
            return count

    def rebuild_if_path_missing(self, note_id: UUID) -> bool:
        """
        Thread-safe rebuild if the given note's path is missing from filesystem.

        This method prevents race conditions where multiple threads simultaneously
        detect a missing file and trigger concurrent rebuilds. Uses RLock to ensure
        only one rebuild occurs. If another process has since written the index,
        it is re-read first, and no scan is done when that explains the missing
        path (the note was moved or removed there).

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
            if self._reload_if_changed():
                path = self.get_path(note_id)
                if path is None or path.exists():
                    return False
            # File missing - rebuild (RLock allows nested call to rebuild())
            self.rebuild()
            return True

    def get_path(self, note_id: UUID) -> Path | None:
        """
        Look up path by UUID.

        A miss re-reads the index file if another process has written it.

        Args:
            note_id: Note UUID

        Returns:
            Full path to note file, or None if not found
        """
        rel_path = self.get_relative_path(note_id)
        if rel_path:
            return self.notes_dir / rel_path
        return None

    def get_relative_path(self, note_id: UUID) -> str | None:
        """
        Look up relative path by UUID.

        A miss re-reads the index file if another process has written it.

        Args:
            note_id: Note UUID

        Returns:
            Relative path from notes_dir, or None if not found
        """
        key = str(note_id)
        rel_path = self._index.get(key)
        if rel_path is None and self._reload_if_changed():
            rel_path = self._index.get(key)
        return rel_path

    def _find_uuid(self, rel_path: str) -> UUID | None:
        for uuid_str, stored_path in self._index.items():
            if stored_path == rel_path:
                try:
                    return UUID(uuid_str)
                except ValueError:
                    return None
        return None

    def get_uuid(self, path: Path) -> UUID | None:
        """
        Reverse lookup: get UUID from path.

        A miss re-reads the index file if another process has written it.

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

        found = self._find_uuid(rel_path)
        if found is None and self._reload_if_changed():
            found = self._find_uuid(rel_path)
        return found

    def add(self, note_id: UUID, path: Path) -> None:
        """
        Add or update a UUID-to-path mapping and persist it (thread- and process-safe).

        Args:
            note_id: Note UUID
            path: Full path to note file
        """
        try:
            rel_path = path.relative_to(self.notes_dir)
        except ValueError:
            logger.error(f"Path {path} is not within notes_dir {self.notes_dir}")
            return

        self._commit([(str(note_id), str(rel_path))])

    def remove(self, note_id: UUID) -> None:
        """
        Remove the mapping for a UUID and persist it (thread- and process-safe).

        The on-disk map is consulted even if this instance does not know the
        UUID, so an entry another process added is removed too.

        Args:
            note_id: Note UUID to remove
        """
        self._commit([(str(note_id), None)])

    def exists(self, note_id: UUID) -> bool:
        """
        Check if UUID exists in index.

        A miss re-reads the index file if another process has written it.

        Args:
            note_id: Note UUID

        Returns:
            True if UUID is in index
        """
        key = str(note_id)
        return key in self._index or (self._reload_if_changed() and key in self._index)

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
