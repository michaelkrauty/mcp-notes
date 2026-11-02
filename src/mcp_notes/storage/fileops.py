"""Atomic file writes and cross-process file locks shared by the storage layer."""

import fcntl
import logging
import os
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


def _fsync_directory(directory: Path) -> None:
    """Flush a directory's entries to disk, so a rename into it survives a crash."""
    try:
        fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError as e:
        logger.debug(f"Cannot open {directory} to fsync it: {e}")
        return
    try:
        os.fsync(fd)
    except OSError as e:
        # Some filesystems refuse fsync on a directory; the rename itself is
        # still atomic there, only its durability is weaker.
        logger.debug(f"Cannot fsync directory {directory}: {e}")
    finally:
        os.close(fd)


def atomic_write(
    path: Path, content: str, encoding: str = "utf-8", *, sync_dir: bool = False
) -> None:
    """
    Atomically write content to a file using temp file + rename.

    This prevents data corruption from concurrent writes or crashes mid-write.
    The rename operation is atomic on POSIX systems, so a reader sees either
    the old file or the new one, never a partial write. The temp file is
    created in the target's directory and removed again if any step fails.

    Args:
        path: File to write
        content: Text to write
        encoding: Text encoding
        sync_dir: Also fsync the parent directory after the rename, so the
            new directory entry is durable and not only the file's data
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
    if sync_dir:
        _fsync_directory(path.parent)


@contextmanager
def file_lock(path: Path, timeout: float = 10.0) -> Iterator[None]:
    """
    Acquire an exclusive file lock for safe read-modify-write operations.

    Uses POSIX flock() which is released automatically when the process exits
    or the file descriptor is closed.

    The lock file is created once and then left in place. ``flock`` is
    associated with the inode, not with the pathname, so unlinking the file
    does not release anybody's lock: a waiter that already opened it keeps its
    lock on the now unreachable inode, while the next arrival creates a fresh
    inode and takes an uncontended lock on that. Two holders, one note, and the
    read-modify-write this lock exists to serialise runs concurrently. Lock
    files are empty, and one accumulates per note UUID ever locked, including
    notes since deleted, so leaving them costs nothing worth that risk.

    An upgrade from a release that still unlinked is not made safe by keeping
    the same pathname; it only avoids guaranteeing separation. A process
    running the old code still unlinks the file out from under a new one.
    Restart every server process to leave the mixed-version window.

    Each acquisition opens its own descriptor, so a holder that enters the
    lock again for the same path blocks on itself until the timeout.

    Args:
        path: Path to file to lock
        timeout: Max seconds to wait for lock (raises TimeoutError if exceeded)

    Yields:
        None (lock is held while in context)
    """
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
