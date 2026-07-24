"""Tests for the per-note file lock that guards read-modify-write updates.

``flock`` is associated with an inode, not with a pathname. Unlinking a lock
file therefore does not end anyone's lock, it only makes the *next* opener
create a fresh inode whose exclusive lock is uncontended. These tests pin the
resulting invariant: the lock file for a note outlives the critical section, so
every contender flocks the same inode.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from uuid import uuid4

import pytest
from git import Repo

from mcp_notes.storage.filesystem import NoteStore, _file_lock
from mcp_notes.storage.git import LOCKS_EXCLUDE, GitManager


@pytest.fixture
def lock_target(tmp_path: Path) -> Path:
    return tmp_path / "locks" / f"{uuid4()}.lock"


def _lock_file_for(target: Path) -> Path:
    """The file ``_file_lock`` actually opens for a given target."""
    return target.with_suffix(target.suffix + ".lock")


class TestLockFileLifetime:
    def test_lock_file_survives_the_critical_section(self, lock_target: Path) -> None:
        """Regression: the lock file was unlinked on release.

        A waiter that had already opened the file kept its lock on the now
        unlinked inode, while the next arrival created a new inode and locked
        that instead. Two holders, one note.
        """
        with _file_lock(lock_target):
            pass

        assert _lock_file_for(lock_target).exists()

    def test_same_inode_is_reused_across_acquisitions(self, lock_target: Path) -> None:
        """Contenders must meet on one inode, or they do not contend at all."""
        with _file_lock(lock_target):
            first = os.stat(_lock_file_for(lock_target)).st_ino

        with _file_lock(lock_target):
            second = os.stat(_lock_file_for(lock_target)).st_ino

        assert first == second

    def test_lock_file_survives_an_exception_in_the_body(self, lock_target: Path) -> None:
        with pytest.raises(RuntimeError, match="boom"), _file_lock(lock_target):
            raise RuntimeError("boom")

        assert _lock_file_for(lock_target).exists()


class TestMutualExclusion:
    def test_only_one_holder_at_a_time_under_contention(self, lock_target: Path) -> None:
        """The property the lock exists for.

        ``flock`` locks are held per open file description, so two descriptors
        in one process exclude each other exactly as two processes would. With
        the lock file unlinked on release this reliably reported several
        simultaneous holders.
        """
        workers = 8
        rounds = 60
        holders = 0
        peak = 0
        violations = 0
        completed = 0
        errors: list[BaseException] = []
        bookkeeping = threading.Lock()

        def worker() -> None:
            nonlocal holders, peak, violations, completed
            try:
                for _ in range(rounds):
                    with _file_lock(lock_target, timeout=30.0):
                        with bookkeeping:
                            holders += 1
                            peak = max(peak, holders)
                            if holders > 1:
                                violations += 1
                        time.sleep(0.001)
                        with bookkeeping:
                            holders -= 1
                            completed += 1
            except BaseException as exc:  # pragma: no cover - reported below
                with bookkeeping:
                    errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # A worker that died early would otherwise leave peak == 1 and look
        # like a pass, so the acquisition count has to be checked too.
        assert not errors, errors
        assert completed == workers * rounds
        assert violations == 0
        assert peak == 1

    def test_a_held_lock_blocks_until_released(self, lock_target: Path) -> None:
        """A second acquirer waits rather than sailing straight through."""
        entered = threading.Event()
        released = threading.Event()
        acquired_second = threading.Event()

        def holder() -> None:
            with _file_lock(lock_target, timeout=30.0):
                entered.set()
                released.wait(timeout=5.0)

        def contender() -> None:
            with _file_lock(lock_target, timeout=30.0):
                acquired_second.set()

        first = threading.Thread(target=holder)
        second = threading.Thread(target=contender)
        first.start()
        assert entered.wait(timeout=5.0)
        second.start()

        assert not acquired_second.wait(timeout=0.3), "lock did not exclude a contender"
        released.set()
        first.join(timeout=5.0)
        second.join(timeout=5.0)
        assert acquired_second.is_set()

    def test_timeout_is_reported_when_the_lock_is_held(self, lock_target: Path) -> None:
        entered = threading.Event()
        release = threading.Event()
        failure: list[BaseException] = []

        def holder() -> None:
            with _file_lock(lock_target, timeout=30.0):
                entered.set()
                release.wait(timeout=5.0)

        thread = threading.Thread(target=holder)
        thread.start()
        try:
            assert entered.wait(timeout=5.0)
            with pytest.raises(TimeoutError):
                with _file_lock(lock_target, timeout=0.2):
                    pass
        except BaseException as exc:  # pragma: no cover - surfaced below
            failure.append(exc)
        finally:
            release.set()
            thread.join(timeout=5.0)
        assert not failure


class TestStartupDoesNotRemoveLocks:
    def test_ensure_directories_keeps_existing_lock_files(self, tmp_path: Path) -> None:
        """Startup used to delete lock files older than an hour.

        Deleting a lock file that another process is holding is the same bug as
        deleting it on release: the holder keeps the orphaned inode while the
        next arrival locks a brand new one. Lock files are empty, one per note,
        so keeping them costs nothing worth this risk.
        """
        store = NoteStore(notes_dir=tmp_path)
        store.ensure_directories()

        # The file that is actually locked, not the path handed to _file_lock,
        # so a future sweep aimed at real lock files cannot regress past this.
        stale = _lock_file_for(store._get_note_lock_path(uuid4()))
        stale.touch()
        ancient = time.time() - 7 * 24 * 3600
        os.utime(stale, (ancient, ancient))

        store.ensure_directories()

        assert stale.exists()


class TestLockDirectoryStaysOutOfGit:
    """Lock files persist now, so a repository that this package did not
    create must still exclude them: untracked clutter is the mild outcome, a
    checkout or sync replacing a held lock's inode is the serious one."""

    @staticmethod
    def _is_ignored(repo: Repo, relative: str) -> bool:
        """Ask git itself, rather than trusting the file we wrote."""
        return bool(repo.ignored(relative))

    def test_adopted_repository_ignores_the_lock_directory(self, tmp_path: Path) -> None:
        repo = Repo.init(tmp_path)
        (tmp_path / ".locks").mkdir()
        (tmp_path / ".locks" / "held.lock").touch()
        assert not self._is_ignored(repo, ".locks/held.lock")

        manager = GitManager(notes_dir=tmp_path)

        assert manager.repo is not None
        assert self._is_ignored(repo, ".locks/held.lock")

    def test_linked_worktree_ignores_the_lock_directory(self, tmp_path: Path) -> None:
        """Regression: the rule was written to the worktree's own git
        directory, which git does not read `info/exclude` from, so it had no
        effect. It belongs in the common directory."""
        origin = tmp_path / "origin"
        origin.mkdir()
        repo = Repo.init(origin)
        (origin / "seed.txt").write_text("seed")
        repo.index.add(["seed.txt"])
        repo.index.commit("seed")

        linked = tmp_path / "linked"
        repo.git.worktree("add", str(linked))
        (linked / ".locks").mkdir()
        (linked / ".locks" / "held.lock").touch()

        manager = GitManager(notes_dir=linked)
        assert manager.repo is not None

        linked_repo = Repo(linked)
        assert self._is_ignored(linked_repo, ".locks/held.lock")

    def test_a_non_utf8_exclude_file_is_not_fatal(self, tmp_path: Path) -> None:
        """An adopted repository's exclude file need not be valid UTF-8."""
        repo = Repo.init(tmp_path)
        exclude_path = tmp_path / ".git" / "info" / "exclude"
        exclude_path.parent.mkdir(parents=True, exist_ok=True)
        exclude_path.write_bytes(b"# caf\xe9\nscratch/\n")
        (tmp_path / ".locks").mkdir()
        (tmp_path / ".locks" / "held.lock").touch()

        manager = GitManager(notes_dir=tmp_path)

        assert manager.repo is not None
        assert self._is_ignored(repo, ".locks/held.lock")
        assert b"# caf\xe9" in exclude_path.read_bytes()

    def test_a_whitespace_prefixed_rule_does_not_count(self, tmp_path: Path) -> None:
        """Leading whitespace is significant to git, so " .locks/" is a
        different rule and must not be mistaken for this one."""
        repo = Repo.init(tmp_path)
        exclude_path = tmp_path / ".git" / "info" / "exclude"
        exclude_path.parent.mkdir(parents=True, exist_ok=True)
        exclude_path.write_text(f" {LOCKS_EXCLUDE}\n", encoding="utf-8")
        (tmp_path / ".locks").mkdir()
        (tmp_path / ".locks" / "held.lock").touch()

        manager = GitManager(notes_dir=tmp_path)

        assert manager.repo is not None
        assert self._is_ignored(repo, ".locks/held.lock")

    def test_a_carriage_return_does_not_hide_a_missing_rule(self, tmp_path: Path) -> None:
        """Git ends a pattern at a newline, not at a lone carriage return.

        `b"scratch/\\r.locks/"` is a single ineffective pattern to git, so the
        rule is genuinely absent and must still be appended.
        """
        repo = Repo.init(tmp_path)
        exclude_path = tmp_path / ".git" / "info" / "exclude"
        exclude_path.parent.mkdir(parents=True, exist_ok=True)
        exclude_path.write_bytes(b"scratch/\r" + LOCKS_EXCLUDE.encode() + b"\n")
        (tmp_path / ".locks").mkdir()
        (tmp_path / ".locks" / "held.lock").touch()

        manager = GitManager(notes_dir=tmp_path)

        assert manager.repo is not None
        assert self._is_ignored(repo, ".locks/held.lock")

    def test_a_bare_repository_is_left_alone(self, tmp_path: Path) -> None:
        """There is no working tree to exclude anything from."""
        bare = tmp_path / "bare.git"
        Repo.init(bare, bare=True)

        exclude_path = bare / "info" / "exclude"
        before = exclude_path.read_bytes() if exclude_path.exists() else None

        manager = GitManager(notes_dir=bare)
        assert manager.repo is not None

        after = exclude_path.read_bytes() if exclude_path.exists() else None
        assert after == before
        assert after is None or LOCKS_EXCLUDE.encode() not in after.splitlines()

    def test_exclude_is_written_once(self, tmp_path: Path) -> None:
        Repo.init(tmp_path)
        for _ in range(3):
            manager = GitManager(notes_dir=tmp_path)
            assert manager.repo is not None

        exclude = (tmp_path / ".git" / "info" / "exclude").read_text(encoding="utf-8")
        assert exclude.splitlines().count(LOCKS_EXCLUDE) == 1

    def test_existing_exclude_rules_are_kept(self, tmp_path: Path) -> None:
        Repo.init(tmp_path)
        exclude_path = tmp_path / ".git" / "info" / "exclude"
        exclude_path.write_text("scratch/\n", encoding="utf-8")

        manager = GitManager(notes_dir=tmp_path)
        assert manager.repo is not None

        lines = exclude_path.read_text(encoding="utf-8").splitlines()
        assert "scratch/" in lines
        assert LOCKS_EXCLUDE in lines
