from __future__ import annotations

import os
import stat
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from dlc_octopus.discovery.observer import ExpectedEntryKind, status_for_expected_entry
from dlc_octopus.discovery.probe import probe_path
from dlc_octopus.schemas.base.observations import NodeStatus
from dlc_octopus.schemas.base.probes import FilesystemEntryKind, PathProbe, PathProbeStatus
from dlc_octopus.schemas.base.symlinks import SymlinkObservation, SymlinkTargetKind


def test_probe_missing_path(tmp_path: Path) -> None:
    probe = probe_path(tmp_path / "missing.mp4")

    assert probe.status == PathProbeStatus.MISSING
    assert probe.entry_kind is None
    assert probe.symlink is None


def test_probe_a_path_the_os_cannot_take(tmp_path: Path) -> None:
    probe = probe_path(tmp_path / "a\x00b")

    assert probe.status == PathProbeStatus.INACCESSIBLE
    assert probe.error_type == "ValueError"


def test_probe_regular_file(tmp_path: Path) -> None:
    path = tmp_path / "video.mp4"
    path.touch()

    probe = probe_path(path)

    assert probe.status == PathProbeStatus.OBSERVED
    assert probe.entry_kind == FilesystemEntryKind.FILE
    assert probe.symlink is None


def test_probe_regular_directory(tmp_path: Path) -> None:
    path = tmp_path / "videos"
    path.mkdir()

    probe = probe_path(path)

    assert probe.status == PathProbeStatus.OBSERVED
    assert probe.entry_kind == FilesystemEntryKind.DIRECTORY
    assert probe.symlink is None


def test_probe_absolute_file_symlink(
    tmp_path: Path,
    create_symlink_or_skip: Callable[..., None],
) -> None:
    target = tmp_path / "source.mp4"
    target.touch()

    link = tmp_path / "linked.mp4"
    create_symlink_or_skip(link, target)

    probe = probe_path(link)

    assert probe.status == PathProbeStatus.OBSERVED
    assert probe.entry_kind == FilesystemEntryKind.SYMLINK
    assert probe.symlink is not None
    assert probe.symlink.raw_target == target
    assert probe.symlink.absolute_target == target.absolute()
    assert probe.symlink.target_kind == SymlinkTargetKind.FILE


def test_probe_relative_file_symlink(
    tmp_path: Path,
    create_symlink_or_skip: Callable[..., None],
) -> None:
    target = tmp_path / "source.mp4"
    target.touch()

    link = tmp_path / "linked.mp4"
    relative_target = Path("source.mp4")
    create_symlink_or_skip(link, relative_target)

    probe = probe_path(link)

    assert probe.entry_kind == FilesystemEntryKind.SYMLINK
    assert probe.symlink is not None
    assert probe.symlink.raw_target == relative_target
    assert probe.symlink.absolute_target == target.absolute()
    assert probe.symlink.target_kind == SymlinkTargetKind.FILE


def test_probe_directory_symlink(
    tmp_path: Path,
    create_symlink_or_skip: Callable[..., None],
) -> None:
    target = tmp_path / "target-directory"
    target.mkdir()

    link = tmp_path / "linked-directory"
    create_symlink_or_skip(link, target, target_is_directory=True)

    probe = probe_path(link)

    assert probe.status == PathProbeStatus.OBSERVED
    assert probe.entry_kind == FilesystemEntryKind.SYMLINK
    assert probe.symlink is not None
    assert probe.symlink.target_kind == SymlinkTargetKind.DIRECTORY


def test_probe_broken_symlink(
    tmp_path: Path,
    create_symlink_or_skip: Callable[..., None],
) -> None:
    target = tmp_path / "missing.mp4"
    link = tmp_path / "linked.mp4"
    create_symlink_or_skip(link, target)

    probe = probe_path(link)

    assert probe.status == PathProbeStatus.OBSERVED
    assert probe.entry_kind == FilesystemEntryKind.SYMLINK
    assert probe.symlink is not None
    assert probe.symlink.absolute_target == target.absolute()
    assert probe.symlink.target_kind == SymlinkTargetKind.MISSING


def test_probe_symlink_to_unsupported_target_type(
    tmp_path: Path,
    create_symlink_or_skip: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A symlink whose target is neither a regular file nor a directory."""

    target = tmp_path / "device"
    target.touch()

    link = tmp_path / "linked-device"
    create_symlink_or_skip(link, target)

    real_stat = Path.stat

    def fake_stat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        result = real_stat(self, *args, **kwargs)

        if self == target.absolute():
            fake_mode = (result.st_mode & ~stat.S_IFMT(result.st_mode)) | stat.S_IFCHR
            return os.stat_result((fake_mode, *result[1:]))

        return result

    monkeypatch.setattr(Path, "stat", fake_stat)

    probe = probe_path(link)

    assert probe.entry_kind == FilesystemEntryKind.SYMLINK
    assert probe.symlink is not None
    assert probe.symlink.target_kind == SymlinkTargetKind.OTHER


def test_probe_unsupported_entry_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-symlink entry that is neither a regular file nor a directory."""

    path = tmp_path / "device"
    path.touch()

    real_lstat = Path.lstat

    def fake_lstat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        result = real_lstat(self, *args, **kwargs)

        if self == path:
            fake_mode = (result.st_mode & ~stat.S_IFMT(result.st_mode)) | stat.S_IFCHR
            return os.stat_result((fake_mode, *result[1:]))

        return result

    monkeypatch.setattr(Path, "lstat", fake_lstat)

    probe = probe_path(path)

    assert probe.status == PathProbeStatus.OBSERVED
    assert probe.entry_kind == FilesystemEntryKind.OTHER
    assert probe.symlink is None


def test_probe_entry_access_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "locked.mp4"
    path.touch()

    def fake_lstat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        if self == path:
            raise PermissionError("Access is denied")

        return Path.lstat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", fake_lstat)

    probe = probe_path(path)

    assert probe.status == PathProbeStatus.INACCESSIBLE
    assert probe.entry_kind is None
    assert probe.symlink is None
    assert probe.error_type == "PermissionError"
    assert probe.error_message


def test_probe_symlink_target_access_failure(
    tmp_path: Path,
    create_symlink_or_skip: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "source.mp4"
    target.touch()

    link = tmp_path / "linked.mp4"
    create_symlink_or_skip(link, target)

    real_stat = Path.stat

    def fake_stat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        if self == target.absolute():
            raise PermissionError("Access is denied")

        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fake_stat)

    probe = probe_path(link)

    assert probe.status == PathProbeStatus.OBSERVED
    assert probe.entry_kind == FilesystemEntryKind.SYMLINK
    assert probe.symlink is not None
    assert probe.symlink.target_kind == SymlinkTargetKind.INACCESSIBLE
    assert probe.error_type == "PermissionError"
    assert probe.error_message


def test_inaccessible_probe_rejects_an_entry_kind() -> None:
    """``INACCESSIBLE`` is reserved for entries that were never observed."""

    with pytest.raises(
        ValidationError,
        match="must not carry an entry kind",
    ):
        PathProbe(
            status=PathProbeStatus.INACCESSIBLE,
            entry_kind=FilesystemEntryKind.FILE,
            error_type="PermissionError",
        )


def test_inaccessible_symlink_target_still_reads_as_unreadable() -> None:
    """An observed entry with an unreadable target stays ``UNREADABLE``.

    Pins the mapping that lets ``INACCESSIBLE`` describe the entry alone,
    without needing a symlink the platform may refuse to create.
    """

    probe = PathProbe(
        status=PathProbeStatus.OBSERVED,
        entry_kind=FilesystemEntryKind.SYMLINK,
        symlink=SymlinkObservation(
            raw_target=Path("target.mp4"),
            absolute_target=Path("/videos/target.mp4"),
            target_kind=SymlinkTargetKind.INACCESSIBLE,
        ),
        error_type="PermissionError",
        error_message="Access is denied",
    )

    assert status_for_expected_entry(probe, expected_kind=ExpectedEntryKind.FILE) == NodeStatus.UNREADABLE
