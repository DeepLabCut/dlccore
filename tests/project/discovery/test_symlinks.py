#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/dlccore
#
# Contributors:
# https://github.com/DeepLabCut/dlccore/graphs/contributors
#
# Licensed under GNU Lesser General Public License v3.0
#

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from dlccore.project.discovery.observer import (
    ExpectedEntryKind,
    classify_unclaimed_entry,
    status_for_expected_entry,
)
from dlccore.project.discovery.probe import probe_path
from dlccore.project.schemas.base.observations import NodeStatus
from dlccore.project.schemas.base.probes import (
    FilesystemEntryKind,
    PathProbe,
    PathProbeStatus,
)
from dlccore.project.schemas.base.resources import (
    DirectoryResourceKind,
    FileResourceKind,
    UnknownResourceKind,
)
from dlccore.project.schemas.base.symlinks import (
    SymlinkObservation,
    SymlinkTargetKind,
)


def test_expected_file_accepts_file_symlink(
    tmp_path: Path,
    create_symlink_or_skip: Callable[..., None],
) -> None:
    target = tmp_path / "source.mp4"
    target.touch()

    link = tmp_path / "linked.mp4"
    create_symlink_or_skip(
        link,
        target,
    )

    probe = probe_path(link)
    status = status_for_expected_entry(probe, expected_kind=ExpectedEntryKind.FILE)

    assert status == NodeStatus.PRESENT
    assert probe.symlink is not None
    assert probe.symlink.target_kind == SymlinkTargetKind.FILE


def test_expected_file_preserves_broken_symlink(
    tmp_path: Path,
    create_symlink_or_skip: Callable[..., None],
) -> None:
    link = tmp_path / "linked.mp4"

    create_symlink_or_skip(
        link,
        tmp_path / "missing.mp4",
    )

    probe = probe_path(link)
    status = status_for_expected_entry(probe, expected_kind=ExpectedEntryKind.FILE)

    assert status == NodeStatus.UNREADABLE
    assert probe.symlink is not None
    assert probe.symlink.target_kind == SymlinkTargetKind.MISSING


def test_expected_file_reports_directory_target_as_wrong_type(
    tmp_path: Path,
    create_symlink_or_skip: Callable[..., None],
) -> None:
    target = tmp_path / "target-directory"
    target.mkdir()

    link = tmp_path / "linked.mp4"
    create_symlink_or_skip(
        link,
        target,
        target_is_directory=True,
    )

    probe = probe_path(link)
    status = status_for_expected_entry(probe, expected_kind=ExpectedEntryKind.FILE)

    assert status == NodeStatus.WRONG_TYPE
    assert probe.symlink is not None
    assert probe.symlink.target_kind == SymlinkTargetKind.DIRECTORY


def test_expected_directory_symlink_is_present(
    tmp_path: Path,
    create_symlink_or_skip: Callable[..., None],
) -> None:
    target = tmp_path / "target-directory"
    target.mkdir()

    link = tmp_path / "linked-directory"
    create_symlink_or_skip(
        link,
        target,
        target_is_directory=True,
    )

    probe = probe_path(link)

    assert status_for_expected_entry(probe, expected_kind=ExpectedEntryKind.DIRECTORY) == NodeStatus.PRESENT
    assert status_for_expected_entry(probe, expected_kind=ExpectedEntryKind.FILE) == NodeStatus.WRONG_TYPE


def test_expected_directory_broken_symlink_is_unreadable(
    tmp_path: Path,
    create_symlink_or_skip: Callable[..., None],
) -> None:
    """A broken directory-expected symlink is UNREADABLE, not a raise."""

    link = tmp_path / "linked-directory"
    create_symlink_or_skip(
        link,
        tmp_path / "missing-directory",
        target_is_directory=True,
    )

    probe = probe_path(link)
    status = status_for_expected_entry(probe, expected_kind=ExpectedEntryKind.DIRECTORY)

    assert status == NodeStatus.UNREADABLE


def test_unknown_file_symlink_is_classified_as_file(
    tmp_path: Path,
    create_symlink_or_skip: Callable[..., None],
) -> None:
    target = tmp_path / "target.bin"
    target.touch()

    link = tmp_path / "unknown-link"
    create_symlink_or_skip(
        link,
        target,
    )

    kind, symlink = classify_unclaimed_entry(probe_path(link))

    assert kind is FileResourceKind.UNKNOWN_FILE
    assert symlink is not None
    assert symlink.target_kind == SymlinkTargetKind.FILE


def test_unknown_broken_symlink_is_classified_unknown_and_retained(
    tmp_path: Path,
    create_symlink_or_skip: Callable[..., None],
) -> None:
    """A dangling link is described rather than refused, and keeps what was seen.

    The kind cannot be determined because the target is gone, but the
    observation still records that it was a link and where it pointed. That
    retention is what separates a diagnosable entry from a merely unclassified
    one, and it is why the undeterminable cases are not all interchangeable.

    Built on a real symlink rather than a constructed probe: this is the one
    place the whole path, ``probe_path`` through classification, runs on bytes.
    """

    link = tmp_path / "unknown-link"

    create_symlink_or_skip(
        link,
        tmp_path / "missing-target",
    )

    kind, symlink = classify_unclaimed_entry(probe_path(link))

    assert kind is UnknownResourceKind.UNKNOWN
    assert symlink is not None
    assert symlink.target_kind == SymlinkTargetKind.MISSING


def test_unknown_directory_symlink_is_classified_as_a_directory() -> None:
    """A symlinked directory the layout does not claim is retained, not refused.

    Relocating a large directory to bulk storage and linking it back is an
    ordinary project layout, and refusing it costs the whole snapshot rather
    than the one entry. The probe is constructed rather than made on disk so
    this runs where creating a symlink needs privileges.
    """

    probe = PathProbe(
        status=PathProbeStatus.OBSERVED,
        entry_kind=FilesystemEntryKind.SYMLINK,
        symlink=SymlinkObservation(
            raw_target=Path("bulk/training-datasets"),
            absolute_target=Path("/bulk/training-datasets"),
            target_kind=SymlinkTargetKind.DIRECTORY,
        ),
    )

    kind, symlink = classify_unclaimed_entry(probe)

    assert kind is DirectoryResourceKind.UNKNOWN_DIRECTORY
    assert symlink is not None
    assert symlink.target_kind == SymlinkTargetKind.DIRECTORY


def _unclaimed_probe(target_kind: SymlinkTargetKind) -> PathProbe:
    return PathProbe(
        status=PathProbeStatus.OBSERVED,
        entry_kind=FilesystemEntryKind.SYMLINK,
        symlink=SymlinkObservation(
            raw_target=Path("gone"),
            absolute_target=Path("/gone"),
            target_kind=target_kind,
        ),
    )


@pytest.mark.parametrize(
    "probe",
    [
        pytest.param(PathProbe(status=PathProbeStatus.MISSING), id="entry-vanished"),
        pytest.param(
            PathProbe(
                status=PathProbeStatus.INACCESSIBLE,
                error_type="PermissionError",
                error_message="denied",
            ),
            id="entry-unreadable",
        ),
        pytest.param(
            PathProbe(status=PathProbeStatus.OBSERVED, entry_kind=FilesystemEntryKind.OTHER),
            id="fifo-socket-device",
        ),
        pytest.param(_unclaimed_probe(SymlinkTargetKind.MISSING), id="symlink-target-missing"),
        pytest.param(_unclaimed_probe(SymlinkTargetKind.INACCESSIBLE), id="symlink-target-unreadable"),
        pytest.param(_unclaimed_probe(SymlinkTargetKind.OTHER), id="symlink-target-other"),
    ],
)
def test_undeterminable_unclaimed_entry_is_classified_rather_than_refused(
    probe: PathProbe,
) -> None:
    """An unclaimed entry is retained whatever it turns out to be.

    Discovery surfaces facts and validation decides, so an entry that cannot be
    identified is a fact to report, not grounds for abandoning the snapshot: a
    file removed mid-scan, an unreadable entry, a device node or a dangling link
    must not cost the parts of the project that are fine.
    """

    kind, _ = classify_unclaimed_entry(probe)

    assert kind is UnknownResourceKind.UNKNOWN


def test_mapping_functions_do_not_touch_the_filesystem(tmp_path: Path) -> None:
    """status_for_expected_entry/classify_unclaimed_entry consume only the probe."""

    path = tmp_path / "video.mp4"
    path.touch()

    probe = probe_path(path)
    path.unlink()

    status = status_for_expected_entry(probe, expected_kind=ExpectedEntryKind.FILE)
    kind, symlink = classify_unclaimed_entry(probe)

    assert status == NodeStatus.PRESENT
    assert kind is FileResourceKind.UNKNOWN_FILE
    assert symlink is None
