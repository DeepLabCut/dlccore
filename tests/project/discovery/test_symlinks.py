from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from deeplabcut.core.config import ProjectConfig

from dlc_octopus.api import inspect_project
from dlc_octopus.discovery.observer import (
    ExpectedEntryKind,
    classify_unclaimed_entry,
    status_for_expected_entry,
)
from dlc_octopus.discovery.probe import probe_path
from dlc_octopus.schemas.base.observations import NodeStatus
from dlc_octopus.schemas.base.probes import (
    FilesystemEntryKind,
    PathProbe,
    PathProbeStatus,
)
from dlc_octopus.schemas.base.resources import (
    DirectoryResourceKind,
    FileResourceKind,
    ResourceManagement,
    UnknownResourceKind,
)
from dlc_octopus.schemas.base.symlinks import (
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


def test_inspection_preserves_valid_video_symlink(
    valid_project: Path,
    tmp_path: Path,
    create_symlink_or_skip: Callable[..., None],
) -> None:
    target = tmp_path / "external.mp4"
    target.touch()

    link = valid_project / "videos" / "external.mp4"
    create_symlink_or_skip(
        link,
        target,
    )

    snapshot = inspect_project(valid_project)
    resource = snapshot.resource_at("videos/external.mp4")

    assert resource is not None
    assert resource.kind == FileResourceKind.SOURCE_VIDEO
    assert resource.symlink is not None
    assert resource.symlink.absolute_target == target.absolute()
    assert resource.symlink.target_kind == SymlinkTargetKind.FILE


def test_inspection_preserves_broken_video_symlink(
    valid_project: Path,
    tmp_path: Path,
    create_symlink_or_skip: Callable[..., None],
) -> None:
    target = tmp_path / "missing.mp4"

    link = valid_project / "videos" / "missing.mp4"
    create_symlink_or_skip(
        link,
        target,
    )

    snapshot = inspect_project(valid_project)
    resource = snapshot.resource_at("videos/missing.mp4")

    assert resource is not None
    assert resource.kind == FileResourceKind.SOURCE_VIDEO
    assert resource.symlink is not None
    assert resource.symlink.absolute_target == target.absolute()
    assert resource.symlink.target_kind == SymlinkTargetKind.MISSING


def test_inspection_retains_an_unclaimed_directory_symlink(
    valid_project: Path,
    tmp_path: Path,
    create_symlink_or_skip: Callable[..., None],
) -> None:
    """End to end: a relocated project directory survives inspection as a resource.

    The mapping is covered by constructed probes above, which run everywhere.
    This covers the wiring - probe, unknown-child collection, ``ProjectResource``
    construction and flattening - and needs a real link, so it skips on Windows.
    """

    target = tmp_path / "relocated"
    target.mkdir()

    link = valid_project / "training-datasets"
    create_symlink_or_skip(
        link,
        target,
        target_is_directory=True,
    )

    snapshot = inspect_project(valid_project)
    resource = snapshot.resource_at("training-datasets")

    assert resource is not None
    assert resource.kind == DirectoryResourceKind.UNKNOWN_DIRECTORY
    assert resource.management == ResourceManagement.UNDETERMINED
    assert resource.symlink is not None
    assert resource.symlink.absolute_target == target.absolute()
    assert resource.symlink.target_kind == SymlinkTargetKind.DIRECTORY


def test_an_unclaimed_entry_that_vanishes_is_not_reported_as_a_resource(
    valid_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``resources`` holds only entries with a filesystem entry, as claimed nodes do.

    An entry can disappear between ``iterdir`` and the probe, and a resource for
    something that no longer exists is a phantom row for any consumer.
    """

    (valid_project / "notes.txt").touch()

    real_probe_path = probe_path

    def fake_probe_path(path: Path) -> PathProbe:
        if path.name == "notes.txt":
            return PathProbe(status=PathProbeStatus.MISSING)

        return real_probe_path(path)

    monkeypatch.setattr(
        "dlc_octopus.discovery.observer.probe_path",
        fake_probe_path,
    )

    snapshot = inspect_project(valid_project)

    assert snapshot.resource_at("notes.txt") is None
    assert snapshot.resource_at("config.yaml") is not None


def test_a_symlinked_videos_directory_is_reported_not_raised(
    tmp_path: Path,
    create_symlink_or_skip: Callable[..., None],
) -> None:
    real_videos = tmp_path / "real_videos"
    real_videos.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    create_symlink_or_skip(project / "videos", real_videos, target_is_directory=True)

    snapshot = inspect_project(project)
    resource = snapshot.resource_at("videos")

    assert resource is not None
    assert resource.kind == DirectoryResourceKind.VIDEOS_DIRECTORY
    assert resource.symlink is not None
    assert resource.symlink.target_kind == SymlinkTargetKind.DIRECTORY


@pytest.mark.parametrize("given", ["root", "config"], ids=["root", "config_yaml"])
def test_a_project_whose_root_is_a_symlink_is_inspected_through_it(
    tmp_path: Path,
    valid_project_config: ProjectConfig,
    create_symlink_or_skip: Callable[..., None],
    given: str,
) -> None:
    real = tmp_path / "real"
    (real / "videos").mkdir(parents=True)
    (real / "labeled-data").mkdir()
    valid_project_config.to_yaml(real / "config.yaml")
    link = tmp_path / "link"
    create_symlink_or_skip(link, real, target_is_directory=True)

    snapshot = inspect_project(link if given == "root" else link / "config.yaml")

    assert snapshot.project_root == link.absolute()
    assert snapshot.resource_at("videos") is not None
    assert snapshot.config_state.value == "valid"


def test_a_symlinked_dataset_is_traversed(
    valid_project: Path,
    tmp_path_factory: pytest.TempPathFactory,
    create_symlink_or_skip: Callable[..., None],
) -> None:
    storage = tmp_path_factory.mktemp("storage") / "session-02"
    storage.mkdir()
    (storage / "img000.png").touch()
    create_symlink_or_skip(valid_project / "labeled-data" / "session-02", storage, target_is_directory=True)

    snapshot = inspect_project(valid_project)
    dataset = snapshot.resource_at("labeled-data/session-02")

    assert dataset is not None
    assert dataset.kind == DirectoryResourceKind.LABELED_DATASET
    assert dataset.symlink is not None
    assert snapshot.resource_at("labeled-data/session-02/img000.png") is not None


def test_a_directory_linked_to_its_own_ancestor_is_unreadable_not_entered(
    valid_project: Path,
    create_symlink_or_skip: Callable[..., None],
) -> None:
    (valid_project / "labeled-data").rmdir()
    create_symlink_or_skip(valid_project / "labeled-data", valid_project, target_is_directory=True)

    snapshot = inspect_project(valid_project)

    (finding,) = (
        finding
        for finding in snapshot.findings
        if finding.code == "PROJECT_NODE_UNREADABLE" and finding.schema_id == "dlc.labeled_data"
    )
    assert finding.evidence["error_type"] == "SymlinkLoop"
    assert snapshot.resource_at("labeled-data/config.yaml") is None
