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

from dlccore.config.utils import (
    read_config_as_dict,
    write_config,
)
from dlccore.project.api import inspect_project
from dlccore.project.discovery.probe import probe_path
from dlccore.project.schemas.base.findings import Reach, Severity
from dlccore.project.schemas.base.probes import (
    FilesystemEntryKind,
    PathProbe,
    PathProbeStatus,
)
from dlccore.project.schemas.base.resources import (
    DirectoryResourceKind,
)
from dlccore.project.schemas.base.symlinks import (
    SymlinkObservation,
    SymlinkTargetKind,
)
from dlccore.project.schemas.base.validation import (
    ValidationCheckStatus,
    ValidationDepth,
)
from dlccore.project.validation import (
    ProjectValidationRequest,
    ResourceValidationRequest,
)
from dlccore.project.validation.deeplabcut import videos

pytestmark = pytest.mark.data_validator("VIDEOS")


def video_filename_request() -> ProjectValidationRequest:
    """Request lightweight validation for the videos area."""

    return ProjectValidationRequest(
        resources=(
            ResourceValidationRequest(
                kind=DirectoryResourceKind.VIDEOS_DIRECTORY,
                depth=ValidationDepth.FILENAME_ONLY,
            ),
        )
    )


def set_configured_videos(
    project_root: Path,
    video_paths: tuple[Path, ...],
) -> None:
    """Replace video_sets in a canonical test config."""

    config_path = project_root / "config.yaml"
    config = read_config_as_dict(config_path)

    config["video_sets"] = {str(video_path): {} for video_path in video_paths}

    write_config(
        config_path,
        config,
    )


def _symlink_probe(
    target: Path,
    target_kind: SymlinkTargetKind,
) -> Callable[[Path], PathProbe]:
    """Return a ``probe_path`` replacement reporting one symlink observation.

    Lets the symlink branches be covered where the platform withholds the
    privilege needed to create a real link.
    """

    def fake_probe_path(_path: Path) -> PathProbe:
        return PathProbe(
            status=PathProbeStatus.OBSERVED,
            entry_kind=FilesystemEntryKind.SYMLINK,
            symlink=SymlinkObservation(
                raw_target=target,
                absolute_target=target.absolute(),
                target_kind=target_kind,
            ),
        )

    return fake_probe_path


def test_configured_videos_present_completes_without_findings(
    valid_project: Path,
) -> None:
    video_path = valid_project / "videos" / "session-01.mp4"
    video_path.touch()

    set_configured_videos(
        valid_project,
        (video_path,),
    )

    snapshot = inspect_project(
        valid_project,
        validation=video_filename_request(),
    )

    video_findings = tuple(
        finding for finding in snapshot.findings if finding.code.startswith("CONFIGURED_VIDEO_PATH_")
    )

    assert video_findings == ()
    assert len(snapshot.validation_checks) == 1

    check = snapshot.validation_checks[0]

    assert check.code == "VIDEO_FILENAME_VALIDATION"
    assert check.status == ValidationCheckStatus.COMPLETED


def test_missing_configured_video_produces_finding(
    valid_project: Path,
) -> None:
    missing_video = valid_project / "videos" / "missing.mp4"

    set_configured_videos(
        valid_project,
        (missing_video,),
    )

    snapshot = inspect_project(
        valid_project,
        validation=video_filename_request(),
    )

    finding = next(finding for finding in snapshot.findings if finding.code == "CONFIGURED_VIDEO_PATH_MISSING")

    assert finding.schema_id == "dlc.project.config"
    assert finding.instance == ()
    assert finding.path == missing_video
    assert finding.field_location == (
        "video_sets",
        str(missing_video),
    )
    assert finding.evidence == {}
    assert snapshot.validation_checks[0].status == ValidationCheckStatus.COMPLETED


def test_keys_under_a_vanished_root_are_one_finding_not_one_per_video(
    valid_project: Path,
    tmp_path: Path,
) -> None:
    vanished_root = tmp_path / "vanished-root"
    names = ("session-01.mp4", "session-02.mp4")

    for name in names:
        (valid_project / "videos" / name).touch()

    set_configured_videos(
        valid_project,
        tuple(vanished_root / "videos" / name for name in names),
    )

    snapshot = inspect_project(
        valid_project,
        validation=video_filename_request(),
    )

    assert snapshot.validation_checks[0].status == ValidationCheckStatus.COMPLETED
    assert [finding for finding in snapshot.findings if finding.code == "CONFIGURED_VIDEO_PATH_MISSING"] == []

    (finding,) = snapshot.findings

    assert finding.code == "CONFIGURED_VIDEOS_UNDER_MISSING_ROOT"
    assert finding.severity == Severity.WARNING
    assert (finding.schema_id, finding.reach, finding.field_location) == (
        "dlc.project.config",
        Reach.ONE_PER_PROJECT,
        ("video_sets",),
    )
    assert finding.evidence == {
        "declared_paths": [str(vanished_root / "videos" / name) for name in names],
        "matched_paths": [f"videos/{name}" for name in names],
    }


def test_a_key_written_with_backslashes_still_names_its_video(
    valid_project: Path,
) -> None:
    """A Windows-written key read on Linux is one path segment there; it still matches by name."""

    (valid_project / "videos" / "session-01.mp4").touch()
    key = r"Z:\vanished-root\videos\session-01.mp4"

    set_configured_videos(valid_project, (Path(key),))

    snapshot = inspect_project(valid_project, validation=video_filename_request())

    (finding,) = snapshot.findings

    assert finding.code == "CONFIGURED_VIDEOS_UNDER_MISSING_ROOT"
    assert finding.evidence["declared_paths"] == [key]


def test_a_key_into_another_copy_of_the_project_is_reported(
    valid_project: Path,
    tmp_path: Path,
) -> None:
    (valid_project / "videos" / "session-01.mp4").write_bytes(b"here")
    other = tmp_path / "other-copy" / "videos" / "session-01.mp4"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"there")

    set_configured_videos(valid_project, (other,))

    snapshot = inspect_project(valid_project, validation=video_filename_request())

    (finding,) = snapshot.findings

    assert finding.code == "CONFIGURED_VIDEO_IN_OTHER_COPY"
    assert finding.severity == Severity.WARNING
    assert finding.field_location == ("video_sets", str(other))
    assert finding.path == other.absolute()
    assert finding.evidence == {"matched_path": "videos/session-01.mp4"}


def test_a_key_to_a_file_with_nothing_in_videos_is_outside_the_project(
    valid_project: Path,
    tmp_path: Path,
) -> None:
    outside = tmp_path / "raw-data" / "session-01.mp4"
    outside.parent.mkdir()
    outside.touch()

    set_configured_videos(valid_project, (outside,))

    snapshot = inspect_project(valid_project, validation=video_filename_request())

    (finding,) = snapshot.findings

    assert finding.code == "CONFIGURED_VIDEO_OUTSIDE_PROJECT"
    assert finding.severity == Severity.ERROR
    assert finding.field_location == ("video_sets", str(outside))


def test_a_video_no_key_matches_is_reported_at_that_video(
    valid_project: Path,
) -> None:
    declared = valid_project / "videos" / "session-01.mp4"
    declared.touch()
    (valid_project / "videos" / "session-02.mp4").touch()

    set_configured_videos(valid_project, (declared,))

    snapshot = inspect_project(valid_project, validation=video_filename_request())

    (finding,) = snapshot.findings

    assert finding.code == "VIDEO_NOT_IN_VIDEO_SETS"
    assert finding.severity == Severity.NOTE
    assert (finding.schema_id, finding.reach, finding.instance) == (
        "dlc.videos.source_video",
        Reach.ONE_OF,
        ("videos", "session-02.mp4"),
    )


@pytest.mark.xfail(
    reason="DeepLabCut builds a training set from the stems of video_sets keys "
    "(merge_annotateddatasets, v3.0.2), so a labeled dataset no key's stem names is left out of training "
    "silently whenever any key matches; it is to be a finding at that dataset. No labeled-data suite is "
    "registered yet, so requesting one raises",
    strict=True,
    raises=(ValueError, AssertionError),
)
def test_a_labeled_dataset_no_key_names_is_reported(
    valid_project: Path,
) -> None:
    declared = valid_project / "videos" / "session-01.mp4"
    declared.touch()

    for stem in ("session-01", "session-07"):
        dataset = valid_project / "labeled-data" / stem
        dataset.mkdir()
        (dataset / "CollectedData_test-scorer.h5").touch()

    set_configured_videos(valid_project, (declared,))

    snapshot = inspect_project(
        valid_project,
        validation=ProjectValidationRequest(
            resources=tuple(
                ResourceValidationRequest(kind=kind, depth=ValidationDepth.FILENAME_ONLY)
                for kind in (DirectoryResourceKind.VIDEOS_DIRECTORY, DirectoryResourceKind.LABELED_DATA_DIRECTORY)
            )
        ),
    )

    assert [
        (finding.schema_id, finding.reach, finding.instance)
        for finding in snapshot.findings
        if finding.instance[:2] == ("labeled-data", "session-07")
    ] == [("dlc.labeled_data.dataset", Reach.ONE_OF, ("labeled-data", "session-07"))]


def test_the_demo_placeholder_key_is_reported_missing_not_raised(
    valid_project: Path,
) -> None:
    placeholder = "WILL BE AUTOMATICALLY UPDATED BY DEMO CODE"

    set_configured_videos(valid_project, (Path(placeholder),))

    snapshot = inspect_project(valid_project, validation=video_filename_request())

    (finding,) = snapshot.findings

    assert snapshot.validation_checks[0].status == ValidationCheckStatus.COMPLETED
    assert finding.code == "CONFIGURED_VIDEO_PATH_MISSING"
    assert finding.field_location == ("video_sets", placeholder)


def test_a_broken_link_in_videos_is_reported_once(
    valid_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The structure finding at the video stands; the key adds no second one.

    The file exists so discovery lists it; both probes are faked as a broken link.
    """

    link = valid_project / "videos" / "session-01.mp4"
    link.touch()
    broken = _symlink_probe(valid_project / "gone.mp4", SymlinkTargetKind.MISSING)
    real_probe_path = probe_path

    def fake_probe_path(path: Path) -> PathProbe:
        return broken(path) if path.name == link.name else real_probe_path(path)

    monkeypatch.setattr("dlccore.project.discovery.observer.probe_path", fake_probe_path)
    monkeypatch.setattr(videos, "probe_path", fake_probe_path)
    set_configured_videos(valid_project, (link,))

    snapshot = inspect_project(valid_project, validation=video_filename_request())

    assert [finding.code for finding in snapshot.findings] == ["PROJECT_SYMLINK_TARGET_MISSING"]


def test_a_resolved_path_never_reaches_a_finding(
    valid_project: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On Windows with an SMB-mounted drive, ``Path.resolve`` returns ``\\\\?\\Volume{GUID}\\...``."""

    for name in ("session-01.mp4", "session-02.mp4", "session-03.mp4", "session-04.mp4"):
        (valid_project / "videos" / name).write_bytes(name.encode())

    other = tmp_path / "other-copy" / "videos" / "session-03.mp4"
    outside = tmp_path / "raw-data" / "session-05.mp4"

    for path in (other, outside):
        path.parent.mkdir(parents=True)
        path.touch()

    set_configured_videos(
        valid_project,
        (
            valid_project / "videos" / "session-01.mp4",
            tmp_path / "vanished-root" / "videos" / "session-02.mp4",
            other,
            outside,
        ),
    )

    real_resolve = Path.resolve
    resolved: list[Path] = []

    def volume_guid_resolve(self: Path, strict: bool = False) -> Path:
        path = real_resolve(self, strict=strict)
        resolved.append(path)

        return Path(r"\\?\Volume{0f1e2d3c-0000-0000-0000-000000000000}") / path.relative_to(path.anchor)

    monkeypatch.setattr(Path, "resolve", volume_guid_resolve)

    snapshot = inspect_project(valid_project, validation=video_filename_request())

    assert resolved
    assert {finding.code for finding in snapshot.findings} == {
        "CONFIGURED_VIDEOS_UNDER_MISSING_ROOT",
        "CONFIGURED_VIDEO_IN_OTHER_COPY",
        "CONFIGURED_VIDEO_OUTSIDE_PROJECT",
        "VIDEO_NOT_IN_VIDEO_SETS",
    }
    assert "Volume{" not in snapshot.model_dump_json()


def test_configured_video_directory_produces_not_file_finding(
    valid_project: Path,
) -> None:
    directory_path = valid_project / "videos" / "not-a-video"
    directory_path.mkdir()

    set_configured_videos(
        valid_project,
        (directory_path,),
    )

    snapshot = inspect_project(
        valid_project,
        validation=video_filename_request(),
    )

    finding = next(finding for finding in snapshot.findings if finding.code == "CONFIGURED_VIDEO_PATH_NOT_FILE")

    assert finding.path == directory_path
    assert finding.field_location == (
        "video_sets",
        str(directory_path),
    )


def test_configured_video_wrong_type_still_produces_not_file_finding(
    valid_project: Path,
) -> None:
    """A schema-matched video path that is a directory still gets its own finding.

    ``videos/session-01.mp4`` matches the source-video naming pattern, so it
    becomes a WRONG_TYPE resource (not an unclaimed one) once discovered -
    this must not be mistaken for a valid, present video.
    """

    video_path = valid_project / "videos" / "session-01.mp4"
    video_path.mkdir()

    set_configured_videos(
        valid_project,
        (video_path,),
    )

    snapshot = inspect_project(
        valid_project,
        validation=video_filename_request(),
    )

    finding = next(finding for finding in snapshot.findings if finding.code == "CONFIGURED_VIDEO_PATH_NOT_FILE")

    assert finding.path == video_path
    assert finding.field_location == (
        "video_sets",
        str(video_path),
    )

    resource = snapshot.resource_at("videos/session-01.mp4")

    assert resource is not None


def test_configured_video_symlink_to_file_is_a_present_file(
    valid_project: Path,
    create_symlink_or_skip: Callable[..., None],
) -> None:
    """A declared video that is a symlink to a file is a file, not a broken path.

    The link sits outside ``videos/`` so it is not indexed as an observed
    source video, which routes the check through its ``probe_path`` fallback.
    """

    target = valid_project / "external.mp4"
    target.touch()

    link = valid_project / "linked.mp4"
    create_symlink_or_skip(
        link,
        target,
    )

    set_configured_videos(
        valid_project,
        (link,),
    )

    snapshot = inspect_project(
        valid_project,
        validation=video_filename_request(),
    )

    assert [finding.code for finding in snapshot.findings] == ["CONFIGURED_VIDEO_OUTSIDE_PROJECT"]


def test_configured_video_symlink_to_file_is_a_present_file_from_probe_alone(
    valid_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback takes a symlink-to-file probe as a file without creating one.

    Mirrors the test above on platforms that withhold symlink privileges, so
    the accepting branch stays covered where the filesystem cannot exercise it.
    """

    link = valid_project / "linked.mp4"
    target = valid_project / "external.mp4"

    monkeypatch.setattr(
        videos,
        "probe_path",
        _symlink_probe(
            target,
            SymlinkTargetKind.FILE,
        ),
    )

    set_configured_videos(
        valid_project,
        (link,),
    )

    snapshot = inspect_project(
        valid_project,
        validation=video_filename_request(),
    )

    assert [finding.code for finding in snapshot.findings] == ["CONFIGURED_VIDEO_OUTSIDE_PROJECT"]


def test_configured_video_broken_symlink_reports_missing_target(
    valid_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken symlink is distinguished from an unreadable path."""

    link = valid_project / "linked.mp4"
    target = valid_project / "missing.mp4"

    monkeypatch.setattr(
        videos,
        "probe_path",
        _symlink_probe(
            target,
            SymlinkTargetKind.MISSING,
        ),
    )

    set_configured_videos(
        valid_project,
        (link,),
    )

    snapshot = inspect_project(
        valid_project,
        validation=video_filename_request(),
    )

    finding = next(
        finding for finding in snapshot.findings if finding.code == "CONFIGURED_VIDEO_PATH_SYMLINK_TARGET_MISSING"
    )

    assert finding.evidence["absolute_target"] == str(target.absolute())


def test_configured_video_unreadable_path_is_not_reported_as_wrong_type(
    valid_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An access failure must not be reported as a type error.

    The path is never created, so project discovery never probes it and only
    the configured-path check reaches the patched probe.
    """

    unreadable = valid_project / "unreadable.mp4"

    def fake_probe_path(_path: Path) -> PathProbe:
        return PathProbe(
            status=PathProbeStatus.INACCESSIBLE,
            error_type="PermissionError",
            error_message="Access is denied",
        )

    monkeypatch.setattr(videos, "probe_path", fake_probe_path)

    set_configured_videos(
        valid_project,
        (unreadable,),
    )

    snapshot = inspect_project(
        valid_project,
        validation=video_filename_request(),
    )

    finding = next(finding for finding in snapshot.findings if finding.code.startswith("CONFIGURED_VIDEO_PATH_"))

    assert finding.code == "CONFIGURED_VIDEO_PATH_UNREADABLE"
    assert finding.evidence["error_type"] == "PermissionError"


def test_relative_configured_video_path_is_resolved_from_project_root(
    valid_project: Path,
) -> None:
    relative_path = Path("videos/session-01.mp4")
    (valid_project / relative_path).touch()

    set_configured_videos(
        valid_project,
        (relative_path,),
    )

    snapshot = inspect_project(
        valid_project,
        validation=video_filename_request(),
    )

    assert not any(finding.code.startswith("CONFIGURED_VIDEO_PATH_") for finding in snapshot.findings)


def test_video_validation_is_unavailable_without_canonical_config(
    tmp_path: Path,
) -> None:
    snapshot = inspect_project(
        tmp_path,
        validation=video_filename_request(),
    )

    assert len(snapshot.validation_checks) == 1

    check = snapshot.validation_checks[0]

    assert check.code == "VIDEO_FILENAME_VALIDATION"
    assert check.status == ValidationCheckStatus.UNAVAILABLE
    assert not any(finding.code.startswith("CONFIGURED_VIDEO_PATH_") for finding in snapshot.findings)


def test_two_videos_resolving_to_one_file_are_reported_not_raised(
    valid_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two symlinks to one master file is a normal disk-saving layout.

    Probes are faked for the two links only, so the collision branch is covered
    where the platform withholds the privilege needed to create a real link.
    """

    master = valid_project / "master.mp4"
    master.touch()

    linked = {"session-01.mp4", "session-02.mp4"}

    for name in sorted(linked):
        (valid_project / "videos" / name).touch()

    real_probe_path = probe_path

    def fake_probe_path(path: Path) -> PathProbe:
        if path.name in linked:
            return _symlink_probe(
                master,
                SymlinkTargetKind.FILE,
            )(path)

        return real_probe_path(path)

    monkeypatch.setattr(
        "dlccore.project.discovery.observer.probe_path",
        fake_probe_path,
    )

    set_configured_videos(
        valid_project,
        (master,),
    )

    snapshot = inspect_project(
        valid_project,
        validation=video_filename_request(),
    )

    collisions = tuple(finding for finding in snapshot.findings if finding.code == "PROJECT_RESOURCE_PATH_COLLISION")

    assert [finding.instance for finding in collisions] == [
        ("videos", "session-01.mp4"),
        ("videos", "session-02.mp4"),
    ]

    for finding in collisions:
        assert finding.severity == Severity.ERROR
        assert (finding.schema_id, finding.reach) == ("dlc.videos.source_video", Reach.ONE_OF)
        assert finding.evidence["collision_path"] == str(master.absolute())
        assert finding.evidence["resource_kind"] == "source_video"
        assert finding.evidence["collision_group"] == ["videos", "session-01.mp4"]


def test_a_collision_reports_each_video_once_with_a_shared_key(
    valid_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    master = valid_project / "master.mp4"
    master.touch()

    linked = {"session-01.mp4", "session-02.mp4"}

    for name in sorted(linked):
        (valid_project / "videos" / name).touch()

    real_probe_path = probe_path

    def fake_probe_path(path: Path) -> PathProbe:
        if path.name in linked:
            return _symlink_probe(master, SymlinkTargetKind.FILE)(path)

        return real_probe_path(path)

    monkeypatch.setattr("dlccore.project.discovery.observer.probe_path", fake_probe_path)
    set_configured_videos(valid_project, (master,))

    snapshot = inspect_project(valid_project, validation=video_filename_request())
    collisions = sorted(
        (finding for finding in snapshot.findings if finding.code == "PROJECT_RESOURCE_PATH_COLLISION"),
        key=lambda finding: finding.instance,
    )

    assert [(finding.schema_id, finding.instance) for finding in collisions] == [
        ("dlc.videos.source_video", ("videos", "session-01.mp4")),
        ("dlc.videos.source_video", ("videos", "session-02.mp4")),
    ]
    assert collisions[0].evidence["collision_group"] == collisions[1].evidence["collision_group"]


def test_three_videos_resolving_to_one_file_are_one_collision(
    valid_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    master = valid_project / "master.mp4"
    master.touch()

    linked = {"session-01.mp4", "session-02.mp4", "session-03.mp4"}

    for name in sorted(linked):
        (valid_project / "videos" / name).touch()

    real_probe_path = probe_path

    def fake_probe_path(path: Path) -> PathProbe:
        if path.name in linked:
            return _symlink_probe(master, SymlinkTargetKind.FILE)(path)

        return real_probe_path(path)

    monkeypatch.setattr("dlccore.project.discovery.observer.probe_path", fake_probe_path)
    set_configured_videos(valid_project, (master,))

    snapshot = inspect_project(valid_project, validation=video_filename_request())
    collisions = [finding for finding in snapshot.findings if finding.code == "PROJECT_RESOURCE_PATH_COLLISION"]

    assert sorted(finding.instance for finding in collisions) == [("videos", name) for name in sorted(linked)]
    assert {tuple(finding.evidence["collision_group"]) for finding in collisions} == {("videos", "session-01.mp4")}


def test_two_real_links_to_one_file_collide(
    valid_project: Path,
    create_symlink_or_skip: Callable[..., None],
) -> None:
    """The unfaked counterpart of the test above."""

    master = valid_project / "master.mp4"
    master.touch()

    for name in ("session-01.mp4", "session-02.mp4"):
        create_symlink_or_skip(
            valid_project / "videos" / name,
            master,
        )

    snapshot = inspect_project(
        valid_project,
        validation=video_filename_request(),
    )

    collisions = tuple(finding for finding in snapshot.findings if finding.code == "PROJECT_RESOURCE_PATH_COLLISION")

    assert len(collisions) == 2
    assert {finding.evidence["collision_path"] for finding in collisions} == {str(master.absolute())}


def test_a_chain_of_links_to_one_file_collides(
    valid_project: Path,
    create_symlink_or_skip: Callable[..., None],
) -> None:
    """Identity is the file a chain ends at; evidence keeps the observed first hop."""

    master = valid_project / "master.mp4"
    master.touch()

    hop = valid_project / "hop.mp4"
    create_symlink_or_skip(
        hop,
        master,
    )
    create_symlink_or_skip(
        valid_project / "videos" / "session-01.mp4",
        master,
    )
    create_symlink_or_skip(
        valid_project / "videos" / "session-02.mp4",
        hop,
    )

    snapshot = inspect_project(
        valid_project,
        validation=video_filename_request(),
    )

    collisions = tuple(finding for finding in snapshot.findings if finding.code == "PROJECT_RESOURCE_PATH_COLLISION")

    by_instance = {finding.instance: finding.evidence for finding in collisions}

    assert by_instance[("videos", "session-01.mp4")]["collision_path"] == str(master.absolute())
    assert by_instance[("videos", "session-02.mp4")]["collision_path"] == str(hop.absolute())
    assert {tuple(evidence["collision_group"]) for evidence in by_instance.values()} == {
        ("videos", "session-01.mp4"),
    }
    assert len(collisions) == 2
