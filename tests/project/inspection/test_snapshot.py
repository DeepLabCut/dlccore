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

from collections.abc import Callable
from pathlib import Path

import pytest

from dlccore.project.api import inspect_project
from dlccore.project.schemas.base.observations import NodeStatus
from dlccore.project.schemas.base.resources import (
    DirectoryResourceKind,
    FileResourceKind,
    ProjectResource,
)
from dlccore.project.schemas.base.validation import (
    ValidationCheck,
    ValidationCheckStatus,
    ValidationDepth,
)
from dlccore.project.schemas.snapshots import ConfigState, ProjectSnapshot
from dlccore.project.validation.results import ValidationResult


def test_inspect_project_returns_expected_snapshot(
    valid_project: Path,
    resources_by_path: Callable[
        [ProjectSnapshot],
        dict[Path, ProjectResource],
    ],
) -> None:
    videos = valid_project / "videos"
    dataset = valid_project / "labeled-data" / "session-01"
    dataset.mkdir()

    (videos / "session-01.mp4").touch()
    (videos / "notes.txt").write_text(
        "user notes",
        encoding="utf-8",
    )
    (dataset / "img001.png").touch()
    (dataset / "CollectedData_test-scorer.h5").touch()

    snapshot = inspect_project(valid_project)
    resources = resources_by_path(snapshot)

    assert snapshot.project_root == valid_project
    assert snapshot.project.status == NodeStatus.PRESENT
    assert resources[Path("config.yaml")].kind == FileResourceKind.PROJECT_CONFIG
    assert resources[Path("videos/session-01.mp4")].kind == FileResourceKind.SOURCE_VIDEO
    assert resources[Path("videos/notes.txt")].kind == FileResourceKind.UNKNOWN_FILE
    assert resources[Path("labeled-data/session-01")].kind == DirectoryResourceKind.LABELED_DATASET


def test_resources_by_path_indexes_all_resources(
    valid_project: Path,
) -> None:
    video = valid_project / "videos" / "session-01.mp4"
    video.touch()

    snapshot = inspect_project(valid_project)

    assert snapshot.resources_by_path[Path("config.yaml")].kind == FileResourceKind.PROJECT_CONFIG
    assert snapshot.resources_by_path[Path("videos/session-01.mp4")].kind == FileResourceKind.SOURCE_VIDEO


def test_resources_of_kind_returns_only_requested_kind(
    valid_project: Path,
) -> None:
    videos = valid_project / "videos"

    (videos / "session-01.mp4").touch()
    (videos / "session-02.avi").touch()
    (videos / "notes.txt").write_text(
        "notes",
        encoding="utf-8",
    )

    snapshot = inspect_project(valid_project)
    source_videos = snapshot.resources_of_kind(FileResourceKind.SOURCE_VIDEO)

    assert tuple(resource.project_relative_path for resource in source_videos) == (
        Path("videos/session-01.mp4"),
        Path("videos/session-02.avi"),
    )
    assert all(resource.kind == FileResourceKind.SOURCE_VIDEO for resource in source_videos)


def test_resources_of_kind_supports_directory_kinds(
    valid_project: Path,
) -> None:
    snapshot = inspect_project(valid_project)

    resources = snapshot.resources_of_kind(DirectoryResourceKind.VIDEOS_DIRECTORY)

    assert len(resources) == 1
    assert resources[0].project_relative_path == Path("videos")


def test_resources_of_kind_returns_empty_tuple_when_absent(
    valid_project: Path,
) -> None:
    snapshot = inspect_project(valid_project)

    assert snapshot.resources_of_kind(FileResourceKind.MACHINE_LABELS) == ()


def test_resource_at_returns_matching_resource(
    valid_project: Path,
) -> None:
    video = valid_project / "videos" / "session-01.mp4"
    video.touch()

    snapshot = inspect_project(valid_project)

    resource = snapshot.resource_at("videos/session-01.mp4")

    assert resource is not None
    assert resource.kind == FileResourceKind.SOURCE_VIDEO


def test_resource_at_returns_none_when_absent(
    valid_project: Path,
) -> None:
    snapshot = inspect_project(valid_project)

    assert snapshot.resource_at("videos/missing.mp4") is None


def test_valid_config_state(
    valid_project: Path,
) -> None:
    snapshot = inspect_project(valid_project)

    assert snapshot.config_state == ConfigState.VALID


def test_missing_config_state(
    tmp_path: Path,
) -> None:
    snapshot = inspect_project(tmp_path)

    assert snapshot.config_state == ConfigState.MISSING


def test_invalid_config_state(
    invalid_project: Path,
) -> None:
    snapshot = inspect_project(invalid_project)

    assert snapshot.config_state == ConfigState.INVALID


def test_resource_projections_do_not_mutate_snapshot(
    valid_project: Path,
) -> None:
    snapshot = inspect_project(valid_project)
    resources_before = snapshot.resources

    snapshot.resources_of_kind(FileResourceKind.PROJECT_CONFIG)
    snapshot.resource_at("config.yaml")
    _ = snapshot.resources_by_path

    assert snapshot.resources == resources_before


def _check(
    code: str,
) -> ValidationCheck:
    """Return a completed check carrying one code."""

    return ValidationCheck(
        code=code,
        kind=DirectoryResourceKind.VIDEOS_DIRECTORY,
        depth=ValidationDepth.FILENAME_ONLY,
        status=ValidationCheckStatus.COMPLETED,
        message="Checked.",
    )


def test_with_validation_result_accumulates_checks_across_calls(
    valid_project: Path,
) -> None:
    """Checks accumulate as findings do, matching ``ValidationResult.merge``."""

    snapshot = inspect_project(valid_project)

    merged = snapshot.with_validation_result(
        ValidationResult(checks=(_check("FIRST"),)),
    ).with_validation_result(
        ValidationResult(checks=(_check("SECOND"),)),
    )

    assert tuple(check.code for check in merged.validation_checks) == (
        "FIRST",
        "SECOND",
    )


@pytest.mark.xfail(
    reason="a consumer is to get the resource kind of any schema_id in one call, including for a node missing "
    "on disk; resource_at covers only existing entries, by path, and resource_kind in structure evidence is "
    "the stop-gap until then. When this passes, drop resource_kind from the structure codes' evidence_keys",
    strict=True,
    raises=AttributeError,
)
def test_the_kind_of_a_schema_id_is_one_call_away(
    tmp_path: Path,
) -> None:
    snapshot = inspect_project(tmp_path)

    assert snapshot.resource_at(Path("config.yaml")) is None
    assert snapshot.kind_of("dlc.project.config") is FileResourceKind.PROJECT_CONFIG
