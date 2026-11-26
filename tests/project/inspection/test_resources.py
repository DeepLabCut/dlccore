from collections.abc import Callable
from pathlib import Path

from dlc_octopus.api import inspect_project
from dlc_octopus.schemas.base.resources import (
    FileResourceKind,
    ProjectResource,
    ResourceManagement,
)
from dlc_octopus.schemas.snapshots import ProjectSnapshot


def test_napari_metadata_is_known_unmanaged(
    valid_project: Path,
    resources_by_path: Callable[
        [ProjectSnapshot],
        dict[Path, ProjectResource],
    ],
) -> None:
    dataset = valid_project / "labeled-data" / "session-01"
    dataset.mkdir()

    metadata = dataset / ".napari-deeplabcut.json"
    metadata.write_text(
        "{}",
        encoding="utf-8",
    )

    snapshot = inspect_project(valid_project)
    resources = resources_by_path(snapshot)
    resource = resources[Path("labeled-data/session-01/.napari-deeplabcut.json")]

    assert resource.kind == FileResourceKind.NAPARI_DLC_METADATA
    assert resource.management == ResourceManagement.KNOWN_UNMANAGED
    assert not any(finding.path == metadata for finding in snapshot.findings)
