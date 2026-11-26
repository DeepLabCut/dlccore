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

from collections.abc import Iterator
from pathlib import Path

import pytest

from dlccore.project.discovery.observer import discover_directory_node, discover_project
from dlccore.project.schemas.base.filesystem import (
    DirectorySchema,
    FileSchema,
    NamePattern,
    NodeSchema,
)
from dlccore.project.schemas.base.observations import (
    NodeStatus,
)
from dlccore.project.schemas.base.probes import FilesystemEntryKind, PathProbeStatus
from dlccore.project.schemas.base.resources import (
    DirectoryResourceKind,
    FileResourceKind,
    ResourceManagement,
)
from dlccore.project.schemas.project_layout.project import PROJECT


def test_discovered_directory_populates_renamed_fields(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.yaml").touch()
    (tmp_path / "videos").mkdir()

    project = discover_project(tmp_path)

    assert project.schema_children
    assert {child.project_relative_path for child in project.schema_children} >= {
        Path("config.yaml"),
        Path("videos"),
    }

    model_fields = type(project).model_fields

    assert "children" not in model_fields
    assert "unknown_children" not in model_fields


def test_discover_project_fixed_children(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("", encoding="utf-8")
    (tmp_path / "videos").mkdir()

    observed = discover_project(
        tmp_path,
    )

    statuses = {child.project_relative_path: child.status for child in observed.schema_children}

    assert statuses[Path("config.yaml")] == NodeStatus.PRESENT
    assert statuses[Path("videos")] == NodeStatus.PRESENT
    assert statuses[Path("labeled-data")] == NodeStatus.MISSING


def test_observe_fixed_child_with_wrong_type(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("", encoding="utf-8")
    (tmp_path / "videos").write_text(
        "not a directory",
        encoding="utf-8",
    )

    observed = discover_directory_node(
        physical_path=tmp_path,
        project_relative_path=Path(),
        project_root=tmp_path,
        schema=PROJECT,
    )

    videos = next(child for child in observed.schema_children if child.project_relative_path == Path("videos"))

    assert videos.status == NodeStatus.WRONG_TYPE
    assert videos.should_be_directory


def test_observe_repeated_video_files(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        "Task: demo",
        encoding="utf-8",
    )

    videos = tmp_path / "videos"
    videos.mkdir()

    (videos / "session-01.mp4").touch()
    (videos / "session-02.avi").touch()
    (videos / "notes.txt").write_text(
        "user notes",
        encoding="utf-8",
    )

    observed = discover_project(
        tmp_path,
    )

    videos_node = next(
        child for child in observed.schema_children if child.kind == DirectoryResourceKind.VIDEOS_DIRECTORY
    )

    source_videos = [child for child in videos_node.schema_children if child.kind == FileResourceKind.SOURCE_VIDEO]

    assert {node.project_relative_path for node in source_videos} == {
        Path("videos/session-01.mp4"),
        Path("videos/session-02.avi"),
    }


def test_observe_labeled_dataset(tmp_path: Path) -> None:
    labeled_data = tmp_path / "labeled-data"
    dataset = labeled_data / "session-01"

    dataset.mkdir(parents=True)

    (dataset / "img001.png").touch()
    (dataset / "CollectedData_Sc.h5").touch()

    observed = discover_project(
        tmp_path,
    )

    labeled_data_node = next(
        child for child in observed.schema_children if child.kind == DirectoryResourceKind.LABELED_DATA_DIRECTORY
    )

    dataset_node = next(
        child for child in labeled_data_node.schema_children if child.kind == DirectoryResourceKind.LABELED_DATASET
    )

    assert dataset_node.project_relative_path == Path("labeled-data/session-01")

    assert {child.kind for child in dataset_node.schema_children} == {
        FileResourceKind.LABELED_FRAME,
        FileResourceKind.HUMAN_LABELS,
        FileResourceKind.NAPARI_DLC_METADATA,
    }


def test_loose_file_in_labeled_data_is_retained_unclaimed(tmp_path: Path) -> None:
    """A file beside the datasets is unidentified, never a wrong-type dataset.

    The dataset schema's name pattern matches every entry, so without entry-kind
    matching any file here is claimed as a labeled dataset and reported
    ``WRONG_TYPE``. Whatever a user leaves in this directory is theirs: it is
    retained and left alone, not judged against a schema it never belonged to.
    """

    labeled_data = tmp_path / "labeled-data"
    dataset = labeled_data / "session-01"

    dataset.mkdir(parents=True)

    (dataset / "img001.png").touch()
    (labeled_data / "notes.txt").write_text("user notes", encoding="utf-8")
    (labeled_data / "export.csv").write_text("", encoding="utf-8")

    observed = discover_project(
        tmp_path,
    )

    labeled_data_node = next(
        child for child in observed.schema_children if child.kind == DirectoryResourceKind.LABELED_DATA_DIRECTORY
    )

    assert {child.project_relative_path for child in labeled_data_node.schema_children} == {
        Path("labeled-data/session-01"),
    }

    assert all(child.status != NodeStatus.WRONG_TYPE for child in labeled_data_node.schema_children)

    unclaimed = {resource.project_relative_path: resource for resource in labeled_data_node.unclaimed_resources}

    assert set(unclaimed) == {
        Path("labeled-data/notes.txt"),
        Path("labeled-data/export.csv"),
    }

    assert {resource.kind for resource in unclaimed.values()} == {FileResourceKind.UNKNOWN_FILE}
    assert {resource.management for resource in unclaimed.values()} == {ResourceManagement.UNDETERMINED}


def _repeated_child_schemas() -> list[tuple[DirectorySchema, NodeSchema]]:
    """Every (parent, repeated child) pair reachable from the project schema."""

    pairs: list[tuple[DirectorySchema, NodeSchema]] = []
    pending: list[NodeSchema] = [PROJECT]

    while pending:
        schema = pending.pop()

        if not isinstance(schema, DirectorySchema):
            continue

        pairs.extend((schema, repeated.child) for repeated in schema.repeated_children)
        pending.extend(schema.children)
        pending.extend(repeated.child for repeated in schema.repeated_children)

    return pairs


def _name_matching(pattern: NamePattern) -> str:
    """Build a filesystem name the pattern accepts."""

    if pattern.exact is not None:
        return pattern.exact

    suffix = sorted(pattern.suffixes)[0] if pattern.suffixes else ""

    return f"{pattern.prefix or ''}entry{suffix}"


@pytest.mark.parametrize(
    ("parent", "child"),
    _repeated_child_schemas(),
    ids=lambda schema: getattr(schema, "schema_id", ""),
)
def test_every_repeated_child_leaves_a_wrong_kind_entry_unclaimed(
    parent: DirectorySchema,
    child: NodeSchema,
    tmp_path: Path,
) -> None:
    """No schema claims an entry of the kind it does not declare.

    Parametrised over the whole layout rather than over the cases that prompted
    the rule, so a schema added later is covered without anyone remembering to
    come back. A new repeated child that mis-claims fails here on the day it
    lands.
    """

    name = _name_matching(child.name)
    entry = tmp_path / name

    if isinstance(child, FileSchema):
        entry.mkdir()
    else:
        entry.write_text("", encoding="utf-8")

    observed = discover_directory_node(
        physical_path=tmp_path,
        project_relative_path=Path(),
        project_root=tmp_path,
        schema=parent,
    )

    assert name not in {node.project_relative_path.name for node in observed.schema_children}
    assert name in {resource.project_relative_path.name for resource in observed.unclaimed_resources}


@pytest.mark.xfail(
    reason="check_labels plot output is claimed as a labeled dataset and its plots as labeled frames",
    strict=True,
    raises=AssertionError,
)
def test_check_labels_plot_output_is_not_a_labeled_dataset(tmp_path: Path) -> None:
    labeled_data = tmp_path / "labeled-data"
    dataset = labeled_data / "session-01"
    plot_output = labeled_data / "session-01_labeled"

    dataset.mkdir(parents=True)
    plot_output.mkdir()

    (dataset / "img001.png").touch()
    (plot_output / "img001_bodypart.png").touch()

    observed = discover_project(
        tmp_path,
    )

    labeled_data_node = next(
        child for child in observed.schema_children if child.kind == DirectoryResourceKind.LABELED_DATA_DIRECTORY
    )

    datasets = {
        child.project_relative_path
        for child in labeled_data_node.schema_children
        if child.kind == DirectoryResourceKind.LABELED_DATASET
    }

    assert datasets == {Path("labeled-data/session-01")}


def test_retain_unknown_video_entries(tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    backups = videos / "old-backups"

    videos.mkdir()
    backups.mkdir()

    (videos / "session-01.mp4").touch()
    (videos / "notes.txt").write_text(
        "user notes",
        encoding="utf-8",
    )

    observed = discover_project(
        tmp_path,
    )

    videos_node = next(
        child for child in observed.schema_children if child.kind == DirectoryResourceKind.VIDEOS_DIRECTORY
    )

    assert {child.project_relative_path for child in videos_node.schema_children} == {
        Path("videos/session-01.mp4"),
    }

    assert {resource.project_relative_path for resource in videos_node.unclaimed_resources} == {
        Path("videos/notes.txt"),
        Path("videos/old-backups"),
    }

    unknown = {resource.project_relative_path: resource.kind for resource in videos_node.unclaimed_resources}

    assert unknown[Path("videos/notes.txt")] == FileResourceKind.UNKNOWN_FILE

    assert unknown[Path("videos/old-backups")] == DirectoryResourceKind.UNKNOWN_DIRECTORY


def test_directory_traversal_failure_is_independent_of_entry_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed ``iterdir()`` marks the node UNREADABLE and keeps why, without touching its probe."""

    videos = tmp_path / "videos"
    videos.mkdir()

    real_iterdir = Path.iterdir

    def fake_iterdir(self: Path, *args: object, **kwargs: object) -> Iterator[Path]:
        if self == videos:
            raise OSError("Directory listing failed")

        return real_iterdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "iterdir", fake_iterdir)

    observed = discover_project(tmp_path)

    videos_node = next(child for child in observed.schema_children if child.project_relative_path == Path("videos"))

    assert videos_node.status == NodeStatus.UNREADABLE
    assert videos_node.traversal_error_type == "OSError"
    assert videos_node.traversal_error_message == "Directory listing failed"
    assert videos_node.probe.status == PathProbeStatus.OBSERVED
    assert videos_node.probe.entry_kind == FilesystemEntryKind.DIRECTORY
    assert videos_node.probe.error_type is None
