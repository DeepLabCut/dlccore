from __future__ import annotations

from pathlib import Path

from dlc_octopus.discovery.observer import (
    ExpectedEntryKind,
    discover_project,
    status_for_expected_entry,
)
from dlc_octopus.schemas.base.filesystem import (
    DirectorySchema,
    FileSchema,
    NamePattern,
    RepeatedNodeSchema,
    index_layout,
)
from dlc_octopus.schemas.base.findings import (
    Severity,
)
from dlc_octopus.schemas.base.observations import (
    NodeStatus,
    ObservedNode,
)
from dlc_octopus.schemas.base.probes import (
    FilesystemEntryKind,
    PathProbe,
    PathProbeStatus,
)
from dlc_octopus.schemas.base.resources import (
    DirectoryResourceKind,
    FileResourceKind,
)
from dlc_octopus.schemas.base.symlinks import (
    SymlinkObservation,
    SymlinkTargetKind,
)
from dlc_octopus.validation.structure import (
    validate_project_structure,
)

_RAW_TARGET = Path("../external.mp4")


def _observed_directory(
    *,
    schema_id: str,
    kind: DirectoryResourceKind,
    project_relative_path: Path,
    schema_children: tuple[ObservedNode, ...],
) -> ObservedNode:
    """Build a present directory node with the given schema children."""

    return ObservedNode(
        schema_id=schema_id,
        kind=kind,
        project_relative_path=project_relative_path,
        probe=PathProbe(
            status=PathProbeStatus.OBSERVED,
            entry_kind=FilesystemEntryKind.DIRECTORY,
        ),
        status=NodeStatus.PRESENT,
        schema_children=schema_children,
    )


def _project_with_symlinked_video(
    *,
    target_kind: SymlinkTargetKind,
    project_root: Path,
) -> ObservedNode:
    """Build an observed project whose only source video is a symlink.

    The node status is derived through ``status_for_expected_entry`` rather
    than assigned, so these tests exercise the real derivation.

    Args:
        target_kind: Observed type of the symlink target.
        project_root: Project root the absolute target is resolved against.

    Returns:
        The observed project root.
    """

    probe = PathProbe(
        status=PathProbeStatus.OBSERVED,
        entry_kind=FilesystemEntryKind.SYMLINK,
        symlink=SymlinkObservation(
            raw_target=_RAW_TARGET,
            absolute_target=project_root / "external.mp4",
            target_kind=target_kind,
        ),
    )

    video = ObservedNode(
        schema_id="dlc.videos.source_video",
        kind=FileResourceKind.SOURCE_VIDEO,
        project_relative_path=Path("videos") / "session-01.mp4",
        probe=probe,
        status=status_for_expected_entry(
            probe,
            expected_kind=ExpectedEntryKind.FILE,
        ),
    )

    videos = _observed_directory(
        schema_id="dlc.videos",
        kind=DirectoryResourceKind.VIDEOS_DIRECTORY,
        project_relative_path=Path("videos"),
        schema_children=(video,),
    )

    return _observed_directory(
        schema_id="dlc.project",
        kind=DirectoryResourceKind.PROJECT_ROOT,
        project_relative_path=Path(),
        schema_children=(videos,),
    )


def test_missing_required_config_produces_finding(
    tmp_path: Path,
) -> None:
    project = discover_project(tmp_path)

    findings = validate_project_structure(
        project,
        project_root=tmp_path,
    )

    missing = [finding for finding in findings if finding.code == "PROJECT_REQUIRED_NODE_MISSING"]

    assert len(missing) == 1
    assert missing[0].severity == Severity.ERROR
    assert missing[0].path == tmp_path / "config.yaml"
    assert missing[0].schema_id == "dlc.project.config"
    assert "schema_id" not in missing[0].evidence
    assert missing[0].instance == ()


def test_a_finding_on_a_repeated_node_names_its_instance(
    tmp_path: Path,
) -> None:
    project = _project_with_symlinked_video(
        target_kind=SymlinkTargetKind.MISSING,
        project_root=tmp_path,
    )

    (finding,) = validate_project_structure(project, project_root=tmp_path)

    assert finding.schema_id == "dlc.videos.source_video"
    assert finding.instance == ("videos", "session-01.mp4")


def test_a_node_inside_a_repeated_instance_carries_that_instance(
    tmp_path: Path,
) -> None:
    # A managed plain child of a repeated dataset belongs to one dataset, so a finding
    # about it must not reach the other datasets. The shipped layout's only such child
    # is known-unmanaged and yields no findings, hence a layout of its own.
    manifest_schema = FileSchema(
        schema_id="dlc.test.dataset.manifest",
        kind=FileResourceKind.NAPARI_DLC_METADATA,
        name=NamePattern(exact="manifest.json"),
    )
    layout = DirectorySchema(
        schema_id="dlc.test",
        kind=DirectoryResourceKind.PROJECT_ROOT,
        name=NamePattern(),
        repeated_children=(
            RepeatedNodeSchema(
                child=DirectorySchema(
                    schema_id="dlc.test.dataset",
                    kind=DirectoryResourceKind.LABELED_DATASET,
                    name=NamePattern(),
                    children=(manifest_schema,),
                ),
            ),
        ),
    )
    manifest = ObservedNode(
        schema_id="dlc.test.dataset.manifest",
        kind=FileResourceKind.NAPARI_DLC_METADATA,
        project_relative_path=Path("ds1") / "manifest.json",
        probe=PathProbe(
            status=PathProbeStatus.INACCESSIBLE,
            error_type="PermissionError",
            error_message="denied",
        ),
        status=NodeStatus.UNREADABLE,
    )
    dataset = _observed_directory(
        schema_id="dlc.test.dataset",
        kind=DirectoryResourceKind.LABELED_DATASET,
        project_relative_path=Path("ds1"),
        schema_children=(manifest,),
    )
    project = _observed_directory(
        schema_id="dlc.test",
        kind=DirectoryResourceKind.PROJECT_ROOT,
        project_relative_path=Path(),
        schema_children=(dataset,),
    )

    index = index_layout(layout)
    (finding,) = validate_project_structure(project, project_root=tmp_path, index=index)

    assert finding.code == "PROJECT_NODE_UNREADABLE"
    assert finding.instance == ("ds1", "manifest.json")


def test_missing_optional_directories_do_not_produce_findings(
    valid_project: Path,
) -> None:
    (valid_project / "videos").rmdir()
    (valid_project / "labeled-data").rmdir()

    project = discover_project(valid_project)

    findings = validate_project_structure(
        project,
        project_root=valid_project,
    )

    assert not any(
        finding.code == "PROJECT_REQUIRED_NODE_MISSING"
        and finding.path
        in {
            valid_project / "videos",
            valid_project / "labeled-data",
        }
        for finding in findings
    )


def test_wrong_type_produces_finding(
    valid_project: Path,
) -> None:
    videos = valid_project / "videos"
    videos.rmdir()
    videos.write_text(
        "not a directory",
        encoding="utf-8",
    )

    project = discover_project(valid_project)

    findings = validate_project_structure(
        project,
        project_root=valid_project,
    )

    wrong_type = next(finding for finding in findings if finding.code == "PROJECT_NODE_WRONG_TYPE")

    assert wrong_type.path == videos
    assert wrong_type.evidence["expected_type"] == "directory"


def test_broken_symlink_produces_symlink_target_missing_finding(
    tmp_path: Path,
) -> None:
    project = _project_with_symlinked_video(
        target_kind=SymlinkTargetKind.MISSING,
        project_root=tmp_path,
    )

    findings = validate_project_structure(
        project,
        project_root=tmp_path,
    )

    assert len(findings) == 1

    finding = findings[0]

    assert finding.code == "PROJECT_SYMLINK_TARGET_MISSING"
    assert finding.severity == Severity.ERROR
    assert finding.path == tmp_path / "videos" / "session-01.mp4"
    assert finding.evidence["raw_target"] == str(_RAW_TARGET)
    assert finding.evidence["absolute_target"] == str(tmp_path / "external.mp4")


def test_symlink_to_directory_produces_symlink_target_wrong_type_finding(
    tmp_path: Path,
) -> None:
    project = _project_with_symlinked_video(
        target_kind=SymlinkTargetKind.DIRECTORY,
        project_root=tmp_path,
    )

    findings = validate_project_structure(
        project,
        project_root=tmp_path,
    )

    assert len(findings) == 1

    finding = findings[0]

    assert finding.code == "PROJECT_SYMLINK_TARGET_WRONG_TYPE"
    assert finding.severity == Severity.ERROR
    assert finding.path == tmp_path / "videos" / "session-01.mp4"
    assert finding.evidence["target_kind"] == SymlinkTargetKind.DIRECTORY.value


def test_symlink_to_file_produces_no_structural_finding(
    tmp_path: Path,
) -> None:
    project = _project_with_symlinked_video(
        target_kind=SymlinkTargetKind.FILE,
        project_root=tmp_path,
    )

    findings = validate_project_structure(
        project,
        project_root=tmp_path,
    )

    assert findings == ()


def test_valid_structure_has_no_structural_findings(
    valid_project: Path,
) -> None:
    project = discover_project(valid_project)

    findings = validate_project_structure(
        project,
        project_root=valid_project,
    )

    assert findings == ()


def test_a_known_unmanaged_node_yields_no_structure_findings(
    tmp_path: Path,
) -> None:
    (tmp_path / "labeled-data" / "m1" / ".napari-deeplabcut.json").mkdir(parents=True)

    project = discover_project(tmp_path)
    pending = [project]
    statuses = {}

    while pending:
        node = pending.pop()
        statuses[node.schema_id] = node.status
        pending.extend(node.schema_children)

    assert statuses["dlc.labeled_data.napari_metadata"] is NodeStatus.WRONG_TYPE

    findings = validate_project_structure(
        project,
        project_root=tmp_path,
    )

    assert [finding for finding in findings if finding.schema_id == "dlc.labeled_data.napari_metadata"] == []
