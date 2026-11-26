"""Labeled-data files, of a snapshot or at a path, read through ``dlc_octopus.io`` with findings about each one."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import JsonValue

from dlc_octopus.content.labeled_data.validate import validate_labeled_data_file
from dlc_octopus.content.reads import FailedRead, present_files_in_scope
from dlc_octopus.discovery.position import LayoutPosition, locate_in_layout
from dlc_octopus.inspection import inspect_project
from dlc_octopus.io import UnreadableFileError
from dlc_octopus.io.labeled_data import LabeledDataFile, read_labeled_data
from dlc_octopus.schemas.base.core import ProjectRelativePath, SchemaId
from dlc_octopus.schemas.base.filesystem import LayoutIndex
from dlc_octopus.schemas.base.finding_codes import LabeledDataFindingCode
from dlc_octopus.schemas.base.findings import Finding
from dlc_octopus.schemas.locations import Location, node_location
from dlc_octopus.schemas.project_layout.labeled_data import HUMAN_LABELS, MACHINE_LABELS, MACHINE_LABELS_CSV
from dlc_octopus.schemas.project_layout.project import PROJECT_INDEX
from dlc_octopus.schemas.scope import Scope, findings_in_scope
from dlc_octopus.schemas.snapshots import ProjectSnapshot
from dlc_octopus.validation.results import ValidationResult

_LABELS = frozenset({HUMAN_LABELS.schema_id, MACHINE_LABELS.schema_id, MACHINE_LABELS_CSV.schema_id})


@dataclass(frozen=True)
class LabeledDataEntry:
    """One labels file read through Octopus, with its layout anchor.

    Attributes:
        schema_id: The layout node of the file.
        project_relative_path: The path as the snapshot observed it.
        data: The file as ``dlc_octopus.io`` read it.
    """

    schema_id: SchemaId
    project_relative_path: ProjectRelativePath
    data: LabeledDataFile


@dataclass(frozen=True)
class ProjectLabeledData:
    """The labels files read, those that could not be, and the snapshot with their findings merged.

    Attributes:
        snapshot: The input snapshot with the findings about the files requested merged in, those read and those
            not; use it, not the input, for any policy decision about them.
        read: One entry per requested file that could be read.
        failed: One entry per requested file that could not be read.
    """

    snapshot: ProjectSnapshot
    read: tuple[LabeledDataEntry, ...] = ()
    failed: tuple[FailedRead, ...] = ()


@dataclass(frozen=True)
class InspectedLabeledDataFile:
    """One labels file inspected from its path, inside a project or not.

    Attributes:
        position: Where the path sits in the layout, as far as it is certain.
        data: The file as ``dlc_octopus.io`` read it; ``None`` exactly when it could not be read.
        findings: Every finding in scope of the file, anchored at ``position.location``.
        snapshot: The project's snapshot, when ``position`` found a project root.
    """

    position: LayoutPosition
    data: LabeledDataFile | None
    findings: tuple[Finding, ...]
    snapshot: ProjectSnapshot | None = None


def read_project_labeled_data(
    snapshot: ProjectSnapshot,
    scope: Scope,
    *,
    index: LayoutIndex = PROJECT_INDEX,
) -> ProjectLabeledData:
    """Read every present human or machine labels file ``scope`` covers in ``snapshot``, without rediscovering.

    Each such file is in ``read`` or in ``failed``, never both. A file that cannot be read yields a
    ``LABELED_DATA_UNREADABLE`` finding, carried by its ``FailedRead``; a file read yields its findings from
    ``validate_labeled_data_file``. Both are merged into the returned snapshot; the input is not modified.

    Args:
        snapshot: A snapshot from ``inspect_project``.
        scope: The nodes whose labels files to read, each within its instance prefix; see ``Scope.covers``.
        index: The layout the snapshot was observed against.

    Raises:
        ValueError: If ``scope`` names a node the layout does not declare.
    """

    read: list[LabeledDataEntry] = []
    failed: list[FailedRead] = []
    about_read: list[Finding] = []

    for node in present_files_in_scope(snapshot, scope, _LABELS & set(scope.file_schema_ids(index=index)), index=index):
        at = node_location(node.schema_id, node.project_relative_path, index=index)
        data, findings = _read(snapshot.project_root / node.project_relative_path, node.kind.value, at)

        if data is None:
            failed.append(FailedRead(node.schema_id, node.project_relative_path, findings))
            continue

        read.append(LabeledDataEntry(node.schema_id, node.project_relative_path, data))
        about_read += findings

    findings = (*(finding for failure in failed for finding in failure.findings), *about_read)

    return ProjectLabeledData(
        snapshot=snapshot.with_validation_result(ValidationResult(findings=findings)),
        read=tuple(read),
        failed=tuple(failed),
    )


def inspect_labeled_data_file(path: Path, *, index: LayoutIndex = PROJECT_INDEX) -> InspectedLabeledDataFile:
    """Read one labels file from its path and report it, placing it in the layout by scanning upward.

    Inside a project, the project is inspected and the file read as ``read_project_labeled_data`` reads it, so
    project-level findings in its scope are included. Outside one, the file gets the same per-file findings.

    Raises:
        ValueError: If the file's name matches no labels node of the layout, or several.
    """

    position = locate_in_layout(path, index=index)
    at = position.location
    node = index.schemas[at["schema_id"]]
    if node.schema_id not in _LABELS:
        raise ValueError(f"{path.name!r} is a {node.schema_id} file, not a labels file.")

    if position.project_root is None:
        data, findings = _read(path, node.kind.value, at)
        return InspectedLabeledDataFile(position, data, findings)

    scope = Scope(node, within=at["instance"])
    result = read_project_labeled_data(inspect_project(position.project_root), scope, index=index)
    data = next((entry.data for entry in result.read), None)
    return InspectedLabeledDataFile(position, data, findings_in_scope(result.snapshot, scope), result.snapshot)


def _read(path: Path, resource_kind: str, at: Location) -> tuple[LabeledDataFile | None, tuple[Finding, ...]]:
    """Read one labels file and return it with its findings, or ``None`` with the finding that it is unreadable."""

    try:
        data = read_labeled_data(path)
    except UnreadableFileError as error:
        evidence: dict[str, JsonValue] = {
            "resource_kind": resource_kind,
            "error_type": type(error.__cause__ or error).__name__,
            "error_message": str(error),
        }
        return None, (Finding(code=LabeledDataFindingCode.LABELED_DATA_UNREADABLE, **at, path=path, evidence=evidence),)
    return data, validate_labeled_data_file(data, at)
