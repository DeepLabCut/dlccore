from __future__ import annotations

from pathlib import Path

from dlc_octopus.schemas.base.filesystem import (
    DirectorySchema,
    LayoutIndex,
    NodeSchema,
)
from dlc_octopus.schemas.base.finding_codes import StructureFindingCode
from dlc_octopus.schemas.base.findings import Finding
from dlc_octopus.schemas.base.observations import NodeStatus, ObservedNode, walk_observed_nodes
from dlc_octopus.schemas.base.resources import ResourceManagement
from dlc_octopus.schemas.base.symlinks import SymlinkTargetKind
from dlc_octopus.schemas.locations import Location, node_location
from dlc_octopus.schemas.project_layout.project import PROJECT_INDEX


def validate_project_structure(
    project: ObservedNode,
    *,
    project_root: Path,
    index: LayoutIndex = PROJECT_INDEX,
) -> tuple[Finding, ...]:
    """Validate an observed project against its layout schema.

    Structural validation reports missing required nodes, nodes with the wrong
    physical type, and nodes that could not be read. It does not interpret
    config fields, validate file contents, or report unknown resources. A
    known-unmanaged node is observed, not validated: it yields no finding here.
    """

    findings: list[Finding] = []

    for node in walk_observed_nodes(project):
        node_schema = index.schemas.get(node.schema_id)

        if node_schema is None:
            raise ValueError(f"Observed node references unknown schema ID {node.schema_id!r}.")

        if node_schema.management == ResourceManagement.KNOWN_UNMANAGED:
            continue

        finding = _finding_for_node(
            node,
            schema=node_schema,
            project_root=project_root,
            location=node_location(node.schema_id, node.project_relative_path, index=index),
        )

        if finding is not None:
            findings.append(finding)

    return tuple(findings)


def _finding_for_node(
    node: ObservedNode,
    *,
    schema: NodeSchema,
    project_root: Path,
    location: Location,
) -> Finding | None:
    """Return a structural finding for one observed node, if needed."""

    path = project_root / node.project_relative_path
    evidence = {
        "resource_kind": node.kind.value,
    }

    if node.status == NodeStatus.MISSING:
        if not schema.required:
            return None

        return Finding(
            code=StructureFindingCode.PROJECT_REQUIRED_NODE_MISSING,
            **location,
            path=path,
            evidence=evidence,
        )

    if node.symlink is not None and node.symlink.target_kind == SymlinkTargetKind.MISSING:
        return Finding(
            code=StructureFindingCode.PROJECT_SYMLINK_TARGET_MISSING,
            **location,
            path=path,
            evidence={
                **evidence,
                "raw_target": str(node.symlink.raw_target),
                "absolute_target": str(node.symlink.absolute_target),
            },
        )

    if node.symlink is not None and node.status == NodeStatus.WRONG_TYPE:
        return Finding(
            code=StructureFindingCode.PROJECT_SYMLINK_TARGET_WRONG_TYPE,
            **location,
            path=path,
            evidence={
                **evidence,
                "raw_target": str(node.symlink.raw_target),
                "absolute_target": str(node.symlink.absolute_target),
                "target_kind": (node.symlink.target_kind.value),
            },
        )

    if node.status == NodeStatus.WRONG_TYPE:
        expected_type = "directory" if isinstance(schema, DirectorySchema) else "file"

        return Finding(
            code=StructureFindingCode.PROJECT_NODE_WRONG_TYPE,
            **location,
            path=path,
            evidence={
                **evidence,
                "expected_type": expected_type,
            },
        )

    if node.status == NodeStatus.UNREADABLE:
        return Finding(
            code=StructureFindingCode.PROJECT_NODE_UNREADABLE,
            **location,
            path=path,
            evidence={
                **evidence,
                "error_type": node.traversal_error_type or node.probe.error_type,
                "error_message": node.traversal_error_message or node.probe.error_message,
            },
        )

    return None
