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

"""What every data entry point shares: which observed files it reads, and how it reports one it could not."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

from dlccore.project.schemas.base.core import ProjectRelativePath, SchemaId
from dlccore.project.schemas.base.filesystem import LayoutIndex
from dlccore.project.schemas.base.findings import Finding
from dlccore.project.schemas.base.observations import NodeStatus, ObservedNode, walk_observed_nodes
from dlccore.project.schemas.locations import node_location
from dlccore.project.schemas.scope import Scope
from dlccore.project.schemas.snapshots import ProjectSnapshot


@dataclass(frozen=True)
class FailedRead:
    """One requested file that could not be read, with the findings saying why.

    Attributes:
        schema_id: The layout node of the file.
        project_relative_path: The path as the snapshot observed it.
        findings: Why it could not be read; also merged into the snapshot the entry point returns.
    """

    schema_id: SchemaId
    project_relative_path: ProjectRelativePath
    findings: tuple[Finding, ...]


def present_files_in_scope(
    snapshot: ProjectSnapshot,
    scope: Scope,
    schema_ids: Collection[SchemaId],
    *,
    index: LayoutIndex,
) -> tuple[ObservedNode, ...]:
    """Return every ``PRESENT`` node of ``schema_ids`` that ``scope`` covers, in walk order, without rediscovering.

    Nodes not ``PRESENT`` are left out: structure validation already reports them.
    """

    return tuple(
        node
        for node in walk_observed_nodes(snapshot.project)
        if node.schema_id in schema_ids
        and node.status is NodeStatus.PRESENT
        and covered(scope, node.schema_id, node.project_relative_path, index)
    )


def covered(scope: Scope, schema_id: SchemaId, project_relative_path: ProjectRelativePath, index: LayoutIndex) -> bool:
    """Return whether ``scope`` covers the node with this schema ID at this path."""

    instance = node_location(schema_id, project_relative_path, index=index)["instance"]

    return scope.covers(schema_id, instance, index=index)
