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

from enum import Enum

from dlccore.project.schemas.base.core import FrozenModel, ProjectRelativePath, SchemaId
from dlccore.project.schemas.base.probes import PathProbe
from dlccore.project.schemas.base.resources import (
    DirectoryResourceKind,
    FileResourceKind,
    ProjectResource,
    ResourceManagement,
)
from dlccore.project.schemas.base.symlinks import (
    SymlinkObservation,
)


class NodeStatus(str, Enum):
    """Observed state of a project node."""

    PRESENT = "present"
    MISSING = "missing"
    UNREADABLE = "unreadable"
    WRONG_TYPE = "wrong_type"


class ObservedNode(FrozenModel):
    """Observed state of one schema-defined project node.

    ``probe`` is the raw filesystem fact the node was built from
    ``status`` is a derived, schema-aware compatibility summary of that
    probe against the node's expected entry kind
    ``symlink`` is a read-only convenience accessor for ``probe.symlink``
    ``traversal_error_type`` and ``traversal_error_message`` say why a present
    directory could not be entered; entry-level failures stay on ``probe``

    This is not a complete recursive filesystem tree.
    ``schema_children`` only contains schema-defined nodes that were observed.
    ``unclaimed_resources`` contains immediate filesystem entries that were
    not claimed by this schema.
    Unknown directories are kept as resources but not traversed.
    """

    schema_id: SchemaId
    kind: FileResourceKind | DirectoryResourceKind
    project_relative_path: ProjectRelativePath
    probe: PathProbe
    status: NodeStatus
    management: ResourceManagement = ResourceManagement.MANAGED
    traversal_error_type: str | None = None
    traversal_error_message: str | None = None

    schema_children: tuple[ObservedNode, ...] = ()
    unclaimed_resources: tuple[ProjectResource, ...] = ()

    @property
    def symlink(self) -> SymlinkObservation | None:
        """Convenience accessor for ``probe.symlink``."""
        return self.probe.symlink

    @property
    def should_be_file(self) -> bool:
        """Whether the project schema defines this node as a file."""
        return isinstance(self.kind, FileResourceKind)

    @property
    def should_be_directory(self) -> bool:
        """Whether the project schema defines this node as a directory."""
        return isinstance(self.kind, DirectoryResourceKind)


def walk_observed_nodes(node: ObservedNode) -> tuple[ObservedNode, ...]:
    """Return a node and its schema-recognized descendants, depth first."""

    nodes = [node]

    for child in node.schema_children:
        nodes.extend(walk_observed_nodes(child))

    return tuple(nodes)
