# schemas/base/observations.py
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from dlc_octopus.schemas.base.core import FrozenModel, ProjectRelativePath, SchemaId
from dlc_octopus.schemas.base.probes import PathProbe
from dlc_octopus.schemas.base.resources import (
    DirectoryResourceKind,
    FileResourceKind,
    ProjectResource,
    ResourceManagement,
)
from dlc_octopus.schemas.base.symlinks import (
    SymlinkObservation,
)

if TYPE_CHECKING:
    from IPython.lib.pretty import RepresentationPrinter


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

    # Display hooks only IPython and Jupyter call; repr, str and serialization stay pydantic's.
    # Imported on call so that schemas never load visualization.

    def _repr_pretty_(self, printer: RepresentationPrinter, cycle: bool) -> None:  # noqa: FBT001 - IPython's signature
        from dlc_octopus.visualization.tree import render_project_tree

        printer.text("ObservedNode(...)" if cycle else render_project_tree(self))

    def _repr_html_(self) -> str:
        from dlc_octopus.visualization.tree import render_project_tree_html

        return render_project_tree_html(self)


def walk_observed_nodes(node: ObservedNode) -> tuple[ObservedNode, ...]:
    """Return a node and its schema-recognized descendants, depth first."""

    nodes = [node]

    for child in node.schema_children:
        nodes.extend(walk_observed_nodes(child))

    return tuple(nodes)
