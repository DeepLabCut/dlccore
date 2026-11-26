"""Where a path outside a known project sits in the layout, scanned upward from the path.

The anchor holds only what is certain: a segment counts when the chain above it reaches a directory with an exact
layout name, or a project root holding its required files. A segment that fits a node by position only is a
``CandidateNode``: returned beside the position, never in a finding. Reads names and the root's required files only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dlc_octopus.schemas.base.core import SchemaId
from dlc_octopus.schemas.base.filesystem import FileSchema, LayoutIndex, NodeSchema
from dlc_octopus.schemas.base.findings import Reach
from dlc_octopus.schemas.locations import Location, node_location
from dlc_octopus.schemas.project_layout.project import PROJECT_INDEX


@dataclass(frozen=True)
class CandidateNode:
    """A path segment that fits a layout node by position only; nothing confirms it.

    Attributes:
        segment: The segment as written.
        fits: The node it would be.
    """

    segment: str
    fits: SchemaId


@dataclass(frozen=True)
class LayoutPosition:
    """Where a path sits in the layout, as far as it is certain.

    Attributes:
        location: The finding anchor: ``ONE_OF`` or ``ONE_PER_PROJECT`` when the certain segments reach the project
            root's children, ``UNROOTED`` with those segments otherwise.
        project_root: The project root, when a parent holds its required files.
        detached: Segments above the certain ones that fit a node by position only, nearest first.
    """

    location: Location
    project_root: Path | None
    detached: tuple[CandidateNode, ...] = ()


def locate_in_layout(path: Path, *, index: LayoutIndex = PROJECT_INDEX) -> LayoutPosition:
    """Place a file in the layout by matching its name, then its parents, against the node chain above it.

    Raises:
        ValueError: If no file node of the layout, or more than one, matches the file's name.
    """

    path = path.absolute()
    leaf = _file_node(path, index)
    chain = index.ancestry(leaf)  # the file's node, then each ancestor up to the root
    certain, matched, project_root = 1, 1, None

    for depth, schema_id in enumerate(chain[1:], start=1):
        if depth - 1 >= len(path.parents):
            break
        directory, node = path.parents[depth - 1], index.schemas[schema_id]
        if index.parents[schema_id] is None:
            if _holds_required_files(directory, node):
                project_root, certain = directory, depth
            break
        if not node.name.matches(Path(directory.name)):
            break
        matched = depth + 1
        if node.name.exact is not None:
            certain = depth + 1

    segments = [path.name, *(parent.name for parent in path.parents)]
    instance = tuple(reversed(segments[:certain]))
    if certain == len(chain) - 1:
        location = node_location(leaf, Path(*instance), index=index)
    else:
        location = Location(schema_id=leaf, reach=Reach.UNROOTED, instance=instance)
    detached = tuple(CandidateNode(segments[depth], chain[depth]) for depth in range(certain, matched))
    return LayoutPosition(location, project_root, detached)


def _file_node(path: Path, index: LayoutIndex) -> SchemaId:
    matches = [
        schema_id for schema_id in index.schema_ids_of(FileSchema) if index.schemas[schema_id].name.matches(path)
    ]
    if len(matches) != 1:
        raise ValueError(f"{path.name!r} matches {len(matches)} file nodes of the layout {matches}; one is needed.")
    return matches[0]


def _holds_required_files(directory: Path, root: NodeSchema) -> bool:
    """Whether ``directory`` holds every required, exactly named file the root declares, and it declares one."""

    required = [
        child.name.exact
        for child in getattr(root, "children", ())
        if isinstance(child, FileSchema) and child.required and child.name.exact
    ]
    return bool(required) and all((directory / name).is_file() for name in required)
