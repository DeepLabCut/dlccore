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

from pathlib import Path
from typing import TypedDict

from dlccore.project.schemas.base.core import LayoutInstance, SchemaId, instance_from_path
from dlccore.project.schemas.base.filesystem import LayoutIndex, NodeSchema
from dlccore.project.schemas.base.findings import Reach
from dlccore.project.schemas.project_layout.project import PROJECT_INDEX


class Location(TypedDict):
    """Where a finding is anchored in the layout, ready to pass to ``Finding``."""

    schema_id: SchemaId
    reach: Reach
    instance: LayoutInstance


def one_per_project(
    node: NodeSchema,
    *,
    index: LayoutIndex = PROJECT_INDEX,
) -> Location:
    """Anchor a finding at the only node of its kind in the project.

    Raises:
        ValueError: If the node is not in the layout, or lies inside a repeated node.
    """

    if _inside_repeated(index, node):
        raise ValueError(f"{node.schema_id!r} lies inside a repeated node, so it has no single instance.")

    return Location(schema_id=node.schema_id, reach=Reach.ONE_PER_PROJECT, instance=())


def one_of(
    node: NodeSchema,
    instance: LayoutInstance,
    *,
    index: LayoutIndex = PROJECT_INDEX,
) -> Location:
    """Anchor a finding at one named instance of a node that is or lies inside a repeated node.

    Raises:
        ValueError: If the node is not in the layout, does not lie inside a repeated node,
            or the instance is empty.
    """

    if not _inside_repeated(index, node):
        raise ValueError(f"{node.schema_id!r} is the only node of its kind, so it has no instances.")

    if not instance:
        raise ValueError(f"{node.schema_id!r} names one instance, but got none.")

    return Location(schema_id=node.schema_id, reach=Reach.ONE_OF, instance=instance)


def all_of(
    node: NodeSchema,
    within: LayoutInstance = (),
    *,
    index: LayoutIndex = PROJECT_INDEX,
) -> Location:
    """Anchor a finding at every instance of a node, optionally within an instance prefix.

    Raises:
        ValueError: If the node is not in the layout, or does not lie inside a repeated node.
    """

    if not _inside_repeated(index, node):
        raise ValueError(f"{node.schema_id!r} is the only node of its kind, so it has no instances.")

    return Location(schema_id=node.schema_id, reach=Reach.ALL_OF, instance=within)


def _inside_repeated(
    index: LayoutIndex,
    node: NodeSchema,
) -> bool:
    """Return whether a layout node is, or lies inside, a repeated node.

    Raises:
        ValueError: If the layout does not declare this node.
    """

    if index.schemas.get(node.schema_id) != node:
        raise ValueError(f"The layout does not declare this {node.schema_id!r} node.")

    return node.schema_id in index.inside_repeated


def node_location(
    schema_id: SchemaId,
    project_relative_path: Path,
    *,
    index: LayoutIndex = PROJECT_INDEX,
) -> Location:
    """Anchor a finding at the observed node with this schema ID and path: one instance if repeated, else the only one.

    Raises:
        ValueError: If the layout does not declare ``schema_id``, or the path is anchored.
    """

    node = index.schemas.get(schema_id)

    if node is None:
        raise ValueError(f"The layout does not declare {schema_id!r}.")

    instance = instance_from_path(project_relative_path)

    if schema_id in index.inside_repeated:
        return one_of(node, instance, index=index)

    return one_per_project(node, index=index)
