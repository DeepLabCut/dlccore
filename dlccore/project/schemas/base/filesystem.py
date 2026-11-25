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

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from pydantic import model_validator
from typing_extensions import Self

from dlccore.project.schemas.base.core import FrozenModel, SchemaId
from dlccore.project.schemas.base.resources import (
    DirectoryResourceKind,
    FileResourceKind,
    ResourceManagement,
)


class NamePattern(FrozenModel):
    """Declarative constraints for a file or directory name.

    Examples:
        - exact: "config.yaml"  -> matches only "config.yaml"
        - prefix: "CollectedData_"  -> matches any name starting with "CollectedData_"
        - suffixes: ".h5" -> matches any name ending with ".h5"
    """

    exact: str | None = None
    prefix: str | None = None
    suffixes: frozenset[str] = frozenset()

    def matches(self, path: Path) -> bool:
        if self.exact is not None and path.name != self.exact:
            return False

        if self.prefix is not None and not path.stem.startswith(self.prefix):
            return False

        return not self.suffixes or path.suffix.lower() in self.suffixes


class FileSchema(FrozenModel):
    """Schema for a file resource.

    Attributes:
        schema_id: The unique identifier for this file schema.
        kind: See FileResourceKind. Importantly, nodes may share a kind, but not an ID.
        name: The name pattern that this file must match, see NamePattern.
        required: Whether this file is required.
        description: A human-readable description of the file.
        management: How this file is managed.

    schema_id should generally be considered immutable and unique across the project.
    Changing that field should not be supported nor necessary.

    Discovery records any missing file; validation reports a missing required one.
    """

    schema_id: SchemaId
    kind: FileResourceKind
    name: NamePattern
    required: bool = False
    description: str = ""
    management: ResourceManagement = ResourceManagement.MANAGED

    @model_validator(mode="after")
    def _check_management(self) -> Self:
        _refuse_required_unmanaged(self)

        return self


class DirectorySchema(FrozenModel):
    schema_id: SchemaId
    kind: DirectoryResourceKind
    name: NamePattern
    required: bool = False
    description: str = ""
    children: tuple[NodeSchema, ...] = ()
    repeated_children: tuple[RepeatedNodeSchema, ...] = ()
    management: ResourceManagement = ResourceManagement.MANAGED

    @model_validator(mode="after")
    def _check_management(self) -> Self:
        _refuse_required_unmanaged(self)

        return self


NodeSchema = FileSchema | DirectorySchema


class RepeatedNodeSchema(FrozenModel):
    child: NodeSchema


DirectorySchema.model_rebuild()


@dataclass(frozen=True)
class LayoutIndex:
    """Every node of one layout by schema ID, with its parent and whether it lies inside a repeated node.

    Built by ``index_layout``, which refuses a schema ID declared at two positions.

    Attributes:
        schemas: Each node's schema, in walk order: depth first, a parent before its children.
        parents: Each node's parent ID; ``None`` for the root.
        inside_repeated: The IDs of every node that is, or lies inside, a repeated node.
    """

    schemas: Mapping[SchemaId, NodeSchema]
    parents: Mapping[SchemaId, SchemaId | None]
    inside_repeated: frozenset[SchemaId]

    def ancestry(self, schema_id: SchemaId) -> tuple[SchemaId, ...]:
        """Return ``schema_id`` followed by its ancestors' IDs, up to the root.

        Raises:
            KeyError: If the layout does not declare ``schema_id``.
        """

        chain: list[SchemaId] = []
        current: SchemaId | None = schema_id

        while current is not None:
            chain.append(current)
            current = self.parents[current]

        return tuple(chain)

    def subtree(self, schema_id: SchemaId) -> tuple[SchemaId, ...]:
        """Return ``schema_id`` and every ID declared beneath it, in walk order.

        Raises:
            KeyError: If the layout does not declare ``schema_id``.
        """

        if schema_id not in self.schemas:
            raise KeyError(schema_id)

        return tuple(other for other in self.schemas if schema_id in self.ancestry(other))

    def schema_ids_of(self, node_type: type[FileSchema] | type[DirectorySchema]) -> tuple[SchemaId, ...]:
        """Return the ID of every node of ``node_type``, in walk order."""

        return tuple(schema_id for schema_id, node in self.schemas.items() if isinstance(node, node_type))


def _refuse_required_unmanaged(schema: NodeSchema) -> None:
    """Refuse a known-unmanaged node declared as required.

    Raises:
        ValueError: If the node is both ``KNOWN_UNMANAGED`` and ``required``.
    """

    if schema.management == ResourceManagement.KNOWN_UNMANAGED and schema.required:
        raise ValueError(f"{schema.schema_id!r} is known-unmanaged, so it cannot be required.")


def child_schemas(schema: NodeSchema) -> tuple[tuple[NodeSchema, bool], ...]:
    """Return a node's declared children, plain children first, each with whether it is a repeated child."""

    if not isinstance(schema, DirectorySchema):
        return ()

    return (
        *((child, False) for child in schema.children),
        *((repeated.child, True) for repeated in schema.repeated_children),
    )


def walk_schema(schema: NodeSchema, *, repeated: bool = False) -> Iterator[tuple[NodeSchema, bool]]:
    """Yield a node and every declared descendant, depth first, each with whether it is a repeated child."""

    yield schema, repeated

    for child, child_repeated in child_schemas(schema):
        yield from walk_schema(child, repeated=child_repeated)


def index_layout(schema: NodeSchema) -> LayoutIndex:
    """Index a layout and every declared descendant by schema ID.

    Raises:
        ValueError: If one schema ID is declared at more than one position, even with equal definitions.
    """

    schemas: dict[SchemaId, NodeSchema] = {}
    parents: dict[SchemaId, SchemaId | None] = {}
    inside_repeated: set[SchemaId] = set()

    def visit(node: NodeSchema, parent: SchemaId | None, *, inside: bool) -> None:
        if node.schema_id in schemas:
            raise ValueError(f"Schema ID {node.schema_id!r} is declared at more than one layout position.")

        schemas[node.schema_id] = node
        parents[node.schema_id] = parent

        if inside:
            inside_repeated.add(node.schema_id)

        for child, repeated in child_schemas(node):
            visit(child, node.schema_id, inside=inside or repeated)

    visit(schema, None, inside=False)

    return LayoutIndex(
        schemas=MappingProxyType(schemas),
        parents=MappingProxyType(parents),
        inside_repeated=frozenset(inside_repeated),
    )
