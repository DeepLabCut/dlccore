"""What a consumer asks Octopus about: layout nodes, each within an instance prefix."""

from __future__ import annotations

from dataclasses import dataclass

from dlc_octopus.schemas.base.core import LayoutInstance, SchemaId
from dlc_octopus.schemas.base.filesystem import FileSchema, LayoutIndex, NodeSchema
from dlc_octopus.schemas.base.findings import Finding
from dlc_octopus.schemas.project_layout.project import PROJECT_CONFIG, PROJECT_INDEX
from dlc_octopus.schemas.snapshots import ProjectSnapshot


@dataclass(frozen=True, init=False)
class Scope:
    """What a consumer asks findings for: layout nodes, each within an instance prefix.

    ``Scope(node, within=...)`` asks for one node; ``a | b`` asks for both.

    Attributes:
        parts: Each requested node with its instance prefix, in the order first asked.
    """

    parts: tuple[tuple[NodeSchema, LayoutInstance], ...]

    def __init__(self, node: NodeSchema, within: LayoutInstance = ()) -> None:
        object.__setattr__(self, "parts", ((node, tuple(within)),))

    def __or__(self, other: Scope) -> Scope:
        """Ask for both scopes; a part asked twice is kept once, where it was first asked."""

        added = tuple(part for part in other.parts if part not in self.parts)
        combined = object.__new__(Scope)
        object.__setattr__(combined, "parts", self.parts + added)

        return combined

    def file_schema_ids(self, *, index: LayoutIndex = PROJECT_INDEX) -> tuple[SchemaId, ...]:
        """Return the file nodes in the subtree of any scoped node, in walk order.

        Only the subtrees: a file at a scoped node's ancestor is not covered, although its findings are in scope.

        Raises:
            ValueError: If a scoped node is not the layout's own node.
        """

        roots = {declared_node(index, node).schema_id for node, _ in self.parts}

        return tuple(
            schema_id for schema_id in index.schema_ids_of(FileSchema) if roots & set(index.ancestry(schema_id))
        )

    def covers(self, schema_id: SchemaId, instance: LayoutInstance, *, index: LayoutIndex = PROJECT_INDEX) -> bool:
        """Return whether the node ``schema_id`` at ``instance`` is covered: in a scoped subtree, within its prefix.

        Raises:
            ValueError: If a scoped node is not the layout's own node.
            KeyError: If the layout does not declare ``schema_id``.
        """

        ancestry = index.ancestry(schema_id)

        return any(
            declared_node(index, node).schema_id in ancestry and instances_overlap(instance, within)
            for node, within in self.parts
        )


def findings_in_scope(
    snapshot: ProjectSnapshot,
    scope: Scope,
    *,
    index: LayoutIndex = PROJECT_INDEX,
) -> tuple[Finding, ...]:
    """Return the findings of ``snapshot`` that concern ``scope``, in snapshot order, each once.

    A finding concerns a scoped node when it is anchored in the node's subtree, at one of its
    ancestors, or at the config: every node depends on the whole config until its fields are
    verified against DeepLabCut. Its instance must also match the scope's ``within``: one is a
    prefix of the other, so ``()`` matches everything and another dataset's instance matches nothing.

    Raises:
        ValueError: If any finding names a ``schema_id`` the layout does not declare, whatever the scope:
            such a finding cannot be placed, so leaving it out would be a silent accept. Also if
            a scoped node is not the layout's own node.
    """

    undeclared = sorted({finding.schema_id for finding in snapshot.findings} - set(index.schemas))

    if undeclared:
        raise ValueError(f"Findings name schema IDs the layout does not declare: {undeclared}.")

    related = [(_related_schema_ids(index, node), within) for node, within in scope.parts]

    return tuple(
        finding
        for finding in snapshot.findings
        if any(finding.schema_id in ids and instances_overlap(finding.instance, within) for ids, within in related)
    )


def declared_node(index: LayoutIndex, node: NodeSchema) -> NodeSchema:
    """Return ``node`` if it is the layout's own node.

    Raises:
        ValueError: If the layout does not declare this node.
    """

    if index.schemas.get(node.schema_id) != node:
        raise ValueError(f"The layout does not declare this {node.schema_id!r} node.")

    return node


def instances_overlap(instance: LayoutInstance, within: LayoutInstance) -> bool:
    """Return whether one instance is a prefix of the other."""

    shorter = min(len(instance), len(within))

    return tuple(instance[:shorter]) == tuple(within[:shorter])


def _related_schema_ids(index: LayoutIndex, node: NodeSchema) -> frozenset[SchemaId]:
    """Return the IDs whose findings concern ``node``: its subtree, its ancestors and the config."""

    schema_id = declared_node(index, node).schema_id

    return frozenset({*index.subtree(schema_id), *index.ancestry(schema_id), PROJECT_CONFIG.schema_id})
