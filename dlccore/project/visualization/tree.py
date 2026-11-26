# dlc_octopus/visualization/tree.py
from __future__ import annotations

from html import escape
from typing import Any

from dlc_octopus.schemas.base.observations import (
    NodeStatus,
    ObservedNode,
)
from dlc_octopus.schemas.base.resources import (
    DirectoryResourceKind,
    FileResourceKind,
    ProjectResource,
    ResourceManagement,
)

_STATUS_MARKERS = {
    NodeStatus.PRESENT: "✓",
    NodeStatus.MISSING: "✗",
    NodeStatus.UNREADABLE: "!",
    NodeStatus.WRONG_TYPE: "!",
}


class _FileBudget:
    """How many more files a rendering may list; ``None`` is unlimited."""

    def __init__(self, remaining: int | None) -> None:
        self.remaining = remaining

    def take(self) -> bool:
        """Spend one file if any remain, and say whether it was spent."""

        if self.remaining is None:
            return True
        if self.remaining == 0:
            return False
        self.remaining -= 1
        return True


def render_project_tree(
    project: ObservedNode,
    *,
    show_schema_ids: bool = False,
    show_unmanaged: bool = False,
    show_unknown: bool = True,
    max_children_per_kind: int | None = 10,
) -> str:
    """Render an observed project as a compact semantic tree."""

    lines: list[str] = []

    _render_node(
        project,
        prefix="",
        is_last=True,
        is_root=True,
        lines=lines,
        show_schema_ids=show_schema_ids,
        show_unmanaged=show_unmanaged,
        show_unknown=show_unknown,
        max_children_per_kind=max_children_per_kind,
    )

    return "\n".join(lines)


def _render_node(
    node: ObservedNode,
    *,
    prefix: str,
    is_last: bool,
    is_root: bool,
    lines: list[str],
    show_schema_ids: bool,
    show_unmanaged: bool,
    show_unknown: bool,
    max_children_per_kind: int | None,
) -> None:
    """Append one observed node and its descendants to rendered lines."""

    connector = "" if is_root else "└── " if is_last else "├── "
    marker = _STATUS_MARKERS[node.status]

    label = _label(node, is_root=is_root)

    details = [
        node.kind.value,
        node.status.value,
    ]

    if node.management != ResourceManagement.MANAGED:
        details.append(node.management.value)

    if show_schema_ids:
        details.append(node.schema_id)

    lines.append(f"{prefix}{connector}{marker} {label} [{', '.join(details)}]")

    eligible_children = _eligible_children(node, show_unmanaged=show_unmanaged)

    visible_children, omitted_count = _compact_children(
        eligible_children,
        maximum=max_children_per_kind,
    )

    entries: list[tuple[str, Any]] = [("node", child) for child in visible_children]

    if omitted_count:
        entries.append(("omitted", omitted_count))

    if show_unknown:
        entries.extend(("unknown", resource) for resource in node.unclaimed_resources)

    child_prefix = prefix if is_root else prefix + ("    " if is_last else "│   ")

    for index, (entry_type, entry) in enumerate(entries):
        entry_is_last = index == len(entries) - 1

        if entry_type == "node":
            _render_node(
                entry,
                prefix=child_prefix,
                is_last=entry_is_last,
                is_root=False,
                lines=lines,
                show_schema_ids=show_schema_ids,
                show_unknown=show_unknown,
                show_unmanaged=show_unmanaged,
                max_children_per_kind=max_children_per_kind,
            )
            continue

        connector = "└── " if entry_is_last else "├── "

        if entry_type == "unknown":
            lines.append(f"{child_prefix}{connector}? {entry.project_relative_path.name} [{entry.kind.value}]")
        else:
            lines.append(f"{child_prefix}{connector}… {entry} additional children omitted")


def render_project_tree_html(
    project: ObservedNode,
    *,
    show_schema_ids: bool = False,
    show_unmanaged: bool = False,
    show_unknown: bool = True,
    max_files: int | None = 500,
) -> str:
    """Render an observed project as a collapsible HTML tree, listing at most ``max_files`` files on disk.

    Directories are ``<details>`` elements, the project root open and the rest closed. Directories, and files that
    are missing or of the wrong type, are always listed. Files on disk count towards ``max_files``, unclaimed ones
    too while ``show_unknown`` lists them; once it is reached,
    each directory reports the number of additional files not shown.

    Raises:
        ValueError: If ``max_files`` is negative.
    """

    if max_files is not None and max_files < 0:
        raise ValueError(f"max_files must be None or at least 0, not {max_files}.")

    budget = _FileBudget(max_files)
    body = _html_node(
        project,
        is_root=True,
        show_schema_ids=show_schema_ids,
        show_unmanaged=show_unmanaged,
        show_unknown=show_unknown,
        budget=budget,
    )

    return f'<div class="dlc-octopus-tree" style="font-family: monospace; line-height: 1.4">{body}</div>'


def _label(node: ObservedNode, *, is_root: bool) -> str:
    """The project root by its kind, any other top node by its project-relative path, a child by its name."""

    if not is_root:
        return node.project_relative_path.name
    if node.kind is DirectoryResourceKind.PROJECT_ROOT:
        return node.kind.value
    return node.project_relative_path.as_posix()


def _html_node(
    node: ObservedNode,
    *,
    is_root: bool,
    show_schema_ids: bool,
    show_unmanaged: bool,
    show_unknown: bool,
    budget: _FileBudget,
) -> str:
    """Render one observed node and its descendants as HTML."""

    label = _label(node, is_root=is_root)
    details = [node.kind.value, node.status.value]

    if node.management != ResourceManagement.MANAGED:
        details.append(node.management.value)

    if show_schema_ids:
        details.append(node.schema_id)

    line = (
        f'<span title="{escape(node.schema_id)}">{_STATUS_MARKERS[node.status]} {escape(label)} '
        f'<span style="opacity: 0.6">[{escape(", ".join(details))}]</span></span>'
    )

    children = _eligible_children(node, show_unmanaged=show_unmanaged)
    unknown = node.unclaimed_resources if show_unknown else ()

    if not children and not unknown:
        return line

    items: list[str] = []
    omitted = 0

    for child in children:
        if child.should_be_file and child.status is NodeStatus.PRESENT and not budget.take():
            omitted += 1
            continue
        items.append(
            _html_node(
                child,
                is_root=False,
                show_schema_ids=show_schema_ids,
                show_unmanaged=show_unmanaged,
                show_unknown=show_unknown,
                budget=budget,
            )
        )

    for resource in unknown:
        if isinstance(resource.kind, FileResourceKind) and not budget.take():
            omitted += 1
            continue
        items.append(_html_unknown(resource))

    if omitted:
        items.append(f"… {omitted} more {'file' if omitted == 1 else 'files'} not shown")

    listed = "".join(f"<li>{item}</li>" for item in items)

    return (
        f"<details{' open' if is_root else ''}><summary>{line}</summary>"
        f'<ul style="list-style: none; margin: 0; padding-left: 1.5em">{listed}</ul></details>'
    )


def _html_unknown(resource: ProjectResource) -> str:
    """Render one resource the layout does not claim."""

    return (
        f"? {escape(resource.project_relative_path.name)} "
        f'<span style="opacity: 0.6">[{escape(resource.kind.value)}]</span>'
    )


def _eligible_children(node: ObservedNode, *, show_unmanaged: bool) -> tuple[ObservedNode, ...]:
    """The node's schema children a rendering may show: known-unmanaged ones only when asked."""

    return tuple(
        child
        for child in node.schema_children
        if show_unmanaged or child.management != ResourceManagement.KNOWN_UNMANAGED
    )


def _compact_children(
    children: tuple[ObservedNode, ...],
    *,
    maximum: int | None,
) -> tuple[tuple[ObservedNode, ...], int]:
    """Limit repetitive leaf nodes while preserving structural nodes.

    Returns:
        A tuple containing the visible children and the number of omitted
        leaf nodes.
    """

    if maximum is None:
        return children, 0

    structural = tuple(child for child in children if child.schema_children or child.unclaimed_resources)

    leaves = tuple(child for child in children if not child.schema_children and not child.unclaimed_resources)

    visible_leaves = leaves[:maximum]
    omitted_count = len(leaves) - len(visible_leaves)

    return (
        structural + visible_leaves,
        omitted_count,
    )
