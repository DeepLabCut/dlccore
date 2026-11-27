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

"""Render what this installation of dlccore recognises and can validate, for people."""

from __future__ import annotations

from typing import Literal

from dlccore.project.schemas.base.resources import ResourceManagement
from dlccore.project.schemas.capabilities import LayoutCapability, ProjectManagerCapabilities
from dlccore.project.visualization.findings import _inline_code, _table_cell

CapabilitiesStyle = Literal["text", "markdown"]


def render_capabilities(capabilities: ProjectManagerCapabilities, *, style: CapabilitiesStyle = "text") -> str:
    """Render capabilities: the layout tree, the baseline checks, then the optional validators.

    Args:
        capabilities: From ``get_project_manager_capabilities``.
        style: ``"text"`` with a drawn tree, or ``"markdown"`` with a nested list and tables.

    Raises:
        ValueError: If ``style`` is not one of the two.
    """

    if style == "text":
        return _render_text(capabilities)
    if style == "markdown":
        return _render_markdown(capabilities)

    raise ValueError(f"Unknown capabilities style {style!r}; expected 'text' or 'markdown'.")


def _render_text(capabilities: ProjectManagerCapabilities) -> str:
    lines = [f"Octopus capabilities (snapshot version {capabilities.snapshot_version})", "", "Layout"]
    _render_layout(capabilities.layout, prefix="", is_last=True, lines=lines)

    lines += ["", "Baseline checks, always run"]
    width = max((len(check.code) for check in capabilities.baseline_validations), default=0)
    lines += [f"  {check.code:<{width}}  {check.description}" for check in capabilities.baseline_validations]

    lines += ["", "Optional validators, on request"]
    if not capabilities.validations:
        lines.append("  none")
    lines += [
        f"  {validation.kind.value}: {', '.join(depth.value for depth in validation.supported_depths)}"
        for validation in capabilities.validations
    ]

    return "\n".join(lines)


def _render_markdown(capabilities: ProjectManagerCapabilities) -> str:
    lines = [
        f"**Octopus capabilities** (snapshot version {_inline_code(capabilities.snapshot_version)})",
        "",
        "**Layout**",
        "",
    ]
    _render_layout_markdown(capabilities.layout, depth=0, lines=lines)

    lines += ["", "**Baseline checks, always run**", ""]
    if capabilities.baseline_validations:
        lines += ["| Check | Description |", "|---|---|"]
        lines += [
            f"| {_inline_code(check.code)} | {_table_cell(check.description)} |"
            for check in capabilities.baseline_validations
        ]
    else:
        lines.append("None.")

    lines += ["", "**Optional validators, on request**", ""]
    if capabilities.validations:
        lines += ["| Resource kind | Depths |", "|---|---|"]
        lines += [
            f"| {_inline_code(validation.kind.value)} "
            f"| {', '.join(depth.value for depth in validation.supported_depths)} |"
            for validation in capabilities.validations
        ]
    else:
        lines.append("None.")

    return "\n".join(lines)


def _details(node: LayoutCapability) -> list[str]:
    details = [node.kind.value, "required" if node.required else "optional"]
    if node.repeated:
        details.append("repeated")
    if node.management != ResourceManagement.MANAGED:
        details.append(node.management.value)

    return details


def _render_layout(node: LayoutCapability, *, prefix: str, is_last: bool, lines: list[str]) -> None:
    """Append one layout node and its descendants to rendered lines."""

    lines.append(f"{prefix}{'└── ' if is_last else '├── '}{node.schema_id} [{', '.join(_details(node))}]")

    child_prefix = prefix + ("    " if is_last else "│   ")
    for index, child in enumerate(node.children):
        _render_layout(child, prefix=child_prefix, is_last=index == len(node.children) - 1, lines=lines)


def _render_layout_markdown(node: LayoutCapability, *, depth: int, lines: list[str]) -> None:
    """Append one layout node and its descendants as a nested Markdown list."""

    lines.append(f"{'  ' * depth}- {_inline_code(node.schema_id)}: {', '.join(_details(node))}")

    for child in node.children:
        _render_layout_markdown(child, depth=depth + 1, lines=lines)
