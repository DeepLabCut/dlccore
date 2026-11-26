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

"""Render findings for people: several lines, one line for logs, or Markdown for notebooks and issues."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from dlccore.project.schemas.base.finding_codes import FindingCode, Severity, iter_finding_codes
from dlccore.project.schemas.base.findings import Finding

FindingStyle = Literal["text", "compact", "markdown"]
CatalogueStyle = Literal["text", "markdown"]

REDACTED_ROOT = "<project>"
_SEVERITY_WIDTH = max(len(severity.value) for severity in Severity)
_MOST_SEVERE_FIRST = tuple(Severity)[::-1]
_FIELD_SEPARATOR = chr(0x203A)  # single right-pointing angle quotation mark, between a file and a field


def render_finding(
    finding: Finding,
    *,
    project_root: Path | None = None,
    style: FindingStyle = "text",
) -> str:
    """Render one finding: where it is, what it says and its code, then its evidence.

    Args:
        finding: The finding to render.
        project_root: The snapshot's project root; a ``path`` under it is shown project-relative.
        style: ``"text"`` over several lines, ``"compact"`` on one line, or ``"markdown"``.

    Raises:
        ValueError: If ``style`` is not one of the three.
    """

    severity = finding.severity.value.upper()
    evidence = _evidence(finding)

    if style == "text":
        indent = " " * (_SEVERITY_WIDTH + 2)
        lines = [
            f"{severity:<{_SEVERITY_WIDTH}}  {_location(finding, project_root)}",
            f"{indent}{finding.code.description}  [{finding.code}]",
        ]
        lines.extend(f"{indent}{key}: {value}" for key, value in evidence)
        return "\n".join(lines)

    if style == "compact":
        details = ", ".join(f"{key}={value}" for key, value in evidence)
        line = f"{severity} {finding.code} {_location(finding, project_root)}: {finding.code.description}"
        return f"{line} ({details})" if details else line

    if style == "markdown":
        lines = [
            f"**{severity}** {_markdown_location(finding, project_root)}: "
            f"{finding.code.description} {_inline_code(str(finding.code))}"
        ]
        lines.extend(f"- {_inline_code(key)}: {_inline_code(value)}" for key, value in evidence)
        return "\n".join(lines)

    raise ValueError(f"Unknown finding style {style!r}; expected 'text', 'compact' or 'markdown'.")


def render_finding_catalogue(*, style: CatalogueStyle = "text") -> str:
    """Render every finding code the project layer can emit, one section per category, in catalogue order.

    Args:
        style: ``"text"``, or ``"markdown"`` with one table per category.

    Raises:
        ValueError: If ``style`` is not one of the two.
    """

    if style not in ("text", "markdown"):
        raise ValueError(f"Unknown catalogue style {style!r}; expected 'text' or 'markdown'.")

    categories: dict[type[FindingCode], list[FindingCode]] = {}
    for code in iter_finding_codes():
        categories.setdefault(type(code), []).append(code)

    sections = []
    for category, codes in categories.items():
        title = _category_title(category)
        summary = (category.__doc__ or "").strip().splitlines()[0].replace("``", "`")
        if style == "text":
            width = max(len(code.value) for code in codes)
            rows = [
                f"  {code.severity.value.upper():<{_SEVERITY_WIDTH}}  {code.value:<{width}}  {code.description}"
                for code in codes
            ]
            sections.append("\n".join([f"{title}: {summary.replace('`', '')}", *rows]))
        else:
            rows = [
                f"| {_inline_code(code.value)} | {code.severity.value} | {_table_cell(code.description)} |"
                for code in codes
            ]
            header = ["| Code | Default severity | Meaning |", "|---|---|---|"]
            sections.append("\n".join([f"**{title}**: {summary}", "", *header, *rows]))

    return "\n\n".join(sections)


def render_findings_markdown(
    findings: Iterable[Finding],
    *,
    project_root: Path | None = None,
    redact_root: bool = True,
) -> str:
    """Render findings as a Markdown table, their evidence folded below it, ready to paste into an issue.

    With ``redact_root``, the project root is replaced by ``<project>`` wherever it appears verbatim, so the
    pasted text does not carry the local path. Paths outside the project root are left as observed.

    Args:
        findings: The findings to render, in the order given.
        project_root: The snapshot's project root.
        redact_root: Whether to replace ``project_root`` in locations and evidence.
    """

    findings = tuple(findings)
    if not findings:
        return "No findings."

    redact = project_root if redact_root else None
    counts = Counter(finding.severity for finding in findings)
    tally = " · ".join(
        f"{counts[severity]} {severity.value}{'' if counts[severity] == 1 else 's'}"
        for severity in _MOST_SEVERE_FIRST
        if counts[severity]
    )
    lines = [
        f"**{len(findings)} finding{'' if len(findings) == 1 else 's'}** · {tally}",
        "",
        "| Severity | Where | What | Code |",
        "|---|---|---|---|",
    ]
    lines.extend(
        f"| {finding.severity.value.upper()} "
        f"| {_table_cell(_redacted(_markdown_location(finding, project_root), redact))} "
        f"| {_table_cell(finding.code.description)} "
        f"| {_inline_code(str(finding.code))} |"
        for finding in findings
    )

    detailed = [finding for finding in findings if finding.evidence]
    if detailed:
        lines.extend(["", "<details><summary>Evidence</summary>", ""])
        lines.extend(
            f"- {_inline_code(str(finding.code))} at "
            f"{_redacted(_markdown_location(finding, project_root), redact)}: "
            + " · ".join(
                f"{_inline_code(key)}: {_inline_code(_redacted(value, redact))}" for key, value in _evidence(finding)
            )
            for finding in detailed
        )
        lines.extend(["", "</details>"])

    return "\n".join(lines)


def _place(finding: Finding, project_root: Path | None) -> str:
    """Return where the finding was observed: its path, else its instance, else its layout node."""

    if finding.path is not None:
        if project_root is not None and finding.path.is_relative_to(project_root):
            return finding.path.relative_to(project_root).as_posix()
        return str(finding.path)

    if finding.instance:
        return "/".join(finding.instance)

    return finding.schema_id


def _field(finding: Finding) -> str:
    return ".".join(str(part) for part in finding.field_location)


def _location(finding: Finding, project_root: Path | None) -> str:
    field = _field(finding)
    place = _place(finding, project_root)
    return f"{place}:{field}" if field else place


def _markdown_location(finding: Finding, project_root: Path | None) -> str:
    field = _field(finding)
    place = _inline_code(_place(finding, project_root))
    return f"{place} {_FIELD_SEPARATOR} {_inline_code(field)}" if field else place


def _evidence(finding: Finding) -> list[tuple[str, str]]:
    """Return the evidence, the code's declared keys first, each value collapsed onto one line."""

    declared = [key for key in finding.code.evidence_keys if key in finding.evidence]
    extra = sorted(key for key in finding.evidence if key not in finding.code.evidence_keys)

    return [(key, " ".join(str(finding.evidence[key]).split())) for key in (*declared, *extra)]


def _redacted(text: str, project_root: Path | None) -> str:
    if project_root is None:
        return text

    for spelling in dict.fromkeys((str(project_root), project_root.as_posix())):
        text = text.replace(spelling, REDACTED_ROOT)

    return text


def _inline_code(text: str) -> str:
    """Wrap ``text`` in a code span that a backtick inside it cannot close."""

    if "`" in text:
        return f"`` {text} ``"
    return f"`{text}`"


def _table_cell(text: str) -> str:
    return text.replace("|", "\\|")


def _category_title(category: type[FindingCode]) -> str:
    """``ConfiguredVideoFindingCode`` reads as "Configured video"."""

    name = category.__name__.removesuffix("FindingCode")
    words = "".join(f" {char}" if char.isupper() else char for char in name).split()

    return " ".join([words[0], *(word.lower() for word in words[1:])])
