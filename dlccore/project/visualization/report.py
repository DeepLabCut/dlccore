from __future__ import annotations

from collections import Counter

from pydantic import JsonValue

from dlc_octopus.schemas.base import ResourceManagement
from dlc_octopus.schemas.base.findings import Finding
from dlc_octopus.schemas.base.validation import ValidationCheckStatus
from dlc_octopus.schemas.project_layout.config.observations import (
    FailedConfigObservation,
    ValidConfigObservation,
)
from dlc_octopus.schemas.snapshots import ProjectSnapshot
from dlc_octopus.visualization.tree import render_project_tree


def render_project_report(
    snapshot: ProjectSnapshot,
    *,
    show_schema_ids: bool = False,
    show_unknown: bool = True,
    show_unmanaged: bool = False,
    show_baseline_note: bool = True,
    max_children_per_kind: int | None = 10,
) -> str:
    """Render a complete human-readable project inspection report."""

    sections = [
        _render_header(snapshot),
        _render_config_summary(snapshot),
        _render_resource_summary(snapshot, show_unmanaged=show_unmanaged),
        _render_checks(snapshot, show_baseline_note=show_baseline_note),
        _render_findings(snapshot),
        _render_layout(
            snapshot,
            show_schema_ids=show_schema_ids,
            show_unknown=show_unknown,
            show_unmanaged=show_unmanaged,
            max_children_per_kind=max_children_per_kind,
        ),
    ]

    return "\n\n".join(sections)


def _render_header(
    snapshot: ProjectSnapshot,
) -> str:
    """Render the report title and project location."""

    return "\n".join(
        [
            "DeepLabCut project inspection",
            "=" * 30,
            f"Project root: {snapshot.project_root}",
        ]
    )


def _render_config_summary(
    snapshot: ProjectSnapshot,
) -> str:
    """Render canonical config observation status."""

    lines = [
        "Configuration",
        "-------------",
    ]
    observation = snapshot.config_observation

    if isinstance(
        observation,
        ValidConfigObservation,
    ):
        declared_path = observation.declared_project_path or "<not declared>"

        lines.extend(
            [
                "Status: valid",
                f"Config file: {observation.config_path}",
                f"Declared project path: {declared_path}",
            ]
        )
    elif isinstance(
        observation,
        FailedConfigObservation,
    ):
        raw_config_retained = "yes" if observation.raw_config is not None else "no"

        lines.extend(
            [
                "Status: failed",
                f"Config file: {observation.config_path}",
                f"Raw config retained for diagnostics: {raw_config_retained}",
            ]
        )
    else:
        lines.append("Status: not found")

    return "\n".join(lines)


def _render_resource_summary(snapshot: ProjectSnapshot, *, show_unmanaged: bool) -> str:
    """Render discovered resource counts grouped by semantic kind."""

    lines = [
        "Resource summary",
        "----------------",
    ]
    visible_resources = tuple(
        resource
        for resource in snapshot.resources
        if show_unmanaged or resource.management != ResourceManagement.KNOWN_UNMANAGED
    )
    kind_counts = Counter(resource.kind.value for resource in visible_resources)

    if not kind_counts:
        lines.append("No physical project resources were discovered.")
        return "\n".join(lines)

    lines.extend(f"{kind}: {count}" for kind, count in sorted(kind_counts.items()))

    return "\n".join(lines)


def _render_checks(
    snapshot: ProjectSnapshot,
    *,
    show_baseline_note: bool,
) -> str:
    """Render which requested checks ran and how each ended, so no findings can be told from not checked."""

    lines = [
        "Checks",
        "------",
    ]

    if show_baseline_note:
        lines.append("(Baseline structure and config checks always run and are not listed.)")

    if not snapshot.validation_checks:
        lines.append("No resource checks were requested.")
        return "\n".join(lines)

    for check in snapshot.validation_checks:
        lines.append(f"- {check.code} on {check.kind.value} ({check.depth.value}): {check.status.value}")

        if check.status is not ValidationCheckStatus.COMPLETED:
            lines.append(f"    {check.message}")

    return "\n".join(lines)


def _render_findings(
    snapshot: ProjectSnapshot,
) -> str:
    """Render findings in a concise human-readable form."""

    lines = [
        "Findings",
        "--------",
    ]

    if not snapshot.findings:
        lines.append("No findings were reported by the checks that ran.")
        return "\n".join(lines)

    for finding in snapshot.findings:
        schema_id = f" at {finding.schema_id}"
        instance = " (" + "/".join(str(part) for part in finding.instance) + ")" if finding.instance else ""
        path = f" [{finding.path}]" if finding.path is not None else ""

        field = " field=" + ".".join(str(part) for part in finding.field_location) if finding.field_location else ""

        lines.append(
            f"- {finding.severity.value.upper()} {finding.code}{schema_id}{instance}{path}{field}: "
            f"{finding.code.description}"
        )
        lines.extend(_render_evidence(finding))

    return "\n".join(lines)


def _render_evidence(
    finding: Finding,
) -> tuple[str, ...]:
    """Render one finding's evidence, the code's declared keys first."""

    declared = [key for key in finding.code.evidence_keys if key in finding.evidence]
    extra = sorted(key for key in finding.evidence if key not in finding.code.evidence_keys)

    return tuple(f"    {key}={_render_evidence_value(finding.evidence[key])}" for key in (*declared, *extra))


def _render_evidence_value(
    value: JsonValue,
) -> str:
    """Collapse one evidence value onto a single line."""

    return " ".join(str(value).split())


def _render_layout(
    snapshot: ProjectSnapshot,
    *,
    show_schema_ids: bool,
    show_unmanaged: bool,
    show_unknown: bool,
    max_children_per_kind: int | None,
) -> str:
    """Render the semantic project layout section."""

    tree = render_project_tree(
        snapshot.project,
        show_schema_ids=show_schema_ids,
        show_unknown=show_unknown,
        show_unmanaged=show_unmanaged,
        max_children_per_kind=max_children_per_kind,
    )

    return "\n".join(
        [
            "Project layout",
            "--------------",
            tree,
        ]
    )
