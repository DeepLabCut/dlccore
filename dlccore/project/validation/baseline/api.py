from __future__ import annotations

from dlc_octopus.schemas.base.findings import Finding
from dlc_octopus.schemas.snapshots import ProjectSnapshot
from dlc_octopus.validation.context import ValidationContext

from .base import BaselineCheckRegistration
from .checklist import BASELINE_CHECKS


def iter_baseline_check_registrations() -> tuple[BaselineCheckRegistration, ...]:
    """Return built-in baseline checks in execution order."""

    return BASELINE_CHECKS


def run_baseline_checks(
    snapshot: ProjectSnapshot,
    context: ValidationContext | None,
) -> tuple[Finding, ...]:
    """Run all built-in baseline checks in execution order."""

    findings: list[Finding] = []

    for registration in BASELINE_CHECKS:
        findings.extend(
            registration.validator(
                snapshot,
                context,
            )
        )

    return tuple(findings)
