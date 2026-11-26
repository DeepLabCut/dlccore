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

from dlccore.project.schemas.base.findings import Finding
from dlccore.project.schemas.snapshots import ProjectSnapshot
from dlccore.project.validation.context import ValidationContext

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
