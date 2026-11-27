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

from dlccore.project.schemas.base.core import FrozenModel
from dlccore.project.schemas.base.findings import Finding
from dlccore.project.schemas.base.validation import ValidationCheck


class ValidationResult(FrozenModel):
    """Findings and execution records produced by validation."""

    findings: tuple[Finding, ...] = ()
    checks: tuple[ValidationCheck, ...] = ()

    def merge(
        self,
        other: ValidationResult,
    ) -> ValidationResult:
        """Combine two immutable validation results."""

        return ValidationResult(
            findings=(self.findings + other.findings),
            checks=(self.checks + other.checks),
        )
