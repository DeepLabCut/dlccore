from __future__ import annotations

from dlc_octopus.schemas.base.core import FrozenModel
from dlc_octopus.schemas.base.findings import Finding
from dlc_octopus.schemas.base.validation import ValidationCheck


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
