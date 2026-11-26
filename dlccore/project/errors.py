"""Data validation-specific errors."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dlc_octopus.schemas.base.findings import Severity

if TYPE_CHECKING:
    from dlc_octopus.schemas.base.findings import Finding
    from dlc_octopus.schemas.snapshots import ProjectSnapshot


class ConfigUnavailableError(RuntimeError):
    """Raised when canonical project config data is unavailable."""


BLOCKING_SEVERITIES = frozenset({Severity.ERROR})


class ProjectValidationError(RuntimeError):
    """Raised when an inspected project contains blocking findings."""

    def __init__(
        self,
        snapshot: ProjectSnapshot,
    ) -> None:
        self.snapshot = snapshot
        self.findings = _blocking_findings(snapshot)

        codes = ", ".join(finding.code for finding in self.findings)

        super().__init__(f"Project validation failed with {len(self.findings)} blocking finding(s): {codes}.")


def _blocking_findings(
    snapshot: ProjectSnapshot,
) -> tuple[Finding, ...]:
    """Return the snapshot's findings whose severity is in ``BLOCKING_SEVERITIES``."""

    return tuple(finding for finding in snapshot.findings if finding.severity in BLOCKING_SEVERITIES)
