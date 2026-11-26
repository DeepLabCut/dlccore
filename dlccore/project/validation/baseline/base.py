from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from dlc_octopus.schemas.base.findings import Finding
from dlc_octopus.schemas.snapshots import ProjectSnapshot
from dlc_octopus.validation.context import ValidationContext

BaselineValidator = Callable[
    [
        ProjectSnapshot,
        ValidationContext | None,
    ],
    tuple[Finding, ...],
]


@dataclass(frozen=True)
class BaselineCheckRegistration:
    """One automatic, lightweight project-level check."""

    code: str
    description: str
    validator: BaselineValidator
