from __future__ import annotations

from enum import Enum
from pathlib import Path

from dlc_octopus.schemas.base.core import FrozenModel
from dlc_octopus.schemas.base.resources import (
    ResourceKind,
)


class ValidationDepth(str, Enum):
    """Supported validation depth for one explicitly requested domain."""

    FILENAME_ONLY = "filename_only"
    FULL = "full"


class ValidationCheckStatus(str, Enum):
    """Outcome of one requested validation check."""

    COMPLETED = "completed"
    UNAVAILABLE = "unavailable"


class ValidationCheck(FrozenModel):
    """Execution record for one explicitly requested validation check."""

    code: str
    kind: ResourceKind
    depth: ValidationDepth
    status: ValidationCheckStatus
    message: str
    path: Path | None = None
