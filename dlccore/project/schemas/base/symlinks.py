from __future__ import annotations

from enum import Enum
from pathlib import Path

from dlc_octopus.schemas.base.core import FrozenModel


class SymlinkTargetKind(str, Enum):
    """Observed type of a symbolic-link target."""

    FILE = "file"
    DIRECTORY = "directory"
    MISSING = "missing"
    OTHER = "other"
    INACCESSIBLE = "inaccessible"


class SymlinkObservation(FrozenModel):
    """Read-only observation of one symbolic link and its target."""

    raw_target: Path
    absolute_target: Path
    target_kind: SymlinkTargetKind
