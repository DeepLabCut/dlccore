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

from enum import Enum
from pathlib import Path

from dlccore.project.schemas.base.core import FrozenModel


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
