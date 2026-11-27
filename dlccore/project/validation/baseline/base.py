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

from collections.abc import Callable
from dataclasses import dataclass

from dlccore.project.schemas.base.findings import Finding
from dlccore.project.schemas.snapshots import ProjectSnapshot
from dlccore.project.validation.context import ValidationContext

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
