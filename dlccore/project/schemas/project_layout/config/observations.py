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

from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from dlccore.config import ProjectConfig
from dlccore.project.schemas.base.core import FrozenModel


class ValidConfigObservation(FrozenModel):
    """A project config successfully loaded through the canonical schema."""

    status: Literal["valid"] = "valid"
    config_path: Path
    declared_project_path: str | None = None

    value: ProjectConfig = Field(
        exclude=True,
        repr=False,
    )


class FailedConfigObservation(FrozenModel):
    """Diagnostic information retained after config loading failed."""

    status: Literal["failed"] = "failed"
    config_path: Path

    raw_config: dict[str, Any] | None = Field(
        default=None,
        exclude=True,
        repr=False,
    )


ConfigObservation = ValidConfigObservation | FailedConfigObservation
