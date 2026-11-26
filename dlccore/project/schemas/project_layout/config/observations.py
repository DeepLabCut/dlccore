# dlc_octopus/schemas/config.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from deeplabcut.core.config import ProjectConfig
from pydantic import Field

from dlc_octopus.schemas.base.core import FrozenModel


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
