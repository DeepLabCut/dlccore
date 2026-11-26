from __future__ import annotations

from deeplabcut.core.config import ProjectConfig

from dlc_octopus.schemas.base.core import FrozenModel
from dlc_octopus.schemas.project_layout.config.observations import (
    ValidConfigObservation,
)
from dlc_octopus.schemas.snapshots import ProjectSnapshot


class ValidationContext(FrozenModel):
    """Canonical inputs available to config-dependent project validators."""

    snapshot: ProjectSnapshot
    config: ProjectConfig
    # PoseConfig
    # and any other validation metadata...


def build_validation_context(
    snapshot: ProjectSnapshot,
) -> ValidationContext | None:
    """Build config-dependent validation context when config is valid."""

    observation = snapshot.config_observation

    if not isinstance(
        observation,
        ValidConfigObservation,
    ):
        return None

    return ValidationContext(
        snapshot=snapshot,
        config=observation.value,
    )
