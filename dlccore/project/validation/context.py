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

from dlccore.config import ProjectConfig
from dlccore.project.schemas.base.core import FrozenModel
from dlccore.project.schemas.project_layout.config.observations import (
    ValidConfigObservation,
)
from dlccore.project.schemas.snapshots import ProjectSnapshot


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
