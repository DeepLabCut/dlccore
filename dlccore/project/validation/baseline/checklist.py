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

from dlccore.project.schemas.base.findings import Finding
from dlccore.project.schemas.snapshots import ProjectSnapshot
from dlccore.project.validation.baseline.base import (
    BaselineCheckRegistration,
)
from dlccore.project.validation.context import ValidationContext
from dlccore.project.validation.deeplabcut.project_path import (
    validate_project_path,
)
from dlccore.project.validation.structure import (
    validate_project_structure,
)


def _config_validation_already_applied(
    snapshot: ProjectSnapshot,
    context: ValidationContext | None,
) -> tuple[Finding, ...]:
    """No-op empty validator: config is validated eagerly, before a snapshot exists.

    This registration exists only so ``get_project_manager_capabilities()``
    accurately advertises that config validation always happens.

    ``inspect_project`` reads and validates the config file via
    ``validate_project_config`` before ``ProjectSnapshot.from_project`` runs,
    so its findings are already present in ``snapshot.findings`` by the time
    baseline checks run here.
    """

    del snapshot, context

    return ()


def _validate_structure(
    snapshot: ProjectSnapshot,
    context: ValidationContext | None,
) -> tuple[Finding, ...]:
    """Run baseline structural validation."""

    del context

    return validate_project_structure(
        snapshot.project,
        project_root=snapshot.project_root,
    )


def _validate_config_project_path(
    snapshot: ProjectSnapshot,
    context: ValidationContext | None,
) -> tuple[Finding, ...]:
    """Compare declared and inspected project paths when applicable."""

    del snapshot

    if context is None:
        return ()

    return validate_project_path(context)


BASELINE_CHECKS = (
    BaselineCheckRegistration(
        code="PROJECT_CONFIG_VALIDATION",
        description=("Reads and validates config.yaml against the canonical schema."),
        validator=_config_validation_already_applied,
    ),
    BaselineCheckRegistration(
        code="PROJECT_STRUCTURE",
        description=("Checks required-node presence, filesystem type, and readability."),
        validator=_validate_structure,
    ),
    BaselineCheckRegistration(
        code="PROJECT_CONFIG_PATH_RELATIONSHIP",
        description=("Compares the project path declared by a valid config with the inspected project root."),
        validator=_validate_config_project_path,
    ),
)
