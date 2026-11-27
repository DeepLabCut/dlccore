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
from typing import TYPE_CHECKING, Literal

from pydantic import model_validator
from typing_extensions import Self

from dlccore.project.errors import ConfigUnavailableError
from dlccore.project.schemas.base.core import FrozenModel
from dlccore.project.schemas.base.findings import (
    Finding,
    Reach,
)
from dlccore.project.schemas.base.observations import ObservedNode
from dlccore.project.schemas.base.resources import (
    ProjectResource,
    ResourceKind,
)
from dlccore.project.schemas.base.validation import ValidationCheck
from dlccore.project.schemas.project_layout.config import (
    ConfigObservation,
    FailedConfigObservation,
    ValidConfigObservation,
)
from dlccore.project.schemas.utils import flatten_observed_project

if TYPE_CHECKING:
    from dlccore.config import ProjectConfig
    from dlccore.project.validation.results import ValidationResult

# 0.x declares the snapshot format unstable: it changes with the vocabulary and carries
# no compatibility promise. Bumping is owed once a consumer reads a snapshot back.
ProjectSnapshotVersion = Literal["0.0"]
PROJECT_SNAPSHOT_VERSION: ProjectSnapshotVersion = "0.0"


class ConfigState(str, Enum):
    MISSING = "missing"
    VALID = "valid"
    INVALID = "invalid"


class ProjectSnapshot(FrozenModel):
    """Read-only description of a DeepLabCut project at inspection time.

    ``project`` is the authoritative schema-derived tree. It includes
    expected nodes that may be missing, unreadable, or have the wrong
    filesystem type.

    ``resources`` is a deterministic flat projection of physical entries
    from that tree. It includes any schema-defined node with an actual
    filesystem entry - present, wrong-type, or unreadable alike - plus
    immediate unclaimed resources. Only nodes with no filesystem entry at
    all remain available exclusively through ``project``.
    """

    version: ProjectSnapshotVersion = PROJECT_SNAPSHOT_VERSION
    project_root: Path
    config_observation: ConfigObservation | None = None
    project: ObservedNode
    resources: tuple[ProjectResource, ...] = ()
    findings: tuple[Finding, ...] = ()
    validation_checks: tuple[ValidationCheck, ...] = ()

    @property
    def resources_by_path(
        self,
    ) -> dict[Path, ProjectResource]:
        """Index the flat resource projection (physical) by project-relative path."""

        return {resource.project_relative_path: resource for resource in self.resources}

    @classmethod
    def from_project(
        cls,
        *,
        project_root: Path,
        project: ObservedNode,
        config_observation: ConfigObservation | None = None,
        findings: tuple[Finding, ...] = (),
        validation_checks: tuple[ValidationCheck, ...] = (),
    ) -> ProjectSnapshot:
        """Build a snapshot from the observed project tree and derive its resources."""
        return cls(
            project_root=project_root,
            project=project,
            config_observation=config_observation,
            findings=findings,
            resources=flatten_observed_project(project),
            validation_checks=validation_checks,
        )

    def with_validation_result(
        self,
        result: ValidationResult,
    ) -> ProjectSnapshot:
        """Return a copy with validation findings and checks merged in.

        Unlike ``model_copy``, this reconstructs the snapshot from its
        current field values, so model validators run again on the result
        """

        return ProjectSnapshot(
            **{
                **dict(self),
                "findings": (self.findings + result.findings),
                "validation_checks": (self.validation_checks + result.checks),
            }
        )

    @model_validator(mode="after")
    def validate_resource_projection(
        self,
    ) -> Self:
        expected = flatten_observed_project(self.project)

        if self.resources != expected:
            raise ValueError("Snapshot resources must be the deterministic projection of the observed project tree.")

        return self

    @model_validator(mode="after")
    def refuse_unrooted_findings(self) -> Self:
        """Refuse a finding outside any known project: the scope query matches instances from the project root."""

        if unrooted := [finding for finding in self.findings if finding.reach is Reach.UNROOTED]:
            raise ValueError(f"A snapshot holds no UNROOTED finding; got {unrooted[0].code} at {unrooted[0].instance}.")

        return self

    def resources_of_kind(
        self,
        kind: ResourceKind,
    ) -> tuple[ProjectResource, ...]:
        """Return resources having the requested semantic kind."""

        return tuple(resource for resource in self.resources if resource.kind == kind)

    def resource_at(
        self,
        project_relative_path: str | Path,
    ) -> ProjectResource | None:
        """Return the resource observed at one project-relative path."""
        target = Path(project_relative_path)
        return next(
            (resource for resource in self.resources if resource.project_relative_path == target),
            None,
        )

    @property
    def config_state(
        self,
    ) -> ConfigState:
        """Return the canonical configuration observation state."""

        if self.config_observation is None:
            return ConfigState.MISSING

        if self.failed_config is not None:
            return ConfigState.INVALID

        return ConfigState.VALID

    @property
    def config(self) -> ProjectConfig:
        """Return canonical config data or fail explicitly."""
        observation = self.config_observation
        if isinstance(
            observation,
            ValidConfigObservation,
        ):
            return observation.value

        if isinstance(
            observation,
            FailedConfigObservation,
        ):
            raise ConfigUnavailableError(
                "The project config was found but could not be loaded through the canonical ProjectConfig schema."
            )

        raise ConfigUnavailableError("The project does not have an observed config.")

    @property
    def failed_config(
        self,
    ) -> FailedConfigObservation | None:
        """Return failed config diagnostics, if loading failed."""
        observation = self.config_observation

        if isinstance(
            observation,
            FailedConfigObservation,
        ):
            return observation

        return None
