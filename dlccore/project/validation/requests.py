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

from pydantic import model_validator
from typing_extensions import Self

from dlccore.project.schemas.base.core import FrozenModel
from dlccore.project.schemas.base.resources import (
    ResourceKind,
)
from dlccore.project.schemas.base.validation import ValidationDepth


class ResourceValidationRequest(FrozenModel):
    """Validation requested for one semantic resource kind."""

    kind: ResourceKind
    depth: ValidationDepth


class ProjectValidationRequest(FrozenModel):
    """Explicit resource validation requested for one project inspection."""

    resources: tuple[ResourceValidationRequest, ...] = ()

    @model_validator(mode="after")
    def validate_unique_resource_kinds(
        self,
    ) -> Self:
        """Reject duplicate requests for the same resource kind."""

        kinds = [request.kind for request in self.resources]

        if len(kinds) != len(set(kinds)):
            raise ValueError("Each resource kind may be requested at most once.")

        return self
