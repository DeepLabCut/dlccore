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

from typing import TYPE_CHECKING

from dlccore.project.schemas.base.core import FrozenModel, SchemaId
from dlccore.project.schemas.base.resources import (
    ResourceKind,
    ResourceManagement,
)
from dlccore.project.schemas.base.validation import (
    ValidationDepth,
)

if TYPE_CHECKING:
    from IPython.lib.pretty import RepresentationPrinter


class LayoutCapability(FrozenModel):
    """One node in the declared DeepLabCut project layout."""

    schema_id: SchemaId
    kind: ResourceKind
    required: bool
    management: ResourceManagement
    repeated: bool = False
    inside_repeated: bool = False
    children: tuple[LayoutCapability, ...] = ()


class BaselineValidationCapability(FrozenModel):
    """One automatic project-level validation capability."""

    code: str
    description: str


class ValidationCapability(FrozenModel):
    """Validation depths available for one resource kind."""

    kind: ResourceKind
    supported_depths: tuple[ValidationDepth, ...]


class ProjectManagerCapabilities(FrozenModel):
    """Versioned description of project-management affordances."""

    snapshot_version: str
    layout: LayoutCapability
    baseline_validations: tuple[BaselineValidationCapability, ...] = ()
    validations: tuple[ValidationCapability, ...] = ()

    def _repr_pretty_(self, printer: RepresentationPrinter, cycle: bool) -> None:
        from dlccore.project.visualization.capabilities import render_capabilities

        printer.text("ProjectManagerCapabilities(...)" if cycle else render_capabilities(self))

    def _repr_markdown_(self) -> str:
        from dlccore.project.visualization.capabilities import render_capabilities

        return render_capabilities(self, style="markdown")
