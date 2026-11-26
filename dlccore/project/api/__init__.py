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

"""Public entry points of ``dlccore.project``; consumer helpers are in ``api.helpers``. Re-exports only."""

from dlccore.project.api import helpers
from dlccore.project.inspection import inspect_project, project_name
from dlccore.project.schemas.base.resources import FileResourceKind
from dlccore.project.schemas.scope import findings_in_scope

__all__ = [
    "FileResourceKind",
    "findings_in_scope",
    "helpers",
    "inspect_project",
    "project_name",
]
