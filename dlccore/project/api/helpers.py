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

"""What a consumer composes with the entry points in ``dlccore.project.api``: scopes and its own finding policy."""

from dlccore.project.policy import get_blocking_findings, unhandled
from dlccore.project.schemas.scope import Scope

__all__ = ["Scope", "get_blocking_findings", "unhandled"]
