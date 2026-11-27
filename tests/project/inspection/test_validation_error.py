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

from dlccore.project.errors import BLOCKING_SEVERITIES
from dlccore.project.schemas.base.finding_codes import Severity


def test_only_error_blocks() -> None:
    assert {Severity.ERROR} == BLOCKING_SEVERITIES
