#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/dlc-core
#
# Contributors:
# https://github.com/DeepLabCut/dlc-core/graphs/contributors
#
# Licensed under GNU Lesser General Public License v3.0
#

import dlc_core


def test_version_comes_from_the_installed_distribution() -> None:
    assert dlc_core.__version__ != "0+unknown"
