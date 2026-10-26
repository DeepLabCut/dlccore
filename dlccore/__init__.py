#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Shared core of the DeepLabCut stack."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dlccore")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0+unknown"
