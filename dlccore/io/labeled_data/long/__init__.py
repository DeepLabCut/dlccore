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

"""Labeled data as one long table: the model every format converts to and from.

Each format's conversion code lives in that format's module. A format behind an optional extra is reached only through
free functions there, never through a ``LabeledData`` method, so the model needs no extra.
"""
