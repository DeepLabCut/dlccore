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

"""Pure checks on labeled data. Never a verdict.

A write refuses or warns on a check, and validation reports the same check as a finding; neither re-derives it.
``file`` holds facts about a file as stored, ``model`` facts about the labels it holds.

Checks are for acting at the moment of use, such as refusing a write; to know a file's condition, use validation,
whose findings carry every relevant check.
"""
