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

"""Lower-case file suffixes and name prefixes per file kind.

Data only, imports stdlib only, so that code deciding which files belong to a project can import
it without loading any reader, and a file is reported as supported only when ``dlccore.io`` can read it.
"""

from __future__ import annotations

from pathlib import PurePath

# Labeled data
H5_SUFFIX = ".h5"
CSV_SUFFIX = ".csv"

LABELED_DATA_TABLE_SUFFIXES = frozenset({H5_SUFFIX, CSV_SUFFIX})
"""A wide labeled-data table: DeepLabCut's H5 and its CSV twin."""

HUMAN_LABELS_PREFIX = "CollectedData_"
"""Human ground-truth labels, ``CollectedData_<scorer>``; matched case-sensitively, as DeepLabCut writes it."""

# Video
VIDEO_SUFFIXES = frozenset({".avi", ".mp4", ".mov", ".mpeg", ".mpg", ".mpv", ".mkv", ".flv", ".qt", ".yuv"})
"""Videos DeepLabCut reads: ``SUPPORTED_VIDEOS`` in ``deeplabcut/utils/auxfun_videos.py`` at ``v3.0.2``."""


def suffix_in(path: PurePath, suffixes: frozenset[str]) -> str | None:
    """Return ``path.suffix.lower()`` if it is in ``suffixes``, else ``None``. ``suffixes`` must be lower case."""

    suffix = path.suffix.lower()
    return suffix if suffix in suffixes else None
