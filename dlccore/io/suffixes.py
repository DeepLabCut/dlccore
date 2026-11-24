"""Lower-case file suffixes and name prefixes per file kind, and a case-insensitive match against the suffixes.

Data only, and imports nothing from ``dlc_octopus``: Octopus's layout imports it, so that a file is claimed by the
layout exactly when ``dlc_octopus.io`` can read it.
"""

from __future__ import annotations

from pathlib import PurePath

H5_SUFFIX = ".h5"
CSV_SUFFIX = ".csv"

LABELED_DATA_TABLE_SUFFIXES = frozenset({H5_SUFFIX, CSV_SUFFIX})
"""A wide labeled-data table: DeepLabCut's H5 and its CSV twin."""

HUMAN_LABELS_PREFIX = "CollectedData_"
"""Human ground-truth labels, ``CollectedData_<scorer>``; matched case-sensitively, as DeepLabCut writes it."""

VIDEO_SUFFIXES = frozenset({".avi", ".mp4", ".mov", ".mpeg", ".mpg", ".mpv", ".mkv", ".flv", ".qt", ".yuv"})
"""Videos DeepLabCut reads: ``SUPPORTED_VIDEOS`` in ``deeplabcut/utils/auxfun_videos.py`` at ``v3.0.2``."""


def suffix_in(path: PurePath, suffixes: frozenset[str]) -> str | None:
    """Return ``path.suffix.lower()`` if it is in ``suffixes``, else ``None``. ``suffixes`` must be lower case."""

    suffix = path.suffix.lower()
    return suffix if suffix in suffixes else None
