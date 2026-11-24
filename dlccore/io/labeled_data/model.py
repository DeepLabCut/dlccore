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

"""DeepLabCut's labeled-data model: what is true in any file format."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# Keypoint data: always x and y, and a likelihood for models
# NOTE: 3D support is obviously missing
POSITION_COORDS = ("x", "y")
LIKELIHOOD_COORD = "likelihood"

# In multi-animal project, this denotes keypoints that do not belong to any individual.
# Extra caution is needed as this is a sentinel value, make sure to treat it accordingly.
# TODO: needs improvement in handling as this is a major source of confusion and potential bugs.
# It should not be used in any other format than the wide table "legacy" format.
UNIQUE_BODYPARTS_INDIVIDUAL = "single"

FramePath = tuple[str, ...]
"""A frame's path components as stored, e.g. ``("labeled-data", "session-01", "img000.png")``."""


class BodypartScope(str, Enum):
    """Whether a bodypart is labeled once per individual, or once per frame as DeepLabCut's ``uniquebodyparts``."""

    INDIVIDUAL = "individual"
    UNIQUE = "unique"


@dataclass(frozen=True)
class Keypoint:
    """Which keypoint: one bodypart, of one individual or unique, from one scorer.

    Attributes:
        scorer: As stored.
        individual: As stored; ``None`` when the source names no individual.
        bodypart: As stored.
    """

    scorer: str
    individual: str | None
    bodypart: str

    @property
    def scope(self) -> BodypartScope:
        """``UNIQUE`` exactly when the individual is DeepLabCut's unique-bodyparts bucket."""

        return BodypartScope.UNIQUE if self.individual == UNIQUE_BODYPARTS_INDIVIDUAL else BodypartScope.INDIVIDUAL
