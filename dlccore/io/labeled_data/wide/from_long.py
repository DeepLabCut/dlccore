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

"""``LabeledData`` as DeepLabCut's current wide table: the one form this package writes."""

from __future__ import annotations

import pandas as pd

from dlccore.io.labeled_data.long.model import FRAME_COLUMNS, LabeledData, identity_columns
from dlccore.io.labeled_data.model import LIKELIHOOD_COORD, POSITION_COORDS
from dlccore.io.labeled_data.wide.header import MULTI_ANIMAL_LEVELS, SINGLE_ANIMAL_LEVELS


def long_to_wide(labels: LabeledData) -> pd.DataFrame:
    """Return the wide table DeepLabCut writes today for these labels.

    One row per frame, indexed by path parts.
    Levels must hold plain text, never categoricals: DeepLabCut and
    napari-deeplabcut group on them without ``observed=``.

    Args:
        labels: The labels to write.
    """

    long = labels.to_long_df()
    keypoint_columns = identity_columns(long)
    coords = (*POSITION_COORDS, LIKELIHOOD_COORD) if labels.has_likelihood else POSITION_COORDS
    n_frames, n_keypoints = long["frame_index"].nunique(), long["keypoint_index"].nunique()

    # Rows are frame-major, so the first frame's rows name the keypoints and every K-th row starts a frame.
    keypoints = long.iloc[:n_keypoints]
    frames = long.iloc[::n_keypoints] if n_keypoints else long.iloc[:0]

    columns = pd.MultiIndex.from_arrays(
        [
            *(keypoints[name].astype(str).repeat(len(coords)).to_numpy() for name in keypoint_columns),
            list(coords) * n_keypoints,
        ],
        names=MULTI_ANIMAL_LEVELS if labels.is_multi_animal else SINGLE_ANIMAL_LEVELS,
    )
    index = pd.MultiIndex.from_arrays([frames[name].astype(str).to_numpy() for name in FRAME_COLUMNS])
    values = long[list(coords)].to_numpy().reshape(n_frames, n_keypoints * len(coords))

    return pd.DataFrame(values, index=index, columns=columns)
