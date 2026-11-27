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

"""Conversion of a wide DataFrame to ``LabeledData``.

Values are taken from the column positions the header classification found, never looked up by level values, so
duplicated or reordered columns cannot pick up the wrong values.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from dlccore.io.labeled_data.checks.file import (
    frames_of_other_depth,
    frames_with_unnamed_parts,
    has_no_frames,
    header_is_unrecognised,
    keypoints_with_unrecognised_coords,
    keypoints_without_the_likelihood_others_store,
    non_numeric_columns,
    row_index_is_unrecognised,
)
from dlccore.io.labeled_data.long.model import (
    FRAME_COLUMNS,
    LabeledData,
    identity_values,
)
from dlccore.io.labeled_data.model import LIKELIHOOD_COORD, POSITION_COORDS
from dlccore.io.labeled_data.wide.header import LabeledDataHeader, LabeledDataHeaderShape
from dlccore.io.labeled_data.wide.row_index import LabeledDataRowIndex, frame_part_as_text
from dlccore.io.labeled_data.wide.table import classify_wide_df


def wide_to_long(
    df: pd.DataFrame,
    header: LabeledDataHeader,
    row_index: LabeledDataRowIndex,
) -> LabeledData:
    """Convert a wide DataFrame, already classified, to ``LabeledData``: one row per frame and keypoint.

    Values are copied as stored. A frame path part stored as an integer (e.g. a folder named ``2019``) becomes
    text, as DeepLabCut's ``guarantee_multiindex_rows`` does; ``row_index`` itself is not changed.

    Args:
        df: The wide DataFrame as read.
        header: The classified header of ``df``.
        row_index: The classified row index of ``df``.

    Raises:
        ValueError: On the first of these problems found:

            - Invalid header or row index;
            - No rows;
            - A frame path is not three parts, or has a part that is neither text nor an integer;
            - A keypoint's coordinates are not ``x``, ``y`` and optionally ``likelihood``;
            - Some keypoints have a likelihood and others do not;
            - A value column is not numeric.
    """

    frames = _frame_parts(row_index)

    if header_is_unrecognised(header):
        raise ValueError(f"Header levels {list(header.level_names)} are not a header DeepLabCut writes.")

    positions = _value_positions(df, header)

    n_frames, n_keypoints = len(frames), len(positions)
    keypoints = [columns.keypoint for columns in header.keypoints]
    is_multi_animal = header.shape is LabeledDataHeaderShape.MULTI_ANIMAL

    def values(coordinate: str) -> np.ndarray:
        # Frame-major, matching the row order below.
        return df.iloc[:, [position[coordinate] for position in positions]].to_numpy(dtype=float).reshape(-1)

    long = {name: _expanded([frame[i] for frame in frames], repeat=n_keypoints) for i, name in enumerate(FRAME_COLUMNS)}
    long |= {
        name: _expanded(column, tile=n_frames)
        for name, column in identity_values(keypoints, multi_animal=is_multi_animal).items()
    }
    long |= {name: values(name) for name in POSITION_COORDS}
    if positions and LIKELIHOOD_COORD in positions[0]:
        long[LIKELIHOOD_COORD] = values(LIKELIHOOD_COORD)
    long["frame_index"] = np.repeat(np.arange(n_frames, dtype=np.int64), n_keypoints)
    long["keypoint_index"] = np.tile(np.arange(n_keypoints, dtype=np.int64), n_frames)

    return LabeledData.from_long_df(pd.DataFrame(long))


def wide_df_to_long(df: pd.DataFrame) -> LabeledData:
    """Classify a wide DataFrame's header and row index, then convert it with ``wide_to_long``.

    Raises:
        ValueError: As ``wide_to_long``.
    """

    header, row_index = classify_wide_df(df)
    return wide_to_long(df, header, row_index)


def _value_positions(df: pd.DataFrame, header: LabeledDataHeader) -> list[dict[str, int]]:
    """Return each keypoint's column position per coordinate, in header order, or raise naming the first check found."""

    if unrecognised := keypoints_with_unrecognised_coords(header):
        columns = unrecognised[0]
        raise ValueError(
            f"{columns.keypoint} has coordinates {list(columns.coords)}; x, y and optionally likelihood are "
            "needed, each once."
        )
    if lacking := keypoints_without_the_likelihood_others_store(header):
        stores = next(columns.keypoint for columns in header.keypoints if LIKELIHOOD_COORD in columns.coords)
        raise ValueError(
            f"Some keypoints store a likelihood and others do not: {stores} does, {lacking[0].keypoint} does not."
        )
    if not_numeric := non_numeric_columns(df):
        raise ValueError(f"Value columns {list(not_numeric)} are not numeric.")

    return [dict(zip(columns.coords, columns.positions, strict=True)) for columns in header.keypoints]


def _expanded(values: list[str], *, repeat: int = 1, tile: int = 1) -> pd.Categorical:
    """Return ``values`` repeated element-wise, then tiled, as a categorical built from integer codes.

    Only the codes are expanded to one per row; the text is hashed once per distinct value, never once per row.
    """

    codes, categories = pd.factorize(np.asarray(values, dtype=object), sort=True)
    return pd.Categorical.from_codes(np.tile(np.repeat(codes, repeat), tile), categories=categories)


def _frame_parts(row_index: LabeledDataRowIndex) -> list[tuple[str, ...]]:
    """Return each frame's three parts as text, or raise naming why the row index has no model."""

    if has_no_frames(row_index):
        raise ValueError("The table has no frames; the model holds keypoints only through frames.")
    if row_index_is_unrecognised(row_index):
        entry = next(entry for entry in row_index.entries if not isinstance(entry, str))
        raise ValueError(f"The row index is neither path parts nor path strings: entry {entry!r} is not text.")

    if other_depth := frames_of_other_depth(row_index):
        frame = row_index.frames[other_depth[0]]
        raise ValueError(f"Frame {frame} has {len(frame)} parts; the model needs {len(FRAME_COLUMNS)}.")
    if unnamed := frames_with_unnamed_parts(row_index):
        frame = row_index.frames[unnamed[0]]
        part = next(part for part in frame if frame_part_as_text(part) is None)
        raise ValueError(f"Frame {frame} has a part {part!r} that is neither text nor an integer.")
    return [tuple(cast("str", frame_part_as_text(part)) for part in frame) for frame in row_index.frames]
