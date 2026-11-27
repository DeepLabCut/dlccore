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

"""Checks on a labeled-data file as stored: H5 keys, header, row index and values.

Each check takes one part of a read (the H5 key selection, the classified header or row index, or the wide
DataFrame) and returns empty, ``None`` or ``False`` when there is nothing to report.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from dlccore.io.labeled_data.long.model import FRAME_COLUMNS
from dlccore.io.labeled_data.model import LIKELIHOOD_COORD, POSITION_COORDS
from dlccore.io.labeled_data.wide.header import KeypointColumns, LabeledDataHeader, LabeledDataHeaderShape
from dlccore.io.labeled_data.wide.row_index import (
    LabeledDataRowIndex,
    RowIndexShape,
    frame_part_as_text,
    split_frame_path,
)

if TYPE_CHECKING:
    from dlccore.io.labeled_data.wide.h5 import H5KeySelection

_COORDINATES = frozenset({*POSITION_COORDS, LIKELIHOOD_COORD})


def unread_h5_keys(selection: H5KeySelection | None) -> tuple[str, ...]:
    """Return the keys of a multi-key H5 file that could not be read.

    Args:
        selection: The key selection of an H5 read; ``None`` for a file that is not H5.

    Returns:
        The unread keys, in the file's order. Empty unless the file holds several keys.
    """

    from dlccore.io.labeled_data.wide.h5 import H5KeySelectionOutcome

    if selection is None or selection.outcome is not H5KeySelectionOutcome.MULTI_KEY:
        return ()
    return tuple(key for key in selection.keys_found if key != selection.selected_key)


def legacy_h5_key(selection: H5KeySelection | None) -> str | None:
    """Return the file's only key if it is a legacy key, such as napari-deeplabcut's ``keypoints``.

    Args:
        selection: The key selection of an H5 read; ``None`` for a file that is not H5.

    Returns:
        The key, or ``None`` if the file is not H5, holds another key, or holds several keys.
    """

    from dlccore.io.labeled_data.wide.h5 import H5KeySelectionOutcome

    return _single_key(selection, H5KeySelectionOutcome.SINGLE_LEGACY)


def unknown_h5_key(selection: H5KeySelection | None) -> str | None:
    """Return the file's only key if no known writer uses it (it is not in ``H5_KEY_REGISTRY``).

    Args:
        selection: The key selection of an H5 read; ``None`` for a file that is not H5.

    Returns:
        The key, or ``None`` if the file is not H5, holds a known key, or holds several keys.
    """

    from dlccore.io.labeled_data.wide.h5 import H5KeySelectionOutcome

    return _single_key(selection, H5KeySelectionOutcome.SINGLE_UNKNOWN)


def _single_key(selection: H5KeySelection | None, outcome: object) -> str | None:
    return selection.selected_key if selection is not None and selection.outcome is outcome else None


def header_is_unrecognised(header: LabeledDataHeader) -> bool:
    """Return whether the header matches neither DeepLabCut layout (single- or multi-animal)."""

    return header.shape is LabeledDataHeaderShape.UNRECOGNISED


def keypoints_with_unrecognised_coords(header: LabeledDataHeader) -> tuple[KeypointColumns, ...]:
    """Return the keypoints whose coordinate columns are not ``x``, ``y`` and optionally ``likelihood``.

    A keypoint is returned if it lacks ``x`` or ``y``, has any other coordinate, or repeats one.
    """

    return tuple(
        columns
        for columns in header.keypoints
        if len(set(columns.coords)) != len(columns.coords)
        or not set(POSITION_COORDS) <= set(columns.coords) <= _COORDINATES
    )


def keypoints_without_the_likelihood_others_store(header: LabeledDataHeader) -> tuple[KeypointColumns, ...]:
    """Return the keypoints with no ``likelihood`` column when at least one other keypoint has one.

    Empty when every keypoint has a likelihood (machine labels) or none does (human labels).
    """

    without = tuple(columns for columns in header.keypoints if LIKELIHOOD_COORD not in columns.coords)
    return without if len(without) < len(header.keypoints) else ()


def non_numeric_columns(df: pd.DataFrame) -> tuple[object, ...]:
    """Return the labels of the columns of a wide DataFrame holding text, bool or complex values."""

    return tuple(label for label, column in df.items() if not _is_real(column))


def _is_real(column: pd.Series) -> bool:
    return (
        pd.api.types.is_numeric_dtype(column)
        and not pd.api.types.is_bool_dtype(column)
        and not pd.api.types.is_complex_dtype(column)
    )


def has_no_frames(row_index: LabeledDataRowIndex) -> bool:
    """Return whether the row index has no rows."""

    return row_index.shape is RowIndexShape.EMPTY


def row_index_is_unrecognised(row_index: LabeledDataRowIndex) -> bool:
    """Return whether the row index is neither a MultiIndex of path parts nor flat path strings."""

    return row_index.shape is RowIndexShape.UNRECOGNISED


def frames_of_other_depth(row_index: LabeledDataRowIndex) -> tuple[int, ...]:
    """Return the positions of the rows whose frame path is not three parts (folder, dataset, image)."""

    return tuple(row for row, frame in enumerate(row_index.frames) if len(frame) != len(FRAME_COLUMNS))


def frames_with_unnamed_parts(row_index: LabeledDataRowIndex) -> tuple[int, ...]:
    """Return the positions of the rows whose frame path has a part that is neither text nor an integer (e.g. NaN)."""

    return tuple(
        row for row, frame in enumerate(row_index.frames) if any(frame_part_as_text(part) is None for part in frame)
    )


def frame_absorbed_into_index_names(row_index: LabeledDataRowIndex) -> tuple[str, ...]:
    """Return the frame path held in the row index's level names, if any.

    When a CSV's first frame has no values, pandas reads that row as the row index's names, so the frame's path ends
    up there. DeepLabCut writes no index names, so names forming a three-part path are taken to be such a frame.

    Returns:
        The path's three parts; empty if the names are not all text or do not form a three-part path. A single name
        is split as a path, as a flat CSV index stores it.
    """

    names = [name for name in row_index.names if isinstance(name, str)]
    if not names or len(names) != len(row_index.names):
        return ()
    parts = split_frame_path(names[0]) if len(names) == 1 else tuple(names)
    return parts if len(parts) == len(FRAME_COLUMNS) else ()


def frames_lost_to_csv(df: pd.DataFrame) -> tuple[object, ...]:
    """Return the row-index entry that writing a wide DataFrame to CSV would lose, if any.

    If the first row has no values, pandas reads it back from the CSV as the row index's names, as DeepLabCut's
    ``convertcsv2h5`` does, and the row is gone. ``frame_absorbed_into_index_names`` finds it in a CSV already read.

    Returns:
        The first row's entry if all its values are NaN; otherwise empty.
    """

    if len(df) and df.iloc[0].isna().all():
        return (df.index[0],)
    return ()
