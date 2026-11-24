"""Facts about a labeled-data file as stored: its keys, header, row index, values, and what its CSV copy keeps.

Each check takes one of ``LabeledDataFile``'s facts, or the table ``to_raw_wide_df()`` returns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from dlc_octopus.io.labeled_data.long.model import FRAME_COLUMNS
from dlc_octopus.io.labeled_data.model import LIKELIHOOD_COORD, POSITION_COORDS
from dlc_octopus.io.labeled_data.wide.header import KeypointColumns, LabeledDataHeader, LabeledDataHeaderShape
from dlc_octopus.io.labeled_data.wide.row_index import (
    LabeledDataRowIndex,
    RowIndexShape,
    frame_part_as_text,
    split_frame_path,
)

if TYPE_CHECKING:
    from dlc_octopus.io.labeled_data.wide.h5 import H5KeySelection

_COORDINATES = frozenset({*POSITION_COORDS, LIKELIHOOD_COORD})


def unread_h5_keys(selection: H5KeySelection | None) -> tuple[str, ...]:
    """The keys found but not read, in the file's order: non-empty exactly when the outcome is ``MULTI_KEY``."""

    from dlc_octopus.io.labeled_data.wide.h5 import H5KeySelectionOutcome

    if selection is None or selection.outcome is not H5KeySelectionOutcome.MULTI_KEY:
        return ()
    return tuple(key for key in selection.keys_found if key != selection.selected_key)


def legacy_h5_key(selection: H5KeySelection | None) -> str | None:
    """The file's one key, when it is a key older DeepLabCut releases wrote; ``None`` otherwise."""

    from dlc_octopus.io.labeled_data.wide.h5 import H5KeySelectionOutcome

    return _single_key(selection, H5KeySelectionOutcome.SINGLE_LEGACY)


def unknown_h5_key(selection: H5KeySelection | None) -> str | None:
    """The file's one key, when no DeepLabCut release is known to write it; ``None`` otherwise."""

    from dlc_octopus.io.labeled_data.wide.h5 import H5KeySelectionOutcome

    return _single_key(selection, H5KeySelectionOutcome.SINGLE_UNKNOWN)


def _single_key(selection: H5KeySelection | None, outcome: object) -> str | None:
    return selection.selected_key if selection is not None and selection.outcome is outcome else None


def header_is_unrecognised(header: LabeledDataHeader) -> bool:
    """Whether the header's level names are neither of the two DeepLabCut writes."""

    return header.shape is LabeledDataHeaderShape.UNRECOGNISED


def keypoints_with_unrecognised_coords(header: LabeledDataHeader) -> tuple[KeypointColumns, ...]:
    """The keypoints whose coordinates are not exactly x, y and optionally likelihood, each once."""

    return tuple(
        columns
        for columns in header.keypoints
        if len(set(columns.coords)) != len(columns.coords)
        or not set(POSITION_COORDS) <= set(columns.coords) <= _COORDINATES
    )


def keypoints_without_the_likelihood_others_store(header: LabeledDataHeader) -> tuple[KeypointColumns, ...]:
    """The keypoints storing no likelihood where another keypoint stores one; empty when all or none do."""

    without = tuple(columns for columns in header.keypoints if LIKELIHOOD_COORD not in columns.coords)
    return without if len(without) < len(header.keypoints) else ()


def non_numeric_columns(table: pd.DataFrame) -> tuple[object, ...]:
    """The column labels of a wide table whose values are not real numbers: text, bool or complex."""

    return tuple(label for label, column in table.items() if not _is_real(column))


def _is_real(column: pd.Series) -> bool:
    return (
        pd.api.types.is_numeric_dtype(column)
        and not pd.api.types.is_bool_dtype(column)
        and not pd.api.types.is_complex_dtype(column)
    )


def has_no_frames(row_index: LabeledDataRowIndex) -> bool:
    """Whether the table has no rows."""

    return row_index.shape is RowIndexShape.EMPTY


def row_index_is_unrecognised(row_index: LabeledDataRowIndex) -> bool:
    """Whether the row index is neither path parts nor path strings."""

    return row_index.shape is RowIndexShape.UNRECOGNISED


def frames_of_other_depth(row_index: LabeledDataRowIndex) -> tuple[int, ...]:
    """The rows whose frame has other than the three parts the model holds."""

    return tuple(row for row, frame in enumerate(row_index.frames) if len(frame) != len(FRAME_COLUMNS))


def frames_with_unnamed_parts(row_index: LabeledDataRowIndex) -> tuple[int, ...]:
    """The rows whose frame has a part that names nothing as ``frame_part_as_text`` reads it: not text or an integer."""

    return tuple(
        row for row, frame in enumerate(row_index.frames) if any(frame_part_as_text(part) is None for part in frame)
    )


def frame_absorbed_into_index_names(row_index: LabeledDataRowIndex) -> tuple[str, ...]:
    """The frame the row index's level names spell, as pandas reads a CSV's first frame when it has no values.

    DeepLabCut writes no index names, so names that spell a frame of the model's three parts are one; empty
    otherwise. Split as a path when there is one name, as a flat CSV index stores it.
    """

    names = [name for name in row_index.names if isinstance(name, str)]
    if not names or len(names) != len(row_index.names):
        return ()
    parts = split_frame_path(names[0]) if len(names) == 1 else tuple(names)
    return parts if len(parts) == len(FRAME_COLUMNS) else ()


def frames_lost_to_csv(frame: pd.DataFrame) -> tuple[object, ...]:
    """The row-index entries of a wide ``frame`` that its CSV reads back without.

    pandas, and DeepLabCut's ``convertcsv2h5`` with it, read a first row without values as the row index's names.
    """

    if len(frame) and frame.iloc[0].isna().all():
        return (frame.index[0],)
    return ()
