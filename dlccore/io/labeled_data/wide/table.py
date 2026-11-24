"""The wide labeled-data table as any container returns it, and its classification.

Containers (``h5``, ``csv``) only open files; everything about the table itself is decided here and in the modules
this one calls, once for every container.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from dlc_octopus.io.labeled_data.wide.header import LabeledDataHeader, classify_labeled_data_header
from dlc_octopus.io.labeled_data.wide.row_index import LabeledDataRowIndex, classify_labeled_data_row_index

if TYPE_CHECKING:
    from dlc_octopus.io.labeled_data.wide.h5 import H5KeySelection
    from dlc_octopus.io.stamp import WriterStamp


@dataclass(frozen=True)
class LabeledDataRead:
    """The raw content of one labeled-data file, whatever container stored it.

    Attributes:
        path: The path as given; never resolved.
        dataframe: The table as the container returned it, unmodified.
        key_selection: For a container holding several tables, the keys found and the one read; ``None`` for a
            container with one table, such as CSV.
        writer_stamp: Who wrote the file, when this package's writer did; ``None`` otherwise, as for every file
            written by anything else and every CSV.
        writer_stamp_error: Why a stamp that is present could not be read, naming the file; the table is read
            regardless.
    """

    path: Path
    dataframe: pd.DataFrame
    key_selection: H5KeySelection | None
    writer_stamp: WriterStamp | None = None
    writer_stamp_error: str | None = None


def classify_labeled_data_read(read: LabeledDataRead) -> tuple[LabeledDataHeader, LabeledDataRowIndex]:
    """Classify the header and row index of a raw read, leaving the DataFrame untouched."""

    return classify_wide_frame(read.dataframe)


def classify_wide_frame(frame: pd.DataFrame) -> tuple[LabeledDataHeader, LabeledDataRowIndex]:
    """Classify the header and row index of a wide table, whether read from a file or built in memory."""

    columns = (column if isinstance(column, tuple) else (column,) for column in frame.columns)

    return (
        classify_labeled_data_header(frame.columns.names, columns),
        classify_labeled_data_row_index(
            frame.index.names, frame.index, is_multi_index=isinstance(frame.index, pd.MultiIndex)
        ),
    )
