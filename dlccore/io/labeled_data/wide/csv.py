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

"""Labeled-data CSV files: which rows hold the header and which columns the row index, and the read itself.

Returns the shared ``table.LabeledDataRead``; classification is the same for every container. Imported explicitly,
so pandas loads only when it is used.
"""

from __future__ import annotations

import csv
from itertools import islice
from pathlib import Path

import pandas as pd

from dlccore.io.errors import UnreadableFileError
from dlccore.io.labeled_data.wide.header import HEADER_DEPTHS, LabeledDataHeaderShape, header_shape
from dlccore.io.labeled_data.wide.table import LabeledDataRead


def _header_rows_and_index_columns(rows: list[list[str]]) -> tuple[int, int]:
    """Return how many rows hold the header and how many columns hold the row index.

    Each header row starts with its level name, followed by one empty cell per further index column. The deepest
    recognised header wins; with none, both are ``0`` and the file is read without a header.
    """

    level_names = [row[0] if row else "" for row in rows]

    for depth in HEADER_DEPTHS:
        if header_shape(level_names[:depth]) is not LabeledDataHeaderShape.UNRECOGNISED:
            after_name = rows[0][1:]
            empty = next((position for position, cell in enumerate(after_name) if cell), len(after_name))
            return depth, 1 + empty

    return 0, 0


def _path_parts_as_text(path: Path, df: pd.DataFrame, header_rows: int) -> pd.MultiIndex:
    """Return a path-parts row index as stored."""

    names = df.index.names
    parts = pd.read_csv(
        path,
        header=None,
        skiprows=header_rows + any(name is not None for name in names),
        usecols=range(len(names)),
        dtype=str,
    )
    return pd.MultiIndex.from_frame(parts, names=names)


def read_labeled_data_csv(path: Path) -> LabeledDataRead:
    """Read a labeled-data CSV as pandas parses it, with the layout detected from the file.

    Values are parsed as ``pandas.read_csv`` parses them, as DeepLabCut's ``convertcsv2h5`` does. Path parts stay
    text, as ``convertcsv2h5`` restores them: a folder named ``2019`` or ``007`` is not a number.

    Args:
        path: A ``CollectedData_*.csv`` or ``machinelabels*.csv`` file.

    Raises:
        UnreadableFileError: If the file cannot be opened or parsed as CSV.
    """

    try:
        with path.open(newline="", encoding="utf-8") as stream:
            header_rows, index_columns = _header_rows_and_index_columns(
                list(islice(csv.reader(stream), max(HEADER_DEPTHS)))
            )

        df = pd.read_csv(path, header=list(range(header_rows)) or None, index_col=list(range(index_columns)) or None)
        if index_columns > 1:
            df.index = _path_parts_as_text(path, df, header_rows)
    except (OSError, ValueError, csv.Error) as error:
        raise UnreadableFileError(f"{path}: cannot be read as CSV: {error}") from error

    return LabeledDataRead(path=path, dataframe=df, key_selection=None)


def write_labeled_data_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a wide table as DeepLabCut's human-readable CSV; the H5 stays authoritative.

    Values are written as pandas formats them, so a float can differ in its last digits after a round trip, and
    the frames ``checks.file.frames_lost_to_csv`` returns are written but read back as something else.
    """

    df.to_csv(path)
