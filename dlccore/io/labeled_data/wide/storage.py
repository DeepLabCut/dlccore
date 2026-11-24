"""Labels stored wide, as DeepLabCut writes them in H5 and CSV: the table, its classification, and the read."""

from __future__ import annotations

import os
import shutil
import uuid
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pandas as pd
from dlc_octopus.io.errors import FileChangedError
from dlc_octopus.io.file_stat import FileStat
from dlc_octopus.io.labeled_data.model import FramePath, Keypoint
from dlc_octopus.io.labeled_data.wide.header import LabeledDataHeader, LabeledDataHeaderShape
from dlc_octopus.io.labeled_data.wide.row_index import LabeledDataRowIndex, RowIndexShape, frame_part_as_text
from dlc_octopus.io.labeled_data.wide.table import LabeledDataRead, classify_labeled_data_read
from dlc_octopus.io.stamp import WriterStamp
from dlc_octopus.io.suffixes import (
    CSV_SUFFIX,
    H5_SUFFIX,
    LABELED_DATA_TABLE_SUFFIXES,
    suffix_in,
)

if TYPE_CHECKING:
    from dlc_octopus.io.labeled_data.long.model import LabeledData
    from dlc_octopus.io.labeled_data.wide.h5 import H5KeySelection


@dataclass(frozen=True, eq=False)
class WideLabeledDataStorage:
    """Labels stored wide: one row per frame; scorer, individual, bodypart and coordinate as column levels.

    Compared and hashed by identity, since a DataFrame has neither value equality nor a hash; compare content
    through ``keypoints``, ``frames`` or ``dataframe`` explicitly.

    Attributes:
        key_selection: For a container holding several tables, the keys found and the one read; ``None`` for a
            container with one table, such as CSV.
        header: The column header, classified without loss.
        row_index: The row index, classified without loss.
        dataframe: The table as the container returned it, unmodified.
        writer_stamp: Who wrote the file, when this package's writer did; ``None`` otherwise.
        writer_stamp_error: Why a stamp that is present could not be read; ``None`` when there is none or it read.
    """

    key_selection: H5KeySelection | None
    header: LabeledDataHeader
    row_index: LabeledDataRowIndex
    dataframe: pd.DataFrame
    writer_stamp: WriterStamp | None = None
    writer_stamp_error: str | None = None

    @property
    def keypoints(self) -> tuple[Keypoint, ...] | None:
        """The keypoints, in order of first appearance; ``None`` when the header is unrecognised."""

        if self.header.shape is LabeledDataHeaderShape.UNRECOGNISED:
            return None
        return tuple(columns.keypoint for columns in self.header.keypoints)

    @property
    def frames(self) -> tuple[FramePath, ...] | None:
        """The frame paths as text, in row order; ``None`` when the row index is unrecognised or a part has no name.

        Parts become text as ``frame_part_as_text`` gives it; ``row_index`` keeps every part as stored.
        """

        if self.row_index.shape is RowIndexShape.UNRECOGNISED:
            return None
        frames = tuple(tuple(frame_part_as_text(part) for part in frame) for frame in self.row_index.frames)
        if any(part is None for frame in frames for part in frame):
            return None
        return cast("tuple[FramePath, ...]", frames)


def wide_suffix(path: Path) -> str | None:
    """Return ``path``'s suffix lowercased if it equals ``.h5`` or ``.csv`` ignoring case; ``None`` otherwise."""

    return suffix_in(path, LABELED_DATA_TABLE_SUFFIXES)


def _read_h5(path: Path) -> LabeledDataRead:
    from dlc_octopus.io.labeled_data.wide.h5 import read_labeled_data_h5

    return read_labeled_data_h5(path)


def _read_csv(path: Path) -> LabeledDataRead:
    from dlc_octopus.io.labeled_data.wide.csv import read_labeled_data_csv

    return read_labeled_data_csv(path)


_READERS: dict[str, Callable[[Path], LabeledDataRead]] = {H5_SUFFIX: _read_h5, CSV_SUFFIX: _read_csv}


def read_wide_labeled_data(path: Path) -> WideLabeledDataStorage:
    """Read a wide labeled-data H5 or CSV file and classify its header and row index.

    Args:
        path: A ``CollectedData_*`` or ``machinelabels*`` file whose suffix equals ``.h5`` or ``.csv`` ignoring case.

    Raises:
        UnreadableFileError: If the file cannot be read.
        ValueError: If the suffix is neither ``.h5`` nor ``.csv``.
    """

    suffix = wide_suffix(path)

    if suffix is None:
        raise ValueError(f"{path}: no wide labeled-data reader for {path.suffix!r}; expected {sorted(_READERS)}.")

    reader = _READERS[suffix]

    read = reader(path)
    header, row_index = classify_labeled_data_read(read)

    return WideLabeledDataStorage(
        key_selection=read.key_selection,
        header=header,
        row_index=row_index,
        dataframe=read.dataframe,
        writer_stamp=read.writer_stamp,
        writer_stamp_error=read.writer_stamp_error,
    )


WRITER_NAME = "dlc_octopus.io.labeled_data"


def write_wide_labeled_data(labels: LabeledData, path: Path, *, replaces: FileStat | None) -> FileStat:
    """Write labels as an H5 file in DeepLabCut's current wide form, with its CSV twin beside it.

    Both files are written to temporary names in the same directory and replaced only once both are complete,
    so neither is ever left half-written. The H5 is stamped with this writer's name and package version. The
    CSV twin is replaced without a check of its own: the H5 is the file the check guards. A replaced file keeps
    its permission bits and a new one gets the umask's; the group is the writer's unless the directory sets it.

    Args:
        labels: What to write.
        path: The H5 file to create or replace. Its suffix must equal ``.h5`` ignoring case. The CSV twin is
            ``csv_twin(path)``.
        replaces: The stat of ``path`` when the caller read it, or ``None`` to create a file that must not exist.

    Returns:
        The stat of the written file, to pass as ``replaces`` to the next write.

    Raises:
        ValueError: If ``path``'s suffix is not ``.h5`` ignoring case, ``path`` is human labels
            (``CollectedData_*``) and the labels have a likelihood, a keypoint's individual is named ``""``, or
            ``path`` holds several keys, which a write of one key would drop.
        UnreadableFileError: If ``replaces`` is given and ``path`` cannot be opened as HDF5, so the keys it would
            drop are unknown.
        FileExistsError: If ``replaces`` is ``None`` and ``path`` exists.
        FileChangedError: If ``path`` changed or vanished since ``replaces`` was taken.

    Warns:
        UserWarning: Once both files are in place, for each frame the CSV twin reads back without.
    """

    from dlc_octopus.io.labeled_data.checks.file import frames_lost_to_csv, unread_h5_keys
    from dlc_octopus.io.labeled_data.checks.model import keypoints_with_empty_individual, likelihood_in_human_labels
    from dlc_octopus.io.labeled_data.wide.csv import write_labeled_data_csv
    from dlc_octopus.io.labeled_data.wide.h5 import select_h5_key, write_labeled_data_h5

    if suffix_in(path, frozenset({H5_SUFFIX})) is None:
        raise ValueError(f"{path}: the labeled-data writer writes an .h5 file and its .csv twin; got {path.suffix!r}.")
    if likelihood_in_human_labels(labels, path):
        # DeepLabCut's training-set creation assumes x and y only (DeepLabCut #3319).
        raise ValueError(f"{path}: human labels never carry a likelihood; drop it first.")
    if unnamed := keypoints_with_empty_individual(labels):
        raise ValueError(
            f"{path}: {unnamed[0]} has an empty individual name; DeepLabCut's header names an individual or has no "
            "individuals level."
        )
    # Any change to the keys since the caller's read fails the stat check before the replace.
    if replaces is not None and path.exists() and (dropped := unread_h5_keys(select_h5_key(path))):
        raise ValueError(f"{path}: holds keys {list(dropped)} besides the one read; a write would drop them.")

    frame = labels.to_wide_df()
    temporaries: list[Path] = []

    try:
        temporaries.append(temporary_h5 := _reserve_temporary(path, H5_SUFFIX))
        temporaries.append(temporary_csv := _reserve_temporary(path, CSV_SUFFIX))
        write_labeled_data_h5(frame, temporary_h5, WriterStamp.current(WRITER_NAME))
        write_labeled_data_csv(frame, temporary_csv)
        _check_unchanged(path, replaces)
        _keep_mode(path, temporary_h5)
        _keep_mode(csv_twin(path), temporary_csv)
        os.replace(temporary_h5, path)
        os.replace(temporary_csv, csv_twin(path))
    finally:
        for temporary in temporaries:
            temporary.unlink(missing_ok=True)

    for entry in frames_lost_to_csv(frame):
        warnings.warn(
            f"{csv_twin(path)}: frame {entry} has no values and comes first, so this CSV reads back without it; "
            f"{path.name} keeps it.",
            UserWarning,
            stacklevel=3,
        )

    return FileStat.of(path)


def csv_twin(path: Path) -> Path:
    """Return ``path`` with its suffix replaced by ``.CSV`` if ``path.suffix.isupper()``, and by ``.csv`` otherwise."""

    return path.with_suffix(CSV_SUFFIX.upper() if path.suffix.isupper() else CSV_SUFFIX)


def _reserve_temporary(path: Path, suffix: str) -> Path:
    """Create an empty file with a name no other call is given, beside ``path`` so a replace stays atomic.

    Created as any new file is, so the umask applies; ``mkstemp`` would make it, and the file it replaces,
    owner-only.
    """

    while True:
        temporary = path.parent / f".{path.stem}.{uuid.uuid4().hex[:8]}.tmp{suffix}"
        try:
            os.close(os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666))
        except FileExistsError:
            continue
        return temporary


def _keep_mode(original: Path, temporary: Path) -> None:
    """Give ``temporary`` the permission bits of the file it is about to replace, if there is one."""

    if original.exists():
        shutil.copymode(original, temporary)


def _check_unchanged(path: Path, replaces: FileStat | None) -> None:
    if replaces is None:
        if path.exists():
            raise FileExistsError(f"{path}: exists; pass the stat taken when it was read to replace it.")
        return
    try:
        current = FileStat.of(path)
    except FileNotFoundError as error:
        raise FileChangedError(f"{path}: was removed since it was read.") from error
    if current != replaces:
        raise FileChangedError(f"{path}: changed since it was read ({replaces} then, {current} now).")
