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

"""One labeled-data file as consumers use it, whatever layout stores it, and the read and write that move it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from dlccore.io.file_stat import FileStat
from dlccore.io.labeled_data.model import FramePath, Keypoint

if TYPE_CHECKING:
    import pandas as pd

    from dlccore.io.labeled_data.long.model import LabeledData
    from dlccore.io.labeled_data.wide.h5 import H5KeySelection
    from dlccore.io.labeled_data.wide.header import LabeledDataHeader
    from dlccore.io.labeled_data.wide.row_index import LabeledDataRowIndex
    from dlccore.io.labeled_data.wide.storage import WideLabeledDataStorage
    from dlccore.io.stamp import WriterStamp


@dataclass(frozen=True)
class LabeledDataFile:
    """One labeled-data file: its table as read, and its labels as the model where they can be interpreted.

    Its facts (``key_selection``, ``writer_stamp``, ``writer_stamp_error``, ``raw_header``, ``raw_row_index``) say
    what the stored file holds, uninterpreted; the checks take them. Nothing else reads the storage behind them.

    Attributes:
        path: The path as given; never resolved.
        stat: The file's stat, taken just before it was read; pass it as ``replaces`` to write the file back.
            ``None`` when it could not be taken, and the writer then refuses to replace the file.
    """

    path: Path
    _stored: WideLabeledDataStorage
    stat: FileStat | None

    @property
    def keypoints(self) -> tuple[Keypoint, ...] | None:
        """The keypoints, in order of first appearance; ``None`` when invalid."""

        return self._stored.keypoints

    @property
    def frames(self) -> tuple[FramePath, ...] | None:
        """The frame paths, in row order; ``None`` when invalid."""

        return self._stored.frames

    @property
    def key_selection(self) -> H5KeySelection | None:
        """The keys of an H5 file and the one read; ``None`` for a container with one table, such as CSV."""

        return self._stored.key_selection

    @property
    def writer_stamp(self) -> WriterStamp | None:
        """Provenance stamp; ``None`` if unavailable."""

        return self._stored.writer_stamp

    @property
    def writer_stamp_error(self) -> str | None:
        """Any provenance reading error; ``None`` if absent."""

        return self._stored.writer_stamp_error

    @property
    def raw_header(self) -> LabeledDataHeader:
        """The column header as stored, classified."""

        return self._stored.header

    @property
    def raw_row_index(self) -> LabeledDataRowIndex:
        """The row index as stored, classified with level names, entries and frame parts."""

        return self._stored.row_index

    @property
    def labeled_data(self) -> LabeledData:
        """Data in long format, built on each access.

        Raises:
            ValueError: Naming the file and why it has no model, e.g. a frame of other than three parts.
        """

        from dlccore.io.labeled_data.wide.to_long import wide_to_long

        try:
            return wide_to_long(self._stored.dataframe, self._stored.header, self._stored.row_index)
        except ValueError as error:
            raise ValueError(f"{self.path}: {error}") from error

    def to_raw_wide_df(self) -> pd.DataFrame:
        """A copy of the table as read.

        Raw, never normalised: legacy keys, a flat row index or a header in another order come back as found.
        For DeepLabCut's current form, use ``labeled_data.to_wide_df()``.
        """

        return self._stored.dataframe.copy()


def read_labeled_data(path: Path) -> LabeledDataFile:
    """Read one labeled-data file.

    Args:
        path: A ``CollectedData_*`` or ``machinelabels*`` file.

    Raises:
        UnreadableFileError: If the file cannot be read.
        ValueError: If no reader handles the suffix.
    """

    from dlccore.io.labeled_data.wide.storage import read_wide_labeled_data

    try:
        stat = FileStat.of(path)
    except OSError:
        stat = None
    stored = read_wide_labeled_data(path)  # raises for an unknown suffix or an unreadable file

    return LabeledDataFile(path, stored, stat)


def write_labeled_data(labels: LabeledData, path: Path, *, replaces: FileStat | None) -> FileStat:
    """Write labels in DeepLabCut's current form: ``path`` (``.h5``) and its ``.csv`` twin.

    Writes are atomic, and ``path`` is replaced only if it corresponds to the state snapshot described by ``replaces``.

    Args:
        labels: Data to write.
        path: H5 file to create or replace. Its suffix must equal ``.h5`` ignoring case. The CSV twin is
            written beside it, with suffix ``.CSV`` if ``path``'s suffix is upper case and ``.csv`` otherwise.
        replaces: ``LabeledDataFile.stat`` of the file being replaced, or ``None`` to create a new file (must not
            already exist).

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
        UserWarning: Once both files are in place, for each frame the CSV twin reads back without: a first frame
            without values, which pandas and DeepLabCut read as the row index's names. The H5 keeps it.
    """

    from dlccore.io.labeled_data.wide.storage import write_wide_labeled_data

    return write_wide_labeled_data(labels, path, replaces=replaces)
