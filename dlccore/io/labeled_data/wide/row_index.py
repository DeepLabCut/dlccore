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

"""DeepLabCut's labeled-data row index: each row's frame path, split into path components as stored."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from numbers import Integral


class RowIndexShape(str, Enum):
    """How a row index stores its frame paths.

    ``PATH_PARTS``: a MultiIndex, one level per path component. ``FLAT_PATHS``: one path string per row.
    ``EMPTY``: no rows, whatever the index type. ``UNRECOGNISED``: anything else.
    """

    PATH_PARTS = "path_parts"
    FLAT_PATHS = "flat_paths"
    EMPTY = "empty"
    UNRECOGNISED = "unrecognised"


@dataclass(frozen=True)
class LabeledDataRowIndex:
    """A labeled-data row index, classified.

    Attributes:
        shape: How the frame paths are stored.
        names: The level names as stored; ``None`` for each unnamed level.
        entries: The entries as stored.
        frames: Each entry's path components as stored. Entries of different depths keep their own length.
            Empty when ``shape`` is ``EMPTY`` or ``UNRECOGNISED``.
    """

    shape: RowIndexShape
    names: tuple[object, ...]
    entries: tuple[object, ...]
    frames: tuple[tuple[object, ...], ...] = ()


# Frame identity rather than wide storage: moves to ``model`` once a second format needs it.
def split_frame_path(entry: str) -> tuple[str, ...]:
    """Split a labeled-data row-index entry into its path components.

    Does not validate: empty and repeated separators give empty components instead of being dropped.

    Args:
        entry: A row-index entry, e.g. ``labeled-data/vid1/img1.png``. Forward slashes and backslashes both separate.
    """

    return tuple(entry.replace("\\", "/").split("/"))


def frame_part_as_text(part: object) -> str | None:
    """Return a row-index part as the folder or file name it stands for.

    A string is returned unchanged. An integer is converted with ``str()``: when a video folder's name is a
    number (e.g. ``labeled-data/123/``), pandas may read that level back as an integer, and DeepLabCut's
    ``guarantee_multiindex_rows`` converts it to a string the same way.

    Returns:
        The name, or ``None`` for any other type. A float or bool cannot be turned back into the original
        name (``1.0`` vs ``"1"``), so no guessing is involved.
    """

    if isinstance(part, str):
        return part
    if isinstance(part, Integral) and not isinstance(part, bool):
        return str(part)
    return None


def classify_labeled_data_row_index(
    names: Iterable[object],
    entries: Iterable[object],
    *,
    is_multi_index: bool,
) -> LabeledDataRowIndex:
    """Classify a row index by how it stores frame paths. Path strings are split but not normalised.

    Args:
        names: The row index's level names.
        entries: The row-index entries, in order; tuples for a MultiIndex.
        is_multi_index: Whether the entries came from a MultiIndex.
    """

    names = tuple(names)
    entries = tuple(entries)

    if not entries:
        return LabeledDataRowIndex(shape=RowIndexShape.EMPTY, names=names, entries=entries)

    if is_multi_index:
        return LabeledDataRowIndex(
            shape=RowIndexShape.PATH_PARTS,
            names=names,
            entries=entries,
            frames=tuple(tuple(entry) for entry in entries),
        )

    if all(isinstance(entry, str) for entry in entries):
        return LabeledDataRowIndex(
            shape=RowIndexShape.FLAT_PATHS,
            names=names,
            entries=entries,
            frames=tuple(split_frame_path(entry) for entry in entries),
        )

    return LabeledDataRowIndex(shape=RowIndexShape.UNRECOGNISED, names=names, entries=entries)
