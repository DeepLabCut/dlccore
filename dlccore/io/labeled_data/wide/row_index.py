"""Frame paths of labeled data: DeepLabCut's row index, split into path components as stored."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from numbers import Integral


class RowIndexShape(str, Enum):
    """The stored row index: path components, path strings, no entries (whatever the index type), or unrecognised."""

    PATH_PARTS = "path_parts"
    FLAT_PATHS = "flat_paths"
    EMPTY = "empty"
    UNRECOGNISED = "unrecognised"


@dataclass(frozen=True)
class LabeledDataRowIndex:
    """A labeled-data row index.

    Attributes:
        shape: How the frame paths are stored.
        names: The stored level names, unmodified; ``None`` for each unnamed level.
        entries: The stored entries, unmodified.
        frames: The path components of each entry, as stored; ragged depths stay ragged. Empty when ``shape``
            is ``EMPTY`` or ``UNRECOGNISED``.
    """

    shape: RowIndexShape
    names: tuple[object, ...]
    entries: tuple[object, ...]
    frames: tuple[tuple[object, ...], ...] = ()


# Frame identity rather than wide storage: moves to ``model`` once a second format needs it.
def split_frame_path(entry: str) -> tuple[str, ...]:
    """Split a labeled-data row-index entry into its path components.

    Splits; does not validate. Empty and repeated separators yield empty components rather than being dropped.

    Args:
        entry: A row-index entry, e.g. ``labeled-data/vid1/img1.png``. Either path separator is accepted.
    """

    return tuple(entry.replace("\\", "/").split("/"))


def frame_part_as_text(part: object) -> str | None:
    """Return a frame part as the name it stands for, or ``None`` when no name can be given without inventing one.

    Text is kept. An integer becomes its digits: pandas infers one for a folder named by a number, and DeepLabCut's
    ``guarantee_multiindex_rows`` casts it back the same way. A float or bool would name a folder that never existed.
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
    """Classify a row index by how its frame paths are stored, splitting path strings without normalising them.

    Args:
        names: The row index's level names
        entries: The row-index entries, in order; tuples for a MultiIndex
        is_multi_index: Whether the entries came from a MultiIndex
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
