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

"""DeepLabCut's wide labeled-data column header, grouped into keypoints without dropping any column."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from dlccore.io.labeled_data.model import Keypoint

SINGLE_ANIMAL_LEVELS = ("scorer", "bodyparts", "coords")
MULTI_ANIMAL_LEVELS = ("scorer", "individuals", "bodyparts", "coords")


class LabeledDataHeaderShape(str, Enum):
    """Column layouts used by a header, if recognised."""

    SINGLE_ANIMAL = "single_animal"
    MULTI_ANIMAL = "multi_animal"
    UNRECOGNISED = "unrecognised"


_HEADER_SHAPES: dict[tuple[str | None, ...], LabeledDataHeaderShape] = {
    SINGLE_ANIMAL_LEVELS: LabeledDataHeaderShape.SINGLE_ANIMAL,
    MULTI_ANIMAL_LEVELS: LabeledDataHeaderShape.MULTI_ANIMAL,
}


@dataclass(frozen=True)
class KeypointColumns:
    """The columns of one keypoint in a wide header.

    Attributes:
        keypoint: The keypoint. The individual is ``None`` in a single-animal header, and the scope is
            ``UNIQUE`` only when the individual is ``"single"``.
        coords: The coordinate name of each column (e.g. ``"x"``), in column order, duplicates kept.
        positions: Each column's index in the header, aligned with ``coords``.
    """

    keypoint: Keypoint
    coords: tuple[str, ...]
    positions: tuple[int, ...]


@dataclass(frozen=True)
class LabeledDataHeader:
    """A wide labeled-data column header, classified.

    Attributes:
        level_names: The header's level names, as stored.
        shape: The DeepLabCut layout these level names match, or ``UNRECOGNISED``.
        keypoints: The keypoints, in order of first appearance; empty when ``shape`` is ``UNRECOGNISED``.
        columns: The raw column tuples, kept only when ``shape`` is ``UNRECOGNISED`` so that nothing is lost.
    """

    level_names: tuple[str | None, ...]
    shape: LabeledDataHeaderShape
    keypoints: tuple[KeypointColumns, ...] = ()
    columns: tuple[tuple[str, ...], ...] = ()


HEADER_DEPTHS: tuple[int, ...] = tuple(sorted({len(level_names) for level_names in _HEADER_SHAPES}, reverse=True))


def header_shape(level_names: Iterable[str | None]) -> LabeledDataHeaderShape:
    """Return the DeepLabCut layout these level names match, or ``UNRECOGNISED``."""

    return _HEADER_SHAPES.get(tuple(level_names), LabeledDataHeaderShape.UNRECOGNISED)


def classify_labeled_data_header(
    level_names: Iterable[str | None],
    columns: Iterable[tuple[str, ...]],
) -> LabeledDataHeader:
    """Group a wide header's columns into keypoints.
    Each column is read on its own and grouped by (scorer, individual, bodypart). The header is never expanded
    from its per-level values (a MultiIndex's ``levels``), which would add keypoints that have no column.

    Reordered headers give the same keypoints at other positions, 
    any missing pair is omitted rather than padded, and duplicated columns are kept.

    Args:
        level_names: The header's level names, outermost first.
        columns: One tuple per column, aligned with ``level_names``.
    """

    level_names = tuple(level_names)
    columns = tuple(tuple(column) for column in columns)
    shape = header_shape(level_names)

    if shape is LabeledDataHeaderShape.UNRECOGNISED:
        return LabeledDataHeader(level_names=level_names, shape=shape, columns=columns)

    groups: dict[tuple[str, str | None, str], list[tuple[str, int]]] = {}

    for position, column in enumerate(columns):
        if shape is LabeledDataHeaderShape.SINGLE_ANIMAL:
            scorer, bodypart, coord = column
            individual = None
        else:
            scorer, individual, bodypart, coord = column

        groups.setdefault((scorer, individual, bodypart), []).append((coord, position))

    keypoints = tuple(
        KeypointColumns(
            keypoint=Keypoint(scorer=scorer, individual=individual, bodypart=bodypart),
            coords=tuple(coord for coord, _position in members),
            positions=tuple(position for _coord, position in members),
        )
        for (scorer, individual, bodypart), members in groups.items()
    )

    return LabeledDataHeader(level_names=level_names, shape=shape, keypoints=keypoints)
