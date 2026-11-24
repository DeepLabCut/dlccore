"""DeepLabCut's wide labeled-data column header, classified by keypoint without loss."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from dlc_octopus.io.labeled_data.model import Keypoint

SINGLE_ANIMAL_LEVELS = ("scorer", "bodyparts", "coords")
MULTI_ANIMAL_LEVELS = ("scorer", "individuals", "bodyparts", "coords")


class LabeledDataHeaderShape(str, Enum):
    """The stored column levels"""

    SINGLE_ANIMAL = "single_animal"
    MULTI_ANIMAL = "multi_animal"
    UNRECOGNISED = "unrecognised"


_HEADER_SHAPES: dict[tuple[str | None, ...], LabeledDataHeaderShape] = {
    SINGLE_ANIMAL_LEVELS: LabeledDataHeaderShape.SINGLE_ANIMAL,
    MULTI_ANIMAL_LEVELS: LabeledDataHeaderShape.MULTI_ANIMAL,
}


@dataclass(frozen=True)
class KeypointColumns:
    """Where one keypoint sits in a wide header.

    Attributes:
        keypoint:Individual is ``None`` if no individuals level, and its scope is
            ``UNIQUE`` exactly when the individual is ``"single"``.
        coords: The coordinate of each column, in column order, duplicates kept.
        positions: Each column's position in the header, aligned with ``coords``.
    """

    keypoint: Keypoint
    coords: tuple[str, ...]
    positions: tuple[int, ...]


@dataclass(frozen=True)
class LabeledDataHeader:
    """A labeled-data column header.

    Attributes:
        level_names: The level names.
        shape: Which header DeepLabCut writes these level names match.
        keypoints: In order of first appearance; empty when ``shape`` is ``UNRECOGNISED``.
        columns: The stored column tuples, kept only when ``shape`` is ``UNRECOGNISED``.
    """

    level_names: tuple[str | None, ...]
    shape: LabeledDataHeaderShape
    keypoints: tuple[KeypointColumns, ...] = ()
    columns: tuple[tuple[str, ...], ...] = ()


HEADER_DEPTHS: tuple[int, ...] = tuple(sorted({len(level_names) for level_names in _HEADER_SHAPES}, reverse=True))


def header_shape(level_names: Iterable[str | None]) -> LabeledDataHeaderShape:
    """Return which header DeepLabCut writes these level names are, or ``UNRECOGNISED``."""

    return _HEADER_SHAPES.get(tuple(level_names), LabeledDataHeaderShape.UNRECOGNISED)


def classify_labeled_data_header(
    level_names: Iterable[str | None],
    columns: Iterable[tuple[str, ...]],
) -> LabeledDataHeader:
    """Classify a column header by keypoint, read by name, never as a grid of distinct level values.

    Columns are grouped by (scorer, individual, bodypart).
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
