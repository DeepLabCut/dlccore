from dataclasses import replace
from pathlib import Path

import pytest

from dlc_octopus.io.labeled_data.model import BodypartScope
from dlc_octopus.io.labeled_data.wide.h5 import read_labeled_data_h5
from dlc_octopus.io.labeled_data.wide.header import (
    LabeledDataHeader,
    LabeledDataHeaderShape,
    classify_labeled_data_header,
)
from dlc_octopus.io.labeled_data.wide.table import classify_labeled_data_read

FIXTURES_DIR = Path(__file__).parents[2] / "fixtures" / "labeled_data"
FIXTURES = sorted(path.name for path in FIXTURES_DIR.glob("*.h5"))
MULTI_ANIMAL = ("scorer", "individuals", "bodyparts", "coords")


def _rebuilt_columns(header: LabeledDataHeader) -> list[tuple[str, ...]]:
    columns: dict[int, tuple[str, ...]] = {}

    for placed in header.keypoints:
        keypoint = placed.keypoint

        for coord, position in zip(placed.coords, placed.positions, strict=True):
            if header.shape is LabeledDataHeaderShape.SINGLE_ANIMAL:
                columns[position] = (keypoint.scorer, keypoint.bodypart, coord)
            else:
                columns[position] = (keypoint.scorer, keypoint.individual, keypoint.bodypart, coord)

    return [columns[position] for position in sorted(columns)]


def _multi_animal(columns: list[tuple[str, str, str, str]]) -> LabeledDataHeader:
    return classify_labeled_data_header(MULTI_ANIMAL, columns)


@pytest.mark.parametrize("filename", FIXTURES)
def test_every_fixture_header_is_rebuilt_from_its_keypoints(filename: str) -> None:
    frame = read_labeled_data_h5(FIXTURES_DIR / filename).dataframe

    header, _row_index = classify_labeled_data_read(read_labeled_data_h5(FIXTURES_DIR / filename))

    assert header.shape is not LabeledDataHeaderShape.UNRECOGNISED
    assert _rebuilt_columns(header) == list(frame.columns)


def test_the_rebuild_sees_a_tampered_keypoint() -> None:
    # Control: a rebuild that ignored the keypoints would pass every fixture above.
    header, _row_index = classify_labeled_data_read(read_labeled_data_h5(FIXTURES_DIR / "single_animal_human.h5"))
    first = header.keypoints[0]
    moved = replace(first, keypoint=replace(first.keypoint, bodypart="elsewhere"))
    tampered = replace(header, keypoints=(moved, *header.keypoints[1:]))

    assert _rebuilt_columns(tampered) != _rebuilt_columns(header)


def test_unique_bodyparts_are_unique_and_never_an_individual() -> None:
    header, _row_index = classify_labeled_data_read(
        read_labeled_data_h5(FIXTURES_DIR / "multi_animal_unique_bodyparts.h5")
    )

    keypoints = [placed.keypoint for placed in header.keypoints]

    assert [(keypoint.individual, keypoint.bodypart, keypoint.scope) for keypoint in keypoints] == [
        ("animalA", "snout", BodypartScope.INDIVIDUAL),
        ("animalA", "tail", BodypartScope.INDIVIDUAL),
        ("animalB", "snout", BodypartScope.INDIVIDUAL),
        ("animalB", "tail", BodypartScope.INDIVIDUAL),
        ("single", "feeder", BodypartScope.UNIQUE),
    ]


def test_a_name_containing_single_is_an_individual() -> None:
    header = _multi_animal([("Sc", "single_mouse", "nose", "x"), ("Sc", "single_mouse", "nose", "y")])

    assert header.keypoints[0].keypoint.scope is BodypartScope.INDIVIDUAL


def test_a_single_animal_header_has_individual_keypoints_without_a_name() -> None:
    header, _row_index = classify_labeled_data_read(read_labeled_data_h5(FIXTURES_DIR / "single_animal_human.h5"))

    assert header.shape is LabeledDataHeaderShape.SINGLE_ANIMAL
    assert {(placed.keypoint.individual, placed.keypoint.scope) for placed in header.keypoints} == {
        (None, BodypartScope.INDIVIDUAL)
    }


def test_reordered_columns_give_the_same_keypoints_at_other_positions() -> None:
    columns = [("Sc", "a", "nose", "x"), ("Sc", "a", "nose", "y"), ("Sc", "b", "nose", "x"), ("Sc", "b", "nose", "y")]
    reordered = [columns[2], columns[0], columns[3], columns[1]]

    in_order, shuffled = _multi_animal(columns), _multi_animal(reordered)

    assert {(k.keypoint, k.coords) for k in in_order.keypoints} == {(k.keypoint, k.coords) for k in shuffled.keypoints}
    assert [k.positions for k in in_order.keypoints] != [k.positions for k in shuffled.keypoints]


def test_a_missing_pair_is_absent_not_padded() -> None:
    header = _multi_animal([("Sc", "a", "nose", "x"), ("Sc", "a", "tail", "x"), ("Sc", "b", "nose", "x")])

    assert [(placed.keypoint.individual, placed.keypoint.bodypart) for placed in header.keypoints] == [
        ("a", "nose"),
        ("a", "tail"),
        ("b", "nose"),
    ]


def test_a_duplicated_column_is_kept() -> None:
    header = _multi_animal([("Sc", "a", "nose", "x"), ("Sc", "a", "nose", "x")])

    (keypoint,) = header.keypoints

    assert (keypoint.coords, keypoint.positions) == (("x", "x"), (0, 1))


def test_coordinates_are_classified_whatever_they_are() -> None:
    header = _multi_animal([("Sc", "a", "nose", "z")])

    assert header.keypoints[0].coords == ("z",)


def test_unrecognised_level_names_keep_the_columns_and_raise_nothing() -> None:
    columns = [("Sc", "nose", "x", "extra")]

    header = classify_labeled_data_header(("scorer", "bodyparts", "coords", "extra"), columns)

    assert (header.shape, header.keypoints, header.columns) == (
        LabeledDataHeaderShape.UNRECOGNISED,
        (),
        (("Sc", "nose", "x", "extra"),),
    )
