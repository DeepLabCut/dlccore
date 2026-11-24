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

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dlccore.io.labeled_data.long.model import LabeledData
from dlccore.io.labeled_data.wide.from_long import long_to_wide
from dlccore.io.labeled_data.wide.storage import read_wide_labeled_data
from dlccore.io.labeled_data.wide.to_long import wide_df_to_long, wide_to_long

FIXTURES_DIR = Path(__file__).parents[2] / "fixtures" / "labeled_data"
FIXTURES = sorted(path.name for path in FIXTURES_DIR.glob("*") if path.suffix in {".h5", ".csv"})
SINGLE_ANIMAL_LEVELS = ["scorer", "bodyparts", "coords"]
MULTI_ANIMAL_LEVELS = ["scorer", "individuals", "bodyparts", "coords"]
PARTS = pd.MultiIndex.from_tuples([("labeled-data", "s1", "img0.png"), ("labeled-data", "s1", "img1.png")])


def _labels(columns: list[tuple[str, ...]], rows: list[list[float]], index: object, names: list[str]) -> LabeledData:
    df = pd.DataFrame(rows, columns=pd.MultiIndex.from_tuples(columns, names=names), index=index)
    return wide_df_to_long(df)


def _single_animal_columns(coords: tuple[str, ...] = ("x", "y")) -> list[tuple[str, ...]]:
    return [("Sc", bodypart, coord) for bodypart in ("nose", "tail") for coord in coords]


def test_every_fixture_is_found() -> None:
    # Control: an empty glob would let the parametrized test below pass by running nothing.
    assert len(FIXTURES) == 10


@pytest.mark.parametrize("name", FIXTURES)
def test_every_fixture_round_trips_through_wide(name: str) -> None:
    storage = read_wide_labeled_data(FIXTURES_DIR / name)
    long = wide_to_long(storage.dataframe, storage.header, storage.row_index)

    wide = long_to_wide(long)
    pd.testing.assert_frame_equal(wide_df_to_long(wide).to_long_df(), long.to_long_df())


def test_the_wide_table_is_deeplabcuts_current_form() -> None:
    # Control for the tests below: the exact table DeepLabCut writes, built by hand.
    rows = [[1.0, 2.0, 3.0, np.nan], [5.0, 6.0, 7.0, 8.0]]
    expected = pd.DataFrame(
        rows, index=PARTS, columns=pd.MultiIndex.from_tuples(_single_animal_columns(), names=SINGLE_ANIMAL_LEVELS)
    )

    wide = long_to_wide(_labels(_single_animal_columns(), rows, PARTS, SINGLE_ANIMAL_LEVELS))

    pd.testing.assert_frame_equal(wide, expected)


def test_levels_are_plain_text_never_categorical() -> None:
    wide = long_to_wide(_labels(_single_animal_columns(), [[1.0, 2.0, 3.0, 4.0]] * 2, PARTS, SINGLE_ANIMAL_LEVELS))

    levels = [*wide.index.levels, *wide.columns.levels]

    assert not any(isinstance(level.dtype, pd.CategoricalDtype) for level in levels)
    assert all(isinstance(value, str) for level in levels for value in level)


def test_flat_paths_are_written_as_path_parts() -> None:
    flat = ["labeled-data/s1/img0.png", "labeled-data/s1/img1.png"]

    wide = long_to_wide(_labels(_single_animal_columns(), [[1.0, 2.0, 3.0, 4.0]] * 2, flat, SINGLE_ANIMAL_LEVELS))

    assert wide.index.equals(PARTS)


def test_a_reordered_header_is_written_one_keypoint_at_a_time_in_coordinate_order() -> None:
    columns = [("Sc", "nose", "y"), ("Sc", "tail", "x"), ("Sc", "nose", "x"), ("Sc", "tail", "y")]

    wide = long_to_wide(_labels(columns, [[2.0, 3.0, 1.0, 4.0]], PARTS[:1], SINGLE_ANIMAL_LEVELS))

    assert list(wide.columns) == _single_animal_columns()
    assert wide.to_numpy().tolist() == [[1.0, 2.0, 3.0, 4.0]]


def test_multi_animal_labels_with_likelihood_get_four_levels_and_three_coordinates() -> None:
    columns = [("Sc", "mouse1", "nose", coord) for coord in ("x", "y", "likelihood")]
    columns += [("Sc", "single", "box", coord) for coord in ("x", "y", "likelihood")]

    wide = long_to_wide(_labels(columns, [[1.0, 2.0, 0.5, 3.0, 4.0, 0.25]], PARTS[:1], MULTI_ANIMAL_LEVELS))

    assert list(wide.columns.names) == MULTI_ANIMAL_LEVELS
    assert list(wide.columns) == columns
    assert wide.to_numpy().tolist() == [[1.0, 2.0, 0.5, 3.0, 4.0, 0.25]]
