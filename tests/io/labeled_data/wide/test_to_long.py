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
from dlccore.io.labeled_data.wide.storage import read_wide_labeled_data
from dlccore.io.labeled_data.wide.to_long import wide_df_to_long, wide_to_long

FIXTURES_DIR = Path(__file__).parents[2] / "fixtures" / "labeled_data"
FIXTURES = sorted(path.name for path in FIXTURES_DIR.glob("*") if path.suffix in {".h5", ".csv"})
SINGLE_ANIMAL_LEVELS = ["scorer", "bodyparts", "coords"]
MULTI_ANIMAL_LEVELS = ["scorer", "individuals", "bodyparts", "coords"]


def _long(columns: list[tuple[str, ...]], rows: list[list[object]], index: object, names: list[str]) -> LabeledData:
    # Built in memory: what is under test is the conversion, not a container.
    df = pd.DataFrame(rows, columns=pd.MultiIndex.from_tuples(columns, names=names), index=index)
    return wide_df_to_long(df)


def _single_animal(rows: list[list[object]], index: object, coords: tuple[str, ...] = ("x", "y")) -> LabeledData:
    columns = [("Sc", bodypart, coord) for bodypart in ("nose", "tail") for coord in coords]
    return _long(columns, rows, index, SINGLE_ANIMAL_LEVELS)


FLAT = ["labeled-data/s1/img0.png", "labeled-data/s1/img1.png"]


def test_every_fixture_is_found() -> None:
    # Control: an empty glob would let the parametrized test below pass by running nothing.
    assert len(FIXTURES) == 10


@pytest.mark.parametrize("name", FIXTURES)
def test_every_fixture_has_a_model_with_one_row_per_frame_and_keypoint(name: str) -> None:
    storage = read_wide_labeled_data(FIXTURES_DIR / name)

    long = wide_to_long(storage.dataframe, storage.header, storage.row_index).to_long_df()

    assert storage.keypoints is not None
    assert len(long) == len(storage.dataframe) * len(storage.keypoints)


def test_values_are_taken_frame_major_and_as_stored() -> None:
    long = _single_animal([[1.0, 2.0, 3.0, np.nan], [5.0, 6.0, 7.0, 8.0]], FLAT).to_long_df()

    assert long[["image", "bodypart"]].astype(str).values.tolist() == [
        ["img0.png", "nose"],
        ["img0.png", "tail"],
        ["img1.png", "nose"],
        ["img1.png", "tail"],
    ]
    np.testing.assert_array_equal(long[["x", "y"]].to_numpy(), [[1.0, 2.0], [3.0, np.nan], [5.0, 6.0], [7.0, 8.0]])


def test_a_reordered_header_gives_the_same_model() -> None:
    # Columns are taken by classified position, never by level values.
    ordered = _single_animal([[1.0, 2.0, 3.0, 4.0]], FLAT[:1]).to_long_df()
    columns = [("Sc", "nose", "x"), ("Sc", "tail", "y"), ("Sc", "nose", "y"), ("Sc", "tail", "x")]

    reordered = _long(columns, [[1.0, 4.0, 2.0, 3.0]], FLAT[:1], SINGLE_ANIMAL_LEVELS).to_long_df()

    pd.testing.assert_frame_equal(reordered, ordered)


def test_multi_animal_keeps_the_individual_and_unique_bodyparts_under_single() -> None:
    columns = [("Sc", "mouse1", "nose", "x"), ("Sc", "mouse1", "nose", "y"), ("Sc", "single", "box", "x")]
    columns.append(("Sc", "single", "box", "y"))

    labels = _long(columns, [[1.0, 2.0, 3.0, 4.0]], FLAT[:1], MULTI_ANIMAL_LEVELS)

    assert labels.is_multi_animal
    assert labels.to_long_df()["individual"].astype(str).tolist() == ["mouse1", "single"]


def test_a_likelihood_is_kept_when_every_keypoint_stores_one() -> None:
    labels = _single_animal([[1.0, 2.0, 0.5, 3.0, 4.0, 0.25]], FLAT[:1], coords=("x", "y", "likelihood"))

    assert labels.has_likelihood
    assert labels.to_long_df()["likelihood"].tolist() == [0.5, 0.25]


def test_path_parts_and_flat_paths_give_the_same_frames() -> None:
    parts = pd.MultiIndex.from_tuples([tuple(path.split("/")) for path in FLAT])
    rows = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]

    pd.testing.assert_frame_equal(_single_animal(rows, parts).to_long_df(), _single_animal(rows, FLAT).to_long_df())


def test_an_integer_frame_part_becomes_text() -> None:
    # A dataset folder named by a number; DeepLabCut's guarantee_multiindex_rows casts it the same way.
    index = pd.MultiIndex.from_tuples([("labeled-data", 2019, "img0.png")])

    long = _single_animal([[1.0, 2.0, 3.0, 4.0]], index).to_long_df()

    assert long["dataset"].astype(str).tolist() == ["2019", "2019"]


def test_storage_frames_and_the_long_model_agree_on_an_integer_frame_part(tmp_path: Path) -> None:
    # Both read frame parts through frame_part_as_text: one policy, so the two can never disagree on a frame.
    index = pd.MultiIndex.from_tuples([("labeled-data", 2019, "img0.png")])
    df = pd.DataFrame(
        [[1.0, 2.0]],
        index=index,
        columns=pd.MultiIndex.from_tuples([("Sc", "nose", "x"), ("Sc", "nose", "y")], names=SINGLE_ANIMAL_LEVELS),
    )
    df.to_hdf(tmp_path / "CollectedData_Sc.h5", key="df_with_missing", mode="w")
    storage = read_wide_labeled_data(tmp_path / "CollectedData_Sc.h5")

    long = wide_to_long(storage.dataframe, storage.header, storage.row_index).to_long_df()

    assert storage.frames == (tuple(long.loc[0, ["folder", "dataset", "image"]].astype(str)),)


@pytest.mark.parametrize(
    ("index", "message"),
    [
        (pd.MultiIndex.from_tuples([("labeled-data", "s1", "extra", "img0.png")]), "4 parts"),
        (pd.MultiIndex.from_tuples([("labeled-data", 2019.5, "img0.png")]), "neither text nor an integer"),
        (pd.MultiIndex.from_tuples([("labeled-data", True, "img0.png")]), "neither text nor an integer"),
        (pd.Index([0]), "entry 0 is not text"),
        (pd.Index([], dtype=object), "no frames"),
    ],
    ids=["four_parts", "float_part", "bool_part", "integer_index", "empty"],
)
def test_a_row_index_without_three_text_parts_has_no_model(index: object, message: str) -> None:
    rows = [[1.0, 2.0, 3.0, 4.0]] * len(index)

    with pytest.raises(ValueError, match=message):
        _single_animal(rows, index)


@pytest.mark.parametrize(
    ("coords", "rows", "message"),
    [
        (("x", "y", "z"), [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], "coordinates"),
        (("x", "y"), [["a", 2.0, 3.0, 4.0]], "not numeric"),
    ],
    ids=["triangulated_xyz", "text_value"],
)
def test_values_that_are_not_x_y_likelihood_numbers_have_no_model(
    coords: tuple[str, ...], rows: list[list[object]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _single_animal(rows, FLAT[:1], coords=coords)


def test_a_likelihood_on_some_keypoints_only_has_no_model() -> None:
    columns = [("Sc", "nose", "x"), ("Sc", "nose", "y"), ("Sc", "nose", "likelihood"), ("Sc", "tail", "x")]
    columns.append(("Sc", "tail", "y"))

    with pytest.raises(ValueError, match="likelihood") as raised:
        _long(columns, [[1.0, 2.0, 0.5, 3.0, 4.0]], FLAT[:1], SINGLE_ANIMAL_LEVELS)

    assert "'nose'" in str(raised.value)
    assert "'tail'" in str(raised.value)


def test_an_unrecognised_header_has_no_model() -> None:
    with pytest.raises(ValueError, match="Header levels"):
        _long([("Sc", "nose", "x")], [[1.0]], FLAT[:1], ["scorer", "bodypart", "coord"])
