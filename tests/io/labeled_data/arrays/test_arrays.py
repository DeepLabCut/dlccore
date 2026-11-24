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

from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dlccore.io.labeled_data import BodypartScope, Keypoint, LabeledData, LabeledDataArrays, read_labeled_data

FIXTURES_DIR = Path(__file__).parents[2] / "fixtures" / "labeled_data"
FIXTURES = sorted(path.name for path in FIXTURES_DIR.glob("*") if path.suffix in {".h5", ".csv"})
FRAMES = (("labeled-data", "s1", "img0.png"), ("labeled-data", "s1", "img1.png"))
KEYPOINTS = tuple(Keypoint("Sc", None, bodypart) for bodypart in ("nose", "tail", "ear"))


def _arrays(**changes: object) -> LabeledDataArrays:
    arrays = LabeledDataArrays(FRAMES, KEYPOINTS, np.zeros((2, 3, 2)), np.zeros((2, 3)))
    return replace(arrays, **changes)


def test_arrays_matching_their_frames_and_keypoints_are_built() -> None:
    # Control for the refusals below.
    assert _arrays(likelihood=None).xy.shape == (2, 3, 2)


@pytest.mark.parametrize(
    "xy",
    [np.zeros((3, 3, 2)), np.zeros((2, 2, 2)), np.zeros((2, 3, 3)), np.zeros((2, 3))],
    ids=["frames", "keypoints", "coordinates", "rank"],
)
def test_xy_must_have_one_pair_per_frame_and_keypoint(xy: np.ndarray) -> None:
    with pytest.raises(ValueError, match="xy"):
        _arrays(xy=xy)


@pytest.mark.parametrize("likelihood", [np.zeros((3, 3)), np.zeros((2, 2)), np.zeros((2, 3, 1))])
def test_likelihood_must_have_one_value_per_frame_and_keypoint(likelihood: np.ndarray) -> None:
    with pytest.raises(ValueError, match="likelihood"):
        _arrays(likelihood=likelihood)


def test_every_fixture_is_found() -> None:
    # Control: an empty glob would let the parametrized tests below pass by running nothing.
    assert len(FIXTURES) == 10


@pytest.mark.parametrize("filename", FIXTURES)
def test_the_arrays_equal_the_long_values_on_every_fixture(filename: str) -> None:
    labels = read_labeled_data(FIXTURES_DIR / filename)
    long = labels.labeled_data.to_long_df()

    arrays = labels.labeled_data.to_arrays()

    assert (arrays.frames, arrays.keypoints) == (labels.frames, labels.keypoints)
    at = (long["frame_index"].to_numpy(), long["keypoint_index"].to_numpy())
    np.testing.assert_array_equal(arrays.xy[at], long[["x", "y"]].to_numpy())
    if "likelihood" in long.columns:
        assert arrays.likelihood is not None
        np.testing.assert_array_equal(arrays.likelihood[at], long["likelihood"].to_numpy())
    else:
        assert arrays.likelihood is None
    # Control: NaN compares equal above, so an all-NaN view would pass without some labeled value.
    assert arrays.labeled.any()


@pytest.mark.parametrize("filename", FIXTURES)
def test_every_fixture_yields_every_cell_in_its_table(filename: str) -> None:
    labels = read_labeled_data(FIXTURES_DIR / filename)
    xs = labels.to_raw_wide_df().xs("x", level="coords", axis=1)

    arrays = labels.labeled_data.to_arrays()

    assert arrays.xy.shape == (len(arrays.frames), len(arrays.keypoints), 2)
    assert arrays.labeled.shape == arrays.xy.shape[:2]
    assert int(arrays.labeled.sum()) == int(xs.notna().to_numpy().sum())


def test_a_missing_pair_is_not_labeled_and_a_present_one_is_its_stored_value() -> None:
    labels = read_labeled_data(FIXTURES_DIR / "single_animal_machine.h5")
    frame_0, frame_3 = labels.to_raw_wide_df().iloc[0], labels.to_raw_wide_df().iloc[3]
    arrays = labels.labeled_data.to_arrays()
    nose, tail = (arrays.keypoints.index(k) for k in arrays.keypoints)

    assert not arrays.labeled[3, nose]
    assert np.isnan(arrays.xy[3, nose]).all()
    assert arrays.likelihood is not None
    assert arrays.likelihood[3, nose] == frame_3[("Sc", "nose", "likelihood")]
    assert arrays.labeled[0, tail]
    assert tuple(arrays.xy[0, tail]) == tuple(frame_0[("Sc", "tailbase", c)] for c in ("x", "y"))


def test_human_labels_have_no_likelihood() -> None:
    arrays = read_labeled_data(FIXTURES_DIR / "single_animal_human.h5").labeled_data.to_arrays()

    assert arrays.likelihood is None


def test_unique_bodyparts_keep_their_scope() -> None:
    arrays = read_labeled_data(FIXTURES_DIR / "multi_animal_unique_bodyparts.h5").labeled_data.to_arrays()

    scopes = {keypoint.individual: keypoint.scope for keypoint in arrays.keypoints}

    assert scopes.pop("single") is BodypartScope.UNIQUE
    assert set(scopes.values()) == {BodypartScope.INDIVIDUAL}


def test_the_arrays_are_read_only() -> None:
    arrays = read_labeled_data(FIXTURES_DIR / "single_animal_machine.h5").labeled_data.to_arrays()
    assert arrays.likelihood is not None

    with pytest.raises(ValueError, match="read-only"):
        arrays.xy[0, 0, 0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        arrays.likelihood[0, 0] = 0.0


def _with_coords(coords: list[str], values: list[object]) -> LabeledData:
    columns = pd.MultiIndex.from_tuples([("Sc", "nose", c) for c in coords], names=["scorer", "bodyparts", "coords"])
    return LabeledData.from_wide_df(pd.DataFrame([values], columns=columns, index=["labeled-data/v1/img0.png"]))


def test_plain_x_and_y_are_interpreted() -> None:
    arrays = _with_coords(["x", "y"], [1.0, 2.0]).to_arrays()

    assert (arrays.xy.tolist(), arrays.labeled.tolist()) == ([[[1.0, 2.0]]], [[True]])


@pytest.mark.parametrize(
    ("x", "y"),
    [(0.0, 0.0), (-1.0, -1.0), (3.5, -0.25)],
    ids=["origin", "minus_one_pair", "negative"],
)
def test_a_pair_with_both_values_is_labeled_whatever_they_are(x: float, y: float) -> None:
    # Negative coordinates and DeepLabCut's (-1, -1) are facts here; what they mean is each consumer's rule.
    arrays = _with_coords(["x", "y"], [x, y]).to_arrays()

    assert arrays.labeled.tolist() == [[True]]
    assert arrays.xy.tolist() == [[[x, y]]]


@pytest.mark.parametrize(
    ("x", "y"), [(float("nan"), float("nan")), (float("nan"), 2.0)], ids=["missing", "half_missing"]
)
def test_a_pair_with_a_missing_value_is_not_labeled_and_keeps_the_other(x: float, y: float) -> None:
    arrays = _with_coords(["x", "y"], [x, y]).to_arrays()

    assert arrays.labeled.tolist() == [[False]]
    np.testing.assert_array_equal(arrays.xy[0, 0], [x, y])


def test_a_keypoint_is_named_by_scorer_individual_and_bodypart_only() -> None:
    assert [field.name for field in fields(Keypoint)] == ["scorer", "individual", "bodypart"]
