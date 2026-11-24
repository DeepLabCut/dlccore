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

import importlib
from pathlib import Path, PurePath

import pandas as pd
import pytest

from dlccore.io.labeled_data import Keypoint, LabeledData, read_labeled_data
from dlccore.io.labeled_data.checks.file import (
    frame_absorbed_into_index_names,
    frames_of_other_depth,
    frames_with_unnamed_parts,
    has_no_frames,
    header_is_unrecognised,
    keypoints_with_unrecognised_coords,
    keypoints_without_the_likelihood_others_store,
    non_numeric_columns,
    row_index_is_unrecognised,
    unread_h5_keys,
)
from dlccore.io.labeled_data.checks.model import (
    keypoints_with_empty_individual,
    likelihood_in_human_labels,
    missing_keypoints,
)
from dlccore.io.labeled_data.wide.h5 import H5KeySelectionOutcome, select_h5_key
from dlccore.io.labeled_data.wide.header import LabeledDataHeader, classify_labeled_data_header
from dlccore.io.labeled_data.wide.row_index import LabeledDataRowIndex, classify_labeled_data_row_index

FIXTURES_DIR = Path(__file__).parents[1] / "fixtures" / "labeled_data"
SINGLE_ANIMAL_LEVELS = ["scorer", "bodyparts", "coords"]
MULTI_ANIMAL_LEVELS = ["scorer", "individuals", "bodyparts", "coords"]


def _labels(keypoints: list[tuple[str, ...]], coords: tuple[str, ...] = ("x", "y")) -> LabeledData:
    names = MULTI_ANIMAL_LEVELS if len(keypoints[0]) == 3 else SINGLE_ANIMAL_LEVELS
    columns = pd.MultiIndex.from_tuples([(*k, c) for k in keypoints for c in coords], names=names)
    return LabeledData.from_wide_df(
        pd.DataFrame([[1.0] * len(columns)], columns=columns, index=["labeled-data/s1/img0.png"])
    )


def _individual(scorer: str, individual: str, bodypart: str) -> Keypoint:
    return Keypoint(scorer, individual, bodypart)


WITH_LIKELIHOOD = _labels([("Sc", "nose")], coords=("x", "y", "likelihood"))


def test_a_likelihood_in_human_labels_is_found() -> None:
    assert likelihood_in_human_labels(WITH_LIKELIHOOD, PurePath("labeled-data/s1/CollectedData_Sc.h5"))


@pytest.mark.parametrize(
    ("labels", "name"),
    [
        (WITH_LIKELIHOOD, "machinelabels-iter0.h5"),
        (WITH_LIKELIHOOD, "collecteddata_Sc.h5"),
        (_labels([("Sc", "nose")]), "CollectedData_Sc.h5"),
    ],
    ids=["machine_labels", "other_case", "no_likelihood"],
)
def test_a_likelihood_elsewhere_or_human_labels_without_one_is_not_found(labels: LabeledData, name: str) -> None:
    assert not likelihood_in_human_labels(labels, PurePath(name))


def test_an_individual_lacking_a_bodypart_another_has_is_found() -> None:
    labels = _labels([("Sc", "mouse1", "nose"), ("Sc", "mouse1", "tail"), ("Sc", "mouse2", "nose")])

    assert missing_keypoints(labels) == (_individual("Sc", "mouse2", "tail"),)


def test_missing_keypoints_come_in_individual_then_bodypart_order() -> None:
    labels = _labels([("Sc", "mouse1", "tail"), ("Sc", "mouse2", "nose"), ("Sc", "mouse3", "ear")])

    assert missing_keypoints(labels) == (
        _individual("Sc", "mouse1", "nose"),
        _individual("Sc", "mouse1", "ear"),
        _individual("Sc", "mouse2", "tail"),
        _individual("Sc", "mouse2", "ear"),
        _individual("Sc", "mouse3", "tail"),
        _individual("Sc", "mouse3", "nose"),
    )


@pytest.mark.parametrize(
    "keypoints",
    [
        [("Sc", "mouse1", "nose"), ("Sc", "mouse2", "nose")],
        [("Sc", "mouse1", "nose"), ("Sc", "mouse2", "nose"), ("Sc", "single", "box")],
        [("Sc1", "mouse1", "nose"), ("Sc2", "mouse2", "tail")],
        [("Sc", "nose"), ("Sc", "tail")],
    ],
    ids=["complete", "unique_bodyparts_apart", "scorers_apart", "single_animal"],
)
def test_no_keypoint_is_missing_where_every_individual_has_its_scorers_bodyparts(
    keypoints: list[tuple[str, ...]],
) -> None:
    assert missing_keypoints(_labels(keypoints)) == ()


@pytest.mark.parametrize(
    "filename", ["multi_animal_human.h5", "multi_animal_machine.h5", "multi_animal_unique_bodyparts.h5"]
)
def test_the_multi_animal_fixtures_miss_no_keypoint(filename: str) -> None:
    assert missing_keypoints(read_labeled_data(FIXTURES_DIR / filename).labeled_data) == ()


def test_the_keys_a_multi_key_file_leaves_unread_are_found() -> None:
    selection = select_h5_key(FIXTURES_DIR / "ambiguous_multi_key.h5")
    assert selection.outcome is H5KeySelectionOutcome.MULTI_KEY  # control: the fixture is the case under test

    unread = unread_h5_keys(selection)

    assert unread
    assert set(unread) == set(selection.keys_found) - {selection.selected_key}


@pytest.mark.parametrize("filename", ["single_animal_human.h5", "single_animal_human_legacy_index.h5"])
def test_a_file_with_one_key_leaves_none_unread(filename: str) -> None:
    assert unread_h5_keys(select_h5_key(FIXTURES_DIR / filename)) == ()


def test_a_container_without_keys_leaves_none_unread() -> None:
    assert unread_h5_keys(None) == ()


def _parts(frames: list[tuple[object, ...]], names: tuple[object, ...] = ()) -> LabeledDataRowIndex:
    return classify_labeled_data_row_index(names, frames, is_multi_index=True)


def test_frames_of_other_than_three_parts_are_found_by_row() -> None:
    frames = [("labeled-data", "s1", "img0.png"), ("s1", "img1.png"), ("labeled-data", "s1", "extra", "img2.png")]

    assert frames_of_other_depth(_parts(frames)) == (1, 2)


def test_frames_of_three_parts_are_not_found() -> None:
    assert frames_of_other_depth(_parts([("labeled-data", "s1", "img0.png"), ("labeled-data", 2019, "img1.png")])) == ()


def test_frames_with_a_part_naming_nothing_are_found_by_row() -> None:
    frames = [("labeled-data", "s1", "img0.png"), ("labeled-data", 2019.5, "img1.png"), ("labeled-data", True, "a")]

    assert frames_with_unnamed_parts(_parts(frames)) == (1, 2)


def test_frames_of_text_and_integer_parts_are_not_found() -> None:
    frames = [("labeled-data", "s1", "img0.png"), ("labeled-data", 2019, "img1.png")]

    assert frames_with_unnamed_parts(_parts(frames)) == ()


def test_a_table_without_rows_has_no_frames_and_one_with_rows_has() -> None:
    assert has_no_frames(classify_labeled_data_row_index((None,), (), is_multi_index=False))
    assert not has_no_frames(_parts([("labeled-data", "s1", "img0.png")]))


def test_a_row_index_of_numbers_is_unrecognised_and_one_of_paths_is_not() -> None:
    assert row_index_is_unrecognised(classify_labeled_data_row_index((None,), (0, 1), is_multi_index=False))
    assert not row_index_is_unrecognised(
        classify_labeled_data_row_index((None,), ("labeled-data/s1/img0.png",), is_multi_index=False)
    )


def _header(coords_by_bodypart: dict[str, tuple[str, ...]], levels: tuple[str, ...] = ()) -> LabeledDataHeader:
    columns = [("Sc", bodypart, coord) for bodypart, coords in coords_by_bodypart.items() for coord in coords]
    return classify_labeled_data_header(levels or SINGLE_ANIMAL_LEVELS, columns)


def test_a_header_with_other_level_names_is_unrecognised_and_deeplabcuts_is_not() -> None:
    assert header_is_unrecognised(_header({"nose": ("x", "y")}, levels=("scorer", "bodypart", "coord")))
    assert not header_is_unrecognised(_header({"nose": ("x", "y")}))


@pytest.mark.parametrize(
    "coords", [("x", "y", "z"), ("x",), ("x", "y", "y"), ("x", "x", "y")], ids=["xyz", "x_only", "y_twice", "x_twice"]
)
def test_coordinates_other_than_x_y_likelihood_once_each_are_found(coords: tuple[str, ...]) -> None:
    header = _header({"nose": ("x", "y"), "tail": coords})

    assert [columns.keypoint.bodypart for columns in keypoints_with_unrecognised_coords(header)] == ["tail"]


@pytest.mark.parametrize("coords", [("x", "y"), ("y", "x"), ("x", "y", "likelihood")], ids=["xy", "yx", "xyl"])
def test_x_y_and_an_optional_likelihood_in_any_order_are_not_found(coords: tuple[str, ...]) -> None:
    assert keypoints_with_unrecognised_coords(_header({"nose": coords})) == ()


def test_a_keypoint_without_the_likelihood_another_stores_is_found() -> None:
    header = _header({"nose": ("x", "y", "likelihood"), "tail": ("x", "y"), "ear": ("x", "y", "likelihood")})

    assert [columns.keypoint.bodypart for columns in keypoints_without_the_likelihood_others_store(header)] == ["tail"]


@pytest.mark.parametrize("coords", [("x", "y"), ("x", "y", "likelihood")], ids=["none_store_one", "all_store_one"])
def test_keypoints_that_agree_on_a_likelihood_are_not_found(coords: tuple[str, ...]) -> None:
    assert keypoints_without_the_likelihood_others_store(_header({"nose": coords, "tail": coords})) == ()


@pytest.mark.parametrize("value", ["1.0", True, 1 + 2j], ids=["text", "bool", "complex"])
def test_a_column_of_values_that_are_not_real_numbers_is_found(value: object) -> None:
    table = pd.DataFrame({("Sc", "nose", "x"): [value], ("Sc", "nose", "y"): [2.0]})

    assert non_numeric_columns(table) == (("Sc", "nose", "x"),)


def test_integer_and_float_columns_are_not_found() -> None:
    assert non_numeric_columns(pd.DataFrame({("Sc", "nose", "x"): [1], ("Sc", "nose", "y"): [float("nan")]})) == ()


def _csv_with_first_frame(tmp_path: Path, *, flat: bool, empty: bool = True) -> Path:
    frames = [("labeled-data", "Sample4", "0155.png"), ("labeled-data", "Sample4", "0156.png")]
    index = pd.Index(["/".join(frame) for frame in frames]) if flat else pd.MultiIndex.from_tuples(frames)
    columns = pd.MultiIndex.from_tuples([("Sc", "nose", "x"), ("Sc", "nose", "y")], names=SINGLE_ANIMAL_LEVELS)
    first = [float("nan")] * 2 if empty else [1.0, 2.0]
    path = tmp_path / "CollectedData_Sc.csv"
    pd.DataFrame([first, [3.0, 4.0]], index=index, columns=columns).to_csv(path)
    return path


@pytest.mark.parametrize("flat", [False, True], ids=["path_parts", "flat_paths"])
def test_a_first_frame_pandas_read_as_index_names_is_found(tmp_path: Path, *, flat: bool) -> None:
    labels = read_labeled_data(_csv_with_first_frame(tmp_path, flat=flat))
    assert len(labels.raw_row_index.entries) == 1  # control: the CSV did lose its first frame

    assert frame_absorbed_into_index_names(labels.raw_row_index) == ("labeled-data", "Sample4", "0155.png")


def test_a_csv_whose_first_frame_has_values_absorbs_nothing(tmp_path: Path) -> None:
    labels = read_labeled_data(_csv_with_first_frame(tmp_path, flat=False, empty=False))

    assert frame_absorbed_into_index_names(labels.raw_row_index) == ()


@pytest.mark.parametrize(
    "names",
    [(None, None, None), ("image",), ("a", "b"), ("labeled-data", None, "img0.png")],
    ids=["unnamed", "one_name", "two_names", "one_unnamed"],
)
def test_index_names_that_spell_no_three_part_frame_are_not_found(names: tuple[object, ...]) -> None:
    assert frame_absorbed_into_index_names(_parts([("labeled-data", "s1", "img1.png")], names=names)) == ()


def test_checks_are_split_into_file_and_model_facts() -> None:
    importlib.import_module("dlccore.io.labeled_data.checks.file")
    importlib.import_module("dlccore.io.labeled_data.checks.model")


def test_a_keypoint_whose_individual_is_named_empty_is_found() -> None:
    labels = _labels([("Sc", "mouse1", "nose"), ("Sc", "", "nose")])

    assert keypoints_with_empty_individual(labels) == (_individual("Sc", "", "nose"),)


@pytest.mark.parametrize("keypoints", [[("Sc", "mouse1", "nose")], [("Sc", "nose")]], ids=["named", "single_animal"])
def test_named_or_absent_individuals_are_not_found(keypoints: list[tuple[str, ...]]) -> None:
    assert keypoints_with_empty_individual(_labels(keypoints)) == ()
