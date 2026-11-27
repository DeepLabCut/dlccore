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

import shutil
from pathlib import Path

import pandas as pd
import pytest

from dlccore.io.errors import UnreadableFileError
from dlccore.io.labeled_data import LabeledData
from dlccore.io.labeled_data.wide.csv import read_labeled_data_csv
from dlccore.io.labeled_data.wide.h5 import read_labeled_data_h5
from dlccore.io.labeled_data.wide.storage import WideLabeledDataStorage, read_wide_labeled_data
from dlccore.io.labeled_data.wide.table import LabeledDataRead, classify_labeled_data_read
from dlccore.io.labeled_data.wide.to_long import wide_to_long

FIXTURES_DIR = Path(__file__).parents[2] / "fixtures" / "labeled_data"
FIXTURES = sorted(path.name for path in FIXTURES_DIR.glob("*") if path.suffix in {".h5", ".csv"})
CONTAINER_READERS = {".h5": read_labeled_data_h5, ".csv": read_labeled_data_csv}


def _wide_columns() -> pd.MultiIndex:
    return pd.MultiIndex.from_product([["Sc"], ["nose", "tail"], ["x", "y"]], names=["scorer", "bodyparts", "coords"])


def _write_h5(path: Path, df: pd.DataFrame) -> Path:
    df.to_hdf(path, key="df_with_missing", mode="w")
    return path


def test_every_fixture_is_found() -> None:
    # Control: an empty glob would let the parametrized tests below pass by running nothing.
    assert [Path(name).suffix for name in FIXTURES].count(".h5") == 8
    assert [Path(name).suffix for name in FIXTURES].count(".csv") == 2


@pytest.mark.parametrize("filename", FIXTURES)
def test_every_fixture_is_interpreted_as_its_container_read_classifies_it(filename: str) -> None:
    path = FIXTURES_DIR / filename
    read = CONTAINER_READERS[path.suffix](path)
    header, row_index = classify_labeled_data_read(read)

    storage = read_wide_labeled_data(path)

    assert storage.keypoints == tuple(columns.keypoint for columns in header.keypoints)
    assert storage.frames == row_index.frames
    assert storage.keypoints is not None
    assert storage.frames is not None
    assert storage.key_selection == read.key_selection
    pd.testing.assert_frame_equal(storage.dataframe, read.dataframe)


@pytest.mark.parametrize("filename", [name for name in FIXTURES if name.endswith(".csv")])
def test_a_csv_is_interpreted_as_its_h5_twin(filename: str) -> None:
    csv_storage = read_wide_labeled_data(FIXTURES_DIR / filename)
    h5_storage = read_wide_labeled_data((FIXTURES_DIR / filename).with_suffix(".h5"))

    assert (csv_storage.keypoints, csv_storage.frames) == (h5_storage.keypoints, h5_storage.frames)


def test_an_unrecognised_header_has_no_keypoints_but_keeps_its_frames(tmp_path: Path) -> None:
    df = pd.DataFrame([[1.0, 2.0]], columns=["nose_x", "nose_y"], index=["labeled-data/v1/img0.png"])

    storage = read_wide_labeled_data(_write_h5(tmp_path / "CollectedData_Sc.h5", df))

    assert storage.keypoints is None
    assert storage.frames == (("labeled-data", "v1", "img0.png"),)
    assert storage.header.columns == (("nose_x",), ("nose_y",))


def test_an_unrecognised_row_index_has_no_frames_but_keeps_its_keypoints(tmp_path: Path) -> None:
    df = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], columns=_wide_columns())

    storage = read_wide_labeled_data(_write_h5(tmp_path / "CollectedData_Sc.h5", df))

    assert storage.frames is None
    assert storage.keypoints is not None
    assert [keypoint.bodypart for keypoint in storage.keypoints] == ["nose", "tail"]


def _storage_with_folder(tmp_path: Path, folder: object) -> WideLabeledDataStorage:
    index = pd.MultiIndex.from_tuples([("labeled-data", folder, "img0.png")])
    df = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], index=index, columns=_wide_columns())
    return read_wide_labeled_data(_write_h5(tmp_path / "CollectedData_Sc.h5", df))


@pytest.mark.parametrize("folder", [2019.5, True], ids=["float", "bool"])
def test_a_path_part_naming_no_folder_leaves_the_frames_uninterpreted_and_unconverted(
    tmp_path: Path, folder: object
) -> None:
    storage = _storage_with_folder(tmp_path, folder)

    assert storage.frames is None
    assert storage.row_index.frames == (("labeled-data", folder, "img0.png"),)


def test_an_integer_path_part_is_read_as_text_and_kept_as_stored(tmp_path: Path) -> None:
    # A folder named by a number, as pandas infers it from a CSV; DeepLabCut casts it back to text.
    storage = _storage_with_folder(tmp_path, 2019)

    assert storage.frames == (("labeled-data", "2019", "img0.png"),)
    assert storage.row_index.frames == (("labeled-data", 2019, "img0.png"),)


def test_the_same_path_part_as_text_is_interpreted(tmp_path: Path) -> None:
    # Control: without it, frames that were never interpreted would pass the test above.
    assert _storage_with_folder(tmp_path, "2019").frames == (("labeled-data", "2019", "img0.png"),)


@pytest.mark.parametrize(("filename", "content"), [("garbage.h5", b"not an hdf5 file"), ("missing.csv", None)])
def test_an_unreadable_file_raises_from_the_original(tmp_path: Path, filename: str, content: bytes | None) -> None:
    path = tmp_path / filename
    if content is not None:
        path.write_bytes(content)

    with pytest.raises(UnreadableFileError, match=filename) as raised:
        read_wide_labeled_data(path)

    assert raised.value.__cause__ is not None


@pytest.mark.parametrize("filename", ["single_animal_legacy_index_dlc.h5", "single_animal_legacy_index_dlc.csv"])
def test_a_suffix_in_upper_case_is_read_as_in_lower_case(tmp_path: Path, filename: str) -> None:
    lower = FIXTURES_DIR / filename
    upper = Path(shutil.copy(lower, tmp_path / f"{lower.stem}{lower.suffix.upper()}"))

    storage = read_wide_labeled_data(upper)

    expected = read_wide_labeled_data(lower)
    assert (storage.keypoints, storage.frames) == (expected.keypoints, expected.frames)


def test_an_unknown_suffix_is_refused_without_reading(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"'\.txt'"):
        read_wide_labeled_data(tmp_path / "CollectedData_Sc.txt")


def test_two_reads_of_one_file_are_distinct_and_hashable() -> None:
    first, second = (read_wide_labeled_data(FIXTURES_DIR / "single_animal_human.h5") for _ in range(2))

    assert first != second
    assert len({first, second}) == 2
    assert second not in [first]


def test_a_read_equals_itself() -> None:
    # Control: identity equality must still hold, or the test above passes for any broken __eq__.
    storage = read_wide_labeled_data(FIXTURES_DIR / "single_animal_human.h5")

    assert storage == storage
    assert {storage} == {storage}


def _in_memory(tmp_path: Path, columns: list[tuple[str, ...]], values: list[object]) -> WideLabeledDataStorage:
    # Built in memory: an H5 file cannot hold a repeated column, and the normaliser is what is under test.
    names = ["scorer", "bodyparts", "coords"]
    df = pd.DataFrame(
        [values], columns=pd.MultiIndex.from_tuples(columns, names=names), index=["labeled-data/v1/img0.png"]
    )
    header, row_index = classify_labeled_data_read(LabeledDataRead(tmp_path / "in-memory.h5", df, None))
    return WideLabeledDataStorage(key_selection=None, header=header, row_index=row_index, dataframe=df)


def _with_coords(tmp_path: Path, coords: list[str], values: list[object]) -> WideLabeledDataStorage:
    return _in_memory(tmp_path, [("Sc", "nose", c) for c in coords], values)


def _model(storage: WideLabeledDataStorage) -> LabeledData:
    return wide_to_long(storage.dataframe, storage.header, storage.row_index)


@pytest.mark.parametrize(
    ("coords", "values"),
    [(["x", "x", "y"], [1.0, 1.5, 2.0]), (["x"], [1.0]), (["x", "y", "z"], [1.0, 2.0, 3.0])],
    ids=["repeated", "missing_y", "triangulated_xyz"],  # xyz: DeepLabCut 3D output, not supported yet
)
def test_coordinates_that_are_not_x_y_likelihood_are_not_interpreted(
    tmp_path: Path, coords: list[str], values: list[object]
) -> None:
    storage = _with_coords(tmp_path, coords, values)

    assert storage.keypoints is not None
    with pytest.raises(ValueError, match="coordinates"):
        _model(storage)


def test_a_likelihood_on_some_keypoints_only_is_not_interpreted(tmp_path: Path) -> None:
    columns = [
        ("Sc", "nose", "x"),
        ("Sc", "nose", "y"),
        ("Sc", "nose", "likelihood"),
        ("Sc", "tail", "x"),
        ("Sc", "tail", "y"),
    ]

    with pytest.raises(ValueError, match="likelihood"):
        _model(_in_memory(tmp_path, columns, [1.0, 2.0, 0.9, 3.0, 4.0]))
    control = _model(_in_memory(tmp_path, columns[:2] + columns[3:], [1.0, 2.0, 3.0, 4.0]))
    assert control.to_arrays().xy.shape == (1, 2, 2)


@pytest.mark.parametrize("x", ["1.0", 1 + 2j], ids=["text", "complex"])
def test_a_non_real_column_is_not_interpreted_or_converted(tmp_path: Path, x: object) -> None:
    storage = _with_coords(tmp_path, ["x", "y"], [x, 2.0])

    with pytest.raises(ValueError, match="not numeric"):
        _model(storage)
    assert storage.dataframe.iloc[0, 0] == x


def test_uninterpretable_frames_give_no_table(tmp_path: Path) -> None:
    df = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], columns=_wide_columns())

    storage = read_wide_labeled_data(_write_h5(tmp_path / "CollectedData_Sc.h5", df))

    assert storage.frames is None
    with pytest.raises(ValueError, match="not text"):
        _model(storage)
