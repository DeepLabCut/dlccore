from pathlib import Path

import dlc_octopus.io.labeled_data as labeled_data
import pandas as pd
import pytest
from dlc_octopus.io.errors import UnreadableFileError
from dlc_octopus.io.labeled_data import LabeledData, LabeledDataFile, read_labeled_data, write_labeled_data
from dlc_octopus.io.labeled_data.wide.storage import WideLabeledDataStorage, read_wide_labeled_data

FIXTURES_DIR = Path(__file__).parents[1] / "fixtures" / "labeled_data"
FIXTURES = sorted(path.name for path in FIXTURES_DIR.glob("*") if path.suffix in {".h5", ".csv"})


def _wide_columns() -> pd.MultiIndex:
    return pd.MultiIndex.from_product([["Sc"], ["nose", "tail"], ["x", "y"]], names=["scorer", "bodyparts", "coords"])


def test_every_fixture_is_found() -> None:
    # Control: an empty glob would let the parametrized test below pass by running nothing.
    assert len(FIXTURES) == 10


@pytest.mark.parametrize("filename", FIXTURES)
def test_every_fixture_reads_as_its_wide_storage_interprets_it(filename: str) -> None:
    path = FIXTURES_DIR / filename

    labels = read_labeled_data(path)
    storage = read_wide_labeled_data(path)

    assert labels.path == path
    assert isinstance(labels._stored, WideLabeledDataStorage)
    assert (labels.keypoints, labels.frames) == (storage.keypoints, storage.frames)
    assert labels.keypoints is not None
    assert labels.frames is not None


def _read_written(tmp_path: Path, frame: pd.DataFrame) -> LabeledDataFile:
    path = tmp_path / "CollectedData_Sc.h5"
    frame.to_hdf(path, key="df_with_missing", mode="w")
    return read_labeled_data(path)


def test_frames_are_kept_when_the_header_is_unrecognised(tmp_path: Path) -> None:
    frame = pd.DataFrame([[1.0, 2.0]], columns=["nose_x", "nose_y"], index=["labeled-data/v1/img0.png"])

    labels = _read_written(tmp_path, frame)

    assert labels.keypoints is None
    assert labels.frames == (("labeled-data", "v1", "img0.png"),)


def test_keypoints_are_kept_when_the_row_index_is_unrecognised(tmp_path: Path) -> None:
    labels = _read_written(tmp_path, pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], columns=_wide_columns()))

    assert labels.frames is None
    assert labels.keypoints is not None
    assert [keypoint.bodypart for keypoint in labels.keypoints] == ["nose", "tail"]


def test_read_errors_propagate_unchanged(tmp_path: Path) -> None:
    with pytest.raises(UnreadableFileError):
        read_labeled_data(tmp_path / "missing.csv")
    with pytest.raises(ValueError, match=r"'\.txt'"):
        read_labeled_data(tmp_path / "CollectedData_Sc.txt")


def test_two_reads_of_one_file_are_distinct_and_hashable() -> None:
    first, second = (read_labeled_data(FIXTURES_DIR / "single_animal_human.h5") for _ in range(2))

    assert first != second
    assert len({first, second}) == 2


def test_the_package_exports_exactly_the_consumer_surface() -> None:
    assert sorted(labeled_data.__all__) == [
        "BodypartScope",
        "FramePath",
        "Keypoint",
        "LABELED_DATA_LONG_VERSION",
        "LabeledData",
        "LabeledDataArrays",
        "LabeledDataFile",
        "read_labeled_data",
        "write_labeled_data",
    ]
    assert labeled_data.LabeledDataFile is LabeledDataFile


def test_a_labeled_data_file_yields_arrays_of_every_frame_and_keypoint() -> None:
    labels = read_labeled_data(FIXTURES_DIR / "single_animal_human.h5")

    arrays = labels.labeled_data.to_arrays()

    assert labels.frames is not None
    assert labels.keypoints is not None
    assert arrays.xy.shape == (len(labels.frames), len(labels.keypoints), 2)
    assert arrays.labeled.size == len(labels.frames) * len(labels.keypoints) > 0


def test_a_legacy_labeled_data_file_is_written_back_in_the_current_deeplabcut_form(tmp_path: Path) -> None:
    table = read_labeled_data(FIXTURES_DIR / "single_animal_human_legacy_index.h5").labeled_data

    path = tmp_path / "CollectedData_Sc.h5"
    write_labeled_data(table, path, replaces=None)

    with pd.HDFStore(path, mode="r") as store:
        assert list(store.keys()) == ["/df_with_missing"]
    written = pd.read_hdf(path, key="df_with_missing")
    assert written.columns.nlevels == 3
    assert isinstance(written.index, pd.MultiIndex)
    assert written.index[0] == ("labeled-data", "session-03", "img000.png")


H5_FIXTURES = [name for name in FIXTURES if name.endswith(".h5") and name != "ambiguous_multi_key.h5"]


def test_the_ambiguous_fixture_is_the_one_h5_left_out() -> None:
    # Control: keyless read_hdf refuses a file with several keys, so the comparison below cannot include it.
    with pytest.raises(ValueError, match="key must be provided"):
        pd.read_hdf(FIXTURES_DIR / "ambiguous_multi_key.h5")
    assert len(H5_FIXTURES) == 7


@pytest.mark.parametrize("filename", H5_FIXTURES)
def test_the_raw_wide_view_is_what_read_hdf_returns(filename: str) -> None:
    # The swap-in guarantee: replacing pd.read_hdf(path) with read_labeled_data(path).to_raw_wide_df() changes nothing.
    path = FIXTURES_DIR / filename

    pd.testing.assert_frame_equal(read_labeled_data(path).to_raw_wide_df(), pd.read_hdf(path), check_exact=True)


def test_the_raw_wide_view_is_a_copy() -> None:
    labels = read_labeled_data(FIXTURES_DIR / "single_animal_human.h5")

    view = labels.to_raw_wide_df()
    view.iloc[0, 0] = -99.0

    assert labels.to_raw_wide_df().iloc[0, 0] != -99.0


@pytest.mark.parametrize("filename", FIXTURES)
def test_every_fixture_has_labeled_data_with_one_row_per_frame_and_keypoint(filename: str) -> None:
    labels = read_labeled_data(FIXTURES_DIR / filename)

    assert labels.frames is not None
    assert labels.keypoints is not None
    assert len(labels.labeled_data.to_long_df()) == len(labels.frames) * len(labels.keypoints)


def test_a_file_without_a_model_names_itself_and_the_reason(tmp_path: Path) -> None:
    index = pd.MultiIndex.from_tuples([("labeled-data", "v1", "extra", "img0.png")])
    labels = _read_written(tmp_path, pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], index=index, columns=_wide_columns()))

    with pytest.raises(ValueError, match=r"CollectedData_Sc\.h5: .*has 4 parts"):
        _ = labels.labeled_data
    assert labels.to_raw_wide_df().shape == (1, 4)


def test_the_package_exports_the_long_model() -> None:
    from dlc_octopus.io.labeled_data.long.model import LabeledData as Model

    assert LabeledData is Model


@pytest.mark.parametrize("filename", FIXTURES)
def test_the_raw_table_and_the_written_table_both_give_back_the_read_labels(filename: str) -> None:
    # from_wide_df interprets any wide form as a read does: the raw table as found, and the converged one written.
    labels = read_labeled_data(FIXTURES_DIR / filename)
    model = labels.labeled_data.to_long_df()

    from_raw = LabeledData.from_wide_df(labels.to_raw_wide_df()).to_long_df()
    from_written = LabeledData.from_wide_df(labels.labeled_data.to_wide_df()).to_long_df()

    pd.testing.assert_frame_equal(from_raw, model)
    pd.testing.assert_frame_equal(from_written, model)


def test_a_wide_table_without_a_model_is_refused_with_the_facts() -> None:
    columns = pd.MultiIndex.from_tuples([("Sc", "nose", "x")], names=["scorer", "bodypart", "coord"])

    with pytest.raises(ValueError, match="Header levels"):
        LabeledData.from_wide_df(pd.DataFrame([[1.0]], index=["labeled-data/v1/img0.png"], columns=columns))


@pytest.mark.parametrize("filename", ["single_animal_human.h5", "multi_animal_unique_bodyparts.csv"])
def test_a_file_exposes_its_stored_facts_read_only(filename: str) -> None:
    labels = read_labeled_data(FIXTURES_DIR / filename)
    stored = labels._stored

    assert labels.key_selection is stored.key_selection
    assert (labels.writer_stamp, labels.writer_stamp_error) == (stored.writer_stamp, stored.writer_stamp_error)
    assert (labels.raw_header, labels.raw_row_index) == (stored.header, stored.row_index)
    with pytest.raises(AttributeError):
        labels.raw_header = stored.header  # type: ignore[misc]


def test_an_h5_file_has_a_key_selection_and_a_csv_none() -> None:
    # Control for the identity checks above: a fact that differs by container is not trivially None.
    assert read_labeled_data(FIXTURES_DIR / "single_animal_human.h5").key_selection is not None
    assert read_labeled_data(FIXTURES_DIR / "multi_animal_unique_bodyparts.csv").key_selection is None
