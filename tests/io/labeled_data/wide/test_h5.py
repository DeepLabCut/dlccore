from pathlib import Path

import pandas as pd
import pytest
import tables

from dlc_octopus.io.errors import UnreadableFileError
from dlc_octopus.io.labeled_data.wide.h5 import (
    H5_KEY_REGISTRY,
    H5KeySelection,
    H5KeySelectionOutcome,
    H5KeyStatus,
    read_labeled_data_h5,
    select_h5_key,
)

FIXTURES_DIR = Path(__file__).parents[2] / "fixtures" / "labeled_data"
FIXTURES = sorted(path.name for path in FIXTURES_DIR.glob("*.h5"))


def _wide_columns() -> pd.MultiIndex:
    return pd.MultiIndex.from_product(
        [["Sc"], ["nose", "tail"], ["x", "y"]],
        names=["scorer", "bodyparts", "coords"],
    )


def _write(path: Path, key: str, frame: pd.DataFrame | None = None) -> Path:
    (pd.DataFrame([[1.0, 2.0]]) if frame is None else frame).to_hdf(path, key=key, mode="w")
    return path


# --- select_h5_key -----------------------------------------------------------------


def test_select_h5_key_single_current(tmp_path: Path) -> None:
    result = select_h5_key(_write(tmp_path / "CollectedData_Sc.h5", "df_with_missing"))

    assert result == H5KeySelection(
        keys_found=("df_with_missing",),
        selected_key="df_with_missing",
        outcome=H5KeySelectionOutcome.SINGLE_CURRENT,
    )


def test_select_h5_key_single_legacy(tmp_path: Path) -> None:
    result = select_h5_key(_write(tmp_path / "CollectedData_Sc.h5", "keypoints"))

    assert result.outcome == H5KeySelectionOutcome.SINGLE_LEGACY
    assert result.selected_key == "keypoints"


def test_select_h5_key_multi_key_selects_current_over_legacy(tmp_path: Path) -> None:
    path = _write(tmp_path / "CollectedData_Sc.h5", "keypoints")
    with pd.HDFStore(path, mode="a") as store:
        store.put("df_with_missing", pd.DataFrame([[1.0, 2.0]]))

    result = select_h5_key(path)

    assert result.outcome == H5KeySelectionOutcome.MULTI_KEY
    assert result.selected_key == "df_with_missing"
    assert set(result.keys_found) == {"df_with_missing", "keypoints"}


def test_select_h5_key_multi_key_selects_legacy_over_unknown(tmp_path: Path) -> None:
    path = _write(tmp_path / "CollectedData_Sc.h5", "some_other_key")
    with pd.HDFStore(path, mode="a") as store:
        store.put("keypoints", pd.DataFrame([[1.0, 2.0]]))

    result = select_h5_key(path)

    assert result.outcome == H5KeySelectionOutcome.MULTI_KEY
    assert result.selected_key == "keypoints"


def test_select_h5_key_single_unknown_key_is_still_selected(tmp_path: Path) -> None:
    """DeepLabCut's merge reads any single key, so refusing one is our defect."""

    result = select_h5_key(_write(tmp_path / "CollectedData_Sc.h5", "some_other_key"))

    assert result.outcome == H5KeySelectionOutcome.SINGLE_UNKNOWN
    assert result.selected_key == "some_other_key"


def test_h5_key_registry_status_matches_design() -> None:
    assert H5_KEY_REGISTRY["df_with_missing"] == H5KeyStatus.CURRENT, (
        "DeepLabCut has written this key since its first commit, 2018-04-10."
    )
    assert H5_KEY_REGISTRY["keypoints"] == H5KeyStatus.LEGACY, (
        "DO NOT DROP THIS KEY UNLESS YOU HAVE A VERY GOOD REASON TO DO SO. "
        "Not a defensive guess: napari-deeplabcut wrote this key itself from 2022-05-12 "
        "until 2026-04-27. Both tools read keyless, so no divergence was noticed. "
        "Files carrying it are labeled data produced by "
        "the GUI DeepLabCut ships for labelling, and a single project may hold both keys, "
        "one per video folder, depending on which tool labeled it."
    )


@pytest.mark.parametrize(
    ("selected_key", "outcome"),
    [(None, H5KeySelectionOutcome.SINGLE_CURRENT), ("df_with_missing", H5KeySelectionOutcome.EMPTY)],
)
def test_a_key_selection_names_a_key_exactly_when_the_file_has_one(
    selected_key: str | None,
    outcome: H5KeySelectionOutcome,
) -> None:
    with pytest.raises(ValueError, match="EMPTY"):
        H5KeySelection(keys_found=(), selected_key=selected_key, outcome=outcome)


# --- read_labeled_data_h5: the raw read --------------------------------------------


@pytest.mark.parametrize("filename", FIXTURES)
def test_the_raw_read_equals_pandas_on_the_selected_key(filename: str) -> None:
    path = FIXTURES_DIR / filename

    result = read_labeled_data_h5(path)

    pd.testing.assert_frame_equal(result.dataframe, pd.read_hdf(path, key=result.key_selection.selected_key))


def test_the_fixtures_are_all_found() -> None:
    # Control: an empty glob would let the parametrized equality test pass by running nothing.
    assert len(FIXTURES) == 8


def test_the_equality_check_sees_a_modified_frame() -> None:
    # Control: assert_frame_equal must fail on a frame differing in one value, or equality proves nothing.
    path = FIXTURES_DIR / "single_animal_human.h5"
    modified = read_labeled_data_h5(path).dataframe.copy()
    modified.iloc[0, 0] += 1

    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(modified, pd.read_hdf(path, key="df_with_missing"))


def test_the_legacy_flat_row_index_is_returned_as_stored() -> None:
    result = read_labeled_data_h5(FIXTURES_DIR / "single_animal_human_legacy_index.h5")

    assert not isinstance(result.dataframe.index, pd.MultiIndex)
    assert all(isinstance(entry, str) for entry in result.dataframe.index)


def test_a_multi_animal_header_with_unique_bodyparts_is_returned_as_stored_not_as_a_grid() -> None:
    columns = read_labeled_data_h5(FIXTURES_DIR / "multi_animal_unique_bodyparts.h5").dataframe.columns

    individuals = columns.get_level_values("individuals")
    bodyparts = columns.get_level_values("bodyparts")
    pairs = list(dict.fromkeys(zip(individuals, bodyparts, strict=True)))

    assert pairs == [
        ("animalA", "snout"),
        ("animalA", "tail"),
        ("animalB", "snout"),
        ("animalB", "tail"),
        ("single", "feeder"),
    ]
    assert len(columns) < columns.levshape[1] * columns.levshape[2] * columns.levshape[3]


@pytest.mark.parametrize("filename", FIXTURES)
def test_reading_leaves_the_file_unchanged(filename: str) -> None:
    path = FIXTURES_DIR / filename
    before = (path.read_bytes(), path.stat().st_mtime_ns)

    read_labeled_data_h5(path)

    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_a_multi_key_fixture_is_read_by_precedence_and_reported() -> None:
    result = read_labeled_data_h5(FIXTURES_DIR / "ambiguous_multi_key.h5")

    assert result.dataframe is not None
    assert result.key_selection.outcome == H5KeySelectionOutcome.MULTI_KEY
    assert result.key_selection.selected_key == "df_with_missing"
    assert set(result.key_selection.keys_found) == {"df_with_missing", "keypoints"}


def test_a_single_unrecognised_key_is_read(tmp_path: Path) -> None:
    path = _write(tmp_path / "CollectedData_Sc.h5", "some_other_key")

    result = read_labeled_data_h5(path)

    assert result.dataframe is not None
    assert result.key_selection.outcome == H5KeySelectionOutcome.SINGLE_UNKNOWN


def test_missing_values_are_preserved(tmp_path: Path) -> None:
    index = pd.MultiIndex.from_tuples([("labeled-data", "session-01", "img001.png")])
    frame = pd.DataFrame([[1.0, 2.0, float("nan"), float("nan")]], index=index, columns=_wide_columns())

    result = read_labeled_data_h5(_write(tmp_path / "CollectedData_Sc.h5", "df_with_missing", frame))

    assert pd.isna(result.dataframe.loc[("labeled-data", "session-01", "img001.png"), ("Sc", "tail", "x")])


def test_an_empty_dataframe_is_read(tmp_path: Path) -> None:
    frame = pd.DataFrame(columns=_wide_columns())

    result = read_labeled_data_h5(_write(tmp_path / "CollectedData_Sc.h5", "df_with_missing", frame))

    assert result.dataframe.empty


# --- read_labeled_data_h5: one exception for a file that cannot be read -----------


@pytest.mark.parametrize(
    ("label", "content"),
    [("missing", None), ("not_hdf5", b"not an hdf5 file at all"), ("zero_bytes", b"")],
    ids=str,
)
def test_an_unreadable_file_raises_from_the_original(tmp_path: Path, label: str, content: bytes | None) -> None:
    path = tmp_path / f"{label}.h5"
    if content is not None:
        path.write_bytes(content)

    with pytest.raises(UnreadableFileError, match=label) as raised:
        read_labeled_data_h5(path)

    assert isinstance(raised.value.__cause__, (OSError, RuntimeError))


def test_a_directory_is_unreadable(tmp_path: Path) -> None:
    path = tmp_path / "CollectedData_Sc.h5"
    path.mkdir()

    with pytest.raises(UnreadableFileError):
        read_labeled_data_h5(path)


def test_a_truncated_file_is_unreadable(tmp_path: Path) -> None:
    path = _write(tmp_path / "CollectedData_Sc.h5", "df_with_missing", pd.DataFrame([[1.0] * 100] * 100))
    content = path.read_bytes()
    path.write_bytes(content[: len(content) // 2])

    with pytest.raises(UnreadableFileError):
        read_labeled_data_h5(path)


def test_an_empty_store_is_selected_as_empty_and_raises_on_read(tmp_path: Path) -> None:
    path = tmp_path / "CollectedData_Sc.h5"
    with pd.HDFStore(path, mode="w"):
        pass

    assert select_h5_key(path) == H5KeySelection(keys_found=(), selected_key=None, outcome=H5KeySelectionOutcome.EMPTY)

    with pytest.raises(UnreadableFileError, match="no pandas key") as raised:
        read_labeled_data_h5(path)

    assert raised.value.__cause__ is None


def test_a_file_holding_only_non_pandas_nodes_has_no_keys(tmp_path: Path) -> None:
    path = tmp_path / "CollectedData_Sc.h5"
    with tables.open_file(path, "w") as h5:
        h5.create_array("/", "values", [1, 2, 3])

    with pytest.raises(UnreadableFileError, match="no pandas key"):
        read_labeled_data_h5(path)


def test_a_payload_pandas_cannot_rebuild_raises_from_the_original(tmp_path: Path) -> None:
    path = _write(tmp_path / "CollectedData_Sc.h5", "df_with_missing")
    with tables.open_file(path, "a") as h5:
        h5.get_node("/df_with_missing")._v_attrs.pandas_type = "not_a_pandas_type"

    with pytest.raises(UnreadableFileError, match="'df_with_missing'") as raised:
        read_labeled_data_h5(path)

    assert isinstance(raised.value.__cause__, TypeError)


def test_a_payload_missing_a_node_raises_from_the_original(tmp_path: Path) -> None:
    path = _write(tmp_path / "CollectedData_Sc.h5", "df_with_missing")
    with tables.open_file(path, "a") as h5:
        h5.remove_node("/df_with_missing", "block0_values")

    with pytest.raises(UnreadableFileError) as raised:
        read_labeled_data_h5(path)

    assert isinstance(raised.value.__cause__, tables.NoSuchNodeError)


def test_a_stored_series_is_not_a_dataframe(tmp_path: Path) -> None:
    path = tmp_path / "CollectedData_Sc.h5"
    pd.Series([1.0, 2.0]).to_hdf(path, key="df_with_missing", mode="w")

    with pytest.raises(UnreadableFileError, match="holds a Series"):
        read_labeled_data_h5(path)
