from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dlc_octopus.io.labeled_data.wide.h5 import read_labeled_data_h5
from dlc_octopus.io.labeled_data.wide.row_index import (
    RowIndexShape,
    classify_labeled_data_row_index,
    frame_part_as_text,
    split_frame_path,
)
from dlc_octopus.io.labeled_data.wide.table import classify_labeled_data_read

FIXTURES_DIR = Path(__file__).parents[2] / "fixtures" / "labeled_data"
FIXTURES = sorted(path.name for path in FIXTURES_DIR.glob("*.h5"))


@pytest.mark.parametrize("filename", FIXTURES)
def test_every_fixture_row_index_is_kept_and_split(filename: str) -> None:
    frame = read_labeled_data_h5(FIXTURES_DIR / filename).dataframe

    _header, row_index = classify_labeled_data_read(read_labeled_data_h5(FIXTURES_DIR / filename))

    assert (row_index.names, row_index.entries) == (tuple(frame.index.names), tuple(frame.index))
    assert all(len(parts) == 3 for parts in row_index.frames)
    assert len(row_index.frames) == len(frame.index)


def test_the_legacy_flat_index_is_flat_paths() -> None:
    _header, row_index = classify_labeled_data_read(
        read_labeled_data_h5(FIXTURES_DIR / "single_animal_human_legacy_index.h5")
    )

    assert row_index.shape is RowIndexShape.FLAT_PATHS


def test_both_separators_split_alike() -> None:
    row_index = classify_labeled_data_row_index(
        [None], [r"labeled-data\vid1\img1.png", "labeled-data/vid1/img2.png"], is_multi_index=False
    )

    assert row_index.frames == (("labeled-data", "vid1", "img1.png"), ("labeled-data", "vid1", "img2.png"))


def test_a_ragged_index_stays_ragged() -> None:
    row_index = classify_labeled_data_row_index(
        [None], ["labeled-data/vid1/img1.png", "img2.png"], is_multi_index=False
    )

    assert row_index.frames == (("labeled-data", "vid1", "img1.png"), ("img2.png",))


def test_a_numeric_index_is_unrecognised_and_kept() -> None:
    row_index = classify_labeled_data_row_index([None], [0, 1], is_multi_index=False)

    assert (row_index.shape, row_index.entries, row_index.frames) == (RowIndexShape.UNRECOGNISED, (0, 1), ())


@pytest.mark.parametrize(
    "index",
    [pd.RangeIndex(0), pd.Index([], dtype=object), pd.MultiIndex.from_tuples([], names=[None, None, None])],
    ids=["range", "object", "multi"],
)
def test_an_index_without_entries_is_empty_whatever_its_type(index: pd.Index) -> None:
    row_index = classify_labeled_data_row_index(index.names, index, is_multi_index=isinstance(index, pd.MultiIndex))

    assert (row_index.shape, row_index.entries, row_index.frames) == (RowIndexShape.EMPTY, (), ())


def test_a_numeric_folder_is_not_stringified() -> None:
    index = pd.MultiIndex.from_tuples([("labeled-data", 2019, "img1.png")])

    row_index = classify_labeled_data_row_index(index.names, index, is_multi_index=True)

    assert row_index.frames == (("labeled-data", 2019, "img1.png"),)


def test_level_names_are_kept_as_stored(tmp_path: Path) -> None:
    # DeepLabCut's convertcsv2h5 stores a CSV's first frame without values as these names.
    columns = pd.MultiIndex.from_product([["Sc"], ["nose"], ["x", "y"]], names=["scorer", "bodyparts", "coords"])
    index = pd.MultiIndex.from_tuples([("labeled-data", "s1", "img1.png")], names=["labeled-data", "s1", "img0.png"])
    pd.DataFrame([[1.0, 2.0]], index=index, columns=columns).to_hdf(tmp_path / "labels.h5", key="df_with_missing")

    _header, row_index = classify_labeled_data_read(read_labeled_data_h5(tmp_path / "labels.h5"))

    assert row_index.names == ("labeled-data", "s1", "img0.png")
    assert row_index.frames == (("labeled-data", "s1", "img1.png"),)


def test_split_frame_path_keeps_empty_components() -> None:
    assert split_frame_path("labeled-data//img1.png") == ("labeled-data", "", "img1.png")


@pytest.mark.parametrize(
    ("part", "text"),
    [("s1", "s1"), ("007", "007"), (2019, "2019"), (np.int64(2019), "2019")],
    ids=["text", "zero_padded_text", "int", "numpy_int"],
)
def test_a_frame_part_with_a_name_is_read_as_that_text(part: object, text: str) -> None:
    assert frame_part_as_text(part) == text


@pytest.mark.parametrize("part", [2019.0, True, np.True_, None], ids=["float", "bool", "numpy_bool", "none"])
def test_a_frame_part_naming_no_folder_has_no_text(part: object) -> None:
    # A float or bool would become a folder name that never existed.
    assert frame_part_as_text(part) is None
