from pathlib import Path

import pandas as pd
import pytest

from dlc_octopus.io.errors import UnreadableFileError
from dlc_octopus.io.labeled_data.wide.csv import read_labeled_data_csv
from dlc_octopus.io.labeled_data.wide.h5 import read_labeled_data_h5
from dlc_octopus.io.labeled_data.wide.header import LabeledDataHeaderShape
from dlc_octopus.io.labeled_data.wide.table import classify_labeled_data_read

FIXTURES_DIR = Path(__file__).parents[2] / "fixtures" / "labeled_data"
CSV_FIXTURES = sorted(path.name for path in FIXTURES_DIR.glob("*.csv"))


def test_the_csv_fixtures_are_all_found_with_their_h5_twins() -> None:
    # Control: an empty glob would let the parametrized parity tests pass by running nothing.
    assert len(CSV_FIXTURES) == 2
    assert all((FIXTURES_DIR / name).with_suffix(".h5").exists() for name in CSV_FIXTURES)


@pytest.mark.parametrize("filename", CSV_FIXTURES)
def test_a_csv_reads_and_classifies_as_its_h5_twin(filename: str) -> None:
    csv_read = read_labeled_data_csv(FIXTURES_DIR / filename)
    h5_read = read_labeled_data_h5((FIXTURES_DIR / filename).with_suffix(".h5"))

    pd.testing.assert_frame_equal(csv_read.dataframe, h5_read.dataframe)
    assert classify_labeled_data_read(csv_read) == classify_labeled_data_read(h5_read)


def test_the_parity_check_sees_a_changed_header(tmp_path: Path) -> None:
    # Control: classifications that ignored the header would compare equal above whatever the file held.
    path = tmp_path / "CollectedData_Sc.csv"
    path.write_text((FIXTURES_DIR / "multi_animal_unique_bodyparts.csv").read_text().replace("feeder", "lever"))
    h5_read = read_labeled_data_h5(FIXTURES_DIR / "multi_animal_unique_bodyparts.h5")

    assert classify_labeled_data_read(read_labeled_data_csv(path)) != classify_labeled_data_read(h5_read)


@pytest.mark.parametrize(
    ("filename", "header_rows", "index_columns"),
    [
        ("multi_animal_unique_bodyparts.csv", 4, 3),
        ("single_animal_legacy_index_dlc.csv", 3, 1),
    ],
)
def test_the_layout_is_read_from_the_file(filename: str, header_rows: int, index_columns: int) -> None:
    frame = read_labeled_data_csv(FIXTURES_DIR / filename).dataframe

    assert (frame.columns.nlevels, frame.index.nlevels) == (header_rows, index_columns)


def test_an_unrecognised_header_is_read_without_one(tmp_path: Path) -> None:
    path = tmp_path / "CollectedData_Sc.csv"
    path.write_text("frame,nose_x,nose_y\nimg0.png,1.0,2.0\n")

    read = read_labeled_data_csv(path)
    header, _row_index = classify_labeled_data_read(read)

    assert read.key_selection is None
    assert header.shape is LabeledDataHeaderShape.UNRECOGNISED
    assert len(read.dataframe) == 2


@pytest.mark.parametrize(
    ("label", "content", "cause"),
    [("missing", None, FileNotFoundError), ("not_utf8", b"\xff\xfe\x00scorer", UnicodeDecodeError)],
    ids=str,
)
def test_an_unreadable_csv_raises_from_the_original(
    tmp_path: Path, label: str, content: bytes | None, cause: type[Exception]
) -> None:
    path = tmp_path / f"{label}.csv"
    if content is not None:
        path.write_bytes(content)

    with pytest.raises(UnreadableFileError, match=label) as raised:
        read_labeled_data_csv(path)

    assert isinstance(raised.value.__cause__, cause)


@pytest.mark.parametrize("index_names", [None, ["root", "folder", "image"]], ids=["unnamed", "named"])
def test_numeric_looking_path_parts_read_as_text_as_in_h5(tmp_path: Path, index_names: list[str] | None) -> None:
    # A named index adds a row of names below the header, which the text read must skip.
    columns = pd.MultiIndex.from_tuples([("Sc", "nose", "x")], names=["scorer", "bodyparts", "coords"])
    index = pd.MultiIndex.from_tuples(
        [("labeled-data", "2019", "img0.png"), ("labeled-data", "007", "img1.png")], names=index_names
    )
    frame = pd.DataFrame([[1.0], [2.0]], index=index, columns=columns)
    frame.to_hdf(tmp_path / "CollectedData_Sc.h5", key="df_with_missing", mode="w")
    frame.to_csv(tmp_path / "CollectedData_Sc.csv")

    h5_read = read_labeled_data_h5(tmp_path / "CollectedData_Sc.h5")
    csv_read = read_labeled_data_csv(tmp_path / "CollectedData_Sc.csv")
    _h5_header, h5_row_index = classify_labeled_data_read(h5_read)
    _csv_header, csv_row_index = classify_labeled_data_read(csv_read)

    pd.testing.assert_frame_equal(csv_read.dataframe, h5_read.dataframe)
    assert h5_row_index.frames == tuple(index)
    assert csv_row_index.frames == tuple(index)
