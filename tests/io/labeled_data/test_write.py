import os
import pickle
import stat
import warnings
from collections.abc import Iterator
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tables
from dlc_octopus.io import FileChangedError, FileStat
from dlc_octopus.io.labeled_data import LabeledData, read_labeled_data, write_labeled_data
from dlc_octopus.io.labeled_data.checks.file import frames_lost_to_csv
from dlc_octopus.io.labeled_data.wide.storage import csv_twin
from dlc_octopus.io.stamp import WRITER_KEY, WRITER_VERSION_KEY, WriterStamp

FIXTURES_DIR = Path(__file__).parents[1] / "fixtures" / "labeled_data"
FIXTURES = sorted(path.name for path in FIXTURES_DIR.glob("*") if path.suffix in {".h5", ".csv"})


def _table(path: Path) -> LabeledData:
    return read_labeled_data(path).labeled_data


def _labels_path(tmp_path: Path, table: LabeledData) -> Path:
    # Human labels never carry a likelihood, so labels that have one are written as machine labels.
    return tmp_path / ("machinelabels-iter0.h5" if table.has_likelihood else "CollectedData_Sc.h5")


def _without_likelihood(table: LabeledData) -> LabeledData:
    return LabeledData.from_long_df(table.to_long_df().drop(columns="likelihood"))


def _assert_same(first: LabeledData, second: LabeledData) -> None:
    a, b = first.to_arrays(), second.to_arrays()
    assert (a.frames, a.keypoints) == (b.frames, b.keypoints)
    np.testing.assert_array_equal(a.xy, b.xy)
    assert (a.likelihood is None) == (b.likelihood is None)
    if a.likelihood is not None and b.likelihood is not None:
        np.testing.assert_array_equal(a.likelihood, b.likelihood)


def test_every_fixture_is_found() -> None:
    # Control: an empty glob would let the round trip below pass by running nothing.
    assert len(FIXTURES) == 10


@pytest.mark.parametrize("filename", FIXTURES)
def test_every_fixture_is_written_and_read_back_unchanged(tmp_path: Path, filename: str) -> None:
    table = _table(FIXTURES_DIR / filename)
    path = _labels_path(tmp_path, table)

    write_labeled_data(table, path, replaces=None)

    _assert_same(_table(path), table)
    with pd.HDFStore(path, mode="r") as store:
        assert list(store.keys()) == ["/df_with_missing"]


@pytest.mark.parametrize("filename", FIXTURES)
def test_the_csv_twin_reads_back_as_the_same_table_within_float_formatting(tmp_path: Path, filename: str) -> None:
    table = _table(FIXTURES_DIR / filename)
    path = _labels_path(tmp_path, table)

    write_labeled_data(table, path, replaces=None)

    twin, arrays = _table(path.with_suffix(".csv")).to_arrays(), table.to_arrays()
    assert (twin.frames, twin.keypoints) == (arrays.frames, arrays.keypoints)
    np.testing.assert_allclose(twin.xy, arrays.xy, rtol=1e-12, equal_nan=True)


def _without_values(table: LabeledData, frame: int) -> LabeledData:
    long = table.to_long_df()
    long.loc[long["frame_index"] == frame, ["x", "y"]] = np.nan
    return LabeledData.from_long_df(long)


@pytest.mark.parametrize("empty", [None, 0, 1], ids=["none", "first", "second"])
def test_frames_lost_to_csv_are_the_frames_pandas_reads_back_without(tmp_path: Path, empty: int | None) -> None:
    table = _table(FIXTURES_DIR / "single_animal_human.h5")
    frame = (table if empty is None else _without_values(table, empty)).to_wide_df()
    frame.to_csv(tmp_path / "labels.csv")

    back = pd.read_csv(tmp_path / "labels.csv", index_col=[0, 1, 2], header=[0, 1, 2])

    assert list(frames_lost_to_csv(frame)) == [entry for entry in frame.index if entry not in back.index]


def test_a_first_frame_without_values_is_kept_in_the_h5_with_a_warning_that_the_csv_twin_loses_it(
    tmp_path: Path,
) -> None:
    table = _without_values(_table(FIXTURES_DIR / "single_animal_human.h5"), 0)
    path = tmp_path / "CollectedData_Sc.h5"

    with pytest.warns(UserWarning, match="reads back without it") as caught:
        write_labeled_data(table, path, replaces=None)

    assert _table(path).to_arrays().frames == table.to_arrays().frames
    assert [Path(warning.filename) for warning in caught] == [Path(__file__)]


def test_a_write_whose_csv_twin_keeps_every_frame_warns_nothing(tmp_path: Path) -> None:
    # Control for the warning above.
    table = _without_values(_table(FIXTURES_DIR / "single_animal_human.h5"), 1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write_labeled_data(table, tmp_path / "CollectedData_Sc.h5", replaces=None)

    assert [warning for warning in caught if "reads back without" in str(warning.message)] == []


@pytest.mark.xfail(
    reason="the CSV twin loses a first frame without values, as pandas and DeepLabCut read it as the row index's "
    "names; owed: a CSV form that keeps it, adopted with DeepLabCut and napari-deeplabcut",
    strict=True,
    raises=AssertionError,
)
def test_the_csv_twin_keeps_a_first_frame_without_values(tmp_path: Path) -> None:
    table = _without_values(_table(FIXTURES_DIR / "single_animal_human.h5"), 0)
    path = tmp_path / "CollectedData_Sc.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        write_labeled_data(table, path, replaces=None)

    assert _table(path.with_suffix(".csv")).to_arrays().frames == table.to_arrays().frames


def test_a_written_file_carries_the_writer_stamp_and_a_fixture_does_not(tmp_path: Path) -> None:
    path = tmp_path / "CollectedData_Sc.h5"
    write_labeled_data(_table(FIXTURES_DIR / "single_animal_human.h5"), path, replaces=None)

    stored = read_labeled_data(path)
    fixture = read_labeled_data(FIXTURES_DIR / "single_animal_human.h5")

    assert stored.writer_stamp == WriterStamp("dlc_octopus.io.labeled_data", version("dlc-octopus"))
    assert fixture.writer_stamp is None
    assert (stored.writer_stamp_error, fixture.writer_stamp_error) == (None, None)


def test_the_stamp_is_stored_as_plain_strings_never_as_a_pickle(tmp_path: Path) -> None:
    path = tmp_path / "CollectedData_Sc.h5"
    write_labeled_data(_table(FIXTURES_DIR / "single_animal_human.h5"), path, replaces=None)
    stamp = WriterStamp.current("dlc_octopus.io.labeled_data")

    raw = path.read_bytes()
    with tables.open_file(str(path)) as h5:
        attributes = h5.get_node("/df_with_missing")._v_attrs
        stored = {name: attributes[name] for name in stamp.as_metadata()}

    assert all(type(value) is np.str_ for value in stored.values())
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        for value in (stamp.as_metadata(), *stamp.as_metadata().values()):
            assert pickle.dumps(value, protocol=protocol) not in raw


@pytest.mark.parametrize("damage", ["wrong_type", "one_key_missing"])
def test_a_file_with_a_malformed_stamp_is_read_and_the_stamp_reported(tmp_path: Path, damage: str) -> None:
    path = tmp_path / "CollectedData_Sc.h5"
    written = _table(FIXTURES_DIR / "single_animal_human.h5")
    write_labeled_data(written, path, replaces=None)
    with tables.open_file(str(path), mode="a") as h5:
        attributes = h5.get_node("/df_with_missing")._v_attrs
        if damage == "wrong_type":
            attributes[WRITER_VERSION_KEY] = 1
        else:
            del attributes[WRITER_KEY]

    stored = read_labeled_data(path)

    assert stored.writer_stamp is None
    assert stored.writer_stamp_error is not None
    assert str(path) in stored.writer_stamp_error
    assert "malformed writer stamp" in stored.writer_stamp_error
    _assert_same(_table(path), written)


def test_the_stamp_is_invisible_to_a_plain_pandas_read(tmp_path: Path) -> None:
    table = _table(FIXTURES_DIR / "multi_animal_machine.h5")
    path = tmp_path / "machinelabels-iter0.h5"

    write_labeled_data(table, path, replaces=None)

    arrays = table.to_arrays()
    assert pd.read_hdf(path).shape == (len(arrays.frames), len(arrays.keypoints) * 3)


def test_a_file_that_exists_is_not_replaced_without_its_stat(tmp_path: Path) -> None:
    path = tmp_path / "CollectedData_Sc.h5"
    path.write_bytes(b"someone else's file")

    with pytest.raises(FileExistsError):
        write_labeled_data(_table(FIXTURES_DIR / "single_animal_human.h5"), path, replaces=None)

    assert path.read_bytes() == b"someone else's file"


def test_a_file_changed_since_it_was_read_is_not_replaced(tmp_path: Path) -> None:
    path = tmp_path / "CollectedData_Sc.h5"
    write_labeled_data(_table(FIXTURES_DIR / "single_animal_human.h5"), path, replaces=None)
    labels = read_labeled_data(path)
    with path.open("ab") as stream:
        stream.write(b"\0")  # someone else's change after our read
    changed = path.read_bytes()

    with pytest.raises(FileChangedError, match=path.name):
        write_labeled_data(_table(FIXTURES_DIR / "multi_animal_human.h5"), path, replaces=labels.stat)

    assert path.read_bytes() == changed
    assert sorted(p.name for p in tmp_path.iterdir()) == ["CollectedData_Sc.csv", "CollectedData_Sc.h5"]


def test_a_file_removed_since_it_was_read_is_not_recreated(tmp_path: Path) -> None:
    path = tmp_path / "CollectedData_Sc.h5"
    write_labeled_data(_table(FIXTURES_DIR / "single_animal_human.h5"), path, replaces=None)
    stat = read_labeled_data(path).stat
    path.unlink()

    with pytest.raises(FileChangedError, match="removed"):
        write_labeled_data(_table(FIXTURES_DIR / "single_animal_human.h5"), path, replaces=stat)

    assert not path.exists()


def test_a_file_unchanged_since_it_was_read_is_replaced_and_the_new_stat_is_returned(tmp_path: Path) -> None:
    # Control for the two refusals above: with the right stat, the write goes through.
    path = tmp_path / "CollectedData_Sc.h5"
    write_labeled_data(_table(FIXTURES_DIR / "single_animal_human.h5"), path, replaces=None)
    labels = read_labeled_data(path)
    machine = _without_likelihood(_table(FIXTURES_DIR / "single_animal_machine.h5"))

    new_stat = write_labeled_data(machine, path, replaces=labels.stat)

    _assert_same(_table(path), machine)
    assert new_stat == FileStat.of(path)
    assert new_stat != labels.stat


def test_a_human_labels_file_is_never_written_with_a_likelihood(tmp_path: Path) -> None:
    machine = _table(FIXTURES_DIR / "single_animal_machine.h5")
    assert machine.has_likelihood

    with pytest.raises(ValueError, match="likelihood"):
        write_labeled_data(machine, tmp_path / "CollectedData_Sc.h5", replaces=None)

    assert list(tmp_path.iterdir()) == []


def test_a_human_labels_file_takes_labels_whose_likelihood_the_caller_dropped(tmp_path: Path) -> None:
    path = tmp_path / "CollectedData_Sc.h5"
    machine = _table(FIXTURES_DIR / "single_animal_machine.h5")

    write_labeled_data(_without_likelihood(machine), path, replaces=None)

    assert not _table(path).has_likelihood


def test_a_machine_labels_file_keeps_its_likelihood(tmp_path: Path) -> None:
    # Control for the refusal above: the rule is about human labels, not about likelihood.
    path = tmp_path / "machinelabels-iter0.h5"
    machine = _table(FIXTURES_DIR / "single_animal_machine.h5")

    write_labeled_data(machine, path, replaces=None)

    _assert_same(_table(path), machine)


def test_an_empty_individual_name_is_refused_and_touches_nothing(tmp_path: Path) -> None:
    wide = _table(FIXTURES_DIR / "multi_animal_human.h5").to_wide_df()
    unnamed = LabeledData.from_wide_df(wide.rename(columns={"animalA": ""}, level="individuals"))

    with pytest.raises(ValueError, match="empty individual"):
        write_labeled_data(unnamed, tmp_path / "CollectedData_Sc.h5", replaces=None)

    assert list(tmp_path.iterdir()) == []


def test_a_file_holding_several_keys_is_not_replaced_and_the_dropped_keys_are_named(tmp_path: Path) -> None:
    path = tmp_path / "CollectedData_Sc.h5"
    path.write_bytes((FIXTURES_DIR / "ambiguous_multi_key.h5").read_bytes())
    labels = read_labeled_data(path)
    before = path.read_bytes()

    with pytest.raises(ValueError, match=r"\['keypoints'\].*drop"):
        write_labeled_data(labels.labeled_data, path, replaces=labels.stat)

    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["CollectedData_Sc.h5"]


def test_a_file_holding_one_legacy_key_is_replaced_under_the_current_key(tmp_path: Path) -> None:
    # Control for the refusal above: a single key is converted, never dropped alongside another.
    path = tmp_path / "CollectedData_Sc.h5"
    pd.read_hdf(FIXTURES_DIR / "single_animal_human.h5").to_hdf(path, key="keypoints", mode="w")
    labels = read_labeled_data(path)

    write_labeled_data(labels.labeled_data, path, replaces=labels.stat)

    with pd.HDFStore(path, mode="r") as store:
        assert list(store.keys()) == ["/df_with_missing"]
    _assert_same(_table(path), labels.labeled_data)


@pytest.mark.parametrize(("name", "twin"), [("x.h5", "x.csv"), ("x.H5", "x.CSV")])
def test_the_csv_twin_takes_the_case_of_the_h5_suffix(name: str, twin: str) -> None:
    # Compare names: WindowsPath equality ignores case, so comparing paths could not fail on Windows.
    assert csv_twin(Path(name)).name == twin


def test_an_upper_case_h5_is_written_read_and_replaced_with_its_twin_in_the_same_case(tmp_path: Path) -> None:
    table = _table(FIXTURES_DIR / "single_animal_human.h5")
    path = tmp_path / "CollectedData_Sc.H5"

    write_labeled_data(table, path, replaces=None)
    write_labeled_data(table, path, replaces=read_labeled_data(path).stat)

    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["CollectedData_Sc.CSV", "CollectedData_Sc.H5"]
    _assert_same(_table(path), table)
    assert _table(tmp_path / "CollectedData_Sc.CSV").to_arrays().keypoints == table.to_arrays().keypoints


def test_only_an_h5_path_is_written(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"'\.csv'"):
        write_labeled_data(_table(FIXTURES_DIR / "single_animal_human.h5"), tmp_path / "x.csv", replaces=None)

    assert list(tmp_path.iterdir()) == []


def test_a_file_without_a_stat_is_read_but_cannot_be_replaced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "CollectedData_Sc.h5"
    write_labeled_data(_table(FIXTURES_DIR / "single_animal_human.h5"), path, replaces=None)
    before = path.read_bytes()

    def no_stat(_path: Path) -> FileStat:
        raise PermissionError("stat denied")

    with monkeypatch.context() as patch:
        patch.setattr(FileStat, "of", no_stat)
        labels = read_labeled_data(path)

    assert labels.stat is None
    assert labels.keypoints is not None
    with pytest.raises(FileExistsError):
        write_labeled_data(_table(FIXTURES_DIR / "multi_animal_human.h5"), path, replaces=labels.stat)
    assert path.read_bytes() == before


def test_files_beside_the_target_survive_a_write(tmp_path: Path) -> None:
    path = tmp_path / "CollectedData_Sc.h5"
    bystanders = [tmp_path / ".CollectedData_Sc.tmp.h5", tmp_path / ".CollectedData_Sc.tmp.csv"]
    for bystander in bystanders:
        bystander.write_bytes(b"not ours")

    write_labeled_data(_table(FIXTURES_DIR / "single_animal_human.h5"), path, replaces=None)

    assert [bystander.read_bytes() for bystander in bystanders] == [b"not ours", b"not ours"]
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        ["CollectedData_Sc.h5", "CollectedData_Sc.csv", *(bystander.name for bystander in bystanders)]
    )


def test_reading_records_the_stat_taken_before_the_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dlc_octopus.io.labeled_data.wide import storage

    path = tmp_path / "CollectedData_Sc.h5"
    path.write_bytes((FIXTURES_DIR / "single_animal_human.h5").read_bytes())
    before = FileStat.of(path)
    read = storage.read_wide_labeled_data

    def read_then_change(source: Path) -> storage.WideLabeledDataStorage:
        stored = read(source)
        with source.open("ab") as stream:
            stream.write(b"\0")
        return stored

    monkeypatch.setattr(storage, "read_wide_labeled_data", read_then_change)

    assert read_labeled_data(path).stat == before
    assert FileStat.of(path) != before


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.fixture
def group_umask() -> Iterator[None]:
    previous = os.umask(0o002)
    yield
    os.umask(previous)


@pytest.mark.skipif(os.name != "posix", reason="permission bits are POSIX")
@pytest.mark.usefixtures("group_umask")
def test_a_new_file_and_its_twin_get_the_umask_default(tmp_path: Path) -> None:
    path = tmp_path / "CollectedData_Sc.h5"

    write_labeled_data(_table(FIXTURES_DIR / "single_animal_human.h5"), path, replaces=None)

    assert (_mode(path), _mode(csv_twin(path))) == (0o664, 0o664)


@pytest.mark.skipif(os.name != "posix", reason="permission bits are POSIX")
@pytest.mark.usefixtures("group_umask")
def test_a_replaced_file_and_its_twin_keep_their_modes(tmp_path: Path) -> None:
    path = tmp_path / "CollectedData_Sc.h5"
    write_labeled_data(_table(FIXTURES_DIR / "single_animal_human.h5"), path, replaces=None)
    path.chmod(0o640)
    csv_twin(path).chmod(0o604)

    write_labeled_data(_table(FIXTURES_DIR / "multi_animal_human.h5"), path, replaces=read_labeled_data(path).stat)

    assert (_mode(path), _mode(csv_twin(path))) == (0o640, 0o604)
