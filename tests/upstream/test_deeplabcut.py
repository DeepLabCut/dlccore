"""Tests whose home is DeepLabCut. Imports ``deeplabcut`` only."""

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import pandas as pd

    from dlc_octopus.io.labeled_data.wide.row_index import LabeledDataRowIndex

pytestmark = pytest.mark.destination("deeplabcut")

LABELED_DATA_FIXTURES = Path(__file__).parents[1] / "io" / "fixtures" / "labeled_data"
_FIXTURE_SCORER = "Sc"

_SUBSTRING_FILTER = (
    "labeled-data folders are filtered with the substring '_labeled', "
    "so a legitimately named dataset folder is invisible to DeepLabCut's own enumeration"
)


def _configured_videos(project_root: Path) -> dict[str, dict]:
    """Return ``video_sets`` as DeepLabCut would read it back from disk."""

    utils = pytest.importorskip("deeplabcut.core.config.utils")

    return utils.read_config_as_dict(project_root / "config.yaml")["video_sets"]


def _declare_videos(project_root: Path, stems: tuple[str, ...]) -> None:
    """Point ``video_sets`` at ``videos/<stem>.mp4`` without creating the files.

    The files are deliberately absent. Reconciliation re-adds a folder that has
    no config entry by opening the matching video, and an empty ``.mp4`` raises
    from ``VideoReader`` — which would make a test fail for a reason that has
    nothing to do with what it is pinning.
    """

    utils = pytest.importorskip("deeplabcut.core.config.utils")

    config_path = project_root / "config.yaml"
    config = utils.read_config_as_dict(config_path)

    config["video_sets"] = {str(project_root / "videos" / f"{stem}.mp4"): {"crop": "0, 100, 0, 100"} for stem in stems}

    utils.write_config(config_path, config)


def _add_dataset_folder(project_root: Path, name: str) -> Path:
    folder = project_root / "labeled-data" / name
    folder.mkdir(parents=True, exist_ok=True)

    return folder


def test_normally_named_folder_keeps_its_video_entry(valid_project: Path) -> None:
    """Control: the reconciliation runs and leaves a matched pair alone.

    Without this, a test asserting that some entry survives cannot distinguish
    "the filter is correct" from "the function never did anything".
    """

    trainingsetmanipulation = pytest.importorskip("deeplabcut.generate_training_dataset.trainingsetmanipulation")

    _declare_videos(valid_project, ("session-01",))
    _add_dataset_folder(valid_project, "session-01")

    trainingsetmanipulation.adddatasetstovideolistandviceversa(str(valid_project / "config.yaml"))

    assert [Path(video).stem for video in _configured_videos(valid_project)] == ["session-01"]


@pytest.mark.xfail(reason=_SUBSTRING_FILTER, strict=True, raises=AssertionError)
def test_a_video_named_labeled_keeps_its_config_entry(valid_project: Path) -> None:
    """A user's video may legitimately be called ``trial_labeled_v2``.

    Its dataset folder is filtered out as if it were ``check_labels`` output,
    the stem is then reported missing, and the entry is deleted from
    ``config.yaml`` on disk.
    """

    trainingsetmanipulation = pytest.importorskip("deeplabcut.generate_training_dataset.trainingsetmanipulation")

    _declare_videos(valid_project, ("trial_labeled_v2",))
    _add_dataset_folder(valid_project, "trial_labeled_v2")

    trainingsetmanipulation.adddatasetstovideolistandviceversa(str(valid_project / "config.yaml"))

    assert [Path(video).stem for video in _configured_videos(valid_project)] == ["trial_labeled_v2"]


@pytest.mark.xfail(
    reason="the removal loop matches with 'in', so dropping Sample1 also drops Sample10",
    strict=True,
    raises=AssertionError,
)
def test_dropping_one_stem_does_not_drop_names_containing_it(valid_project: Path) -> None:
    """``Sample1`` has no folder; ``Sample10`` has one and must survive.

    ``horse10`` ships ``Sample1`` through ``Sample20``, so this is the shape of
    a real project rather than a contrived name.
    """

    trainingsetmanipulation = pytest.importorskip("deeplabcut.generate_training_dataset.trainingsetmanipulation")

    _declare_videos(valid_project, ("Sample1", "Sample10"))
    _add_dataset_folder(valid_project, "Sample10")

    trainingsetmanipulation.adddatasetstovideolistandviceversa(str(valid_project / "config.yaml"))

    assert "Sample10" in [Path(video).stem for video in _configured_videos(valid_project)]


@pytest.mark.xfail(reason=_SUBSTRING_FILTER, strict=True, raises=AssertionError)
def test_comparison_does_not_report_a_labeled_named_folder_as_missing(
    valid_project: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The read-only report should agree that the folder is present."""

    trainingsetmanipulation = pytest.importorskip("deeplabcut.generate_training_dataset.trainingsetmanipulation")

    _declare_videos(valid_project, ("trial_labeled_v2",))
    _add_dataset_folder(valid_project, "trial_labeled_v2")

    trainingsetmanipulation.comparevideolistsanddatafolders(str(valid_project / "config.yaml"))

    assert "is missing as a folder" not in capsys.readouterr().out


@pytest.mark.xfail(
    reason="labeled-data is enumerated with an unfiltered iterdir(), "
    "so any directory holding a CollectedData file is merged whatever its name",
    strict=True,
    raises=AssertionError,
)
def test_a_directory_the_config_does_not_name_is_not_merged(valid_project: Path) -> None:
    """A copy of a dataset folder is merged as though it were a second dataset.

    ``merge_windowsannotationdataONlinuxsystem`` lists ``labeled-data/`` with
    ``d.is_dir()`` and no filter, so a directory belongs to the training set by
    holding a ``CollectedData`` file rather than by being named in the config.

    Reached from ``create_training_dataset`` through ``merge_annotateddatasets``'
    empty-result branch.

    ``check_labels`` output is not an instance of this: ``<stem>_labeled/`` holds
    rendered images only, so the loop skips it.
    """

    pd = pytest.importorskip("pandas")
    conversioncode = pytest.importorskip("deeplabcut.utils.conversioncode")
    utils = pytest.importorskip("deeplabcut.core.config.utils")

    config = utils.read_config_as_dict(valid_project / "config.yaml")
    scorer = config["scorer"]

    def write_collected_data(folder: Path, frame: str) -> None:
        columns = pd.MultiIndex.from_tuples(
            [(scorer, "nose", "x"), (scorer, "nose", "y")],
            names=["scorer", "bodyparts", "coords"],
        )
        frame_index = [("labeled-data", folder.name, frame)]
        pd.DataFrame([[1.0, 2.0]], columns=columns, index=pd.MultiIndex.from_tuples(frame_index)).to_hdf(
            folder / f"CollectedData_{scorer}.h5",
            key="df_with_missing",
            mode="w",
        )

    dataset = _add_dataset_folder(valid_project, "session-01")
    copy = _add_dataset_folder(valid_project, "session-01_backup")

    write_collected_data(dataset, "img001.png")
    write_collected_data(copy, "img001.png")

    merged = conversioncode.merge_windowsannotationdataONlinuxsystem(config)

    merged_folders = {entry[1] for entry in pd.concat(merged).index}

    assert merged_folders == {"session-01"}


@pytest.mark.xfail(
    reason="read_config writes project_path as the config path's parent as given, so 'config.yaml' writes '.'",
    strict=True,
    raises=AssertionError,
)
def test_reading_a_moved_project_by_a_relative_path_writes_an_absolute_project_path(
    valid_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public calls such as ``check_labels("config.yaml")`` from inside a moved project write ``project_path: .``."""

    utils = pytest.importorskip("deeplabcut.core.config.utils")

    config_path = valid_project / "config.yaml"
    config = utils.read_config_as_dict(config_path)
    config["project_path"] = str(valid_project.parent / "moved-from")
    utils.write_config(config_path, config)

    monkeypatch.chdir(valid_project)
    utils.read_config("config.yaml")

    assert Path(utils.read_config_as_dict(config_path)["project_path"]) == valid_project.absolute()


@pytest.mark.parametrize(
    "entries",
    [
        ["labeled-data/session-01/img001.png", "labeled-data/session-01/img002.png"],
        [r"labeled-data\session-01\img001.png", r"labeled-data\session-01\img002.png"],
    ],
    ids=["posix", "windows"],
)
def test_split_frame_path_matches_guarantee_multiindex_rows(entries: list[str]) -> None:
    """The one shared definition of a frame path agrees with DeepLabCut's on string paths of equal depth.

    DeepLabCut pads a ragged index with ``NaN`` and stringifies a numeric folder; ``split_frame_path`` does
    neither, so those cases are deliberately left out.
    """

    pd = pytest.importorskip("pandas")
    conversioncode = pytest.importorskip("deeplabcut.utils.conversioncode")

    from dlc_octopus.io.labeled_data.wide.row_index import split_frame_path

    frame = pd.DataFrame({"x": range(len(entries))}, index=entries)
    conversioncode.guarantee_multiindex_rows(frame)

    assert list(frame.index) == [split_frame_path(entry) for entry in entries]


def _fixture_dataset(project_root: Path, **config_fields: object) -> Path:
    """Name ``session-01`` in ``video_sets`` under the fixtures' scorer, and return its dataset folder."""

    utils = pytest.importorskip("deeplabcut.core.config.utils")

    _declare_videos(project_root, ("session-01",))
    config_path = project_root / "config.yaml"
    config = utils.read_config_as_dict(config_path)
    config.update(scorer=_FIXTURE_SCORER, **config_fields)
    utils.write_config(config_path, config)

    return _add_dataset_folder(project_root, "session-01")


def _convertcsv2h5(project_root: Path, csv_source: Path) -> "pd.DataFrame":
    """Return the table ``convertcsv2h5`` writes for ``csv_source``."""

    pd = pytest.importorskip("pandas")
    conversioncode = pytest.importorskip("deeplabcut.utils.conversioncode")

    collected_data = _fixture_dataset(project_root) / f"CollectedData_{_FIXTURE_SCORER}.csv"
    shutil.copyfile(csv_source, collected_data)
    conversioncode.convertcsv2h5(str(project_root / "config.yaml"), userfeedback=False)

    return pd.read_hdf(collected_data.with_suffix(".h5"))


def _read_and_classify_csv(path: Path) -> tuple["pd.DataFrame", "LabeledDataRowIndex"]:
    from dlc_octopus.io.labeled_data.wide.csv import read_labeled_data_csv
    from dlc_octopus.io.labeled_data.wide.table import classify_labeled_data_read

    read = read_labeled_data_csv(path)
    _header, row_index = classify_labeled_data_read(read)

    return read.dataframe, row_index


@pytest.mark.parametrize("filename", sorted(path.name for path in LABELED_DATA_FIXTURES.glob("*.csv")))
def test_read_labeled_data_csv_matches_convertcsv2h5(valid_project: Path, filename: str) -> None:
    """Same columns and values, and each row's path parts as DeepLabCut stores them; a flat index stays flat in ours."""

    pd = pytest.importorskip("pandas")

    converted = _convertcsv2h5(valid_project, LABELED_DATA_FIXTURES / filename)
    dataframe, row_index = _read_and_classify_csv(LABELED_DATA_FIXTURES / filename)

    pd.testing.assert_frame_equal(dataframe.reset_index(drop=True), converted.reset_index(drop=True))
    assert list(row_index.frames) == list(converted.index)


def test_read_labeled_data_csv_keeps_a_numeric_folder_as_text_like_convertcsv2h5(valid_project: Path) -> None:
    pd = pytest.importorskip("pandas")

    columns = pd.MultiIndex.from_tuples([(_FIXTURE_SCORER, "nose", "x")], names=["scorer", "bodyparts", "coords"])
    index = pd.MultiIndex.from_tuples([("labeled-data", "2019", "img0.png")])
    source = valid_project / "source.csv"
    pd.DataFrame([[1.0]], index=index, columns=columns).to_csv(source)

    converted = _convertcsv2h5(valid_project, source)
    _dataframe, row_index = _read_and_classify_csv(source)

    assert list(row_index.frames) == list(converted.index)


_MULTI_KEY_FIXTURE = "ambiguous_multi_key.h5"
_SINGLE_KEY_H5_FIXTURES = sorted(
    path.name for path in LABELED_DATA_FIXTURES.glob("*.h5") if path.name != _MULTI_KEY_FIXTURE
)


def _deeplabcut_read(path: Path) -> "pd.DataFrame":
    """Read a labels file as ``merge_annotateddatasets`` does: the file's only key, rows as path parts."""

    pd = pytest.importorskip("pandas")
    conversioncode = pytest.importorskip("deeplabcut.utils.conversioncode")

    frame = pd.read_hdf(path)
    conversioncode.guarantee_multiindex_rows(frame)
    return frame


def _rewrite(source: Path, tmp_path: Path, *, move_one_point: bool = False) -> Path:
    import numpy as np

    from dlc_octopus.io.labeled_data import LabeledData, read_labeled_data, write_labeled_data

    table = read_labeled_data(source).labeled_data
    if move_one_point:
        long = table.to_long_df()
        long.loc[np.flatnonzero(long["x"].notna())[0], "x"] += 1.0
        table = LabeledData.from_long_df(long)

    # Human labels never carry a likelihood, so labels that have one are rewritten as machine labels.
    human = f"CollectedData_{_FIXTURE_SCORER}.h5"
    target = tmp_path / ("machinelabels-iter0.h5" if table.has_likelihood else human)
    write_labeled_data(table, target, replaces=None)
    return target


def test_every_single_key_h5_fixture_is_found() -> None:
    # Control: an empty glob would let the rewrite test pass by running nothing.
    assert len(_SINGLE_KEY_H5_FIXTURES) == 7


@pytest.mark.parametrize("filename", _SINGLE_KEY_H5_FIXTURES)
def test_deeplabcut_reads_a_rewritten_labels_file_as_the_original(tmp_path: Path, filename: str) -> None:
    pd = pytest.importorskip("pandas")
    source = LABELED_DATA_FIXTURES / filename

    pd.testing.assert_frame_equal(
        _deeplabcut_read(_rewrite(source, tmp_path)), _deeplabcut_read(source), check_exact=True
    )


def test_deeplabcut_sees_one_moved_point_in_a_rewritten_labels_file(tmp_path: Path) -> None:
    # Control: the comparison above can fail.
    pd = pytest.importorskip("pandas")
    source = LABELED_DATA_FIXTURES / "single_animal_human.h5"

    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(
            _deeplabcut_read(_rewrite(source, tmp_path, move_one_point=True)),
            _deeplabcut_read(source),
            check_exact=True,
        )


def test_deeplabcut_reads_the_rewrite_of_a_multi_key_labels_file_it_cannot_read(tmp_path: Path) -> None:
    source = LABELED_DATA_FIXTURES / _MULTI_KEY_FIXTURE

    with pytest.raises(ValueError, match="key must be provided"):
        _deeplabcut_read(source)
    assert not _deeplabcut_read(_rewrite(source, tmp_path)).empty


@pytest.mark.xfail(
    reason="convertcsv2h5 counts header rows by finding 'individuals' anywhere in the second line, "
    "so a single-animal bodypart containing it is read as a fourth header row",
    strict=True,
    raises=AssertionError,
)
def test_convertcsv2h5_reads_a_bodypart_containing_individuals_as_single_animal(valid_project: Path) -> None:
    pd = pytest.importorskip("pandas")

    columns = pd.MultiIndex.from_product(
        [[_FIXTURE_SCORER], ["nose", "individuals_marker"], ["x", "y"]], names=["scorer", "bodyparts", "coords"]
    )
    index = pd.MultiIndex.from_tuples([("labeled-data", "session-01", f"img{frame}.png") for frame in range(3)])
    source = valid_project / "source.csv"
    pd.DataFrame([[0.0, 1.0, 2.0, 3.0]] * 3, index=index, columns=columns).to_csv(source)

    assert list(_convertcsv2h5(valid_project, source).columns) == list(columns)
