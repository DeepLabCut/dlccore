import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tables

from dlc_octopus.api import findings_in_scope, inspect_project
from dlc_octopus.content import ProjectLabeledData, read_project_labeled_data
from dlc_octopus.io.labeled_data import LabeledData, read_labeled_data, write_labeled_data
from dlc_octopus.io.stamp import WRITER_VERSION_KEY
from dlc_octopus.schemas.base.finding_codes import LabeledDataFindingCode, Severity
from dlc_octopus.schemas.base.findings import Reach
from dlc_octopus.schemas.project_layout.labeled_data import HUMAN_LABELS, LABELED_DATASET
from dlc_octopus.schemas.project_layout.videos import VIDEOS
from dlc_octopus.schemas.scope import Scope
from dlc_octopus.schemas.snapshots import ProjectSnapshot

FIXTURES_DIR = Path(__file__).parents[1] / "io" / "fixtures" / "labeled_data"
UNREADABLE = LabeledDataFindingCode.LABELED_DATA_UNREADABLE
HUMAN_H5 = "CollectedData_test-scorer.h5"
HUMAN_CSV = "CollectedData_test-scorer.csv"
MACHINE_H5 = "machinelabels-iter0.h5"


def _session(project: Path, name: str = "session-01") -> Path:
    folder = project / "labeled-data" / name
    folder.mkdir()
    shutil.copy(FIXTURES_DIR / "single_animal_legacy_index_dlc.h5", folder / HUMAN_H5)
    shutil.copy(FIXTURES_DIR / "single_animal_legacy_index_dlc.csv", folder / HUMAN_CSV)
    shutil.copy(FIXTURES_DIR / "single_animal_machine.h5", folder / MACHINE_H5)
    (folder / "img000.png").write_bytes(b"not read: a frame, not labels")
    return folder


def _uninterpretable(path: Path) -> None:
    frame = pd.DataFrame([[1.0, 2.0]], columns=["nose_x", "nose_y"], index=["labeled-data/session-01/img000.png"])
    frame.to_hdf(path, key="df_with_missing", mode="w")


def _names(result: ProjectLabeledData) -> tuple[list[str], list[str]]:
    return (
        sorted(entry.project_relative_path.name for entry in result.read),
        sorted(failure.project_relative_path.name for failure in result.failed),
    )


def test_every_present_labels_file_in_scope_is_read_and_nothing_else(valid_project: Path) -> None:
    _session(valid_project)
    snapshot = inspect_project(valid_project)

    result = read_project_labeled_data(snapshot, Scope(LABELED_DATASET))

    assert sorted((entry.schema_id, entry.project_relative_path.name) for entry in result.read) == [
        ("dlc.labeled_data.human_labels", HUMAN_CSV),
        ("dlc.labeled_data.human_labels", HUMAN_H5),
        ("dlc.labeled_data.machine_labels", MACHINE_H5),
    ]
    assert result.failed == ()
    assert result.snapshot == snapshot
    for entry in result.read:
        direct = read_labeled_data(valid_project / entry.project_relative_path)
        assert (entry.data.keypoints, entry.data.frames) == (direct.keypoints, direct.frames)


def test_an_unreadable_file_is_one_failed_read_with_a_merged_finding_and_the_rest_are_read(
    valid_project: Path,
) -> None:
    (_session(valid_project) / HUMAN_H5).write_bytes(b"not an hdf5 file")
    snapshot = inspect_project(valid_project)

    result = read_project_labeled_data(snapshot, Scope(LABELED_DATASET))

    assert _names(result) == ([HUMAN_CSV, MACHINE_H5], [HUMAN_H5])
    (failure,) = result.failed
    (finding,) = failure.findings
    assert failure.project_relative_path == Path("labeled-data", "session-01", HUMAN_H5)
    assert (finding.code, finding.severity, finding.reach, finding.instance) == (
        UNREADABLE,
        Severity.ERROR,
        Reach.ONE_OF,
        ("labeled-data", "session-01", HUMAN_H5),
    )
    assert finding.path == valid_project / "labeled-data" / "session-01" / HUMAN_H5
    assert finding.evidence["resource_kind"] == "human_labels"
    assert finding.evidence["error_type"] == "HDF5ExtError"
    assert HUMAN_H5 in finding.evidence["error_message"]
    assert result.snapshot.findings == (*snapshot.findings, finding)


def test_a_labels_file_with_an_upper_case_suffix_is_read_with_the_rest(valid_project: Path) -> None:
    folder = _session(valid_project)
    upper = (folder / HUMAN_H5).rename(folder / f"{Path(HUMAN_H5).stem}.H5")
    snapshot = inspect_project(valid_project)

    result = read_project_labeled_data(snapshot, Scope(LABELED_DATASET))

    assert _names(result) == (sorted([HUMAN_CSV, upper.name, MACHINE_H5]), [])
    (entry,) = [entry for entry in result.read if entry.project_relative_path.name == upper.name]
    assert entry.schema_id == HUMAN_LABELS.schema_id
    assert entry.data.keypoints is not None


def test_the_merged_finding_is_in_the_returned_snapshot_only(valid_project: Path) -> None:
    (_session(valid_project) / HUMAN_H5).write_bytes(b"not an hdf5 file")
    snapshot = inspect_project(valid_project)
    file_scope = Scope(HUMAN_LABELS, within=("labeled-data", "session-01", HUMAN_H5))

    result = read_project_labeled_data(snapshot, Scope(LABELED_DATASET))

    assert UNREADABLE in [finding.code for finding in findings_in_scope(result.snapshot, file_scope)]
    assert UNREADABLE not in [finding.code for finding in findings_in_scope(snapshot, file_scope)]


def test_an_uninterpretable_file_is_read_and_reported_not_failed(valid_project: Path) -> None:
    _uninterpretable(_session(valid_project) / HUMAN_H5)
    snapshot = inspect_project(valid_project)

    result = read_project_labeled_data(snapshot, Scope(LABELED_DATASET))

    (entry,) = [entry for entry in result.read if entry.project_relative_path.name == HUMAN_H5]
    assert entry.data.keypoints is None
    assert entry.data.frames is not None
    assert result.failed == ()
    assert _human_labels_findings(valid_project) == [LabeledDataFindingCode.LABELED_DATA_HEADER_UNRECOGNISED]


def test_a_scope_within_one_dataset_reads_that_dataset_only(valid_project: Path) -> None:
    _session(valid_project, "session-01")
    _session(valid_project, "session-02")
    snapshot = inspect_project(valid_project)
    assert len(read_project_labeled_data(snapshot, Scope(LABELED_DATASET)).read) == 6  # control: both are read

    result = read_project_labeled_data(snapshot, Scope(LABELED_DATASET, within=("labeled-data", "session-02")))

    assert {entry.project_relative_path.parent.name for entry in result.read} == {"session-02"}
    assert len(result.read) == 3


def test_a_scope_holding_no_labels_files_reads_nothing(valid_project: Path) -> None:
    _session(valid_project)
    snapshot = inspect_project(valid_project)

    result = read_project_labeled_data(snapshot, Scope(VIDEOS))

    assert (result.read, result.failed, result.snapshot) == ((), (), snapshot)


def test_a_node_the_layout_does_not_declare_is_refused(valid_project: Path) -> None:
    impostor = HUMAN_LABELS.model_copy(update={"description": "not the layout's own"})

    with pytest.raises(ValueError, match="does not declare"):
        read_project_labeled_data(inspect_project(valid_project), Scope(impostor))


def test_the_public_api_reexports_labeled_data_reading() -> None:
    import dlc_octopus.api as api
    import dlc_octopus.content as content

    assert (api.read_project_labeled_data, api.ProjectLabeledData, api.LabeledDataEntry, api.FailedRead) == (
        content.read_project_labeled_data,
        content.ProjectLabeledData,
        content.LabeledDataEntry,
        content.FailedRead,
    )


def test_an_uninterpretable_labels_file_read_through_octopus_is_reported(valid_project: Path) -> None:
    _uninterpretable(_session(valid_project) / HUMAN_H5)

    result = read_project_labeled_data(inspect_project(valid_project), Scope(LABELED_DATASET))

    file_scope = Scope(HUMAN_LABELS, within=("labeled-data", "session-01", HUMAN_H5))
    codes = [finding.code for finding in findings_in_scope(result.snapshot, file_scope)]
    assert [code for code in codes if code is not UNREADABLE]


def _without_frames(path: Path) -> pd.DataFrame:
    columns = pd.MultiIndex.from_product(
        [["test-scorer"], ["nose"], ["x", "y"]], names=["scorer", "bodyparts", "coords"]
    )
    frame = pd.DataFrame([], columns=columns, index=pd.MultiIndex.from_tuples([], names=[None] * 3), dtype=float)
    frame.to_hdf(path, key="df_with_missing", mode="w")
    return frame


def test_a_labels_file_without_frames_is_read_through_octopus_as_found(valid_project: Path) -> None:
    # DeepLabCut v3.0.2 and napari-deeplabcut v0.4.0 read such a file; only the model is refused.
    path = _session(valid_project) / HUMAN_H5
    stored = _without_frames(path)

    result = read_project_labeled_data(inspect_project(valid_project), Scope(LABELED_DATASET))

    (entry,) = [entry for entry in result.read if entry.project_relative_path.name == HUMAN_H5]
    pd.testing.assert_frame_equal(entry.data.to_raw_wide_df(), stored)
    with pytest.raises(ValueError, match="no frames"):
        _ = entry.data.labeled_data


def test_a_labels_file_without_frames_read_through_octopus_is_reported(valid_project: Path) -> None:
    _without_frames(_session(valid_project) / HUMAN_H5)

    result = read_project_labeled_data(inspect_project(valid_project), Scope(LABELED_DATASET))

    file_scope = Scope(HUMAN_LABELS, within=("labeled-data", "session-01", HUMAN_H5))
    codes = [finding.code for finding in findings_in_scope(result.snapshot, file_scope)]
    assert [code for code in codes if code is not UNREADABLE]


def test_a_labels_file_with_a_malformed_writer_stamp_read_through_octopus_is_reported(valid_project: Path) -> None:
    path = _session(valid_project) / HUMAN_H5
    table = read_labeled_data(path).labeled_data
    path.unlink()
    write_labeled_data(table, path, replaces=None)
    with tables.open_file(str(path), mode="a") as h5:
        h5.get_node("/df_with_missing")._v_attrs[WRITER_VERSION_KEY] = 1

    result = read_project_labeled_data(inspect_project(valid_project), Scope(LABELED_DATASET))

    assert HUMAN_H5 in [entry.project_relative_path.name for entry in result.read]
    file_scope = Scope(HUMAN_LABELS, within=("labeled-data", "session-01", HUMAN_H5))
    codes = [finding.code for finding in findings_in_scope(result.snapshot, file_scope)]
    assert [code for code in codes if code is not UNREADABLE]


def _human_labels_findings(project: Path, name: str = HUMAN_H5) -> list[LabeledDataFindingCode]:
    result = read_project_labeled_data(inspect_project(project), Scope(LABELED_DATASET))

    assert name in [entry.project_relative_path.name for entry in result.read]
    file_scope = Scope(HUMAN_LABELS, within=("labeled-data", "session-01", name))
    codes = [finding.code for finding in findings_in_scope(result.snapshot, file_scope)]
    return [code for code in codes if code is not UNREADABLE]


def test_human_labels_carrying_a_likelihood_read_through_octopus_are_reported(valid_project: Path) -> None:
    path = _session(valid_project) / HUMAN_H5
    shutil.copy(FIXTURES_DIR / "single_animal_machine.h5", path)
    assert read_labeled_data(path).labeled_data.to_arrays().likelihood is not None

    assert _human_labels_findings(valid_project)


def test_a_missing_keypoint_pair_read_through_octopus_is_reported(valid_project: Path) -> None:
    path = _session(valid_project) / HUMAN_H5
    frame = pd.read_hdf(FIXTURES_DIR / "multi_animal_human.h5")
    pair = (frame.columns.get_level_values("individuals") == "animalB") & (
        frame.columns.get_level_values("bodyparts") == "tailbase"
    )
    frame.loc[:, ~pair].to_hdf(path, key="df_with_missing", mode="w")
    assert len(read_labeled_data(path).keypoints) == 3

    assert _human_labels_findings(valid_project)


def test_a_csv_twin_missing_a_frame_of_its_h5_read_through_octopus_is_reported(valid_project: Path) -> None:
    path = _session(valid_project) / HUMAN_H5
    long = read_labeled_data(path).labeled_data.to_long_df()
    long.loc[long["frame_index"] == 0, ["x", "y"]] = np.nan
    with pytest.warns(UserWarning, match="reads back without it"):
        write_labeled_data(LabeledData.from_long_df(long), path, replaces=read_labeled_data(path).stat)
    assert len(read_labeled_data(path.with_name(HUMAN_CSV)).frames) == long["frame_index"].nunique() - 1

    assert _human_labels_findings(valid_project, HUMAN_CSV)


def _flat_row_index(path: Path) -> None:
    shutil.copy(FIXTURES_DIR / "single_animal_human_legacy_index.h5", path)


def _integer_frame_part(path: Path) -> None:
    frame = pd.read_hdf(FIXTURES_DIR / "single_animal_human.h5")
    frame.index = pd.MultiIndex.from_tuples([(folder, 2019, image) for folder, _, image in frame.index])
    frame.to_hdf(path, key="df_with_missing", mode="w")


def _codes_at(snapshot: ProjectSnapshot, dataset: str) -> list[LabeledDataFindingCode]:
    file_scope = Scope(HUMAN_LABELS, within=("labeled-data", dataset, HUMAN_H5))
    return sorted(finding.code for finding in findings_in_scope(snapshot, file_scope))


@pytest.mark.xfail(
    reason="normalisation changes are to surface as findings at FULL depth (octopus-contracts.md, 'Normalisation is "
    "allowed only at FULL depth'); deferred past PR E2 (decided 2026-10-20): a file the model reads only after "
    "normalising it reports nothing the same labels in DeepLabCut's current form do not",
    strict=True,
    raises=AssertionError,
)
@pytest.mark.parametrize("legacy", [_flat_row_index, _integer_frame_part], ids=["flat_row_index", "integer_part"])
def test_a_normalisation_on_read_through_octopus_is_reported(
    valid_project: Path, legacy: Callable[[Path], None]
) -> None:
    found = _session(valid_project, "session-01") / HUMAN_H5
    legacy(found)
    current = _session(valid_project, "session-02") / HUMAN_H5
    current.unlink()
    write_labeled_data(read_labeled_data(found).labeled_data, current, replaces=None)

    result = read_project_labeled_data(inspect_project(valid_project), Scope(LABELED_DATASET))

    assert _codes_at(result.snapshot, "session-01") != _codes_at(result.snapshot, "session-02")


@pytest.mark.xfail(
    reason="an infinite coordinate has no meaning in a labels file and is always reported at FULL depth (design, "
    "canonical data model, decided 2026-10-14); no check or code reports it yet",
    strict=True,
    raises=AssertionError,
)
def test_an_infinite_coordinate_read_through_octopus_is_reported(valid_project: Path) -> None:
    path = _session(valid_project) / HUMAN_H5
    frame = pd.read_hdf(path)
    frame.iloc[0, 0] = np.inf
    frame.to_hdf(path, key="df_with_missing", mode="w")

    assert _human_labels_findings(valid_project)


@pytest.mark.xfail(
    reason="whether a frame's first path part is labeled-data is a project rule, reported by Octopus, never "
    "checked by the model (design, canonical data model, decided 2026-10-16); no check or code reports it yet",
    strict=True,
    raises=AssertionError,
)
def test_a_frame_outside_labeled_data_read_through_octopus_is_reported(valid_project: Path) -> None:
    path = _session(valid_project) / HUMAN_H5
    frame = pd.read_hdf(path)
    frame.index = [entry.replace("labeled-data", "images", 1) for entry in frame.index]
    frame.to_hdf(path, key="df_with_missing", mode="w")
    assert read_labeled_data(path).labeled_data  # control: the file still has a model

    assert _human_labels_findings(valid_project)
