import inspect
import shutil
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from dlc_octopus.api import findings_in_scope, inspect_project
from dlc_octopus.content import read_project_labeled_data
from dlc_octopus.content.labeled_data import file, model
from dlc_octopus.content.labeled_data.report import NOT_REPORTED, Report
from dlc_octopus.content.labeled_data.validate import validate_labeled_data_file
from dlc_octopus.io.labeled_data import LabeledData, LabeledDataFile, read_labeled_data
from dlc_octopus.io.labeled_data.checks import file as file_checks
from dlc_octopus.io.labeled_data.checks import model as model_checks
from dlc_octopus.schemas.base.finding_codes import LabeledDataFindingCode
from dlc_octopus.schemas.base.finding_codes import LabeledDataFindingCode as Code
from dlc_octopus.schemas.base.findings import Finding
from dlc_octopus.schemas.locations import Location, node_location
from dlc_octopus.schemas.project_layout.labeled_data import HUMAN_LABELS, LABELED_DATASET
from dlc_octopus.schemas.scope import Scope

FIXTURES_DIR = Path(__file__).parents[1] / "io" / "fixtures" / "labeled_data"
HUMAN_H5 = "CollectedData_test-scorer.h5"
STAND_IN = LabeledDataFindingCode.LABELED_DATA_UNREADABLE  # any declared code no report emits


def _project_with(project: Path, write: Callable[[Path], None]) -> Path:
    folder = project / "labeled-data" / "session-01"
    folder.mkdir()
    write(folder / HUMAN_H5)
    return project


def _readable(path: Path) -> None:
    shutil.copy(FIXTURES_DIR / "single_animal_human.h5", path)


def _without_model(path: Path) -> None:
    frame = pd.DataFrame([[1.0, 2.0]], columns=["nose_x", "nose_y"], index=["labeled-data/session-01/img000.png"])
    frame.to_hdf(path, key="df_with_missing", mode="w")


def _stand_in(at: Location, data: LabeledDataFile) -> tuple[Finding, ...]:
    evidence = {"resource_kind": "stand_in", "error_type": "none", "error_message": str(data.path.name)}
    return (Finding(code=STAND_IN, **at, path=data.path, evidence=evidence),)


def _codes(project: Path) -> list[str]:
    result = read_project_labeled_data(inspect_project(project), Scope(LABELED_DATASET))
    scope = Scope(HUMAN_LABELS, within=("labeled-data", "session-01", HUMAN_H5))
    return [finding.evidence["resource_kind"] for finding in findings_in_scope(result.snapshot, scope)]


@pytest.mark.parametrize(
    ("write", "model_runs"), [(_readable, True), (_without_model, False)], ids=["model", "no_model"]
)
def test_every_file_read_is_reported_and_only_a_file_with_a_model_gets_model_reports(
    valid_project: Path, monkeypatch: pytest.MonkeyPatch, write: Callable[[Path], None], *, model_runs: bool
) -> None:
    def file_report(data: LabeledDataFile, at: Location) -> tuple[Finding, ...]:
        return _stand_in(at, data)

    def model_report(data: LabeledDataFile, labels: LabeledData, at: Location) -> tuple[Finding, ...]:
        assert labels.to_arrays().keypoints
        return _stand_in(at, data)

    monkeypatch.setattr(file, "REPORTS", (Report(file_checks.unread_h5_keys, STAND_IN, file_report),))
    monkeypatch.setattr(model, "REPORTS", (Report(model_checks.missing_keypoints, STAND_IN, model_report),))

    assert _codes(_project_with(valid_project, write)) == ["stand_in"] * (2 if model_runs else 1)


def test_a_file_a_deeplabcut_writer_produced_gets_no_finding(valid_project: Path) -> None:
    assert _codes(_project_with(valid_project, _readable)) == []


def _public_checks() -> list[Callable[..., object]]:
    return [
        function
        for module in (file_checks, model_checks)
        for name, function in inspect.getmembers(module, inspect.isfunction)
        if function.__module__ == module.__name__ and not name.startswith("_")
    ]


def test_the_public_checks_are_found() -> None:
    # Control: an empty list would let the completeness test below pass by checking nothing.
    assert {file_checks.unread_h5_keys, model_checks.missing_keypoints} <= set(_public_checks())


def test_every_check_is_reported_by_exactly_one_code() -> None:
    reports = (*file.REPORTS, *model.REPORTS)
    reported = [report.check for report in reports]
    unreported = [check.__name__ for check in _public_checks() if check not in {*reported, *NOT_REPORTED}]
    codes = [report.code for report in reports]
    shared = [check.__name__ for check in set(reported) if reported.count(check) > 1]
    shared += [str(code) for code in set(codes) if codes.count(code) > 1]

    assert (unreported, shared) == ([], [])


def test_a_check_is_either_reported_or_listed_as_not_reported_never_both() -> None:
    assert {report.check for report in (*file.REPORTS, *model.REPORTS)} & NOT_REPORTED == set()


SINGLE = ["scorer", "bodyparts", "coords"]
MULTI = ["scorer", "individuals", "bodyparts", "coords"]
ROWS = pd.MultiIndex.from_tuples([("labeled-data", "s1", "img0.png"), ("labeled-data", "s1", "img1.png")])


def _h5(path: Path, frame: pd.DataFrame, key: str = "df_with_missing") -> None:
    frame.to_hdf(path, key=key, mode="w")


def _frame(
    columns: list[tuple[str, ...]], *, levels: list[str] = SINGLE, index: object = ROWS, value: object = 1.0
) -> pd.DataFrame:
    header = pd.MultiIndex.from_tuples(columns, names=levels)
    return pd.DataFrame([[value] * len(columns)] * len(index), index=index, columns=header)  # type: ignore[arg-type]


XY = [("Sc", "nose", "x"), ("Sc", "nose", "y")]

CASES: list[tuple[str, Callable[[Path], None], LabeledDataFindingCode]] = [
    ("multi_key", lambda p: shutil.copy(FIXTURES_DIR / "ambiguous_multi_key.h5", p), Code.LABELED_DATA_H5_MULTI_KEY),
    ("legacy_key", lambda p: _h5(p, _frame(XY), key="keypoints"), Code.LABELED_DATA_H5_LEGACY_KEY),
    ("unknown_key", lambda p: _h5(p, _frame(XY), key="labels"), Code.LABELED_DATA_H5_UNKNOWN_KEY),
    ("coords", lambda p: _h5(p, _frame([*XY, ("Sc", "nose", "z")])), Code.LABELED_DATA_COORDS_UNRECOGNISED),
    (
        "partial_likelihood",
        lambda p: _h5(p, _frame([*XY, ("Sc", "nose", "likelihood"), ("Sc", "tail", "x"), ("Sc", "tail", "y")])),
        Code.LABELED_DATA_LIKELIHOOD_PARTIAL,
    ),
    ("not_numeric", lambda p: _h5(p, _frame(XY, value="a")), Code.LABELED_DATA_VALUES_NOT_NUMERIC),
    ("row_index", lambda p: _h5(p, _frame(XY, index=pd.Index([0, 1]))), Code.LABELED_DATA_ROW_INDEX_UNRECOGNISED),
    (
        "frame_depth",
        lambda p: _h5(p, _frame(XY, index=pd.MultiIndex.from_tuples([("s1", "img0.png"), ("s1", "img1.png")]))),
        Code.LABELED_DATA_FRAME_DEPTH,
    ),
    (
        "frame_part",
        lambda p: _h5(p, _frame(XY, index=pd.MultiIndex.from_tuples([("labeled-data", 1.5, "a.png")]))),
        Code.LABELED_DATA_FRAME_PART_UNNAMED,
    ),
    (
        "empty_individual",
        lambda p: _h5(p, _frame([("Sc", "", "nose", "x"), ("Sc", "", "nose", "y")], levels=MULTI)),
        Code.LABELED_DATA_INDIVIDUAL_EMPTY,
    ),
]


@pytest.mark.parametrize(("write", "code"), [(write, code) for _id, write, code in CASES], ids=[c[0] for c in CASES])
def test_each_condition_is_reported_by_its_code_and_no_other(
    tmp_path: Path, write: Callable[[Path], None], code: LabeledDataFindingCode
) -> None:
    path = tmp_path / "CollectedData_Sc.h5"
    write(path)
    at = node_location(HUMAN_LABELS.schema_id, Path("labeled-data", "s1", path.name))

    findings = validate_labeled_data_file(read_labeled_data(path), at)

    assert [finding.code for finding in findings] == [code]
    assert set(findings[0].evidence) == set(code.evidence_keys)


def test_every_code_reported_in_this_package_is_tested() -> None:
    # The six codes the read-through-Octopus tests flip are not repeated here.
    tested = {code for _id, _write, code in CASES}
    flipped = {
        Code.LABELED_DATA_HEADER_UNRECOGNISED,
        Code.LABELED_DATA_NO_FRAMES,
        Code.LABELED_DATA_WRITER_STAMP_MALFORMED,
        Code.LABELED_DATA_HUMAN_LIKELIHOOD,
        Code.LABELED_DATA_KEYPOINT_MISSING,
        Code.LABELED_DATA_CSV_FRAME_ABSORBED,
    }

    assert {report.code for report in (*file.REPORTS, *model.REPORTS)} == tested | flipped


@pytest.mark.xfail(
    reason="not every way the model can fail to build is a file check: a bodypart level that is not text passes "
    "every check, the model raises, and validate_labeled_data_file returns no finding saying why",
    raises=AssertionError,
    strict=True,
)
def test_a_file_whose_model_cannot_be_built_has_a_file_check_saying_why(tmp_path: Path) -> None:
    stored = pd.read_hdf(FIXTURES_DIR / "single_animal_human.h5")
    numbered = {name: i for i, name in enumerate(stored.columns.get_level_values("bodyparts").unique())}
    stored.columns = pd.MultiIndex.from_tuples(
        [(scorer, numbered[bodypart], coord) for scorer, bodypart, coord in stored.columns],
        names=stored.columns.names,
    )
    path = tmp_path / "CollectedData_Sc.h5"
    stored.to_hdf(path, key="df_with_missing", mode="w")
    data = read_labeled_data(path)
    with pytest.raises(ValueError, match="bodypart"):  # control: the model cannot be built
        _ = data.labeled_data

    at = node_location(HUMAN_LABELS.schema_id, Path("labeled-data", "s1", path.name))

    assert validate_labeled_data_file(data, at)
