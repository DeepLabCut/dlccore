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

from dlccore.project.api import inspect_labeled_data_file
from dlccore.project.schemas.base.finding_codes import LabeledDataFindingCode as Code
from dlccore.project.schemas.base.findings import Reach

FIXTURES_DIR = Path(__file__).parents[2] / "io" / "fixtures" / "labeled_data"
NAME = "CollectedData_test-scorer.h5"


def _uninterpretable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([[1.0, 2.0]], columns=["nose_x", "nose_y"], index=["labeled-data/session-01/img000.png"])
    frame.to_hdf(path, key="df_with_missing", mode="w")
    return path


def test_a_file_and_its_copy_outside_the_project_get_the_same_findings(valid_project: Path, tmp_path: Path) -> None:
    inside = _uninterpretable(valid_project / "labeled-data" / "session-01" / NAME)
    copy = tmp_path / "backup" / "labeled-data" / "session-01" / NAME
    copy.parent.mkdir(parents=True)
    shutil.copy(inside, copy)

    found_inside, found_copy = inspect_labeled_data_file(inside), inspect_labeled_data_file(copy)

    assert found_inside.snapshot is not None
    assert found_copy.snapshot is None
    assert [(f.code, f.reach, f.instance) for f in found_copy.findings] == [
        (f.code, f.reach, f.instance) for f in found_inside.findings
    ]
    assert [f.code for f in found_copy.findings] == [Code.LABELED_DATA_HEADER_UNRECOGNISED]  # control: not vacuous


def test_a_detached_copy_gets_the_same_codes_and_its_guess_never_reaches_a_finding(tmp_path: Path) -> None:
    path = _uninterpretable(tmp_path / "Downloads" / "session-01" / NAME)

    inspected = inspect_labeled_data_file(path)

    assert [f.code for f in inspected.findings] == [Code.LABELED_DATA_HEADER_UNRECOGNISED]
    assert {(f.reach, f.instance) for f in inspected.findings} == {(Reach.UNROOTED, (NAME,))}
    assert inspected.position.detached  # control: there was a guess to leak


def test_an_unreadable_file_outside_a_project_has_no_data_and_says_why(tmp_path: Path) -> None:
    path = tmp_path / NAME
    path.write_bytes(b"not an HDF5 file")

    inspected = inspect_labeled_data_file(path)

    assert inspected.data is None
    assert [f.code for f in inspected.findings] == [Code.LABELED_DATA_UNREADABLE]


def test_a_readable_file_outside_a_project_is_returned(tmp_path: Path) -> None:
    path = tmp_path / NAME
    shutil.copy(FIXTURES_DIR / "single_animal_human.h5", path)

    inspected = inspect_labeled_data_file(path)

    assert inspected.data is not None
    assert inspected.findings == ()
