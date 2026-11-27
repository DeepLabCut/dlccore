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

from __future__ import annotations

from pathlib import Path

import pytest

from dlccore.config.utils import write_config
from dlccore.project import api
from dlccore.project.errors import ConfigUnavailableError

# DeepLabCut's own example project: examples/openfield-Pranav-2018-10-30, config date "Oct30".
TASK, SCORER, CONFIG_DATE = "openfield", "Pranav", "Oct30"
NAME = "openfield-Pranav-2018-10-30"


def _project(
    root: Path,
    folder: str,
    config_data: dict,
    *,
    project_path: str,
    task: str = TASK,
    scorer: str = SCORER,
    date: str = CONFIG_DATE,
) -> Path:
    project = root / folder
    (project / "videos").mkdir(parents=True)
    (project / "labeled-data").mkdir()
    write_config(
        project / "config.yaml",
        {**config_data, "Task": task, "scorer": scorer, "date": date, "project_path": project_path},
    )
    return project


def test_the_name_comes_from_the_folder(tmp_path: Path, valid_config_data: dict) -> None:
    project = _project(tmp_path, NAME, valid_config_data, project_path="WILL BE AUTOMATICALLY UPDATED BY DEMO CODE")

    assert api.project_name(api.inspect_project(project)) == NAME


def test_a_renamed_folder_has_no_name_whatever_the_config_path_says(tmp_path: Path, valid_config_data: dict) -> None:
    project = _project(tmp_path, "horse10", valid_config_data, project_path=rf"C:\Users\lab\{NAME}")

    assert api.project_name(api.inspect_project(project)) is None


def test_a_relative_project_root_names_the_inspected_folder_after_a_chdir(
    tmp_path: Path, valid_config_data: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path, NAME, valid_config_data, project_path="placeholder")
    same_day_next_year = _project(
        tmp_path, "openfield-Pranav-2019-10-30", valid_config_data, project_path="placeholder"
    )
    monkeypatch.chdir(project)
    snapshots = [api.inspect_project("."), api.inspect_project("config.yaml")]

    monkeypatch.chdir(same_day_next_year)

    assert [api.project_name(snapshot) for snapshot in snapshots] == [NAME, NAME]


def test_a_folder_date_that_disagrees_with_the_config_gives_no_name(tmp_path: Path, valid_config_data: dict) -> None:
    project = _project(tmp_path, "openfield-Pranav-2018-10-31", valid_config_data, project_path="placeholder")

    assert api.project_name(api.inspect_project(project)) is None


@pytest.mark.parametrize(
    ("stamp", "config_date"),
    [("20181030", "Oct30"), ("2018-02-30", "Feb30")],
    ids=["not_dashed", "not_a_calendar_date"],
)
def test_a_folder_stamp_that_is_not_a_date_gives_no_name(
    tmp_path: Path, valid_config_data: dict, stamp: str, config_date: str
) -> None:
    project = _project(
        tmp_path, f"{TASK}-{SCORER}-{stamp}", valid_config_data, project_path="placeholder", date=config_date
    )

    assert api.project_name(api.inspect_project(project)) is None


def test_a_hyphenated_task_is_matched_exactly(tmp_path: Path, valid_config_data: dict) -> None:
    name = "reaching-task-Linus-2026-10-06"
    project = _project(
        tmp_path, name, valid_config_data, project_path="placeholder", task="reaching-task", scorer="Linus", date="Oct6"
    )

    assert api.project_name(api.inspect_project(project)) == name


def test_a_project_without_a_loaded_config_has_no_canonical_name(invalid_project: Path) -> None:
    snapshot = api.inspect_project(invalid_project)

    with pytest.raises(ConfigUnavailableError):
        api.project_name(snapshot)
