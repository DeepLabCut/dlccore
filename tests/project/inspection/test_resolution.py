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

from pathlib import Path

import pytest

from dlccore.project.api import inspect_project
from dlccore.project.schemas.base.finding_codes import ConfigFindingCode
from dlccore.project.schemas.snapshots import (
    ConfigState,
)


def test_inspect_from_config_matches_inspect_from_root(
    valid_project: Path,
) -> None:
    from_root = inspect_project(valid_project)
    from_config = inspect_project(valid_project / "config.yaml")

    assert from_config == from_root


def test_inspect_rejects_existing_non_config_file(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text(
        "notes",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"config\.yaml",
    ):
        inspect_project(file_path)


def test_inspect_rejects_missing_project_root(
    tmp_path: Path,
) -> None:
    missing_root = tmp_path / "missing-project"

    with pytest.raises(
        FileNotFoundError,
        match="does not exist",
    ):
        inspect_project(missing_root)


def test_missing_config_path_resolves_to_parent(
    tmp_path: Path,
) -> None:
    snapshot = inspect_project(tmp_path / "config.yaml")

    finding_codes = {finding.code for finding in snapshot.findings}

    assert snapshot.project_root == tmp_path
    assert snapshot.config_state == ConfigState.MISSING
    assert "PROJECT_REQUIRED_NODE_MISSING" in finding_codes
    assert not any(isinstance(code, ConfigFindingCode) for code in finding_codes)
