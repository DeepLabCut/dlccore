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

from dlccore.project.api import inspect_project
from dlccore.project.schemas.base.finding_codes import ConfigFindingCode, StructureFindingCode


def test_inspect_project_includes_missing_required_node_finding(
    tmp_path: Path,
) -> None:
    snapshot = inspect_project(tmp_path)
    finding_codes = {finding.code for finding in snapshot.findings}

    assert "PROJECT_REQUIRED_NODE_MISSING" in finding_codes
    assert not any(isinstance(code, ConfigFindingCode) for code in finding_codes)


def test_inspect_project_reports_wrong_type(
    valid_project: Path,
) -> None:
    videos = valid_project / "videos"
    videos.rmdir()
    videos.write_text(
        "not a directory",
        encoding="utf-8",
    )

    snapshot = inspect_project(valid_project)
    finding = next(finding for finding in snapshot.findings if finding.code == "PROJECT_NODE_WRONG_TYPE")

    assert finding.path == videos
    assert finding.evidence["expected_type"] == "directory"

    resource = snapshot.resource_at("videos")

    assert resource is not None


def test_valid_project_has_no_structural_findings(
    valid_project: Path,
) -> None:
    snapshot = inspect_project(valid_project)

    assert not any(isinstance(finding.code, StructureFindingCode) for finding in snapshot.findings)
