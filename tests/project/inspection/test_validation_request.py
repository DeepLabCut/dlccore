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
from dlccore.project.schemas.base.resources import (
    DirectoryResourceKind,
)
from dlccore.project.schemas.base.validation import (
    ValidationDepth,
)
from dlccore.project.validation import (
    ProjectValidationRequest,
    ResourceValidationRequest,
)


def test_inspection_without_request_runs_baseline_validation(
    tmp_path: Path,
) -> None:
    snapshot = inspect_project(tmp_path)
    codes = {finding.code for finding in snapshot.findings}

    assert "PROJECT_REQUIRED_NODE_MISSING" in codes
    assert not any(isinstance(code, ConfigFindingCode) for code in codes)
    assert snapshot.validation_checks == ()


def test_unregistered_resource_validation_is_rejected(
    valid_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dlccore.project.validation.project.get_validator_registration",
        lambda _kind: None,
    )

    request = ProjectValidationRequest(
        resources=(
            ResourceValidationRequest(
                kind=DirectoryResourceKind.VIDEOS_DIRECTORY,
                depth=ValidationDepth.FILENAME_ONLY,
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match="No validator is available for resource kind",
    ):
        inspect_project(
            valid_project,
            validation=request,
        )
