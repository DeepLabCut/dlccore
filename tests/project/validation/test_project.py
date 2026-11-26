from __future__ import annotations

from pathlib import Path

import pytest

from dlc_octopus.api import inspect_project
from dlc_octopus.schemas.base.resources import DirectoryResourceKind
from dlc_octopus.schemas.base.validation import ValidationDepth
from dlc_octopus.validation import validate_project
from dlc_octopus.validation.requests import ProjectValidationRequest, ResourceValidationRequest


def test_validate_project_returns_structural_findings(
    tmp_path: Path,
) -> None:
    snapshot = inspect_project(tmp_path)

    result = validate_project(snapshot)

    assert "PROJECT_REQUIRED_NODE_MISSING" in {finding.code for finding in result.findings}


def test_validate_project_returns_no_findings_for_valid_structure(
    valid_project: Path,
) -> None:
    snapshot = inspect_project(valid_project)

    result = validate_project(snapshot)

    assert result.findings == ()
    assert result.checks == ()


def test_inspect_project_includes_validation_findings(
    tmp_path: Path,
) -> None:
    snapshot = inspect_project(tmp_path)
    codes = {finding.code for finding in snapshot.findings}

    assert "PROJECT_REQUIRED_NODE_MISSING" in codes


def _videos_at(depth: ValidationDepth) -> ProjectValidationRequest:
    return ProjectValidationRequest(
        resources=(ResourceValidationRequest(kind=DirectoryResourceKind.VIDEOS_DIRECTORY, depth=depth),)
    )


def test_a_depth_the_suite_does_not_support_is_refused_naming_the_supported_ones(valid_project: Path) -> None:
    snapshot = inspect_project(valid_project)
    validate_project(snapshot, request=_videos_at(ValidationDepth.FILENAME_ONLY))  # control: a supported depth runs

    with pytest.raises(ValueError, match=r"'full' is not supported .* Supported depths are: filename_only\."):
        validate_project(snapshot, request=_videos_at(ValidationDepth.FULL))
