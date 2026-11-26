from __future__ import annotations

from pathlib import Path

import pytest

from dlc_octopus.api import inspect_project
from dlc_octopus.errors import BLOCKING_SEVERITIES, ProjectValidationError, _blocking_findings
from dlc_octopus.schemas.base.finding_codes import ConfigFindingCode, Severity

_UNPARSEABLE_CONFIG = "a:\n  b: [\n"


def test_only_error_blocks() -> None:
    assert {Severity.ERROR} == BLOCKING_SEVERITIES


def test_inspect_project_does_not_raise_without_blocking_findings(
    valid_project: Path,
) -> None:
    snapshot = inspect_project(
        valid_project,
        raise_on_validation_error=True,
    )

    assert _blocking_findings(snapshot) == ()


def test_inspect_project_does_not_raise_unless_asked(
    invalid_project: Path,
) -> None:
    snapshot = inspect_project(invalid_project)

    assert _blocking_findings(snapshot)


def test_inspect_project_raises_on_an_error_finding(
    invalid_project: Path,
) -> None:
    with pytest.raises(ProjectValidationError) as raised:
        inspect_project(
            invalid_project,
            raise_on_validation_error=True,
        )

    assert ConfigFindingCode.CONFIG_VALIDATION_ERROR in {finding.code for finding in raised.value.findings}


def test_inspect_project_raises_on_an_unparseable_config(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.yaml").write_text(
        _UNPARSEABLE_CONFIG,
        encoding="utf-8",
    )

    with pytest.raises(ProjectValidationError) as raised:
        inspect_project(
            tmp_path,
            raise_on_validation_error=True,
        )

    codes = {finding.code for finding in raised.value.findings}

    assert ConfigFindingCode.CONFIG_PARSE_FAILED in codes
    assert raised.value.findings[0].severity is Severity.ERROR


def test_project_validation_error_names_the_blocking_codes(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.yaml").write_text(
        _UNPARSEABLE_CONFIG,
        encoding="utf-8",
    )

    with pytest.raises(ProjectValidationError, match="CONFIG_PARSE_FAILED"):
        inspect_project(
            tmp_path,
            raise_on_validation_error=True,
        )


def test_project_validation_error_retains_the_snapshot(
    invalid_project: Path,
) -> None:
    with pytest.raises(ProjectValidationError) as raised:
        inspect_project(
            invalid_project,
            raise_on_validation_error=True,
        )

    assert raised.value.snapshot.project_root == invalid_project
