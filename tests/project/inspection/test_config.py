from pathlib import Path

import pytest

from dlc_octopus.api import inspect_project
from dlc_octopus.errors import (
    ConfigUnavailableError,
)


def test_valid_config_is_available(
    valid_project: Path,
) -> None:
    snapshot = inspect_project(valid_project)

    assert snapshot.config.scorer == "test-scorer"
    assert snapshot.config_observation is not None
    assert snapshot.failed_config is None
    assert snapshot.findings == ()


def test_missing_config_is_distinct_from_failed_config(
    tmp_path: Path,
) -> None:
    snapshot = inspect_project(tmp_path)

    with pytest.raises(ConfigUnavailableError):
        _ = snapshot.config

    assert snapshot.config_observation is None
    assert snapshot.failed_config is None
    assert "PROJECT_REQUIRED_NODE_MISSING" in {finding.code for finding in snapshot.findings}


def test_invalid_config_is_available_only_for_diagnostics(
    invalid_project: Path,
) -> None:
    snapshot = inspect_project(invalid_project)

    with pytest.raises(ConfigUnavailableError):
        _ = snapshot.config

    assert snapshot.failed_config is not None
    assert snapshot.failed_config.raw_config is not None
    assert any(finding.code == "CONFIG_VALIDATION_ERROR" for finding in snapshot.findings)
