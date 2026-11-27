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

import inspect
from pathlib import Path

import numpy as np
import pytest

from dlccore.config import ProjectConfig, versioning
from dlccore.config.utils import write_config
from dlccore.config.versioning import CURRENT_CONFIG_VERSION
from dlccore.project.schemas.base.findings import Finding, Severity
from dlccore.project.schemas.project_layout.config.observations import (
    ConfigObservation,
    FailedConfigObservation,
    ValidConfigObservation,
)
from dlccore.project.validation.config import validate_project_config


def test_validate_valid_project_config(
    valid_project: Path,
) -> None:
    config_path = valid_project / "config.yaml"

    observation, findings = validate_project_config(
        config_path,
    )

    assert isinstance(
        observation,
        ValidConfigObservation,
    )
    assert observation.config_path == config_path
    assert observation.declared_project_path == str(valid_project)
    assert isinstance(
        observation.value,
        ProjectConfig,
    )
    assert observation.value.scorer == "test-scorer"
    assert findings == ()


def test_validate_project_config_does_not_repair_declared_path(
    tmp_path: Path,
    valid_config_data: dict,
) -> None:
    config_path = tmp_path / "config.yaml"
    declared_path = tmp_path / "declared-elsewhere"

    config_data = dict(valid_config_data)
    config_data["project_path"] = str(declared_path)

    write_config(
        config_path,
        config_data,
    )

    observation, findings = validate_project_config(
        config_path,
    )

    assert isinstance(
        observation,
        ValidConfigObservation,
    )
    assert observation.declared_project_path == str(declared_path)
    assert observation.value.project_path == declared_path
    assert findings == ()


def _validate_with_null_keys(
    tmp_path: Path, valid_config_data: dict, *fields: str
) -> tuple[ConfigObservation, tuple[Finding, ...]]:
    config_path = tmp_path / "config.yaml"

    write_config(
        config_path,
        {**valid_config_data, **dict.fromkeys(fields)},
    )

    return validate_project_config(
        config_path,
    )


def test_validate_project_config_accepts_a_legacy_null_key_as_deeplabcut_does(
    tmp_path: Path,
    valid_config_data: dict,
) -> None:
    # As in DeepLabCut's own examples/openfield-Pranav-2018-10-30: `multianimalproject:` left bare.
    observation, findings = _validate_with_null_keys(tmp_path, valid_config_data, "multianimalproject")

    assert isinstance(
        observation,
        ValidConfigObservation,
    )
    assert observation.value.multianimalproject is False
    assert [(str(f.code), f.severity, f.field_location, f.evidence) for f in findings] == [
        ("CONFIG_NULL_KEY_DEFAULTED", Severity.NOTE, ("multianimalproject",), {"default": False})
    ]


def test_each_defaulted_null_key_is_reported_once(
    tmp_path: Path,
    valid_config_data: dict,
) -> None:
    _, findings = _validate_with_null_keys(tmp_path, valid_config_data, "skeleton", "multianimalproject")
    reported = [(f.field_location, f.evidence["default"]) for f in findings]

    assert sorted(reported) == [(("multianimalproject",), False), (("skeleton",), [])]


def test_a_null_key_whose_field_accepts_none_is_not_reported(
    tmp_path: Path,
    valid_config_data: dict,
) -> None:
    observation, findings = _validate_with_null_keys(tmp_path, valid_config_data, "identity")

    assert isinstance(observation, ValidConfigObservation)
    assert observation.value.identity is None
    assert findings == ()


def test_validate_invalid_project_config_retains_raw_data(
    invalid_project: Path,
) -> None:
    config_path = invalid_project / "config.yaml"

    observation, findings = validate_project_config(
        config_path,
    )

    assert isinstance(
        observation,
        FailedConfigObservation,
    )
    assert observation.config_path == config_path
    assert observation.raw_config is not None
    assert np.isnan(observation.raw_config["pcutoff"])

    assert findings
    assert all(finding.code == "CONFIG_VALIDATION_ERROR" for finding in findings)
    assert all(finding.severity == Severity.ERROR for finding in findings)
    assert all(finding.schema_id == "dlc.project.config" for finding in findings)
    assert any(finding.field_location == ("pcutoff",) for finding in findings)


def test_validate_missing_config_returns_read_failure(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"

    observation, findings = validate_project_config(
        config_path,
    )

    assert isinstance(
        observation,
        FailedConfigObservation,
    )
    assert observation.config_path == config_path
    assert observation.raw_config is None

    assert len(findings) == 1
    assert findings[0].code == "CONFIG_READ_FAILED"
    assert findings[0].severity == Severity.ERROR
    assert findings[0].schema_id == "dlc.project.config"
    assert findings[0].path == config_path


def test_validate_empty_config_returns_parse_failure(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "",
        encoding="utf-8",
    )

    observation, findings = validate_project_config(
        config_path,
    )

    assert isinstance(
        observation,
        FailedConfigObservation,
    )
    assert observation.raw_config is None

    assert len(findings) == 1
    assert findings[0].code == "CONFIG_PARSE_FAILED"
    assert findings[0].severity == Severity.ERROR
    assert findings[0].schema_id == "dlc.project.config"


def test_validate_project_config_does_not_modify_file(
    valid_project: Path,
) -> None:
    config_path = valid_project / "config.yaml"

    before = (
        config_path.read_bytes(),
        config_path.stat().st_mtime_ns,
    )

    validate_project_config(
        config_path,
    )

    after = (
        config_path.read_bytes(),
        config_path.stat().st_mtime_ns,
    )

    assert after == before


def test_validate_config_from_a_newer_deeplabcut_is_reported_not_raised(
    tmp_path: Path,
    valid_config_data: dict,
) -> None:
    """A config version with no registered migration escapes ``from_dict`` as a bare ValueError."""

    config_path = tmp_path / "config.yaml"

    config_data = dict(valid_config_data)
    config_data["config_version"] = CURRENT_CONFIG_VERSION + 2

    write_config(
        config_path,
        config_data,
    )

    observation, findings = validate_project_config(
        config_path,
    )

    assert isinstance(
        observation,
        FailedConfigObservation,
    )
    assert observation.raw_config is not None

    assert len(findings) == 1
    assert findings[0].code == "CONFIG_LOAD_FAILED"
    assert findings[0].severity == Severity.ERROR
    assert findings[0].schema_id == "dlc.project.config"
    assert findings[0].evidence["error_type"] == "ValueError"
    assert "No migration registered" in findings[0].evidence["error_message"]


def _migration_findings_for(tmp_path: Path, valid_config_data: dict, **fields: object) -> list[Finding]:
    config_data = dict(valid_config_data)
    config_data.pop("config_version", None)
    config_data.update(fields)
    write_config(tmp_path / "config.yaml", config_data)

    observation, findings = validate_project_config(tmp_path / "config.yaml")

    assert isinstance(observation, ValidConfigObservation)

    return [finding for finding in findings if finding.field_location == ("config_version",)]


def test_a_config_at_the_current_version_reports_no_migration(
    tmp_path: Path,
    valid_config_data: dict,
) -> None:
    assert _migration_findings_for(tmp_path, valid_config_data, config_version=CURRENT_CONFIG_VERSION) == []


def test_an_older_config_migrated_up_is_reported(
    tmp_path: Path,
    valid_config_data: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Uses DeepLabCut's registered ProjectConfig 0 -> 1 migration, with the current version patched to 1."""

    monkeypatch.setattr(versioning, "CURRENT_CONFIG_VERSION", 1)

    (finding,) = _migration_findings_for(tmp_path, valid_config_data)

    assert finding.code == "CONFIG_VERSION_UPGRADED"
    assert finding.severity == Severity.NOTE
    assert finding.evidence == {"declared_version": 0, "loaded_version": 1}


def test_a_newer_config_migrated_down_is_reported(
    tmp_path: Path,
    valid_config_data: dict,
) -> None:
    """DeepLabCut registers a ProjectConfig 1 -> 0 migration, so a version-1 config loads today as version 0."""

    (finding,) = _migration_findings_for(tmp_path, valid_config_data, config_version=CURRENT_CONFIG_VERSION + 1)

    assert finding.code == "CONFIG_VERSION_DOWNGRADED"
    assert finding.severity == Severity.WARNING
    assert finding.evidence == {
        "declared_version": CURRENT_CONFIG_VERSION + 1,
        "loaded_version": CURRENT_CONFIG_VERSION,
    }


def test_validate_project_config_takes_no_unused_project_root() -> None:
    assert "project_root" not in inspect.signature(validate_project_config).parameters


_MISSING = object()


def _config_with(tmp_path: Path, valid_config_data: dict, **fields: object) -> Path:
    data = dict(valid_config_data)
    for field, value in fields.items():
        if value is _MISSING:
            data.pop(field, None)
        else:
            data[field] = value
    write_config(tmp_path / "config.yaml", data)
    return tmp_path / "config.yaml"


@pytest.mark.parametrize(
    ("bodyparts", "declared"),
    [("", {"bodyparts": ""}), ([], {"bodyparts": []}), (_MISSING, {})],
)
def test_a_single_animal_config_without_bodyparts_declares_no_keypoints(
    tmp_path: Path, valid_config_data: dict, bodyparts: object, declared: dict
) -> None:
    observation, findings = validate_project_config(_config_with(tmp_path, valid_config_data, bodyparts=bodyparts))

    assert isinstance(observation, ValidConfigObservation)  # reported, not refused
    (finding,) = findings
    assert (str(finding.code), finding.severity, finding.field_location) == (
        "CONFIG_NO_KEYPOINTS",
        Severity.WARNING,
        ("bodyparts",),
    )
    assert finding.evidence == {"multianimal_project": False, "declared": declared}


@pytest.mark.parametrize("value", [[], _MISSING])
def test_a_multi_animal_config_without_keypoints_declares_no_keypoints(
    tmp_path: Path, valid_config_data: dict, value: object
) -> None:
    path = _config_with(
        tmp_path,
        valid_config_data,
        multianimalproject=True,
        individuals=["a", "b"],
        bodyparts="MULTI!",
        multianimalbodyparts=value,
        uniquebodyparts=value,
    )

    _, findings = validate_project_config(path)

    assert [(str(f.code), f.evidence["multianimal_project"]) for f in findings] == [("CONFIG_NO_KEYPOINTS", True)]


@pytest.mark.parametrize(
    "fields",
    [
        {"bodyparts": ["snout"]},
        {
            "multianimalproject": True,
            "individuals": ["a"],
            "bodyparts": "MULTI!",
            "multianimalbodyparts": ["snout"],
            "uniquebodyparts": [],
        },
        {
            "multianimalproject": True,
            "individuals": ["a"],
            "bodyparts": "MULTI!",
            "multianimalbodyparts": [],
            "uniquebodyparts": ["arena"],
        },
    ],
)
def test_a_config_with_any_keypoint_declares_some(tmp_path: Path, valid_config_data: dict, fields: dict) -> None:
    # Control: the check fires on the absence of keypoints only.
    _, findings = validate_project_config(_config_with(tmp_path, valid_config_data, **fields))

    assert findings == ()
