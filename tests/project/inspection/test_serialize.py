from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from dlc_octopus.api import inspect_project
from dlc_octopus.capabilities import get_project_manager_capabilities
from dlc_octopus.discovery.probe import probe_path
from dlc_octopus.schemas.base.probes import (
    FilesystemEntryKind,
    PathProbe,
    PathProbeStatus,
)
from dlc_octopus.schemas.base.resources import (
    ProjectResource,
    UnknownResourceKind,
)
from dlc_octopus.schemas.snapshots import (
    PROJECT_SNAPSHOT_VERSION,
    ProjectSnapshot,
)

_VERSION_FIELD_NAMES = frozenset({"version", "snapshot_version"})

_FINDING_WIRE_KEYS = frozenset(
    {
        "code",
        "severity",
        "schema_id",
        "reach",
        "instance",
        "path",
        "field_location",
        "evidence",
    }
)


def test_project_snapshot_is_json_serializable(
    valid_project: Path,
) -> None:
    serialized = inspect_project(valid_project).model_dump_json()

    assert '"project_root"' in serialized
    assert '"project"' in serialized
    assert '"resources"' in serialized
    assert '"config_observation"' in serialized


def test_project_snapshot_nodes_include_probe(
    valid_project: Path,
) -> None:
    serialized = inspect_project(valid_project).model_dump_json()

    assert '"probe"' in serialized


def test_runtime_config_is_excluded_from_json(
    valid_project: Path,
) -> None:
    serialized = inspect_project(valid_project).model_dump_json()

    assert '"config_path"' in serialized
    assert '"value"' not in serialized


def test_failed_raw_config_is_excluded_from_json(
    invalid_project: Path,
) -> None:
    serialized = inspect_project(invalid_project).model_dump_json()

    assert '"raw_config"' not in serialized


def test_project_snapshot_includes_format_version(
    valid_project: Path,
) -> None:
    snapshot = inspect_project(valid_project)

    assert snapshot.version == PROJECT_SNAPSHOT_VERSION
    assert (
        snapshot.model_dump(
            mode="json",
        )["version"]
        == PROJECT_SNAPSHOT_VERSION
    )


def test_project_snapshot_json_includes_format_version(
    valid_project: Path,
) -> None:
    serialized = inspect_project(valid_project).model_dump_json()

    assert f'"version":"{PROJECT_SNAPSHOT_VERSION}"' in serialized


@pytest.mark.xfail(
    reason="the snapshot format is 0.x and unstable; bumping is owed once a consumer reads one back",
    strict=True,
    raises=AssertionError,
)
def test_snapshot_format_version_has_left_zero() -> None:
    """The prototype declares no compatibility, and must stop doing so before one is relied on.

    ``0.x`` is accurate while the vocabulary still moves and nothing reads a
    stored snapshot. This passes the day the version is raised, which fails the
    suite and asks whether the format is now a contract.
    """

    assert not PROJECT_SNAPSHOT_VERSION.startswith("0.")


def test_project_snapshot_with_findings_round_trips(
    tmp_path: Path,
) -> None:
    snapshot = inspect_project(tmp_path)

    assert snapshot.findings

    restored = ProjectSnapshot.model_validate(json.loads(snapshot.model_dump_json()))

    assert restored.findings == snapshot.findings


def test_findings_serialize_a_pinned_key_set(
    tmp_path: Path,
) -> None:
    # Changing this set changes the wire format; once the format has left 0.x, bump PROJECT_SNAPSHOT_VERSION with it.
    snapshot = inspect_project(tmp_path)

    assert snapshot.findings

    for finding in snapshot.findings:
        assert finding.model_dump(mode="json").keys() == _FINDING_WIRE_KEYS


def test_serialized_findings_carry_no_message(
    tmp_path: Path,
) -> None:
    serialized = inspect_project(tmp_path).model_dump_json()

    assert '"message"' not in serialized


def test_project_snapshot_rejects_unknown_format_version(
    tmp_path: Path,
) -> None:
    payload = inspect_project(tmp_path).model_dump()
    payload["version"] = "2.0"

    with pytest.raises(ValidationError) as raised:
        ProjectSnapshot.model_validate(payload)

    errors = raised.value.errors()

    assert [error["type"] for error in errors] == ["literal_error"]
    assert errors[0]["loc"] == ("version",)


def test_an_unknown_resource_kind_round_trips(
    valid_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``UnknownResourceKind`` is wire-visible, so it must survive the union on the way back."""

    (valid_project / "device-node").touch()

    real_probe_path = probe_path

    def fake_probe_path(path: Path) -> PathProbe:
        if path.name == "device-node":
            return PathProbe(
                status=PathProbeStatus.OBSERVED,
                entry_kind=FilesystemEntryKind.OTHER,
            )

        return real_probe_path(path)

    monkeypatch.setattr(
        "dlc_octopus.discovery.observer.probe_path",
        fake_probe_path,
    )

    snapshot = inspect_project(valid_project)

    assert '"unknown"' in snapshot.model_dump_json()

    resource = snapshot.resource_at("device-node")

    assert resource is not None
    assert resource.kind is UnknownResourceKind.UNKNOWN

    # The whole snapshot cannot round-trip while the config is excluded; see
    # test_project_snapshot_of_a_valid_project_round_trips.
    restored = ProjectResource.model_validate_json(resource.model_dump_json())

    assert restored.kind is UnknownResourceKind.UNKNOWN


@pytest.mark.xfail(
    reason="a snapshot from a newer catalogue is to be refused by its version before any finding "
    "code is resolved; today an unknown code fails resolution alongside the version",
    strict=True,
    raises=AssertionError,
)
def test_a_newer_catalogue_is_refused_by_version(
    tmp_path: Path,
) -> None:
    payload = inspect_project(tmp_path).model_dump(mode="json")
    payload["version"] = "2.0"
    payload["findings"][0]["code"] = "A_CODE_FROM_A_NEWER_CATALOGUE"

    with pytest.raises(ValidationError) as raised:
        ProjectSnapshot.model_validate(payload)

    assert [error["loc"] for error in raised.value.errors()] == [("version",)]


@pytest.mark.xfail(
    reason="the config is excluded to keep `deeplabcut` out of the snapshot read path; "
    "it can be carried once the canonical config schema is in dlccore",
    strict=True,
    raises=ValidationError,
)
def test_project_snapshot_of_a_valid_project_round_trips(
    valid_project: Path,
) -> None:
    snapshot = inspect_project(valid_project)

    restored = ProjectSnapshot.model_validate(json.loads(snapshot.model_dump_json()))

    assert restored.config_observation == snapshot.config_observation


@pytest.mark.owed_before_release(
    reason="the snapshot serializes `version` while capabilities advertises `snapshot_version`",
    raises=AssertionError,
)
def test_snapshot_and_capabilities_name_the_version_field_alike(
    valid_project: Path,
) -> None:
    snapshot_fields = _VERSION_FIELD_NAMES & inspect_project(valid_project).model_dump().keys()
    capability_fields = _VERSION_FIELD_NAMES & get_project_manager_capabilities().model_dump().keys()

    assert snapshot_fields == capability_fields
