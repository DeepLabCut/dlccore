from __future__ import annotations

import pytest

from dlc_octopus.capabilities import (
    get_project_manager_capabilities,
)
from dlc_octopus.schemas.base.resources import (
    DirectoryResourceKind,
)
from dlc_octopus.schemas.base.validation import (
    ValidationDepth,
)
from dlc_octopus.schemas.capabilities import LayoutCapability
from dlc_octopus.schemas.snapshots import (
    PROJECT_SNAPSHOT_VERSION,
    ProjectSnapshot,
)
from dlc_octopus.validation.context import (
    ValidationContext,
)
from dlc_octopus.validation.registry import (
    ValidatorRegistration,
)
from dlc_octopus.validation.results import (
    ValidationResult,
)


def test_capabilities_include_snapshot_version() -> None:
    capabilities = get_project_manager_capabilities()

    assert capabilities.snapshot_version == PROJECT_SNAPSHOT_VERSION


def test_capabilities_advertise_config_validation_as_baseline() -> None:
    capabilities = get_project_manager_capabilities()

    codes = {baseline.code for baseline in capabilities.baseline_validations}

    assert "PROJECT_CONFIG_VALIDATION" in codes


def test_capabilities_describe_declared_layout() -> None:
    capabilities = get_project_manager_capabilities()

    assert capabilities.layout.kind == DirectoryResourceKind.PROJECT_ROOT

    child_kinds = {child.kind for child in capabilities.layout.children}

    assert DirectoryResourceKind.VIDEOS_DIRECTORY in child_kinds
    assert DirectoryResourceKind.LABELED_DATA_DIRECTORY in child_kinds


def test_capabilities_mark_repeated_layout_nodes() -> None:
    capabilities = get_project_manager_capabilities()

    labeled_data = next(
        child for child in capabilities.layout.children if child.kind == DirectoryResourceKind.LABELED_DATA_DIRECTORY
    )

    assert any(
        child.repeated and child.kind == DirectoryResourceKind.LABELED_DATASET for child in labeled_data.children
    )


def test_capabilities_derive_validation_depths_from_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def validator(
        snapshot: ProjectSnapshot,
        context: ValidationContext | None,
        depth: ValidationDepth,
    ) -> ValidationResult:
        del snapshot, context, depth

        return ValidationResult()

    registration = ValidatorRegistration(
        code="TEST_ONLY_DONOTUSE",
        validator=validator,
        supported_depths=(ValidationDepth.FILENAME_ONLY,),
    )

    monkeypatch.setattr(
        "dlc_octopus.capabilities.iter_validator_registrations",
        lambda: (
            (
                DirectoryResourceKind.VIDEOS_DIRECTORY,
                registration,
            ),
        ),
    )

    capabilities = get_project_manager_capabilities()

    assert len(capabilities.validations) == 1
    assert capabilities.validations[0].kind == (DirectoryResourceKind.VIDEOS_DIRECTORY)
    assert capabilities.validations[0].supported_depths == (ValidationDepth.FILENAME_ONLY,)


def test_capabilities_are_json_serializable() -> None:
    capabilities = get_project_manager_capabilities()

    serialized = capabilities.model_dump_json()

    assert '"snapshot_version"' in serialized
    assert '"layout"' in serialized
    assert '"validations"' in serialized


def _published_schema_ids(node: LayoutCapability) -> set[str]:
    """Return every schema_id the published layout declares."""

    ids = {node.schema_id}

    for child in node.children:
        ids |= _published_schema_ids(child)

    return ids


@pytest.mark.xfail(
    reason="a training-dataset shuffle is a logical layout node with no resource kind, keyed by "
    "(train_percent, shuffle_index), so shuffle findings take it as their schema_id rather than their dataset; "
    "due with Stack B's training-data work, which observes shuffles outside the tree today",
    strict=True,
    raises=AssertionError,
)
def test_a_training_shuffle_is_a_published_logical_node() -> None:
    layout = get_project_manager_capabilities().layout

    assert "dlc.training_data.dataset.shuffle" in _published_schema_ids(layout)


def test_capabilities_flag_nodes_inside_a_repeated_node() -> None:
    flags: dict[str, bool] = {}
    pending = [get_project_manager_capabilities().layout]

    while pending:
        node = pending.pop()
        flags[node.schema_id] = node.inside_repeated
        pending.extend(node.children)

    assert flags["dlc.project.config"] is False
    assert flags["dlc.videos.source_video"] is True
    assert flags["dlc.labeled_data.dataset"] is True
    assert flags["dlc.labeled_data.napari_metadata"] is True
