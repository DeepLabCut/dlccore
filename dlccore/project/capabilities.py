from __future__ import annotations

from dlc_octopus.schemas.base.filesystem import (
    NodeSchema,
    child_schemas,
    index_layout,
)
from dlc_octopus.schemas.capabilities import (
    BaselineValidationCapability,
    LayoutCapability,
    ProjectManagerCapabilities,
    ValidationCapability,
)
from dlc_octopus.schemas.project_layout.project import PROJECT
from dlc_octopus.schemas.snapshots import (
    PROJECT_SNAPSHOT_VERSION,
)
from dlc_octopus.validation.baseline import (
    iter_baseline_check_registrations,
)
from dlc_octopus.validation.registry import (
    iter_validator_registrations,
)


def get_project_manager_capabilities() -> ProjectManagerCapabilities:
    """Return the layout and validations supported by this installation."""

    baseline_validations = tuple(
        BaselineValidationCapability(
            code=registration.code,
            description=registration.description,
        )
        for registration in iter_baseline_check_registrations()
    )

    validations = tuple(
        ValidationCapability(
            kind=kind,
            supported_depths=tuple(registration.supported_depths),
        )
        for kind, registration in iter_validator_registrations()
    )

    return ProjectManagerCapabilities(
        snapshot_version=PROJECT_SNAPSHOT_VERSION,
        layout=_describe_layout(PROJECT, inside_repeated=index_layout(PROJECT).inside_repeated),
        baseline_validations=baseline_validations,
        validations=validations,
    )


def _describe_layout(
    schema: NodeSchema,
    *,
    inside_repeated: frozenset[str],
    repeated: bool = False,
) -> LayoutCapability:
    """Project one schema node into public capability metadata."""

    return LayoutCapability(
        schema_id=schema.schema_id,
        kind=schema.kind,
        required=schema.required,
        management=schema.management,
        repeated=repeated,
        inside_repeated=schema.schema_id in inside_repeated,
        children=tuple(
            _describe_layout(child, inside_repeated=inside_repeated, repeated=child_repeated)
            for child, child_repeated in child_schemas(schema)
        ),
    )
