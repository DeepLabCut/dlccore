from __future__ import annotations

from dlc_octopus.schemas.snapshots import ProjectSnapshot
from dlc_octopus.validation.baseline import run_baseline_checks
from dlc_octopus.validation.context import (
    build_validation_context,
)
from dlc_octopus.validation.registry import (
    get_validator_registration,
)
from dlc_octopus.validation.requests import (
    ProjectValidationRequest,
)
from dlc_octopus.validation.results import ValidationResult


def validate_project(
    snapshot: ProjectSnapshot,
    *,
    request: ProjectValidationRequest | None = None,
) -> ValidationResult:
    """Run baseline checks and explicitly requested resource validators."""

    context = build_validation_context(snapshot)
    result = ValidationResult(
        findings=run_baseline_checks(
            snapshot,
            context,
        )
    )

    if request is None:
        return result

    for resource_request in request.resources:
        registration = get_validator_registration(resource_request.kind)

        if registration is None:
            raise ValueError(f"No validator is available for resource kind {resource_request.kind.value!r}.")

        if resource_request.depth not in registration.supported_depths:
            supported = ", ".join(depth.value for depth in registration.supported_depths)

            raise ValueError(
                "Validation depth "
                f"{resource_request.depth.value!r} is not supported for resource kind "
                f"{resource_request.kind.value!r}. Supported depths are: {supported or '<NONE>'}."
            )
        result = result.merge(
            registration.validator(
                snapshot,
                context=context,
                depth=resource_request.depth,
            )
        )

    return result
