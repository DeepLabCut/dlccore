from __future__ import annotations

import pytest
from pydantic import ValidationError

from dlc_octopus.schemas.base.resources import (
    DirectoryResourceKind,
)
from dlc_octopus.schemas.base.validation import (
    ValidationDepth,
)
from dlc_octopus.validation import (
    ProjectValidationRequest,
    ResourceValidationRequest,
)


def test_validation_request_targets_resource_kinds() -> None:
    request = ProjectValidationRequest(
        resources=(
            ResourceValidationRequest(
                kind=DirectoryResourceKind.VIDEOS_DIRECTORY,
                depth=ValidationDepth.FILENAME_ONLY,
            ),
            ResourceValidationRequest(
                kind=(DirectoryResourceKind.LABELED_DATA_DIRECTORY),
                depth=ValidationDepth.FULL,
            ),
        )
    )

    assert request.resources[0].kind == (DirectoryResourceKind.VIDEOS_DIRECTORY)
    assert request.resources[1].depth == ValidationDepth.FULL


def test_validation_request_rejects_duplicate_resource_kinds() -> None:
    with pytest.raises(
        ValidationError,
        match=("Each resource kind may be requested at most once"),
    ):
        ProjectValidationRequest(
            resources=(
                ResourceValidationRequest(
                    kind=(DirectoryResourceKind.VIDEOS_DIRECTORY),
                    depth=ValidationDepth.FILENAME_ONLY,
                ),
                ResourceValidationRequest(
                    kind=(DirectoryResourceKind.VIDEOS_DIRECTORY),
                    depth=ValidationDepth.FULL,
                ),
            )
        )
