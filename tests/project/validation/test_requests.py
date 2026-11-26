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

import pytest
from pydantic import ValidationError

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
