from pathlib import Path

import pytest
from pydantic import ValidationError

from dlc_octopus.schemas.base.resources import (
    DirectoryResourceKind,
    FileResourceKind,
    ProjectResource,
    ResourceManagement,
)


@pytest.mark.parametrize(
    ("kind", "management"),
    [
        (
            FileResourceKind.SOURCE_VIDEO,
            ResourceManagement.MANAGED,
        ),
        (
            FileResourceKind.NAPARI_DLC_METADATA,
            ResourceManagement.KNOWN_UNMANAGED,
        ),
        (
            FileResourceKind.UNKNOWN_FILE,
            ResourceManagement.UNDETERMINED,
        ),
        (
            DirectoryResourceKind.UNKNOWN_DIRECTORY,
            ResourceManagement.UNDETERMINED,
        ),
    ],
)
def test_resource_kind_and_management_accept_coherent_combinations(
    kind: FileResourceKind | DirectoryResourceKind,
    management: ResourceManagement,
) -> None:
    resource = ProjectResource(
        project_relative_path=Path("resource"),
        kind=kind,
        management=management,
    )

    assert resource.kind == kind
    assert resource.management == management


@pytest.mark.parametrize(
    ("kind", "management"),
    [
        (
            FileResourceKind.UNKNOWN_FILE,
            ResourceManagement.MANAGED,
        ),
        (
            DirectoryResourceKind.UNKNOWN_DIRECTORY,
            ResourceManagement.KNOWN_UNMANAGED,
        ),
        (
            FileResourceKind.SOURCE_VIDEO,
            ResourceManagement.UNDETERMINED,
        ),
        (
            FileResourceKind.NAPARI_DLC_METADATA,
            ResourceManagement.UNDETERMINED,
        ),
    ],
)
def test_resource_kind_and_management_reject_incoherent_combinations(
    kind: FileResourceKind | DirectoryResourceKind,
    management: ResourceManagement,
) -> None:
    with pytest.raises(
        ValidationError,
        match=("Unknown resource kinds must use undetermined management"),
    ):
        ProjectResource(
            project_relative_path=Path("resource"),
            kind=kind,
            management=management,
        )
