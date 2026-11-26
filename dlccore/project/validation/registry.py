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

from collections.abc import Callable
from dataclasses import dataclass

from dlccore.project.schemas.base.resources import (
    DirectoryResourceKind,
    ResourceKind,
)
from dlccore.project.schemas.base.validation import (
    ValidationDepth,
)
from dlccore.project.schemas.snapshots import ProjectSnapshot
from dlccore.project.validation.context import (
    ValidationContext,
)
from dlccore.project.validation.deeplabcut.videos import (
    validate_videos,
)
from dlccore.project.validation.results import (
    ValidationResult,
)

ResourceValidator = Callable[
    [
        ProjectSnapshot,
        ValidationContext | None,
        ValidationDepth,
    ],
    ValidationResult,
]


@dataclass(frozen=True)
class ValidatorRegistration:
    """Registration for one resource-validation suite."""

    code: str
    validator: ResourceValidator
    supported_depths: tuple[ValidationDepth, ...]


_VALIDATOR_REGISTRATIONS = (
    (
        DirectoryResourceKind.VIDEOS_DIRECTORY,
        ValidatorRegistration(
            code="VIDEOS",
            validator=validate_videos,
            supported_depths=(ValidationDepth.FILENAME_ONLY,),
        ),
    ),
)


def index_registrations(
    registrations: tuple[
        tuple[
            ResourceKind,
            ValidatorRegistration,
        ],
        ...,
    ],
) -> dict[
    ResourceKind,
    ValidatorRegistration,
]:
    """Index registrations by resource kind.

    Raises:
        ValueError: If one resource kind is registered more than once.
    """

    index: dict[
        ResourceKind,
        ValidatorRegistration,
    ] = {}

    for kind, registration in registrations:
        if kind in index:
            raise ValueError(
                f"Resource kind {kind.value!r} is registered twice: {index[kind].code!r} and {registration.code!r}."
            )

        index[kind] = registration

    return index


_REGISTRATIONS_BY_KIND = index_registrations(_VALIDATOR_REGISTRATIONS)


def get_validator_registration(
    kind: ResourceKind,
) -> ValidatorRegistration | None:
    """Return the validator registered for one resource kind."""

    return _REGISTRATIONS_BY_KIND.get(kind)


def iter_validator_registrations() -> tuple[
    tuple[
        ResourceKind,
        ValidatorRegistration,
    ],
    ...,
]:
    """Return validator registrations in deterministic order."""

    return _VALIDATOR_REGISTRATIONS
