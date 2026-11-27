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

from dlccore.project.schemas.base.resources import DirectoryResourceKind
from dlccore.project.schemas.base.validation import ValidationDepth
from dlccore.project.validation.registry import (
    ValidatorRegistration,
    get_validator_registration,
    index_registrations,
    iter_validator_registrations,
)
from dlccore.project.validation.results import ValidationResult


def _never_dispatched(
    *_args: object,
    **_kwargs: object,
) -> ValidationResult:
    """Stand-in validator; these tests index registrations without running them."""

    return ValidationResult()


def _registration(
    code: str,
) -> ValidatorRegistration:
    """Return a registration whose validator reports nothing."""

    return ValidatorRegistration(
        code=code,
        validator=_never_dispatched,
        supported_depths=(ValidationDepth.FILENAME_ONLY,),
    )


def test_each_resource_kind_is_registered_once() -> None:
    kinds = tuple(kind for kind, _ in iter_validator_registrations())

    assert kinds
    assert len(kinds) == len(set(kinds))


def test_indexing_rejects_a_resource_kind_registered_twice() -> None:
    """The positive control for the test above, which cannot fail on a manifest of one."""

    duplicated = (
        (
            DirectoryResourceKind.VIDEOS_DIRECTORY,
            _registration("FIRST"),
        ),
        (
            DirectoryResourceKind.VIDEOS_DIRECTORY,
            _registration("SECOND"),
        ),
    )

    with pytest.raises(ValueError, match="registered twice") as raised:
        index_registrations(duplicated)

    message = str(raised.value)

    assert "FIRST" in message
    assert "SECOND" in message


def test_indexing_keeps_distinct_kinds() -> None:
    distinct = (
        (
            DirectoryResourceKind.VIDEOS_DIRECTORY,
            _registration("VIDEOS"),
        ),
        (
            DirectoryResourceKind.LABELED_DATA_DIRECTORY,
            _registration("LABELED_DATA"),
        ),
    )

    index = index_registrations(distinct)

    assert set(index) == {
        DirectoryResourceKind.VIDEOS_DIRECTORY,
        DirectoryResourceKind.LABELED_DATA_DIRECTORY,
    }


def test_lookup_returns_none_for_an_unregistered_kind() -> None:
    assert get_validator_registration(DirectoryResourceKind.LABELED_DATA_DIRECTORY) is None
