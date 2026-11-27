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

import gc
from collections.abc import Callable
from pathlib import Path

import pytest

from dlccore.project.schemas.base.finding_codes import FindingCode, Severity


def _create_symlink_or_skip(
    link: Path,
    target: Path,
    *,
    target_is_directory: bool = False,
) -> None:
    """Create a symlink or skip when the platform disallows it."""

    try:
        link.symlink_to(
            target,
            target_is_directory=target_is_directory,
        )
    except (
        NotImplementedError,
        OSError,
    ) as error:
        pytest.skip(f"Symlink creation is unavailable: {error}")


@pytest.fixture
def create_symlink_or_skip() -> Callable[..., None]:
    """Return a helper that creates a symlink or skips the test cleanly."""

    return _create_symlink_or_skip


def _declare_example_codes() -> type[FindingCode]:
    class ExampleFindingCode(FindingCode):
        """Declared for one test; each real namespace lands with the check that first emits it."""

        TEST_NODE_WRONG_TYPE = (
            "TEST_NODE_WRONG_TYPE",
            "A project resource exists at the expected location but has the wrong filesystem type.",
            Severity.ERROR,
            ("resource_kind", "expected_type"),
        )
        TEST_NODE_UNREADABLE = (
            "TEST_NODE_UNREADABLE",
            "A project-layout node could not be read.",
            Severity.ERROR,
            ("resource_kind", "error_type", "error_message"),
        )

    return ExampleFindingCode


_example_codes_declared = False


@pytest.fixture
def example_codes() -> type[FindingCode]:
    """A finding-code namespace that exists for one test only, so the catalogue every other test sees stays closed."""

    global _example_codes_declared
    _example_codes_declared = True

    return _declare_example_codes()


def pytest_runtest_setup() -> None:
    """Collect the example namespace an earlier test declared; an enum class lingers in a reference cycle."""

    global _example_codes_declared

    if _example_codes_declared:
        gc.collect()
        _example_codes_declared = False
