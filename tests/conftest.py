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
from importlib import metadata
from pathlib import Path

import pytest

from dlccore.config import ProjectConfig
from dlccore.project.schemas.base.finding_codes import FindingCode, Severity
from dlccore.project.schemas.snapshots import PROJECT_SNAPSHOT_VERSION


def _owed_is_deferred(package_version: str, snapshot_version: str) -> bool:
    """Whether prototype maintenance may still wait: only while both versions are 0.x."""

    return package_version.startswith("0.") and snapshot_version.startswith("0.")


@pytest.fixture
def owed_is_deferred() -> Callable[[str, str], bool]:
    """The rule behind ``owed_before_release``, exposed so it can be tested."""

    return _owed_is_deferred


def _expected_failures_without_raises(items: list[pytest.Item]) -> list[str]:
    """Node ids of tests with an ``xfail`` or ``owed_before_release`` marker that does not name its exception.

    Without ``raises=``, any exception satisfies an xfail, so a test broken by a rename or a fixture error
    still counts as the expected failure.
    """

    return [
        item.nodeid
        for item in items
        if any(
            "raises" not in marker.kwargs
            for name in ("xfail", "owed_before_release")
            for marker in item.iter_markers(name)
        )
    ]


@pytest.fixture
def expected_failures_without_raises() -> Callable[[list[pytest.Item]], list[str]]:
    """The rule behind the collection check on ``raises=``, exposed so it can be tested."""

    return _expected_failures_without_raises


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Refuse an expected failure that does not name its exception, and turn every ``owed_before_release`` test
    into a strict xfail while the package is 0.x.

    Raises:
        pytest.UsageError: If a collected ``xfail`` or ``owed_before_release`` marker has no ``raises=``.
    """

    missing = _expected_failures_without_raises(items)

    if missing:
        raise pytest.UsageError(
            "Every xfail and owed_before_release marker must declare raises=:\n  " + "\n  ".join(missing)
        )

    if not _owed_is_deferred(metadata.version("dlccore"), PROJECT_SNAPSHOT_VERSION):
        return

    for item in items:
        marker = item.get_closest_marker("owed_before_release")

        if marker is not None:
            item.add_marker(
                pytest.mark.xfail(
                    reason=f"owed before release: {marker.kwargs['reason']}",
                    strict=True,
                    raises=marker.kwargs["raises"],
                )
            )


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


@pytest.fixture
def valid_project_config(
    tmp_path: Path,
) -> ProjectConfig:
    """Return a canonical valid project configuration for tests."""

    return ProjectConfig(
        Task="test-project",
        scorer="test-scorer",
        date="2026-01-01",
        project_path=tmp_path,
        bodyparts=["nose", "tail"],
    )


@pytest.fixture
def valid_project(
    tmp_path: Path,
    valid_project_config: ProjectConfig,
) -> Path:
    """Create a minimal project containing a canonical config."""

    (tmp_path / "videos").mkdir()
    (tmp_path / "labeled-data").mkdir()

    valid_project_config.to_yaml(
        tmp_path / "config.yaml",
    )

    return tmp_path
