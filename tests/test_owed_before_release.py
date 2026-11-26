from __future__ import annotations

from collections.abc import Callable

import pytest


@pytest.mark.parametrize(
    ("package_version", "snapshot_version", "deferred"),
    [
        ("0.0.1", "0.0", True),
        ("1.0.0", "0.0", False),
        ("0.0.1", "1.0", False),
        ("1.0.0", "1.0", False),
    ],
)
def test_owed_work_waits_only_while_both_versions_are_zero(
    owed_is_deferred: Callable[[str, str], bool],
    package_version: str,
    snapshot_version: str,
    *,
    deferred: bool,
) -> None:
    assert owed_is_deferred(package_version, snapshot_version) is deferred


def test_every_owed_test_is_xfailed_while_deferred(request: pytest.FixtureRequest) -> None:
    owed = [item for item in request.session.items if item.get_closest_marker("owed_before_release")]

    if not owed:
        pytest.skip("no owed_before_release test was selected in this run")

    assert all(item.get_closest_marker("xfail") is not None for item in owed)
