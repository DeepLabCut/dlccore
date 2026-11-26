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

import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

Rule = Callable[[list], list[str]]


@dataclass(frozen=True)
class _Item:
    nodeid: str
    marks: tuple[pytest.Mark, ...]

    def iter_markers(self, name: str) -> Iterator[pytest.Mark]:
        return (mark for mark in self.marks if mark.name == name)


def _item(*marks: pytest.MarkDecorator) -> _Item:
    return _Item(nodeid="test_x.py::test_x", marks=tuple(mark.mark for mark in marks))


@pytest.mark.parametrize(
    "marks",
    [
        (pytest.mark.xfail(reason="owed", strict=True),),
        (pytest.mark.owed_before_release(reason="owed"),),
        (pytest.mark.xfail(reason="class", strict=True), pytest.mark.xfail(reason="test", raises=AssertionError)),
    ],
    ids=["xfail", "owed_before_release", "one_of_several_xfails"],
)
def test_an_expected_failure_without_raises_is_flagged(
    expected_failures_without_raises: Rule,
    marks: tuple[pytest.MarkDecorator, ...],
) -> None:
    assert expected_failures_without_raises([_item(*marks)]) == ["test_x.py::test_x"]


def test_expected_failures_that_name_their_exception_pass(expected_failures_without_raises: Rule) -> None:
    items = [
        _item(pytest.mark.xfail(reason="owed", strict=True, raises=AssertionError)),
        _item(pytest.mark.owed_before_release(reason="owed", raises=AttributeError)),
        _item(pytest.mark.parametrize("x", [1])),
    ]

    assert expected_failures_without_raises(items) == []


def test_every_collected_expected_failure_names_its_exception(
    expected_failures_without_raises: Rule,
    request: pytest.FixtureRequest,
) -> None:
    items = request.session.items
    expected = [
        item for item in items if item.get_closest_marker("xfail") or item.get_closest_marker("owed_before_release")
    ]

    if not expected:
        pytest.skip("no xfail or owed_before_release test was selected in this run")

    assert expected_failures_without_raises(items) == []


def test_an_xfail_raising_another_exception_fails(tmp_path: Path) -> None:
    # Control: raises= only helps if pytest reports a different exception as a failure, not as the expected one.
    (tmp_path / "test_probe.py").write_text(
        "import pytest\n\n"
        "@pytest.mark.xfail(raises=AssertionError, strict=True)\n"
        "def test_wrong_reason():\n"
        "    undefined_name\n\n"
        "@pytest.mark.xfail(raises=AssertionError, strict=True)\n"
        "def test_right_reason():\n"
        "    assert False\n",
        encoding="utf-8",
    )
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")

    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--rootdir", str(tmp_path), str(tmp_path)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )

    assert "1 failed, 1 xfailed" in run.stdout, run.stdout + run.stderr
