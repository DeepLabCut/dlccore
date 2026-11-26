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

from pathlib import Path

import pytest

from dlccore.project.discovery.observer import discover_project
from dlccore.project.schemas.base.observations import ObservedNode


@pytest.mark.xfail(
    reason="discover_directory_node and _unreadable update nodes with model_copy, which skips validation",
    raises=AssertionError,
    strict=True,
)
def test_discovery_builds_every_node_through_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "config.yaml").write_text("Task: demo", encoding="utf-8")
    (tmp_path / "videos").mkdir()
    (tmp_path / "labeled-data" / "session-01").mkdir(parents=True)

    copied: list[dict] = []
    original = ObservedNode.model_copy

    def spy(self: ObservedNode, *, update: dict | None = None, deep: bool = False) -> ObservedNode:
        copied.append(update or {})
        return original(self, update=update, deep=deep)

    monkeypatch.setattr(ObservedNode, "model_copy", spy)

    discover_project(tmp_path)

    assert copied == []
