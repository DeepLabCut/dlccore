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

from collections.abc import Callable
from pathlib import Path

import pytest

from dlccore.project.discovery.observer import discover_project
from dlccore.project.schemas.base.observations import NodeStatus, ObservedNode


@pytest.fixture
def model_copy_updates(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Record the ``update`` of every ``ObservedNode.model_copy`` call."""

    updates: list[dict] = []
    original = ObservedNode.model_copy

    def spy(self: ObservedNode, *, update: dict | None = None, deep: bool = False) -> ObservedNode:
        updates.append(update or {})
        return original(self, update=update, deep=deep)

    monkeypatch.setattr(ObservedNode, "model_copy", spy)
    return updates


@pytest.mark.xfail(
    reason="discover_directory_node attaches children with model_copy, which skips validation",
    raises=AssertionError,
    strict=True,
)
def test_discovery_builds_directory_nodes_through_validation(tmp_path: Path, model_copy_updates: list[dict]) -> None:
    (tmp_path / "config.yaml").write_text("Task: demo", encoding="utf-8")
    (tmp_path / "videos").mkdir()
    (tmp_path / "labeled-data" / "session-01").mkdir(parents=True)

    discover_project(tmp_path)

    assert model_copy_updates == []


@pytest.mark.xfail(
    reason="_unreadable marks a node unreadable with model_copy, which skips validation",
    raises=AssertionError,
    strict=True,
)
def test_discovery_builds_unreadable_nodes_through_validation(
    tmp_path: Path,
    model_copy_updates: list[dict],
    create_symlink_or_skip: Callable[..., None],
) -> None:
    (tmp_path / "config.yaml").write_text("Task: demo", encoding="utf-8")
    (tmp_path / "labeled-data").mkdir()
    create_symlink_or_skip(tmp_path / "videos", tmp_path, target_is_directory=True)

    project = discover_project(tmp_path)

    videos = next(child for child in project.schema_children if child.project_relative_path.name == "videos")
    if (videos.status, videos.traversal_error_type) != (NodeStatus.UNREADABLE, "SymlinkLoop"):
        pytest.fail(f"the symlink loop was not reached: {videos.status}, {videos.traversal_error_type}")

    assert [update for update in model_copy_updates if "status" in update] == []
