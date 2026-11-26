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
from pathlib import Path

import pytest

from dlccore.project.schemas.base.resources import ProjectResource
from dlccore.project.schemas.snapshots import ProjectSnapshot


@pytest.fixture
def resources_by_path() -> Callable[
    [ProjectSnapshot],
    dict[Path, ProjectResource],
]:
    """Return a helper that indexes snapshot resources by project path."""

    def index(
        snapshot: ProjectSnapshot,
    ) -> dict[Path, ProjectResource]:
        return {resource.project_relative_path: resource for resource in snapshot.resources}

    return index


@pytest.fixture
def file_state() -> Callable[
    [Path],
    dict[Path, tuple[bytes, int]],
]:
    """Return a helper that captures file contents and modification times."""

    def capture(
        project_root: Path,
    ) -> dict[Path, tuple[bytes, int]]:
        return {
            path.relative_to(project_root): (
                path.read_bytes(),
                path.stat().st_mtime_ns,
            )
            for path in project_root.rglob("*")
            if path.is_file()
        }

    return capture
