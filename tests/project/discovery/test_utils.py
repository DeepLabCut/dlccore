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

from collections.abc import Iterator
from pathlib import Path

import pytest

from dlccore.project.discovery.observer import discover_project
from dlccore.project.schemas.base.resources import DirectoryResourceKind, FileResourceKind
from dlccore.project.schemas.utils import flatten_observed_project


def test_flatten_observed_project(tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    labeled_data = tmp_path / "labeled-data"
    dataset = labeled_data / "session-01"

    videos.mkdir()
    dataset.mkdir(parents=True)

    (tmp_path / "config.yaml").write_text(
        "Task: demo",
        encoding="utf-8",
    )
    (videos / "session-01.mp4").touch()
    (videos / "notes.txt").write_text(
        "user notes",
        encoding="utf-8",
    )
    (dataset / "img001.png").touch()
    (dataset / "CollectedData_Sc.h5").touch()

    observed = discover_project(
        tmp_path,
    )

    resources = flatten_observed_project(observed)

    resources_by_path = {resource.project_relative_path: resource.kind for resource in resources}

    assert resources_by_path[Path("config.yaml")] == (FileResourceKind.PROJECT_CONFIG)
    assert resources_by_path[Path("videos")] == (DirectoryResourceKind.VIDEOS_DIRECTORY)
    assert resources_by_path[Path("videos/session-01.mp4")] == (FileResourceKind.SOURCE_VIDEO)
    assert resources_by_path[Path("videos/notes.txt")] == (FileResourceKind.UNKNOWN_FILE)
    assert resources_by_path[Path("labeled-data/session-01")] == (DirectoryResourceKind.LABELED_DATASET)
    assert resources_by_path[Path("labeled-data/session-01/img001.png")] == FileResourceKind.LABELED_FRAME


def test_flat_projection_excludes_missing_nodes(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.yaml").write_text(
        "Task: demo",
        encoding="utf-8",
    )

    observed = discover_project(
        tmp_path,
    )

    resources = flatten_observed_project(observed)

    paths = {resource.project_relative_path for resource in resources}

    assert Path("config.yaml") in paths
    assert Path("videos") not in paths
    assert Path("labeled-data") not in paths


def test_flat_projection_excludes_project_root_by_default(
    tmp_path: Path,
) -> None:
    observed = discover_project(
        tmp_path,
    )

    resources = flatten_observed_project(observed)

    assert all(resource.project_relative_path != Path() for resource in resources)


def test_flat_projection_includes_wrong_type_nodes(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.yaml").write_text(
        "Task: demo",
        encoding="utf-8",
    )
    (tmp_path / "videos").write_text(
        "not a directory",
        encoding="utf-8",
    )

    observed = discover_project(
        tmp_path,
    )

    resources = flatten_observed_project(observed)
    resources_by_path = {resource.project_relative_path: resource.kind for resource in resources}

    assert resources_by_path[Path("videos")] == DirectoryResourceKind.VIDEOS_DIRECTORY


def test_flat_projection_includes_unreadable_nodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()

    (tmp_path / "config.yaml").write_text(
        "Task: demo",
        encoding="utf-8",
    )

    real_iterdir = Path.iterdir

    def fake_iterdir(self: Path, *args: object, **kwargs: object) -> Iterator[Path]:
        if self == videos:
            raise OSError("Directory listing failed")

        return real_iterdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "iterdir", fake_iterdir)

    observed = discover_project(
        tmp_path,
    )

    resources = flatten_observed_project(observed)
    resources_by_path = {resource.project_relative_path: resource.kind for resource in resources}

    assert resources_by_path[Path("videos")] == DirectoryResourceKind.VIDEOS_DIRECTORY
