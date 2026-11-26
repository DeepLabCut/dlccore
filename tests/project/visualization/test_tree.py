from pathlib import Path

import pytest

from dlc_octopus.api import inspect_project
from dlc_octopus.schemas.base.observations import ObservedNode
from dlc_octopus.visualization import render_project_tree, render_project_tree_html


def test_render_tree_shows_project_structure(valid_project: Path) -> None:
    videos = valid_project / "videos"
    (videos / "session-01.mp4").touch()

    snapshot = inspect_project(valid_project)
    rendered = snapshot.render_tree()

    assert "project_root" in rendered
    assert "config.yaml" in rendered
    assert "videos" in rendered
    assert "session-01.mp4" in rendered
    assert "[source_video, present]" in rendered


def test_render_tree_shows_missing_fixed_children(
    valid_project: Path,
) -> None:
    (valid_project / "videos").rmdir()
    (valid_project / "labeled-data").rmdir()
    rendered = inspect_project(valid_project).render_tree()

    assert "✗ videos [videos_directory, missing]" in rendered
    assert "✗ labeled-data [labeled_data_directory, missing]" in rendered


def test_render_tree_shows_unknown_resources(
    valid_project: Path,
) -> None:
    videos = valid_project / "videos"
    (videos / "notes.txt").write_text(
        "notes",
        encoding="utf-8",
    )

    rendered = inspect_project(valid_project).render_tree()

    assert "? notes.txt [unknown_file]" in rendered


def test_render_tree_can_hide_unknown_resources(
    valid_project: Path,
) -> None:
    videos = valid_project / "videos"

    (videos / "notes.txt").write_text(
        "notes",
        encoding="utf-8",
    )

    rendered = inspect_project(valid_project).render_tree(
        show_unknown=False,
    )

    assert "notes.txt" not in rendered


def test_render_tree_can_show_schema_ids(
    valid_project: Path,
) -> None:
    rendered = inspect_project(valid_project).render_tree(
        show_schema_ids=True,
    )

    assert "dlc.project" in rendered
    assert "dlc.project.config" in rendered


def test_render_tree_reports_omitted_leaf_children(
    tmp_path: Path,
) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()

    for index in range(5):
        (videos / f"session-{index:02d}.mp4").touch()

    rendered = inspect_project(tmp_path).render_tree(
        max_children_per_kind=2,
    )

    assert "session-00.mp4" in rendered
    assert "session-01.mp4" in rendered
    assert "session-02.mp4" not in rendered
    assert "3 additional children omitted" in rendered


def test_render_tree_without_child_limit(
    tmp_path: Path,
) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()

    for index in range(3):
        (videos / f"session-{index:02d}.mp4").touch()

    rendered = inspect_project(tmp_path).render_tree(
        max_children_per_kind=None,
    )

    assert "session-00.mp4" in rendered
    assert "session-01.mp4" in rendered
    assert "session-02.mp4" in rendered
    assert "additional children omitted" not in rendered


def _project_with_videos(root: Path, *names: str) -> ObservedNode:
    videos = root / "videos"
    videos.mkdir()
    for name in names:
        (videos / name).touch()
    return inspect_project(root).project


def test_html_tree_lists_directories_as_collapsible_details(tmp_path: Path) -> None:
    html = render_project_tree_html(_project_with_videos(tmp_path, "session-00.mp4"))

    assert html.count("<details open>") == 1  # the project root
    assert "<details><summary>" in html  # control: other directories start closed
    assert "session-00.mp4" in html
    assert "[source_video, present]" in html


def test_html_tree_caps_the_files_it_lists_and_counts_the_rest(tmp_path: Path) -> None:
    project = _project_with_videos(tmp_path, *(f"session-{i:02d}.mp4" for i in range(5)))

    capped = render_project_tree_html(project, max_files=0)
    uncapped = render_project_tree_html(project, max_files=None)

    assert "session-" not in capped
    assert "… 5 more files not shown" in capped
    assert all(f"session-{i:02d}.mp4" in uncapped for i in range(5))
    assert "not shown" not in uncapped


def test_html_tree_spends_the_cap_across_directories(tmp_path: Path) -> None:
    project = _project_with_videos(tmp_path, *(f"session-{i:02d}.mp4" for i in range(5)))

    html = render_project_tree_html(project, max_files=2)

    assert sum(f"session-{i:02d}.mp4" in html for i in range(5)) == 2
    assert "… 3 more files not shown" in html


def test_html_tree_counts_unclaimed_files_towards_the_cap_only_when_listing_them(tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "session-00.mp4").touch()
    (videos / "notes.txt").write_text("notes", encoding="utf-8")
    project = inspect_project(tmp_path).project

    assert "… 1 more file not shown" in render_project_tree_html(project, max_files=1)
    assert "not shown" not in render_project_tree_html(project, max_files=1, show_unknown=False)


def test_html_tree_escapes_names(tmp_path: Path) -> None:
    html = render_project_tree_html(_project_with_videos(tmp_path, "a&b.mp4"))

    assert "a&amp;b.mp4" in html
    assert "a&b.mp4" not in html


def test_html_tree_shows_unknown_resources_unless_asked_not_to(tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "notes.txt").write_text("notes", encoding="utf-8")
    project = inspect_project(tmp_path).project

    assert "? notes.txt" in render_project_tree_html(project)
    assert "notes.txt" not in render_project_tree_html(project, show_unknown=False)


def test_html_tree_refuses_a_negative_cap(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_files"):
        render_project_tree_html(_project_with_videos(tmp_path), max_files=-1)


def test_an_observed_project_displays_as_its_tree(tmp_path: Path) -> None:
    project = _project_with_videos(tmp_path, "session-00.mp4")

    class Printer:
        text = ""

        def text(self, value: str) -> None:  # noqa: F811 - records what IPython would print
            Printer.text = value

    printer = Printer()
    project._repr_pretty_(printer, cycle=False)

    assert project._repr_html_() == render_project_tree_html(project)
    assert Printer.text == render_project_tree(project)


def test_a_node_displayed_alone_is_named_by_its_project_relative_path(tmp_path: Path) -> None:
    project = _project_with_videos(tmp_path, "session-00.mp4", "session-01.mp4")
    (videos,) = (child for child in project.schema_children if child.project_relative_path == Path("videos"))
    first, second = videos.schema_children

    assert render_project_tree(first).endswith("videos/session-00.mp4 [source_video, present]")
    assert render_project_tree(first) != render_project_tree(second)
    assert "videos/session-00.mp4 " in render_project_tree_html(first)
    assert render_project_tree(project).splitlines()[0].endswith("project_root [project_root, present]")  # control
