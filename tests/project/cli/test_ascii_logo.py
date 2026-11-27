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

"""The `--version` logo: an octopus whose arms each stand for one DeepLabCut library.

An arm's keypoints are drawn only once that library, installed, declares a dependency on dlccore. The head and the
hub are always drawn.
"""

import re
from collections.abc import Callable
from pathlib import Path

import pytest

from dlccore.project.cli import ascii_logo
from dlccore.project.cli.ascii_logo import (
    ADOPTERS,
    ALWAYS_SHOWN,
    ARM_LABELS,
    COPYRIGHT,
    COPYRIGHT_ASCII,
    CORE_ARM,
    EDGE_GLYPHS,
    KEYPOINT_MARKER,
    TITLE_BOTTOM,
    TITLE_TOP,
    adopting_arms,
    keypoints_path,
    load_keypoints,
    logo_banner,
    shown_groups,
)

_, DOWN_RIGHT, _, UP_RIGHT = EDGE_GLYPHS

# Which library each arm stands for. Changing an arm's meaning means changing this table, on purpose.
ARM_MEANINGS = {
    "arm-1": "napari-deeplabcut",
    "arm-2": "scene-pipelines[dlc]",
    "arm-3": "deeplabcut (core)",
    "arm-5": "deeplabcut[gui]",
    "arm-6": "deeplabcut-live",
    "arm-7": "deeplabcut-live-gui",
}


def _installed(**requirements: list[str]) -> Callable[[str], list[str] | None]:
    """Fake installed distributions: name -> requirement lines; anything else is not installed."""
    return lambda distribution: requirements.get(distribution)


def test_a_missing_keypoint_file_loads_as_none(tmp_path: Path) -> None:
    assert load_keypoints(tmp_path / "absent.json") is None


@pytest.mark.parametrize(
    "content",
    ['{"group": []}', '{"groups": [{"points": [], "lines": []}]}', '{"groups": [{"color": 3}]}', "not json"],
)
def test_a_malformed_keypoint_file_raises_value_error(tmp_path: Path, content: str) -> None:
    path = tmp_path / "keypoints.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError):
        load_keypoints(path)


def test_each_arm_stands_for_the_library_listed_here() -> None:
    meanings = {arm: f"{name}[{extra}]" if extra else name for arm, (name, extra) in ADOPTERS.items()}

    assert meanings == {arm: meaning.split(" ")[0] for arm, meaning in ARM_MEANINGS.items()}


def test_the_shipped_logo_has_exactly_the_head_the_hub_and_one_group_per_arm() -> None:
    groups = load_keypoints(keypoints_path())

    assert groups is not None
    assert sorted(group.name for group in groups) == sorted([*ALWAYS_SHOWN, *ARM_MEANINGS])
    assert all(group.points for group in groups)


def test_no_library_installed_lights_no_arm() -> None:
    assert adopting_arms(_installed()) == frozenset()


def test_an_installed_library_that_does_not_depend_on_dlccore_lights_no_arm() -> None:
    assert adopting_arms(_installed(deeplabcut=["numpy", "dlccore-extras"])) == frozenset()


@pytest.mark.parametrize("line", ["dlccore", "DLCCore>=0.1", "dlccore ; python_version >= '3'"])
def test_a_library_depending_on_dlccore_lights_its_arm(line: str) -> None:
    assert adopting_arms(_installed(**{"napari-deeplabcut": [line]})) == {"arm-1"}


def test_a_dependency_under_the_dlc_extra_lights_the_scene_pipelines_arm() -> None:
    assert adopting_arms(_installed(**{"scene-pipelines": ["dlccore ; extra == 'dlc'"]})) == {"arm-2"}


def test_a_dependency_under_the_gui_extra_lights_only_the_gui_arm() -> None:
    arms = adopting_arms(_installed(deeplabcut=["dlccore ; extra == 'gui'"]))

    assert arms == {"arm-5"}


def test_a_core_dependency_lights_both_core_and_gui_arms() -> None:
    # A GUI install pulls in core's dependencies, so it uses dlccore too.
    assert adopting_arms(_installed(deeplabcut=["dlccore"])) == {"arm-3", "arm-5"}


def test_every_library_depending_on_dlccore_lights_every_arm() -> None:
    every = {name: ["dlccore"] for name, _ in ADOPTERS.values()}

    assert adopting_arms(_installed(**every)) == set(ARM_MEANINGS)


def test_head_and_hub_are_kept_and_only_lit_arms_are_kept() -> None:
    groups = load_keypoints(keypoints_path())
    assert groups is not None

    kept = shown_groups(groups, frozenset({"arm-2"}))

    assert sorted(group.name for group in kept) == sorted([*ALWAYS_SHOWN, "arm-2"])


def _banner(
    monkeypatch: pytest.MonkeyPatch,
    arms: frozenset[str],
    *,
    width: int = 120,
    unicode: bool = True,
    color: ascii_logo.ColorMode = "never",
    version_line: str | None = None,
) -> str:
    """The banner on a simulated TTY of `width` columns, with `arms` adopting dlccore."""
    monkeypatch.setattr(ascii_logo, "terminal_is_wide_enough", lambda *_: True)
    monkeypatch.setattr(ascii_logo, "get_terminal_width", lambda *_, **__: width)
    monkeypatch.setattr(ascii_logo, "_stdout_can_encode", lambda *_: unicode)
    monkeypatch.setattr(ascii_logo, "adopting_arms", lambda: arms)

    banner = logo_banner(color=color, version_line=version_line)

    assert banner is not None
    return banner


def _markers_in_banner(monkeypatch: pytest.MonkeyPatch, arms: frozenset[str]) -> int:
    return _banner(monkeypatch, arms).count(KEYPOINT_MARKER)


def _labelled_arms(banner: str) -> set[str]:
    """Arms whose label appears whole in `banner`; `DLC` does not match inside `DLC-GUI`."""
    return {arm for arm, text in ARM_LABELS.items() if re.search(rf"(?<![\w-]){re.escape(text)}(?![\w-])", banner)}


def test_the_banner_draws_more_keypoints_as_arms_light_up(monkeypatch: pytest.MonkeyPatch) -> None:
    none = _markers_in_banner(monkeypatch, frozenset())
    one = _markers_in_banner(monkeypatch, frozenset({"arm-1"}))
    every = _markers_in_banner(monkeypatch, frozenset(ARM_MEANINGS))

    assert 0 < none < one < every


def test_turning_the_toggle_off_draws_every_arm_whatever_is_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    every = _markers_in_banner(monkeypatch, frozenset(ARM_MEANINGS))
    monkeypatch.setattr(ascii_logo, "ONLY_ADOPTING_ARMS", False)

    assert _markers_in_banner(monkeypatch, frozenset()) == every


def test_a_diagonal_edge_leaves_its_keypoint_on_the_next_row_not_beside_it(monkeypatch: pytest.MonkeyPatch) -> None:
    # The keypoint sits near the top of its cell and the edge drops under a row in its first step: the first up-right
    # diagonal still belongs below-left of the keypoint, not beside it.
    monkeypatch.setattr(ascii_logo, "_stdout_can_encode", lambda *_: True)
    group = ascii_logo.KeypointGroup(name="arm-3", color=(0, 0, 0), points=(), lines=(((0.60, 0.511), (0.43, 0.61)),))
    chars = [[" "] * 10 for _ in range(10)]

    chars, _ = ascii_logo._draw_keypoints(chars, [group], (0, 0, 9, 9), (10, 10), aspect=0.5)

    assert chars[5][5] == " "
    assert chars[6][5] == UP_RIGHT


@pytest.mark.parametrize("width", [60, 80, 120])
def test_no_diagonal_glyph_sits_right_beside_the_hub(monkeypatch: pytest.MonkeyPatch, width: int) -> None:
    monkeypatch.setattr(ascii_logo, "_stdout_can_encode", lambda *_: True)
    groups = load_keypoints(keypoints_path())
    chars, colors = ascii_logo._ascii_cells(
        ascii_logo.logo_path(), width=width, fine=True, crop_content=True, crop_pad=1, keypoints=groups
    )
    hub = next(group for group in groups if group.name == "hub")
    row, col = next(
        (i, j)
        for i, line in enumerate(chars)
        for j, ch in enumerate(line)
        if ch == KEYPOINT_MARKER and colors[i][j] == hub.color
    )

    assert {chars[row][col - 1], chars[row][col + 1]}.isdisjoint({UP_RIGHT, DOWN_RIGHT})


ANSI = re.compile(r"\x1b\[[0-9;]*m")
NOT_IN_RAMP = "="


def test_every_arm_has_a_label() -> None:
    assert set(ARM_LABELS) == {*ADOPTERS, CORE_ARM}


@pytest.mark.parametrize("arms", [frozenset(), frozenset({"arm-1", "arm-2"}), frozenset(ADOPTERS)])
def test_a_label_is_drawn_exactly_by_the_drawn_arms(monkeypatch: pytest.MonkeyPatch, arms: frozenset[str]) -> None:
    assert _labelled_arms(_banner(monkeypatch, arms)) == {*arms, CORE_ARM}


def test_turning_labels_off_draws_none(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _labelled_arms(_banner(monkeypatch, frozenset(ARM_LABELS))) == set(ARM_LABELS)
    monkeypatch.setattr(ascii_logo, "SHOW_ARM_LABELS", False)

    assert _labelled_arms(_banner(monkeypatch, frozenset(ARM_LABELS))) == set()


def test_labels_never_cover_the_skeleton(monkeypatch: pytest.MonkeyPatch) -> None:
    def skeleton(banner: str) -> list[int]:
        return [banner.count(glyph) for glyph in EDGE_GLYPHS + KEYPOINT_MARKER]

    labelled = _banner(monkeypatch, frozenset(ARM_LABELS))
    monkeypatch.setattr(ascii_logo, "SHOW_ARM_LABELS", False)
    unlabelled = _banner(monkeypatch, frozenset(ARM_LABELS))

    assert labelled != unlabelled
    assert skeleton(labelled) == skeleton(unlabelled)


@pytest.mark.parametrize("width", [60, 61, 69, 70, 79, 80, 99, 120])
def test_every_label_fits_and_no_line_is_wider_than_the_terminal(monkeypatch: pytest.MonkeyPatch, width: int) -> None:
    for color in ("never", "always"):
        lines = ANSI.sub("", _banner(monkeypatch, frozenset(ARM_LABELS), width=width, color=color)).splitlines()

        assert _labelled_arms("\n".join(lines)) == set(ARM_LABELS)
        assert max(len(line) for line in lines) == width  # the logo reaches the edge, so the bound is tight


def test_colour_on_and_off_carry_the_same_characters(monkeypatch: pytest.MonkeyPatch) -> None:
    off = _banner(monkeypatch, frozenset(ARM_LABELS), color="never")
    on = _banner(monkeypatch, frozenset(ARM_LABELS), color="always")

    assert on != off
    assert [line.rstrip() for line in ANSI.sub("", on).splitlines()] == off.splitlines()


def test_the_ascii_fallback_is_pure_ascii_and_keeps_the_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    assert not _banner(monkeypatch, frozenset(ARM_LABELS)).isascii()

    banner = _banner(monkeypatch, frozenset(ARM_LABELS), unicode=False)

    assert banner.isascii()
    assert banner.splitlines()[-1].strip() == COPYRIGHT_ASCII
    assert _labelled_arms(banner) == set(ARM_LABELS)


def test_a_label_too_wide_for_the_terminal_is_dropped_not_clipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(ARM_LABELS, "arm-1", NOT_IN_RAMP * 20)
    assert NOT_IN_RAMP * 20 in _banner(monkeypatch, frozenset({"arm-1"}), width=60)

    monkeypatch.setitem(ARM_LABELS, "arm-1", NOT_IN_RAMP * 61)
    banner = _banner(monkeypatch, frozenset({"arm-1"}), width=60)

    assert NOT_IN_RAMP not in banner
    assert max(len(line) for line in banner.splitlines()) <= 60


def test_a_label_lands_by_the_last_point_not_the_point_farthest_from_the_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    # Cells are (row, column) = (20 y, 40 x): hub at (10, 20), farthest point at (10, 4), last point at (10, 12).
    monkeypatch.setattr(ascii_logo, "_stdout_can_encode", lambda *_: True)
    points = ((0.45, 0.5), (0.1, 0.5), (0.3, 0.5))
    arm = ascii_logo.KeypointGroup(name="arm-1", color=(0, 0, 0), points=points, lines=(points,))
    hub = ascii_logo.KeypointGroup(name="hub", color=(0, 0, 0), points=((0.5, 0.5),), lines=())
    chars = [[" "] * 40 for _ in range(20)]

    chars, _ = ascii_logo._draw_keypoints(
        chars, [arm, hub], (0, 0, 39, 19), (40, 20), aspect=0.5, labels={"arm-1": "abc"}
    )

    row, col = next((i, "".join(line).index("abc")) for i, line in enumerate(chars) if "abc" in "".join(line))
    assert abs(row - 10) <= 1
    assert col == 11  # "abc" is centred on column 12; by the farthest point it would sit near 4
    assert chars[10][12] == KEYPOINT_MARKER


def test_the_title_is_drawn_until_switched_off(monkeypatch: pytest.MonkeyPatch) -> None:
    assert TITLE_TOP[1] in _banner(monkeypatch, frozenset())
    monkeypatch.setattr(ascii_logo, "SHOW_TITLE", False)

    assert TITLE_TOP[1] not in _banner(monkeypatch, frozenset())


def test_the_version_line_is_drawn_directly_under_the_copyright(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = _banner(monkeypatch, frozenset(), version_line="dlccore 1.2.3").splitlines()

    assert [line.strip() for line in lines[-2:]] == [COPYRIGHT, "dlccore 1.2.3"]


def test_without_the_copyright_the_version_line_follows_a_blank_row(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ascii_logo, "SHOW_COPYRIGHT", False)

    lines = _banner(monkeypatch, frozenset(), version_line="dlccore 1.2.3").splitlines()

    assert [line.strip() for line in lines[-2:]] == ["", "dlccore 1.2.3"]


def test_the_copyright_is_the_last_line_until_switched_off(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _banner(monkeypatch, frozenset()).splitlines()[-1].strip() == COPYRIGHT
    monkeypatch.setattr(ascii_logo, "SHOW_COPYRIGHT", False)

    assert COPYRIGHT not in _banner(monkeypatch, frozenset())


def test_the_title_leaves_the_logo_and_its_labels_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ascii_logo, "SHOW_COPYRIGHT", False)
    monkeypatch.setattr(ascii_logo, "SHOW_TITLE", False)
    logo = _banner(monkeypatch, frozenset(ARM_LABELS)).rstrip("\n ").splitlines()
    monkeypatch.setattr(ascii_logo, "SHOW_TITLE", True)

    titled = _banner(monkeypatch, frozenset(ARM_LABELS)).splitlines()

    top = len(TITLE_TOP) + 1
    assert _labelled_arms("\n".join(logo)) == set(ARM_LABELS)
    assert titled[top : top + len(logo)] == logo
    assert [line.strip() for line in titled[-len(TITLE_BOTTOM) :]] == [row.strip() for row in TITLE_BOTTOM]


def test_no_arm_label_is_a_placeholder() -> None:
    assert not [text for text in ARM_LABELS.values() if text.startswith("[PH]")]
