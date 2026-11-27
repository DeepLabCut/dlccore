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

"""Ported from DeepLabCut-live-GUI dlclivegui/assets/ascii_art.py by its sole author."""

from __future__ import annotations

import itertools
import json
import math
import os
import re
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata, resources
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import numpy as np

ColorMode = Literal["auto", "always", "never"]
Cells = list[list[str]]
Colors = list[list[tuple[int, int, int]]]

ASCII_RAMP_SIMPLE = "@%#*+=-:. "
ASCII_RAMP_FINE = "@$B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
LOGO_FILENAME = "logo.png"
KEYPOINTS_FILENAME = "logo_keypoints.json"
BODY_RGB = (128, 128, 128)
EDGE_DIM = 0.55
"""Weight of the group colour against `BODY_RGB` in skeleton edges."""
EDGE_GLYPHS = "─╲│╱"  # noqa: RUF001
EDGE_GLYPHS_ASCII = "-\\|/"
KEYPOINT_MARKER = "●"
KEYPOINT_MARKER_ASCII = "O"
TITLE_RGB = (176, 176, 176)
TITLE_LETTER_WIDTH = 10
TITLE_TOP = tuple(
    r"""
    ___       ___       ___
   /\  \     /\__\     /\  \
  /::\  \   /:/  /    /::\  \
 /:/\:\__\ /:/__/    /:/\:\__\
 \:\/:/  / \:\  \    \:\ \/__/
  \::/  /   \:\__\    \:\__\
   \/__/     \/__/     \/__/
""".strip("\n").splitlines()
)
"""Letters from the FIGlet font smisome1 (Lennert Stock, Kent Nassen, M. Lindsey), as are those of `TITLE_BOTTOM`."""
TITLE_BOTTOM = tuple(
    r"""
    ___       ___       ___       ___       ___       ___       ___
   /\  \     /\  \     /\  \     /\  \     /\  \     /\__\     /\  \
  /::\  \   /::\  \    \:\  \   /::\  \   /::\  \   /:/ _/_   /::\  \
 /:/\:\__\ /:/\:\__\   /::\__\ /:/\:\__\ /::\:\__\ /:/_/\__\ /\:\:\__\
 \:\/:/  / \:\ \/__/  /:/\/__/ \:\/:/  / \/\::/  / \:\/:/  / \:\:\/__/
  \::/  /   \:\__\    \/__/     \::/  /     \/__/   \::/  /   \::/  /
   \/__/     \/__/               \/__/               \/__/     \/__/
""".strip("\n").splitlines()
)
DISTRIBUTION = "dlccore"
ONLY_ADOPTING_ARMS = True
"""Draw an arm only once its library depends on `DISTRIBUTION`; False draws every arm."""
HUB = "hub"
CORE_ARM = "arm-4"
"""dlccore's arm: the package itself, not an adopter, so drawn like the head and hub."""
ALWAYS_SHOWN = frozenset({"head", HUB, CORE_ARM})
"""Keypoint groups drawn whatever is installed."""
ADOPTERS = {
    "arm-1": ("napari-deeplabcut", None),
    "arm-2": ("scene-pipelines", "dlc"),
    "arm-3": ("deeplabcut", None),
    "arm-5": ("deeplabcut", "gui"),
    "arm-6": ("deeplabcut-live", None),
    "arm-7": ("deeplabcut-live-gui", None),
}
"""Each arm's distribution and extra; the arm is drawn once that distribution, with that extra, requires dlccore."""
ARM_LABELS = {
    "arm-1": "napari-DLC",
    "arm-2": "SCENE-pipelines",
    "arm-3": "DLC",
    "arm-4": "dlccore",
    "arm-5": "DLC-GUI",
    "arm-6": "DLC-Live",
    "arm-7": "DLC-Live-GUI",
}
"""Text drawn by each arm's tip, in the arm's colour."""
SHOW_ARM_LABELS = True
"""Draw `ARM_LABELS` by the tips of the drawn arms; False draws no labels."""
COPYRIGHT = "© A. & M.W. Mathis Labs · deeplabcut.org"
"""From DeepLabCut's source file header; centred under the banner in `TITLE_RGB`."""
COPYRIGHT_ASCII = "(c) A. & M.W. Mathis Labs - deeplabcut.org"
SHOW_COPYRIGHT = True
"""End the banner with `COPYRIGHT`, or `COPYRIGHT_ASCII` where stdout cannot encode it."""
SHOW_TITLE = True
"""Draw `TITLE_TOP` and `TITLE_BOTTOM`; False draws the logo alone. Labels are not placed around the title."""


@dataclass(frozen=True)
class KeypointGroup:
    """One skeleton group of the logo, in 0-1 image coordinates (x right, y down)."""

    name: str
    color: tuple[int, int, int]
    points: tuple[tuple[float, float], ...]
    lines: tuple[tuple[tuple[float, float], ...], ...]


def enable_windows_ansi_support() -> None:
    """Enable ANSI escape support in Windows terminals; no-op elsewhere."""
    if os.name == "nt":
        os.system("")


def get_terminal_width(default: int = 80) -> int:
    """Return terminal width in columns, or `default` if stdout is not a TTY."""
    try:
        if not sys.stdout.isatty():
            return default
        return shutil.get_terminal_size((default, 24)).columns
    except (OSError, ValueError):
        return default


def should_use_color(mode: ColorMode = "auto") -> bool:
    """Return whether to emit ANSI color: `auto` means a TTY with NO_COLOR unset."""
    if mode == "never":
        return False
    if mode == "always":
        return True
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def terminal_is_wide_enough(min_width: int = 60) -> bool:
    """Return whether stdout is a TTY at least `min_width` columns wide."""
    if not sys.stdout.isatty():
        return False
    return get_terminal_width() >= min_width


def _resource_path(filename: str) -> Path:
    """Return `filename` as shipped with this package."""
    return Path(str(resources.files(__package__) / filename))


def logo_path() -> Path:
    """Return the path of the logo image."""
    return _resource_path(LOGO_FILENAME)


def keypoints_path() -> Path:
    """Return the path of the logo keypoint file."""
    return _resource_path(KEYPOINTS_FILENAME)


def _parse_group(group: dict) -> KeypointGroup:
    r, g, b = group["color"]
    return KeypointGroup(
        name=str(group["name"]),
        color=(int(r), int(g), int(b)),
        points=tuple((float(x), float(y)) for x, y in group["points"]),
        lines=tuple(tuple((float(x), float(y)) for x, y in line) for line in group["lines"]),
    )


def load_keypoints(path: Path) -> list[KeypointGroup] | None:
    """Return the logo keypoint groups in draw order, or None if `path` does not exist.

    Raises:
        ValueError: If the file is not a valid keypoint file.
    """
    if not path.is_file():
        return None
    try:
        return [_parse_group(g) for g in json.loads(path.read_text(encoding="utf-8"))["groups"]]
    except (KeyError, TypeError) as e:
        raise ValueError(f"Invalid logo keypoint file {path}: {e!r}") from e


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _installed_requirements(distribution: str) -> list[str] | None:
    """Return the requirement lines of an installed `distribution`, or None if it is not installed."""
    try:
        return metadata.requires(distribution) or []
    except metadata.PackageNotFoundError:
        return None


def adopting_arms(requires: Callable[[str], list[str] | None] = _installed_requirements) -> frozenset[str]:
    """Return the arms of `ADOPTERS` whose distribution, with its extra, requires `DISTRIBUTION`.

    Raises:
        ImportError: If packaging is not installed.
        ValueError: If a requirement line cannot be parsed.
    """
    from packaging.requirements import Requirement

    arms = set()
    for arm, (distribution, extra) in ADOPTERS.items():
        for line in requires(distribution) or ():
            requirement = Requirement(line)
            if _normalize(requirement.name) != DISTRIBUTION:
                continue
            if requirement.marker is None or requirement.marker.evaluate({"extra": extra or ""}):
                arms.add(arm)
    return frozenset(arms)


def shown_groups(groups: list[KeypointGroup], arms: frozenset[str]) -> list[KeypointGroup]:
    """Return `groups` keeping `ALWAYS_SHOWN` and the given `arms` only."""
    return [group for group in groups if group.name in ALWAYS_SHOWN or group.name in arms]


def _to_bgr(img: np.ndarray) -> np.ndarray:
    import cv2 as cv

    if img.ndim == 2:
        return cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    raise ValueError(f"Unsupported image shape for BGR conversion: {img.shape!r}")


def _composite_over_color(img: np.ndarray, bg_bgr: tuple[int, int, int]) -> np.ndarray:
    import numpy as np

    if img.ndim == 3 and img.shape[2] == 4:
        alpha = (img[..., 3].astype(np.float32) / 255.0)[..., None]
        bgr = img[..., :3].astype(np.float32)
        bg = np.empty_like(bgr)
        bg[...] = bg_bgr
        out = alpha * bgr + (1.0 - alpha) * bg
        return np.clip(out, 0, 255).astype(np.uint8)
    return _to_bgr(img)


def _content_box(img_bgra: np.ndarray, alpha_thresh: int, pad: int) -> tuple[int, int, int, int]:
    """Return the inclusive (x0, y0, x1, y1) box around pixels with alpha above `alpha_thresh`."""
    import numpy as np

    mask = img_bgra[..., 3] > alpha_thresh
    h, w = mask.shape
    if not mask.any():
        return 0, 0, w - 1, h - 1
    ys, xs = np.where(mask)
    return max(0, xs.min() - pad), max(0, ys.min() - pad), min(w - 1, xs.max() + pad), min(h - 1, ys.max() + pad)


def _resize_for_terminal(img: np.ndarray, width: int, aspect: float) -> np.ndarray:
    import cv2 as cv

    h, w = img.shape[:2]
    width = max(20, int(width))
    new_h = max(1, int((h / w) * width * aspect))
    return cv.resize(img, (width, new_h), interpolation=cv.INTER_AREA)


def _map_luminance_to_chars(gray: np.ndarray, ramp: str) -> list[str]:
    import numpy as np

    ramp_arr = np.array(list(ramp), dtype="<U1")
    idx = (gray.astype(np.float32) / 255.0 * (len(ramp) - 1)).astype(np.int32)
    return ["".join(row.tolist()) for row in ramp_arr[idx]]


def _stdout_can_encode(text: str) -> bool:
    try:
        text.encode(sys.stdout.encoding or "ascii")
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def _edge_glyph(dx: float, dy: float, aspect: float, glyphs: str) -> str:
    """Return the glyph of `glyphs` (horizontal, down-right, vertical, up-right) nearest a cell step."""
    angle = math.degrees(math.atan2(dy / aspect, dx)) % 180
    return glyphs[int((angle + 22.5) % 180 // 45)]


def _place_label(
    text: str, tip: tuple[int, int], shape: tuple[int, int], blocked: set[tuple[int, int]], *, aspect: float
) -> tuple[int, int] | None:
    """Return the (row, column) nearest `tip` where `text` and one cell either side avoid `blocked`, or None.

    Distance runs from `tip` to the nearest character of `text`, a row counting as `1 / aspect` columns; ties go to
    the placement whose middle is nearest `tip`'s column.
    """
    (n_rows, n_cols), (ti, tj) = shape, tip

    def distance(cell: tuple[int, int]) -> tuple[float, float]:
        i, j = cell
        return math.hypot((i - ti) / aspect, max(j - tj, 0, tj - j - len(text) + 1)), abs(j + (len(text) - 1) / 2 - tj)

    for i, j in sorted(itertools.product(range(n_rows), range(n_cols - len(text) + 1)), key=distance):
        if all((i, k) not in blocked for k in range(max(0, j - 1), min(n_cols, j + len(text) + 1))):
            return i, j
    return None


def _draw_keypoints(
    chars: Cells,
    keypoints: list[KeypointGroup],
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    aspect: float,
    labels: dict[str, str] | None = None,
) -> tuple[Cells, Colors]:
    """Draw keypoint edges and markers over body `chars`, which cover `box` of an image of `image_size`.

    Each group named in `labels` then gets its text over the body as near its last point as the skeleton allows.
    """
    n_rows, n_cols = len(chars), len(chars[0])
    x0, y0, x1, y1 = box
    sx, sy = image_size[0] / (x1 - x0 + 1) * n_cols, image_size[1] / (y1 - y0 + 1) * n_rows
    ox, oy = x0 / image_size[0] * sx, y0 / image_size[1] * sy

    unicode_ok = _stdout_can_encode(EDGE_GLYPHS + KEYPOINT_MARKER)
    glyphs = EDGE_GLYPHS if unicode_ok else EDGE_GLYPHS_ASCII
    marker = KEYPOINT_MARKER if unicode_ok else KEYPOINT_MARKER_ASCII
    diagonals = {glyphs[1], glyphs[3]}
    # Several edges leave the hub; a diagonal beside it on its own row reads as leaving it upward.
    beside_hub = {
        (math.floor(y * sy - oy), math.floor(x * sx - ox) + side)
        for group in keypoints
        if group.name == HUB
        for x, y in group.points
        for side in (-1, 1)
    }
    colors = [[BODY_RGB] * n_cols for _ in chars]
    skeleton: set[tuple[int, int]] = set()

    def put(col: float, row: float, ch: str, rgb: tuple[int, int, int]) -> None:
        i, j = int(row), int(col)
        if 0 <= i < n_rows and 0 <= j < n_cols:
            chars[i][j], colors[i][j] = ch, rgb
            skeleton.add((i, j))

    for group in keypoints:
        edge_rgb = tuple(round(c * EDGE_DIM + b * (1 - EDGE_DIM)) for c, b in zip(group.color, BODY_RGB, strict=True))
        for line in group.lines:
            cells = [(x * sx - ox, y * sy - oy) for x, y in line]
            for (ax, ay), (bx, by) in itertools.pairwise(cells):
                ch = _edge_glyph(bx - ax, by - ay, aspect, glyphs)
                (ci, cj), (di, dj) = (math.floor(ay), math.floor(ax)), (math.floor(by), math.floor(bx))
                n = max(1, abs(di - ci), abs(dj - cj))
                for k in range(n + 1):
                    i, j = math.floor(ci + (di - ci) * k / n + 0.5), math.floor(cj + (dj - cj) * k / n + 0.5)
                    if ch not in diagonals or (i, j) not in beside_hub:
                        put(j, i, ch, edge_rgb)
        for x, y in group.points:
            put(x * sx - ox, y * sy - oy, marker, group.color)

    for group in keypoints:
        text = (labels or {}).get(group.name)
        if text is None or not group.points:
            continue
        x, y = group.points[-1]  # an arm's points run from the hub to its tip
        at = _place_label(text, (int(y * sy - oy), int(x * sx - ox)), (n_rows, n_cols), skeleton, aspect=aspect)
        if at is None:
            continue
        i, j = at
        for k in range(max(0, j - 1), min(n_cols, j + len(text) + 1)):
            chars[i][k] = " "
            skeleton.add((i, k))
        for k, ch in enumerate(text):
            chars[i][j + k], colors[i][j + k] = ch, group.color
    return chars, colors


def _emit(chars: Cells, colors: Colors, *, use_color: bool) -> list[str]:
    """Join character cells into lines, with 24-bit ANSI color if `use_color`."""
    if not use_color:
        return ["".join(row).rstrip() for row in chars]
    lines = []
    for char_row, color_row in zip(chars, colors, strict=True):
        parts: list[str] = []
        prev = None
        for ch, rgb in zip(char_row, color_row, strict=True):
            if rgb != prev and ch != " ":
                parts.append(f"\x1b[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m")
                prev = rgb
            parts.append(ch)
        lines.append("".join(parts).rstrip() + "\x1b[0m")
    return lines


def _ascii_cells(
    image_path: Path,
    *,
    width: int,
    aspect: float = 0.5,
    fine: bool = False,
    invert: bool = False,
    crop_content: bool = False,
    alpha_thresh: int = 1,
    crop_pad: int = 0,
    bg_bgr: tuple[int, int, int] = (255, 255, 255),
    keypoints: list[KeypointGroup] | None = None,
    labels: dict[str, str] | None = None,
) -> tuple[Cells, Colors]:
    """Return the image as character cells and their RGB colors.

    Arguments are as for `generate_ascii_lines`, plus `labels`: text drawn by the tip of each named keypoint group.
    """
    import cv2 as cv

    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    img = cv.imread(str(image_path), cv.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to load image with OpenCV: {image_path}")

    image_size = (img.shape[1], img.shape[0])
    box = (0, 0, image_size[0] - 1, image_size[1] - 1)
    has_alpha = img.ndim == 3 and img.shape[2] == 4
    if crop_content and has_alpha:
        box = _content_box(img, alpha_thresh, crop_pad)
        img = img[box[1] : box[3] + 1, box[0] : box[2] + 1, :]
    if keypoints is not None and has_alpha:
        img = img.copy()
        img[..., :3] = 0  # the skeleton is redrawn from `keypoints`
    img_bgr = _resize_for_terminal(_composite_over_color(img, bg_bgr), width, aspect)

    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    if invert:
        gray = 255 - gray
    chars = [list(row) for row in _map_luminance_to_chars(gray, ASCII_RAMP_FINE if fine else ASCII_RAMP_SIMPLE)]
    if keypoints is None:
        return chars, [[(r, g, b) for b, g, r in row] for row in img_bgr.tolist()]
    return _draw_keypoints(chars, keypoints, box, image_size, aspect=aspect, labels=labels)


def generate_ascii_lines(
    image_path: Path,
    *,
    width: int,
    aspect: float = 0.5,
    color: ColorMode = "auto",
    fine: bool = False,
    invert: bool = False,
    crop_content: bool = False,
    alpha_thresh: int = 1,
    crop_pad: int = 0,
    bg_bgr: tuple[int, int, int] = (255, 255, 255),
    keypoints: list[KeypointGroup] | None = None,
) -> list[str]:
    """Load an image and return it as ASCII art lines.

    Args:
        image_path: Input image.
        width: Output width in characters.
        aspect: Character cell height/width ratio.
        color: ANSI color mode.
        fine: Use the 70-character ramp instead of the 10-character one.
        invert: Invert the luminance mapping.
        crop_content: Crop to pixels with alpha above `alpha_thresh`, if the image has alpha.
        alpha_thresh: Alpha threshold for cropping (0-255).
        crop_pad: Pixels of padding kept around the cropped content.
        bg_bgr: Background color transparency is composited over.
        keypoints: If set, the body is drawn in grey from its silhouette and these skeletons over it.

    Returns:
        One string per terminal row.

    Raises:
        ImportError: If numpy or OpenCV is not installed.
        FileNotFoundError: If `image_path` does not exist.
        RuntimeError: If OpenCV cannot decode the image.
    """
    chars, colors = _ascii_cells(
        image_path,
        width=width,
        aspect=aspect,
        fine=fine,
        invert=invert,
        crop_content=crop_content,
        alpha_thresh=alpha_thresh,
        crop_pad=crop_pad,
        bg_bgr=bg_bgr,
        keypoints=keypoints,
    )
    use_color = should_use_color(color)
    if use_color:
        enable_windows_ansi_support()
    return _emit(chars, colors, use_color=use_color)


def _place_title(rows: tuple[str, ...], chars: Cells, colors: Colors, *, top: int) -> None:
    """Draw `rows` centred, `top` rows down, blanking the cells each letter covers plus one either side."""
    text_width = max(len(row) for row in rows)
    left = (len(chars[0]) - text_width) // 2
    for i, row in enumerate(rows):
        if top + i == len(chars):
            chars.append([" "] * len(chars[0]))
            colors.append([TITLE_RGB] * len(chars[0]))
        row = row.ljust(text_width)
        for x0 in range(0, text_width, TITLE_LETTER_WIDTH):
            strokes = [j for j in range(x0, min(x0 + TITLE_LETTER_WIDTH, text_width)) if row[j] != " "]
            if not strokes:
                continue
            for j in range(max(0, strokes[0] - 1), min(text_width, strokes[-1] + 2)):
                chars[top + i][left + j], colors[top + i][left + j] = row[j], TITLE_RGB


def _add_title(chars: Cells, colors: Colors) -> tuple[Cells, Colors]:
    """Put `TITLE_TOP` above the logo and `TITLE_BOTTOM` below it, a blank row apart; omit what does not fit."""
    width = len(chars[0])
    while chars and not "".join(chars[-1]).strip():
        chars, colors = chars[:-1], colors[:-1]
    if max(len(row) for row in TITLE_TOP) > width:
        return chars, colors
    gap = len(TITLE_TOP) + 1
    chars = [[" "] * width for _ in range(gap)] + chars
    colors = [[TITLE_RGB] * width for _ in range(gap)] + colors
    _place_title(TITLE_TOP, chars, colors, top=0)
    if max(len(row) for row in TITLE_BOTTOM) <= width:
        chars.append([" "] * width)
        colors.append([TITLE_RGB] * width)
        _place_title(TITLE_BOTTOM, chars, colors, top=len(chars))
    return chars, colors


def _add_footer(chars: Cells, colors: Colors, text: str, *, gap: bool = True) -> tuple[Cells, Colors]:
    """Put `text` centred below the last drawn row, a blank row apart unless not `gap`; omit it if it does not fit."""
    width = len(chars[0])
    while chars and not "".join(chars[-1]).strip():
        chars, colors = chars[:-1], colors[:-1]
    if len(text) > width:
        return chars, colors
    left = (width - len(text)) // 2
    row = [" "] * left + list(text) + [" "] * (width - left - len(text))
    blank = [[" "] * width] if gap else []
    return [*chars, *blank, row], [*colors, *([[TITLE_RGB] * width] if gap else []), [TITLE_RGB] * width]


def logo_banner(
    *, color: ColorMode = "auto", min_width: int = 60, max_width: int = 120, version_line: str | None = None
) -> str | None:
    """Return the titled logo as ASCII art, or None if the terminal is not a TTY or too narrow.

    Args:
        color: When to colour the output.
        min_width: Narrowest terminal the logo is drawn in.
        max_width: Widest the logo is drawn.
        version_line: Drawn last, centred in the copyright's colour, directly under the copyright when shown.

    Raises:
        ImportError: If numpy or OpenCV is not installed.
        FileNotFoundError: If the logo is missing.
        RuntimeError: If OpenCV cannot decode the logo.
        ValueError: If the logo keypoint file is invalid, or an installed requirement cannot be parsed.
    """
    if not terminal_is_wide_enough(min_width):
        return None
    width = max(min(get_terminal_width(default=max_width), max_width), min_width)
    keypoints = load_keypoints(keypoints_path())
    if keypoints is not None and ONLY_ADOPTING_ARMS:
        keypoints = shown_groups(keypoints, adopting_arms())
    chars, colors = _ascii_cells(
        logo_path(),
        width=width,
        fine=True,
        crop_content=True,
        crop_pad=1,
        keypoints=keypoints,
        labels=ARM_LABELS if SHOW_ARM_LABELS else None,
    )
    if SHOW_TITLE:
        chars, colors = _add_title(chars, colors)
    if SHOW_COPYRIGHT:
        chars, colors = _add_footer(chars, colors, COPYRIGHT if _stdout_can_encode(COPYRIGHT) else COPYRIGHT_ASCII)
    if version_line:
        chars, colors = _add_footer(chars, colors, version_line, gap=not SHOW_COPYRIGHT)
    use_color = should_use_color(color)
    if use_color:
        enable_windows_ansi_support()
    return "\n".join(_emit(chars, colors, use_color=use_color))
