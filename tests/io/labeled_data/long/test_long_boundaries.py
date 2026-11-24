"""``LabeledData``'s table is private to ``long/``, and its public names change only by a reviewed edit here."""

import ast
from pathlib import Path

import dlc_octopus
from dlc_octopus.io.labeled_data.long.model import LabeledData

PACKAGE_ROOT = Path(dlc_octopus.__file__).parent
LONG = PACKAGE_ROOT / "io" / "labeled_data" / "long"
PRIVATE_FRAME = "_frame"


def _modules_touching_the_frame() -> set[Path]:
    return {
        path.relative_to(PACKAGE_ROOT)
        for path in PACKAGE_ROOT.rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Attribute) and node.attr == PRIVATE_FRAME
    }


def test_the_scan_sees_the_frame_in_long() -> None:
    # Control: a scan that matched nothing would let the test below pass.
    assert (LONG / "model.py").relative_to(PACKAGE_ROOT) in _modules_touching_the_frame()


def test_the_private_frame_is_touched_only_in_long() -> None:
    # Everything outside long/ goes through the public API, so the table's layout can change behind it.
    outside = {path for path in _modules_touching_the_frame() if LONG.relative_to(PACKAGE_ROOT) not in path.parents}

    assert outside == set()


def test_the_public_surface_is_pinned() -> None:
    # Adding a name is a reviewed diff here, not drift: operations belong in ops unless they need the frame.
    public = {name for name in dir(LabeledData) if not name.startswith("_")}

    assert public == {
        "from_long_df",
        "from_wide_df",
        "has_likelihood",
        "is_multi_animal",
        "to_arrays",
        "to_long_df",
        "to_wide_df",
    }
