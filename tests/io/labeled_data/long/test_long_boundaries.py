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

"""``LabeledData``'s table is private to ``long/``, and its public names change only by a reviewed edit here."""

import ast
from pathlib import Path

import dlccore
from dlccore.io.labeled_data.long.model import LabeledData

PACKAGE_ROOT = Path(dlccore.__file__).parent
LONG = PACKAGE_ROOT / "io" / "labeled_data" / "long"
PRIVATE_TABLE = "_df"


def _modules_touching_the_table() -> set[Path]:
    return {
        path.relative_to(PACKAGE_ROOT)
        for path in PACKAGE_ROOT.rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Attribute) and node.attr == PRIVATE_TABLE
    }


def test_the_scan_sees_the_table_in_long() -> None:
    # Control: a scan that matched nothing would let the test below pass.
    assert (LONG / "model.py").relative_to(PACKAGE_ROOT) in _modules_touching_the_table()


def test_the_private_table_is_touched_only_in_long() -> None:
    # Everything outside long/ goes through the public API, so the table's layout can change behind it.
    outside = {path for path in _modules_touching_the_table() if LONG.relative_to(PACKAGE_ROOT) not in path.parents}

    assert outside == set()


def test_the_public_surface_is_pinned() -> None:
    # Adding a name is a reviewed diff here, not drift: an operation that does not need the private table belongs
    # outside the class.
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
