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

import re
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Files ported from DeepLabCut are excluded from ruff until linted in a separate PR",
)
def test_no_file_is_excluded_from_ruff() -> None:
    if not PYPROJECT.is_file():
        pytest.skip("Needs the source tree")
    # Text search, not tomllib: Python 3.10 has no TOML parser in the standard library
    excluded = re.search(r"^extend-exclude\s*=\s*\[(.*?)\]", PYPROJECT.read_text(encoding="utf-8"), re.M | re.S)
    assert excluded is None, f"Excluded from ruff: {excluded.group(1)}"
