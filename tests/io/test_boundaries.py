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

"""``dlccore.io`` imports nothing from ``dlccore`` outside itself, and its suffixes only the standard library."""

import ast
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

import dlccore

PACKAGE_ROOT = Path(dlccore.__file__).parent
IO = "dlccore.io"
SUFFIXES = f"{IO}.suffixes"


def _within(name: str, package: str) -> bool:
    return name == package or name.startswith(f"{package}.")


def _imported_modules(source: str, module: str, *, is_package: bool) -> set[str]:
    """Return every module a source may load by a static import, ``from a import b`` counting ``a.b`` too."""

    found: set[str] = set()

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = module.split(".") if is_package else module.split(".")[:-1]
                base = ".".join([*parts[: len(parts) - (node.level - 1)], *([node.module] if node.module else [])])
            else:
                base = node.module or ""
            found.add(base)
            found.update(f"{base}.{alias.name}" for alias in node.names)

    return found


def _package_modules() -> Iterator[tuple[str, set[str]]]:
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        parts = path.relative_to(PACKAGE_ROOT.parent).with_suffix("").parts
        is_package = parts[-1] == "__init__"
        module = ".".join(parts[:-1] if is_package else parts)
        yield module, _imported_modules(path.read_text(encoding="utf-8"), module, is_package=is_package)


def test_io_imports_nothing_from_dlccore_outside_io() -> None:
    offending = {
        module: sorted(name for name in imports if _within(name, "dlccore") and not _within(name, IO))
        for module, imports in _package_modules()
        if _within(module, IO)
    }

    assert {module: names for module, names in offending.items() if names} == {}


def test_the_scan_reaches_io() -> None:
    # Control: io's own __init__ imports its errors module, so an empty scan cannot pass the test above.
    assert f"{IO}.errors" in dict(_package_modules())[IO]


def test_io_suffixes_imports_only_the_standard_library() -> None:
    imports = dict(_package_modules())[SUFFIXES]

    assert imports  # control: an empty scan would pass vacuously
    assert {name for name in imports if name.split(".")[0] not in sys.stdlib_module_names} == set()


@pytest.mark.parametrize(
    "source",
    ["import dlccore.io.fingerprints\n", "from dlccore import io\n", "from ..io import fingerprints\n"],
)
def test_every_import_form_is_seen(source: str) -> None:
    found = _imported_modules(source, "dlccore.config.utils", is_package=False)

    assert any(_within(name, IO) for name in found)


def test_a_relative_import_in_a_package_init_resolves_inside_it() -> None:
    found = _imported_modules("from . import fingerprints\n", IO, is_package=True)

    assert f"{IO}.fingerprints" in found
