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

"""``dlccore.io``'s import boundaries: nothing from ``dlccore`` outside io, suffixes only the standard library, and
no format module loaded until it is imported.

``dlccore.project`` reaches io only through ``dlccore.project.content``, except for the layout, which imports the
data in ``dlccore.io.suffixes``.
"""

import ast
import subprocess
import sys
from collections.abc import Iterator
from functools import cache
from pathlib import Path

import pytest

import dlccore

PACKAGE_ROOT = Path(dlccore.__file__).parent
IO = "dlccore.io"
SUFFIXES = f"{IO}.suffixes"
PROJECT = "dlccore.project"
CONTENT = f"{PROJECT}.content"
SCHEMAS = f"{PROJECT}.schemas"


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


def _may_import_from_io(module: str, name: str) -> bool:
    """Content may import any of io; the layout may import its suffixes, which are data only."""

    return _within(module, CONTENT) or (_within(module, SCHEMAS) and _within(name, SUFFIXES))


def test_only_content_imports_io_and_the_layout_only_its_suffixes() -> None:
    offending = {
        module: sorted(name for name in imports if _within(name, IO) and not _may_import_from_io(module, name))
        for module, imports in _package_modules()
        if _within(module, PROJECT)
    }

    assert {module: names for module, names in offending.items() if names} == {}


def test_the_layout_may_import_no_other_part_of_io() -> None:
    # Control: the exception for suffixes must not let the layout import io's readers.
    assert _may_import_from_io(f"{SCHEMAS}.project_layout.videos", SUFFIXES)
    assert not _may_import_from_io(f"{SCHEMAS}.project_layout.videos", f"{IO}.labeled_data")
    assert not _may_import_from_io(f"{PROJECT}.validation.structure", SUFFIXES)


def test_the_scan_sees_the_layout_import_io_suffixes() -> None:
    # Control: the layout does import suffixes, so the exception above is exercised.
    assert SUFFIXES in dict(_package_modules())[f"{SCHEMAS}.project_layout.videos"]


def _modules_spelling(literal: str) -> set[str]:
    """Modules holding ``literal`` as a code constant; docstrings, which only mention it, are longer strings."""

    return {
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in PACKAGE_ROOT.rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Constant) and node.value == literal
    }


def test_the_human_labels_prefix_is_spelled_once_in_io() -> None:
    # The layout imports it from io.suffixes, so the project layer and the I/O layer can never claim different files.
    assert _modules_spelling("CollectedData_") == {"io/suffixes.py"}


def test_the_scan_sees_content_import_io() -> None:
    # Control: content does import io, so the scan must report it, or "only content imports io" proves nothing.
    assert any(_within(name, IO) for name in dict(_package_modules())[f"{CONTENT}.fingerprints"])


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


_FRESH_IMPORTS = (IO, f"{IO}.labeled_data.wide", f"{IO}.labeled_data.wide.h5", f"{IO}.labeled_data")
_UNLOAD_AND_IMPORT_EACH = f"""
import importlib, sys
for module in {_FRESH_IMPORTS!r}:
    for name in [name for name in sys.modules if name.split(".")[0] == "dlccore"]:
        del sys.modules[name]
    importlib.import_module(module)
    print("LOADED", module, *(name for name in sys.modules if name.split(".")[0] == "dlccore"))
"""


@cache
def _loaded_by_importing() -> dict[str, frozenset[str]]:
    """Return, per module, the ``dlccore`` modules importing it loads, from one fresh interpreter per session.

    ``dlccore`` is unloaded before each import; third-party modules stay cached, so only ours are reported.
    """

    result = subprocess.run([sys.executable, "-c", _UNLOAD_AND_IMPORT_EACH], capture_output=True, text=True, check=True)
    lines = [line.split()[1:] for line in result.stdout.splitlines() if line.startswith("LOADED ")]

    return {module: frozenset(loaded) for module, *loaded in lines}


def test_importing_io_loads_its_fingerprints() -> None:
    # Control: the fresh interpreter does load what io's __init__ imports, so an absence below means something.
    assert f"{IO}.fingerprints" in _loaded_by_importing()[IO]


def test_each_fresh_import_starts_without_the_previous_ones() -> None:
    # Control: the h5 module is imported just before its grandparent package, which does not import it.
    assert f"{IO}.labeled_data.wide.h5" in _loaded_by_importing()[f"{IO}.labeled_data.wide.h5"]
    assert f"{IO}.labeled_data.wide.h5" not in _loaded_by_importing()[f"{IO}.labeled_data"]


def test_importing_io_does_not_load_the_video_module() -> None:
    assert f"{IO}.video" not in _loaded_by_importing()[IO]


@pytest.mark.parametrize("package", [f"{IO}.labeled_data", f"{IO}.labeled_data.wide"])
def test_importing_a_labeled_data_package_does_not_load_the_h5_module(package: str) -> None:
    assert f"{IO}.labeled_data.wide.h5" not in _loaded_by_importing()[package]


_FILE_MODULE = PACKAGE_ROOT / "io" / "labeled_data" / "file.py"


def _reads_of_stored(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(isinstance(node, ast.Attribute) and node.attr == "_stored" for node in ast.walk(tree))


def test_only_the_file_class_reads_its_storage() -> None:
    assert _reads_of_stored(_FILE_MODULE)  # control: the class itself does, so a zero below means something
    readers = [path for path in PACKAGE_ROOT.rglob("*.py") if path != _FILE_MODULE and _reads_of_stored(path)]

    assert readers == []
