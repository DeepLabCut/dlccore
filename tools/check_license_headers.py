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

"""Check that every Python file is covered by exactly one NOTICE.yml entry and starts with its header.

Reads only; use update_license_headers.py for writing.
"""

import subprocess
import sys
from collections import Counter
from itertools import takewhile
from pathlib import Path

from update_license_headers import load_config, walk_directory


def python_files() -> set[str]:
    """Return the repository's Python files that exist on disk, tracked or untracked but not ignored."""
    command = ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.py"]
    names = subprocess.run(command, capture_output=True, text=True, check=True).stdout.splitlines()
    return {name for name in names if Path(name).is_file()}


def leading_comment(path: Path) -> list[str]:
    """Return the file's first comment block, with comment markers and one following space removed."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.lstrip("#").removeprefix(" ").rstrip() for line in takewhile(lambda line: line.startswith("#"), lines)]


def has_header(path: Path, header: list[str]) -> bool:
    """Return whether the header lines appear, in order and contiguous, in the file's first comment block."""
    comment = leading_comment(path)
    return any(comment[start : start + len(header)] == header for start in range(len(comment) - len(header) + 1))


def main(input_file: str = "NOTICE.yml") -> int:
    """Print each file that is uncovered, covered twice or missing its header; return 1 if there is any, else 0."""
    coverage = Counter()
    missing = []
    for entry in load_config(input_file):
        header = [line.rstrip() for line in entry["header"].splitlines()]
        names = [Path(name).as_posix() for name in walk_directory(entry)]
        coverage.update(names)
        missing += [name for name in names if not has_header(Path(name), header)]

    problems = [f"{name}: missing or outdated license header (see {input_file})" for name in missing]
    problems += [f"{name}: matched by no entry in {input_file}" for name in python_files() - coverage.keys()]
    problems += [f"{name}: matched by {count} entries in {input_file}" for name, count in coverage.items() if count > 1]

    for problem in sorted(problems):
        print(problem)

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
