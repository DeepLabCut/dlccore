from collections.abc import Callable
from pathlib import Path

from dlc_octopus.api import inspect_project


def test_inspect_project_does_not_modify_files(
    valid_project: Path,
    file_state: Callable[
        [Path],
        dict[Path, tuple[bytes, int]],
    ],
) -> None:
    before = file_state(valid_project)

    inspect_project(valid_project)

    after = file_state(valid_project)

    assert after == before
