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

from collections.abc import Callable
from pathlib import Path

import pytest


def _create_symlink_or_skip(
    link: Path,
    target: Path,
    *,
    target_is_directory: bool = False,
) -> None:
    """Create a symlink or skip when the platform disallows it."""

    try:
        link.symlink_to(
            target,
            target_is_directory=target_is_directory,
        )
    except (
        NotImplementedError,
        OSError,
    ) as error:
        pytest.skip(f"Symlink creation is unavailable: {error}")


@pytest.fixture
def create_symlink_or_skip() -> Callable[..., None]:
    """Return a helper that creates a symlink or skips the test cleanly."""

    return _create_symlink_or_skip
