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

"""Low-level, typed filesystem probing.

``probe_path()`` is the single primitive that inspects one path on disk and
classifies it. It records raw filesystem facts only - it never decides
schema validity, finding codes, or remediation.
"""

from __future__ import annotations

import stat
from pathlib import Path

from dlccore.project.schemas.base.probes import (
    FilesystemEntryKind,
    PathProbe,
    PathProbeStatus,
)
from dlccore.project.schemas.base.symlinks import SymlinkObservation, SymlinkTargetKind


def probe_path(path: Path) -> PathProbe:
    """Observe one filesystem path without following symlinks unconditionally."""

    try:
        entry_mode = path.lstat().st_mode
    except FileNotFoundError:
        return PathProbe(status=PathProbeStatus.MISSING)
    except (OSError, ValueError) as error:  # ValueError: a path the OS cannot take, such as one holding NUL
        return PathProbe(
            status=PathProbeStatus.INACCESSIBLE,
            error_type=type(error).__name__,
            error_message=str(error),
        )

    if stat.S_ISLNK(entry_mode):
        return _probe_symlink(path)

    if stat.S_ISREG(entry_mode):
        return PathProbe(status=PathProbeStatus.OBSERVED, entry_kind=FilesystemEntryKind.FILE)

    if stat.S_ISDIR(entry_mode):
        return PathProbe(status=PathProbeStatus.OBSERVED, entry_kind=FilesystemEntryKind.DIRECTORY)

    return PathProbe(status=PathProbeStatus.OBSERVED, entry_kind=FilesystemEntryKind.OTHER)


def _probe_symlink(path: Path) -> PathProbe:
    """Observe a confirmed symlink entry and its lexical/resolved target."""

    try:
        raw_target = path.readlink()
        absolute_target = (raw_target if raw_target.is_absolute() else path.parent / raw_target).absolute()
    except OSError as error:
        return PathProbe(
            status=PathProbeStatus.INACCESSIBLE,
            error_type=type(error).__name__,
            error_message=str(error),
        )

    try:
        target_mode = absolute_target.stat().st_mode
    except FileNotFoundError:
        return PathProbe(
            status=PathProbeStatus.OBSERVED,
            entry_kind=FilesystemEntryKind.SYMLINK,
            symlink=SymlinkObservation(
                raw_target=raw_target,
                absolute_target=absolute_target,
                target_kind=SymlinkTargetKind.MISSING,
            ),
        )
    except OSError as error:
        return PathProbe(
            status=PathProbeStatus.OBSERVED,
            entry_kind=FilesystemEntryKind.SYMLINK,
            symlink=SymlinkObservation(
                raw_target=raw_target,
                absolute_target=absolute_target,
                target_kind=SymlinkTargetKind.INACCESSIBLE,
            ),
            error_type=type(error).__name__,
            error_message=str(error),
        )

    if stat.S_ISREG(target_mode):
        target_kind = SymlinkTargetKind.FILE
    elif stat.S_ISDIR(target_mode):
        target_kind = SymlinkTargetKind.DIRECTORY
    else:
        target_kind = SymlinkTargetKind.OTHER

    return PathProbe(
        status=PathProbeStatus.OBSERVED,
        entry_kind=FilesystemEntryKind.SYMLINK,
        symlink=SymlinkObservation(
            raw_target=raw_target,
            absolute_target=absolute_target,
            target_kind=target_kind,
        ),
    )
