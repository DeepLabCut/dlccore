"""File fingerprints: SHA-256 of whole files."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from dlc_octopus.io.errors import UnreadableFileError
from dlc_octopus.io.file_stat import FileStat

_CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class FileFingerprintResult:
    """The file fingerprint of one file.

    Attributes:
        path: The path as given; never resolved.
        sha256: Hex SHA-256 of the full bytes.
        size_bytes: Bytes hashed.
    """

    path: Path
    sha256: str
    size_bytes: int


def fingerprint_file(path: Path) -> FileFingerprintResult:
    """Hash one file's full bytes, through a symlink.

    Args:
        path: The file to hash.

    Raises:
        UnreadableFileError: If the file is missing, is not a regular file, cannot be read, or changed while it
            was read.
    """

    try:
        before = _signature(path)
    except OSError as error:
        raise UnreadableFileError(f"{path}: cannot be inspected: {error}") from error

    if before is None:
        raise UnreadableFileError(f"{path}: is not a regular file.")

    hasher = hashlib.sha256()
    size = 0

    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(_CHUNK_BYTES), b""):
                hasher.update(chunk)
                size += len(chunk)
    except OSError as error:
        raise UnreadableFileError(f"{path}: cannot be read: {error}") from error

    try:
        after = _signature(path)
    except OSError as error:
        raise UnreadableFileError(f"{path}: changed while it was read: {error}") from error

    if after != before or size != before.size_bytes:
        raise UnreadableFileError(f"{path}: changed while it was read.")

    return FileFingerprintResult(path=path, sha256=hasher.hexdigest(), size_bytes=size)


def _signature(path: Path) -> FileStat | None:
    """Return the stat of the regular file at ``path``, through symlinks, or ``None`` if it is not one."""

    status = os.stat(path)

    if not stat.S_ISREG(status.st_mode):
        return None

    return FileStat.from_os_stat(status)
