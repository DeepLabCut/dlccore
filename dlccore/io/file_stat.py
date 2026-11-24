"""What a file looked like when it was read, so a later read or write can tell whether it changed since."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FileStat:
    """A file's identity on disk as ``os.stat`` reports it, providing basic evidence that content has not changed.

    Not a full content check. The inode and device catch a file replaced by another, as an atomic save does, even
    at the same size and modification time.
    Risks :
    - A same-size rewrite in place within the file system's timestamp resolution (up to 2 s on FAT and some network
      shares), and a change another machine made while a network file system still serves cached attributes.
    - An inode a server does not keep stable makes the stat differ, so a check refuses rather than misses.

    Attributes:
        size_bytes: Size.
        mtime_ns: Modification time in nanoseconds; exact, where a ``datetime`` would round to microseconds.
        inode: File index on its device.
        device: Device or volume.
    """

    size_bytes: int
    mtime_ns: int
    inode: int
    device: int

    @classmethod
    def of(cls, path: Path) -> FileStat:
        """Return the current stat of ``path``, through symlinks.

        Raises:
            OSError: If the file cannot be stat'ed, such as when it does not exist.
        """

        return cls.from_os_stat(os.stat(path))

    @classmethod
    def from_os_stat(cls, status: os.stat_result) -> FileStat:
        """Return the identity fields of an ``os.stat`` result."""

        return cls(size_bytes=status.st_size, mtime_ns=status.st_mtime_ns, inode=status.st_ino, device=status.st_dev)
