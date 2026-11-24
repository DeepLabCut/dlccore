"""DLC Project Management read-only I/O, built here until it moves to dlccore.

Paths in, typed results out. Imports nothing from ``dlc_octopus`` outside this package;
Octopus reaches it only through ``dlc_octopus.content``, except for the layout's data from ``suffixes``.
"""

from dlc_octopus.io.errors import FileChangedError, UnreadableFileError
from dlc_octopus.io.file_stat import FileStat
from dlc_octopus.io.fingerprints import FileFingerprintResult, fingerprint_file
from dlc_octopus.io.video import Video

__all__ = [
    "FileChangedError",
    "FileFingerprintResult",
    "FileStat",
    "UnreadableFileError",
    "Video",
    "fingerprint_file",
]
