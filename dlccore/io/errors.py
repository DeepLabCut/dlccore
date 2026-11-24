"""The exceptions for a file that could not be read, and for one that was not replaced because it changed."""


class UnreadableFileError(OSError):
    """A file could not be read.

    Chained from the library's exception when one caused it; raised directly when a check of ours finds the file
    unusable, such as not a regular file or changed while read. The message names the file and says what failed.
    """


class FileChangedError(OSError):
    # Please ensure the message names the file and explains that it changed since it was read.
    """A file was not replaced, as the content changed since the previous read."""
