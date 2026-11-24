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

"""Shared custom I/O exceptions."""


class UnreadableFileError(OSError):
    """A file could not be read.

    Chained from other libs exceptions if applicable; raised directly when a check here fails,
    such as file changed while reading. The message should name the file and failure.
    """


class FileChangedError(OSError):
    """A file was not replaced, as the content changed since the previous read.

    Should name the file and nature of change.
    """
