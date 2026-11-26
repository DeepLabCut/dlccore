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

"""Provenance stored by each container as plain strings in metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version

# The distribution recorded with the stamp
_DISTRIBUTION = "dlccore"

WRITER_KEY = "dlc_writer"
WRITER_VERSION_KEY = "dlc_writer_version"


@dataclass(frozen=True)
class WriterStamp:
    """File provenance and the package version.

    Stored as plain strings only, never as a serialized object, for portability and safety.

    Attributes:
        writer: Which writer, e.g. ``"dlccore.io.labeled_data"``
        version: The version of this package used to write
    """

    writer: str
    version: str

    @classmethod
    def current(cls, writer: str) -> WriterStamp:
        """Return the stamp for ``writer`` at the installed version of this package."""

        return cls(writer=writer, version=version(_DISTRIBUTION))

    def as_metadata(self) -> dict[str, str]:
        """Return the stamp as the plain-string key-value pairs every container stores."""

        return {WRITER_KEY: self.writer, WRITER_VERSION_KEY: self.version}

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> WriterStamp | None:
        """Return the stamp stored in a container's metadata, or ``None`` if neither key is present.

        Raises:
            ValueError: If either key is present but the two are not plain strings.
        """

        if WRITER_KEY not in metadata and WRITER_VERSION_KEY not in metadata:
            return None
        writer, stamp_version = metadata.get(WRITER_KEY), metadata.get(WRITER_VERSION_KEY)
        if not isinstance(writer, str) or not isinstance(stamp_version, str):
            raise ValueError(
                f"malformed writer stamp: {WRITER_KEY}={writer!r}, {WRITER_VERSION_KEY}={stamp_version!r}; "
                "expected two plain strings."
            )
        return cls(writer=str(writer), version=str(stamp_version))
