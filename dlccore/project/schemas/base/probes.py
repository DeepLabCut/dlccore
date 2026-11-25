# dlc_octopus/schemas/base/probes.py
"""Typed filesystem probe vocabulary.

A ``PathProbe`` records raw filesystem facts about one path: whether it is
missing, what kind of entry it is, and symlink-target details. It never
assigns validation findings, severity, remediation, or project-level
validity, those remain the responsibility of discovery and validation.
"""

from __future__ import annotations

from enum import Enum

from pydantic import model_validator
from typing_extensions import Self

from dlc_octopus.schemas.base.core import FrozenModel
from dlc_octopus.schemas.base.symlinks import SymlinkObservation


class FilesystemEntryKind(str, Enum):
    """Observed kind of a directory entry itself (not its symlink target)."""

    FILE = "file"
    DIRECTORY = "directory"
    SYMLINK = "symlink"
    OTHER = "other"


class PathProbeStatus(str, Enum):
    """Outcome of attempting to observe one directory entry.

    Describes the entry only. A symlink target that could not be read is
    recorded by ``SymlinkObservation.target_kind``, not here.
    """

    OBSERVED = "observed"
    MISSING = "missing"
    INACCESSIBLE = "inaccessible"


class PathProbe(FrozenModel):
    """Read-only, schema-neutral observation of one filesystem path."""

    status: PathProbeStatus
    entry_kind: FilesystemEntryKind | None = None
    symlink: SymlinkObservation | None = None
    error_type: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_probe_shape(self) -> Self:
        """Enforce invariants between status, entry kind, and symlink details."""

        if self.status == PathProbeStatus.MISSING and (self.entry_kind is not None or self.symlink is not None):
            raise ValueError("A missing probe must not carry an entry kind or symlink details.")

        has_symlink_kind = self.entry_kind == FilesystemEntryKind.SYMLINK
        has_symlink_details = self.symlink is not None

        if has_symlink_kind != has_symlink_details:
            raise ValueError(
                "A symlink probe must include SymlinkObservation details, and only a symlink probe may include them."
            )

        if self.status == PathProbeStatus.INACCESSIBLE and self.error_type is None:
            raise ValueError("An inaccessible probe must retain structured failure information.")

        if self.status == PathProbeStatus.INACCESSIBLE and self.entry_kind is not None:
            raise ValueError("An inaccessible probe must not carry an entry kind, the entry itself was not observed.")

        return self
