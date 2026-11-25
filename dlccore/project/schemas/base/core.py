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

import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, StringConstraints


class FrozenModel(BaseModel):
    """Common behavior for all results and schema models.

    - Unexpected fields are forbidden
    - Cannot be modified after creation
    - Serializable via pydantic
      - Can produce JSON
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


def path_anchor(
    value: str | Path,
) -> str:
    """Return the drive and root a path carries under either filesystem, or ``""``.

    ``Path`` takes the host filesystem, so a Windows drive is invisible on POSIX.
    Both filesystems are asked, because a path is written on one platform and read on another.
    """

    text = str(value)

    for filesystem in (PureWindowsPath, PurePosixPath):
        anchor = filesystem(text).anchor

        if anchor:
            return anchor

    return ""


def is_absolute_path(
    value: str | Path,
) -> bool:
    r"""Return whether a path is absolute under either filesystem.

    Stricter than carrying an anchor: ``C:project`` and ``\\project`` are anchored but relative.
    """

    text = str(value)

    return PureWindowsPath(text).is_absolute() or PurePosixPath(text).is_absolute()


def reject_anchored_path(
    value: Path,
) -> Path:
    """Reject a path carrying a drive or a root, whatever the host.

    Raises:
        ValueError: If the path is anchored under either filesystem.
    """

    anchor = path_anchor(value)

    if anchor:
        raise ValueError(f"must be project-relative, but {str(value)!r} carries the anchor {anchor!r}")

    return value


ProjectRelativePath = Annotated[Path, AfterValidator(reject_anchored_path)]

# Lowercase hex, as ``hashlib`` writes it: a consumer keying on it must not see one file twice.
Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


def reject_non_segment(value: tuple[str | int, ...]) -> tuple[str | int, ...]:
    r"""Reject instance segments that only a caller's mistake produces: empty, joined or anchored.

    ``..`` is user data, reported by validation rather than refused here, as for
    ``ProjectRelativePath``. ``\`` and ``:`` are legal inside POSIX file names, so
    the anchor is checked on the joined segments, under both filesystems.

    Raises:
        ValueError: If a segment is empty or contains ``/``, or the segments carry a drive or root.
    """

    for segment in value:
        if isinstance(segment, str) and (segment == "" or "/" in segment):
            raise ValueError(f"{segment!r} is not one project-relative path segment")

    if value:
        reject_anchored_path(Path("/".join(str(segment) for segment in value)))

    return value


# Project-relative path segments, then a logical node's key as ints (a shuffle's 95, 2).
LayoutInstance = Annotated[tuple[str | int, ...], AfterValidator(reject_non_segment)]


def instance_from_path(path: Path) -> tuple[str, ...]:
    """Return a project-relative path's segments, as a layout instance.

    Raises:
        ValueError: If the path is anchored under either filesystem.
    """

    return reject_anchored_path(path).parts


_SCHEMA_ID = re.compile(r"dlc(\.[a-z0-9_]+)+")


def check_schema_id_format(value: str) -> str:
    """Reject a malformed layout schema ID; whether it names a declared node is not checked here.

    Raises:
        ValueError: If the ID is not ``dlc`` followed by dot-separated lowercase segments.
    """

    if _SCHEMA_ID.fullmatch(value) is None:
        raise ValueError(f"{value!r} is not a layout schema ID of the form 'dlc.<segment>[.<segment>...]'")

    return value


SchemaId = Annotated[str, AfterValidator(check_schema_id_format)]
