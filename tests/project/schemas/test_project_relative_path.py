"""Identity paths must stay project-relative.

An anchored path is machine-local. `Path.resolve()` on Windows against a mapped
network drive can return a volume-GUID path such as ``\\\\?\\Volume{...}\\...``;
DeepLabCut issue #3348 is that behaviour rewriting a project's ``project_path``
into an unusable form, and DeepLabCut's own ``auxiliaryfunctions`` now falls back
to ``Path.absolute()`` because of it.

These paths are identity for a snapshot and eventually for database primary
keys so the constraint is on the value rather than on any one call: no
resolution mechanism, present or future, can make an identity machine-local if an
anchored path cannot be stored in the first place.
"""

from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest
from pydantic import ValidationError

from dlc_octopus.schemas.base.core import reject_anchored_path
from dlc_octopus.schemas.base.observations import NodeStatus, ObservedNode
from dlc_octopus.schemas.base.probes import (
    FilesystemEntryKind,
    PathProbe,
    PathProbeStatus,
)
from dlc_octopus.schemas.base.resources import FileResourceKind, ProjectResource

_ANCHORED = [
    "Z:/shared/DLC/project",
    "//?/Volume{80D426CE-0000-0000-691B-0D2800000040}/DLC/project",
    "//server/share/project",
    "C:rel_to_drive",
    "/absolute/posix",
]

_RELATIVE = [
    "videos/vid1.mp4",
    "labeled-data/vid1/img001.png",
    "config.yaml",
]

_PROBE = PathProbe(
    status=PathProbeStatus.OBSERVED,
    entry_kind=FilesystemEntryKind.FILE,
)


@pytest.mark.parametrize("raw", _ANCHORED, ids=str)
def test_reject_anchored_path_rejects_an_anchor(raw: str) -> None:
    with pytest.raises(ValueError, match="project-relative"):
        reject_anchored_path(Path(raw))


@pytest.mark.parametrize("raw", _RELATIVE, ids=str)
def test_reject_anchored_path_accepts_a_relative_path(raw: str) -> None:
    assert reject_anchored_path(Path(raw)) == Path(raw)


def test_drive_relative_path_is_rejected_though_it_is_not_absolute() -> None:
    """``C:rel`` resolves against that drive's working directory.

    ``is_absolute()`` reports ``False`` for it, so a check written against
    absoluteness rather than the anchor lets it through.
    """

    drive_relative = Path("C:rel_to_drive")

    assert not drive_relative.is_absolute()

    with pytest.raises(ValueError, match="project-relative"):
        reject_anchored_path(drive_relative)


@pytest.mark.parametrize("raw", _ANCHORED, ids=str)
def test_anchor_is_rejected_whatever_filesystem_the_host_uses(raw: str) -> None:
    """The verdict must come from the value, not from the interpreter reading it.

    A snapshot is written on one platform and validated on another, so handing
    the validator a deliberately POSIX path stands in for running it
    on Linux. ``PurePosixPath("C:rel_to_drive").anchor`` is empty, so a check
    reading ``Path.anchor`` accepts a Windows drive everywhere but on Windows.
    """

    with pytest.raises(ValueError, match="project-relative"):
        reject_anchored_path(PurePosixPath(raw))

    with pytest.raises(ValueError, match="project-relative"):
        reject_anchored_path(PureWindowsPath(raw))


@pytest.mark.parametrize("raw", _RELATIVE, ids=str)
def test_relative_path_is_accepted_whatever_filesystem_the_host_uses(raw: str) -> None:
    assert reject_anchored_path(PurePosixPath(raw)) is not None
    assert reject_anchored_path(PureWindowsPath(raw)) is not None


def test_posix_filename_holding_a_colon_is_accepted() -> None:
    """POSIX allows a colon in a filename, and asking both filesystems must not forbid it."""

    assert reject_anchored_path(PurePosixPath("labeled-data/vid:1/img.png")) is not None


def test_leading_single_letter_colon_is_rejected_on_every_host() -> None:
    """The one accepted cost of asking both filesystems.

    A POSIX directory literally named ``c:foo`` at the head of the path parses
    as a Windows drive and is refused. Accepting genuine ``C:rel`` identity is
    the worse failure, so this is deliberate rather than overlooked.
    """

    with pytest.raises(ValueError, match="project-relative"):
        reject_anchored_path(PurePosixPath("c:foo/bar.png"))


def test_escaping_path_is_accepted_here() -> None:
    """``..`` is a user-data state, not a programmer error.

    A path escaping the project root is reported by validation, never clamped
    at construction; only an anchor means this code built the model wrong.
    """

    assert reject_anchored_path(Path("../outside")) == Path("../outside")


def test_project_resource_rejects_an_anchored_path() -> None:
    with pytest.raises(ValidationError) as raised:
        ProjectResource(
            project_relative_path=Path("Z:/shared/DLC/videos/vid1.mp4"),
            kind=FileResourceKind.SOURCE_VIDEO,
        )

    assert raised.value.errors()[0]["loc"] == ("project_relative_path",)


def test_observed_node_rejects_an_anchored_path() -> None:
    with pytest.raises(ValidationError) as raised:
        ObservedNode(
            schema_id="dlc.project.videos.video",
            kind=FileResourceKind.SOURCE_VIDEO,
            project_relative_path=Path("Z:/shared/DLC/videos/vid1.mp4"),
            probe=_PROBE,
            status=NodeStatus.PRESENT,
        )

    assert raised.value.errors()[0]["loc"] == ("project_relative_path",)
