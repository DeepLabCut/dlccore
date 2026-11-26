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

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from pydantic import JsonValue

from dlccore.project.discovery.observer import ExpectedEntryKind, status_for_expected_entry
from dlccore.project.discovery.probe import probe_path
from dlccore.project.schemas.base.core import instance_from_path
from dlccore.project.schemas.base.finding_codes import (
    ConfiguredVideoFindingCode,
    StructureFindingCode,
)
from dlccore.project.schemas.base.findings import (
    Finding,
)
from dlccore.project.schemas.base.observations import (
    NodeStatus,
    ObservedNode,
)
from dlccore.project.schemas.base.probes import (
    PathProbe,
)
from dlccore.project.schemas.base.resources import (
    DirectoryResourceKind,
    FileResourceKind,
)
from dlccore.project.schemas.base.symlinks import (
    SymlinkTargetKind,
)
from dlccore.project.schemas.base.validation import (
    ValidationCheck,
    ValidationCheckStatus,
    ValidationDepth,
)
from dlccore.project.schemas.locations import one_of, one_per_project
from dlccore.project.schemas.project_layout.project import PROJECT_CONFIG
from dlccore.project.schemas.project_layout.videos import SOURCE_VIDEO
from dlccore.project.schemas.snapshots import ProjectSnapshot
from dlccore.project.validation.context import ValidationContext
from dlccore.project.validation.results import ValidationResult


@dataclass(frozen=True)
class _ObservedVideos:
    """The source videos in ``videos/``, indexed for reconciliation with config keys.

    Attributes:
        present: Present videos, in layout order.
        by_identity: Present videos by the file they resolve to.
        by_name: Present videos directly under ``videos/``, by file name.
        physical_paths: The lexical path of every source-video node, of any status.
        collision_findings: One finding per video resolving to the same file as another.
    """

    present: tuple[ObservedNode, ...]
    by_identity: dict[Path, tuple[ObservedNode, ...]]
    by_name: dict[str, ObservedNode]
    physical_paths: frozenset[Path]
    collision_findings: tuple[Finding, ...]


def validate_videos(
    snapshot: ProjectSnapshot,
    context: ValidationContext | None,
    depth: ValidationDepth,
) -> ValidationResult:
    """Run the requested video-validation suite."""

    if context is None:
        return ValidationResult(
            checks=(
                ValidationCheck(
                    code="VIDEO_FILENAME_VALIDATION",
                    kind=(DirectoryResourceKind.VIDEOS_DIRECTORY),
                    depth=depth,
                    status=(ValidationCheckStatus.UNAVAILABLE),
                    message=(
                        "Configured video paths could not be "
                        "checked because a canonical project "
                        "config was unavailable."
                    ),
                ),
            ),
        )

    findings = check_configured_video_paths(
        snapshot,
        context,
    )

    return ValidationResult(
        findings=findings,
        checks=(
            ValidationCheck(
                code="VIDEO_FILENAME_VALIDATION",
                kind=(DirectoryResourceKind.VIDEOS_DIRECTORY),
                depth=depth,
                status=(ValidationCheckStatus.COMPLETED),
                message=("Reconciled config-declared video paths with the videos directory."),
            ),
        ),
    )


def check_configured_video_paths(
    snapshot: ProjectSnapshot,
    context: ValidationContext,
) -> tuple[Finding, ...]:
    """Reconcile config-declared video paths with the videos in ``videos/``.

    Each key is classified by the first rule that applies: it resolves to a
    video in ``videos/``; it ends in ``videos/<name>`` for a video here and does
    not exist, or exists as a different file; it exists elsewhere; it is
    missing, broken, unreadable or not a file. Keys are reported as written.
    """

    observed = _index_observed_videos(snapshot)
    findings: list[Finding] = list(observed.collision_findings)
    matched: set[Path] = set()
    under_missing_root: list[tuple[str, ObservedNode]] = []

    for declared_value in sorted(context.config.video_sets):
        checked_path = _absolute_configured_path(
            declared_value,
            project_root=snapshot.project_root,
        )

        identity = _identity(checked_path)
        if identity in observed.by_identity:
            matched.update(node.project_relative_path for node in observed.by_identity[identity])
            continue

        probe = probe_path(checked_path)
        node_status = status_for_expected_entry(probe, expected_kind=ExpectedEntryKind.FILE)
        # Either separator, so a key written on another platform still points to the video
        segments = [segment for segment in re.split(r"[\\/]", declared_value) if segment]
        named = observed.by_name.get(segments[-1]) if segments[-2:-1] == ["videos"] else None

        if named is not None and node_status in (NodeStatus.MISSING, NodeStatus.PRESENT):
            matched.add(named.project_relative_path)

            if node_status == NodeStatus.MISSING:
                under_missing_root.append((declared_value, named))
                continue

            findings.append(
                Finding(
                    code=ConfiguredVideoFindingCode.CONFIGURED_VIDEO_IN_OTHER_COPY,
                    **one_per_project(PROJECT_CONFIG),
                    path=checked_path,
                    field_location=("video_sets", declared_value),
                    evidence={"matched_path": named.project_relative_path.as_posix()},
                )
            )
            continue

        if node_status == NodeStatus.PRESENT:
            findings.append(
                Finding(
                    code=ConfiguredVideoFindingCode.CONFIGURED_VIDEO_OUTSIDE_PROJECT,
                    **one_per_project(PROJECT_CONFIG),
                    path=checked_path,
                    field_location=("video_sets", declared_value),
                )
            )
            continue

        code = _configured_video_path_code(
            node_status,
            probe,
        )

        if (
            code == ConfiguredVideoFindingCode.CONFIGURED_VIDEO_PATH_SYMLINK_TARGET_MISSING
            and checked_path in observed.physical_paths
        ):
            continue

        evidence: dict[str, JsonValue] = {}

        if "error_type" in code.evidence_keys:
            evidence["error_type"] = probe.error_type
            evidence["error_message"] = probe.error_message

        if probe.symlink is not None:
            evidence["raw_target"] = str(probe.symlink.raw_target)
            evidence["absolute_target"] = str(probe.symlink.absolute_target)

        findings.append(
            Finding(
                code=code,
                **one_per_project(PROJECT_CONFIG),
                path=checked_path,
                field_location=(
                    "video_sets",
                    declared_value,
                ),
                evidence=evidence,
            )
        )

    if under_missing_root:
        findings.append(
            Finding(
                code=ConfiguredVideoFindingCode.CONFIGURED_VIDEOS_UNDER_MISSING_ROOT,
                **one_per_project(PROJECT_CONFIG),
                field_location=("video_sets",),
                evidence={
                    "declared_paths": [declared_value for declared_value, _node in under_missing_root],
                    "matched_paths": [node.project_relative_path.as_posix() for _value, node in under_missing_root],
                },
            )
        )

    findings.extend(
        Finding(
            code=ConfiguredVideoFindingCode.VIDEO_NOT_IN_VIDEO_SETS,
            **one_of(SOURCE_VIDEO, instance_from_path(node.project_relative_path)),
            path=snapshot.project_root / node.project_relative_path,
            evidence={"resource_kind": FileResourceKind.SOURCE_VIDEO.value},
        )
        for node in observed.present
        if node.project_relative_path not in matched
    )

    return tuple(findings)


def _configured_video_path_code(
    node_status: NodeStatus,
    probe: PathProbe,
) -> ConfiguredVideoFindingCode:
    """Return the finding code for a rejected video path.

    A broken symlink is reported ahead of the unreadable case because it
    reaches here as ``UNREADABLE`` yet has a distinct cause and remedy.
    """

    if node_status == NodeStatus.MISSING:
        return ConfiguredVideoFindingCode.CONFIGURED_VIDEO_PATH_MISSING

    if probe.symlink is not None and probe.symlink.target_kind == SymlinkTargetKind.MISSING:
        return ConfiguredVideoFindingCode.CONFIGURED_VIDEO_PATH_SYMLINK_TARGET_MISSING

    if node_status == NodeStatus.UNREADABLE:
        return ConfiguredVideoFindingCode.CONFIGURED_VIDEO_PATH_UNREADABLE

    return ConfiguredVideoFindingCode.CONFIGURED_VIDEO_PATH_NOT_FILE


def _absolute_configured_path(
    declared_value: str,
    *,
    project_root: Path,
) -> Path:
    """Return the lexical absolute path for one config declaration."""

    declared_path = Path(declared_value)

    return (declared_path if declared_path.is_absolute() else project_root / declared_path).absolute()


def _index_observed_videos(
    snapshot: ProjectSnapshot,
) -> _ObservedVideos:
    r"""Index the source videos in ``videos/`` for reconciliation with config keys.

    Built from ``snapshot.project`` rather than ``snapshot.resources`` so that
    a wrong-type or unreadable video node - which now also appears as a
    physical resource - is not mistaken for a valid, present one.

    Two videos may legitimately resolve to one file, so a collision is reported
    rather than raised: once per colliding video, at that video, the findings of
    one collision sharing ``collision_group``, the first member's instance.

    Collisions compare fully resolved files, so a chain of links still collides.
    A resolved path is for comparison only and never reaches a finding.
    """

    by_name: dict[str, ObservedNode] = {}
    collisions: dict[
        Path,
        list[tuple[ObservedNode, Path, Path]],
    ] = {}
    findings: list[Finding] = []
    present: list[ObservedNode] = []
    physical_paths: set[Path] = set()

    for node in _iter_source_videos(snapshot.project):
        physical_path = (snapshot.project_root / node.project_relative_path).absolute()
        physical_paths.add(physical_path)

        if node.status != NodeStatus.PRESENT:
            continue

        present.append(node)
        observed_target = physical_path

        if node.symlink is not None and node.symlink.target_kind == SymlinkTargetKind.FILE:
            observed_target = node.symlink.absolute_target

        collisions.setdefault(_identity(observed_target), []).append((node, physical_path, observed_target))

        if len(node.project_relative_path.parts) == 2:
            by_name.setdefault(node.project_relative_path.name, node)

    for members in collisions.values():
        if len(members) < 2:
            continue

        first, _path, _target = members[0]
        group: list[JsonValue] = list(instance_from_path(first.project_relative_path))

        findings.extend(
            Finding(
                code=StructureFindingCode.PROJECT_RESOURCE_PATH_COLLISION,
                **one_of(SOURCE_VIDEO, instance_from_path(member.project_relative_path)),
                path=physical_path,
                evidence={
                    "resource_kind": FileResourceKind.SOURCE_VIDEO.value,
                    "collision_path": str(observed_target),
                    "collision_group": group,
                },
            )
            for member, physical_path, observed_target in members
        )

    return _ObservedVideos(
        present=tuple(present),
        by_identity={
            identity: tuple(node for node, _path, _target in members) for identity, members in collisions.items()
        },
        by_name=by_name,
        physical_paths=frozenset(physical_paths),
        collision_findings=tuple(findings),
    )


def _identity(path: Path) -> Path:
    """Return the file a path resolves to, for comparison only."""

    try:
        return path.resolve()
    except OSError:
        return path


def _iter_source_videos(
    node: ObservedNode,
) -> Iterator[ObservedNode]:
    """Yield source-video nodes, of any status, from a schema-observed subtree."""

    if node.kind == FileResourceKind.SOURCE_VIDEO:
        yield node

    for child in node.schema_children:
        yield from _iter_source_videos(child)
