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

import os
from enum import Enum
from pathlib import Path

from dlccore.project.discovery.probe import probe_path
from dlccore.project.schemas.base.filesystem import (
    DirectorySchema,
    FileSchema,
    NodeSchema,
)
from dlccore.project.schemas.base.observations import (
    NodeStatus,
    ObservedNode,
)
from dlccore.project.schemas.base.probes import (
    FilesystemEntryKind,
    PathProbe,
    PathProbeStatus,
)
from dlccore.project.schemas.base.resources import (
    DirectoryResourceKind,
    FileResourceKind,
    ProjectResource,
    ResourceKind,
    ResourceManagement,
    UnknownResourceKind,
)
from dlccore.project.schemas.base.symlinks import (
    SymlinkObservation,
    SymlinkTargetKind,
)
from dlccore.project.schemas.project_layout.project import PROJECT

_UNREADABLE_SYMLINK_TARGET_KINDS = frozenset(
    {
        SymlinkTargetKind.MISSING,
        SymlinkTargetKind.INACCESSIBLE,
    }
)


class ExpectedEntryKind(str, Enum):
    """Filesystem type required by a schema node."""

    FILE = "file"
    DIRECTORY = "directory"


def status_for_expected_entry(
    probe: PathProbe,
    *,
    expected_kind: ExpectedEntryKind,
) -> NodeStatus:
    """Derive the ``NodeStatus`` of a schema-expected entry from its probe.

    Consumes only ``PathProbe`` facts, does not touch the filesystem.
    """

    if probe.status == PathProbeStatus.MISSING:
        return NodeStatus.MISSING

    if probe.entry_kind is None:
        # Entry-level access failure (PathProbeStatus.INACCESSIBLE).
        return NodeStatus.UNREADABLE

    if probe.entry_kind == FilesystemEntryKind.SYMLINK:
        symlink = probe.symlink
        assert symlink is not None  # guaranteed by PathProbe invariants

        if symlink.target_kind in _UNREADABLE_SYMLINK_TARGET_KINDS:
            return NodeStatus.UNREADABLE

        expected_target = (
            SymlinkTargetKind.DIRECTORY if expected_kind == ExpectedEntryKind.DIRECTORY else SymlinkTargetKind.FILE
        )

        return NodeStatus.PRESENT if symlink.target_kind == expected_target else NodeStatus.WRONG_TYPE

    matches_expected_kind = (
        expected_kind == ExpectedEntryKind.FILE and probe.entry_kind == FilesystemEntryKind.FILE
    ) or (expected_kind == ExpectedEntryKind.DIRECTORY and probe.entry_kind == FilesystemEntryKind.DIRECTORY)

    return NodeStatus.PRESENT if matches_expected_kind else NodeStatus.WRONG_TYPE


def classify_unclaimed_entry(
    probe: PathProbe,
) -> tuple[
    ResourceKind,
    SymlinkObservation | None,
]:
    """Return which resource kind an unclaimed entry corresponds to, from its probe.

    Consumes only ``PathProbe``, does not touch the filesystem.

    A symlink to a directory is retained as one, rather than refused: the target
    kind already says what it is, and an unknown directory is recorded without
    its contents being interpreted, so nothing follows the link.

    An entry whose kind cannot be determined is classified as
    ``UnknownResourceKind.UNKNOWN`` rather than refused.
    """

    if probe.entry_kind is None:
        return UnknownResourceKind.UNKNOWN, None

    if probe.entry_kind == FilesystemEntryKind.SYMLINK:
        symlink = probe.symlink
        assert symlink is not None  # guaranteed by PathProbe invariants

        if symlink.target_kind == SymlinkTargetKind.FILE:
            return FileResourceKind.UNKNOWN_FILE, symlink

        if symlink.target_kind == SymlinkTargetKind.DIRECTORY:
            return DirectoryResourceKind.UNKNOWN_DIRECTORY, symlink

        return UnknownResourceKind.UNKNOWN, symlink  # broken / unreachable

    if probe.entry_kind == FilesystemEntryKind.FILE:
        return FileResourceKind.UNKNOWN_FILE, None

    if probe.entry_kind == FilesystemEntryKind.DIRECTORY:
        return DirectoryResourceKind.UNKNOWN_DIRECTORY, None

    return UnknownResourceKind.UNKNOWN, None


def expected_kind_for(
    schema: NodeSchema,
) -> ExpectedEntryKind:
    """Return the filesystem kind a schema node declares.

    Raises:
        TypeError: If the schema is neither a file nor a directory node.
    """

    if isinstance(
        schema,
        FileSchema,
    ):
        return ExpectedEntryKind.FILE

    if isinstance(
        schema,
        DirectorySchema,
    ):
        return ExpectedEntryKind.DIRECTORY

    raise TypeError(f"Unsupported schema node type: {type(schema).__name__}")


def entry_matches_expected_kind(
    probe: PathProbe,
    *,
    expected_kind: ExpectedEntryKind,
) -> bool:
    """Whether a plain entry is of the kind a schema declares.

    Only regular files and directories are judged here. Symlinks, access
    failures and unsupported entry types are left to
    ``status_for_expected_entry``, which reports them rather than dropping
    them.
    """

    if probe.entry_kind == FilesystemEntryKind.FILE:
        return expected_kind == ExpectedEntryKind.FILE

    if probe.entry_kind == FilesystemEntryKind.DIRECTORY:
        return expected_kind == ExpectedEntryKind.DIRECTORY

    return True


def discover_node(
    *,
    physical_path: Path,
    project_relative_path: Path,
    schema: NodeSchema,
) -> ObservedNode:
    """Observe one path against one layout-schema node."""

    expected_kind = expected_kind_for(schema)

    probe = probe_path(physical_path)
    status = status_for_expected_entry(
        probe,
        expected_kind=expected_kind,
    )

    return ObservedNode(
        schema_id=schema.schema_id,
        project_relative_path=(project_relative_path),
        probe=probe,
        status=status,
        kind=schema.kind,
        management=schema.management,
    )


def discover_schema_node(
    *,
    physical_path: Path,
    project_relative_path: Path,
    project_root: Path,
    schema: NodeSchema,
    ancestors: frozenset[tuple[int, int]] = frozenset(),
) -> ObservedNode:
    """Observe a schema node, recursively when the schema is a directory."""

    if isinstance(schema, DirectorySchema):
        return discover_directory_node(
            physical_path=physical_path,
            project_relative_path=project_relative_path,
            project_root=project_root,
            schema=schema,
            ancestors=ancestors,
        )

    return discover_node(
        physical_path=physical_path,
        project_relative_path=project_relative_path,
        schema=schema,
    )


def discover_fixed_children(
    *,
    physical_directory: Path,
    project_relative_directory: Path,
    project_root: Path,
    schema: DirectorySchema,
    ancestors: frozenset[tuple[int, int]] = frozenset(),
) -> tuple[ObservedNode, ...]:
    """Observe children with exact names declared by a directory schema.

    Fixed children occupy predetermined locations in the project layout,
    such as ``config.yaml``, ``videos/``, and ``labeled-data/``.

    This function does not discover dynamically named children, such as
    source videos or labeled-dataset directories. Those are described by
    ``DirectorySchema.repeated_children`` and observed separately.

    Example:
        project_root/
        ├── config.yaml             # fixed file
        ├── videos/                 # fixed directory
        │   └── session-01.mp4      # repeated child
        └── labeled-data/           # fixed directory
            └── session-01/         # repeated child
    """

    observations: list[ObservedNode] = []

    for child_schema in schema.children:
        exact_name = child_schema.name.exact

        if exact_name is None:
            raise ValueError(f"Fixed child schema {child_schema.schema_id!r} must define an exact name.")

        observations.append(
            discover_schema_node(
                physical_path=physical_directory / exact_name,
                project_relative_path=(project_relative_directory / exact_name),
                project_root=project_root,
                schema=child_schema,
                ancestors=ancestors,
            )
        )

    return tuple(observations)


def discover_repeated_children(
    *,
    entries: tuple[Path, ...],
    project_relative_directory: Path,
    project_root: Path,
    schema: DirectorySchema,
    ancestors: frozenset[tuple[int, int]] = frozenset(),
) -> tuple[ObservedNode, ...]:
    """Observe dynamically named children declared by a schema.

    A schema claims an entry only when the name matches *and* the entry is of
    the kind the schema declares. An entry of the wrong kind is left unclaimed
    so it is retained as an unidentified resource, rather than reported against
    a schema it never belonged to.
    """

    fixed_names = {child.name.exact for child in schema.children if child.name.exact is not None}

    observations: list[ObservedNode] = []

    for entry in entries:
        if entry.name in fixed_names:
            continue

        probe = probe_path(entry)

        matching_schemas = tuple(
            repeated.child
            for repeated in schema.repeated_children
            if repeated.child.name.matches(entry)
            and entry_matches_expected_kind(probe, expected_kind=expected_kind_for(repeated.child))
        )

        if not matching_schemas:
            continue

        if len(matching_schemas) > 1:
            matching_ids = [matching_schema.schema_id for matching_schema in matching_schemas]

            raise ValueError(f"Filesystem entry {entry!s} matches multiple repeated-child schemas: {matching_ids}")

        observations.append(
            discover_schema_node(
                physical_path=entry,
                project_relative_path=(project_relative_directory / entry.name),
                project_root=project_root,
                schema=matching_schemas[0],
                ancestors=ancestors,
            )
        )

    return tuple(observations)


def claimed_child_names(
    *,
    fixed_children: tuple[ObservedNode, ...],
    repeated_children: tuple[ObservedNode, ...],
) -> frozenset:
    """Return immediate filesystem names claimed by schema observations.

    Missing fixed children are excluded because no corresponding filesystem
    entry exists. Present, unreadable, and wrong-type entries remain claimed
    to prevent them from also being reported as unknown.
    """

    return frozenset(
        child.project_relative_path.name
        for child in fixed_children + repeated_children
        if child.status != NodeStatus.MISSING
    )


def _observe_unknown_resource(
    *,
    physical_path: Path,
    project_root: Path,
) -> ProjectResource | None:
    """Observe one unclaimed immediate filesystem entry.

    Returns ``None`` for an entry that vanished between listing and probing, so
    ``resources`` holds only entries with a filesystem entry, as claimed nodes do.
    An inaccessible entry does exist, and is retained.
    """

    probe = probe_path(physical_path)

    if probe.status == PathProbeStatus.MISSING:
        return None

    kind, symlink = classify_unclaimed_entry(probe)

    return ProjectResource(
        project_relative_path=(physical_path.relative_to(project_root)),
        kind=kind,
        management=(ResourceManagement.UNDETERMINED),
        symlink=symlink,
    )


def discover_unknown_children(
    *,
    entries: tuple[Path, ...],
    claimed_names: frozenset[str],
    project_root: Path,
) -> tuple[ProjectResource, ...]:
    """Retain immediate entries not claimed by a schema."""

    observed = (
        _observe_unknown_resource(
            physical_path=entry,
            project_root=project_root,
        )
        for entry in entries
        if entry.name not in claimed_names
    )

    return tuple(resource for resource in observed if resource is not None)


def discover_directory_node(
    *,
    physical_path: Path,
    project_relative_path: Path,
    project_root: Path,
    schema: DirectorySchema,
    ancestors: frozenset[tuple[int, int]] = frozenset(),
) -> ObservedNode:
    """Observe a directory and classify its immediate contents.

    Fixed and repeated children are observed according to the layout schema.
    Entries that remain unclaimed are retained as unknown resources.

    Directory schemas are observed recursively, through symlinks, while unknown
    directories are recorded without interpreting their contents. A directory that
    is one of its own ancestors, reached through a link, is unreadable and not entered.
    """

    observed = discover_node(
        physical_path=physical_path,
        project_relative_path=project_relative_path,
        schema=schema,
    )

    if observed.status != NodeStatus.PRESENT:
        return observed

    try:
        identity = _directory_identity(physical_path)
    except OSError as error:
        return _unreadable(observed, type(error).__name__, str(error))

    if identity in ancestors:
        return _unreadable(observed, "SymlinkLoop", f"{physical_path} is the same directory as one of its ancestors")

    ancestors = ancestors | {identity}

    try:
        entries = tuple(
            sorted(
                physical_path.iterdir(),
                key=lambda entry: entry.name.casefold(),
            )
        )
    except OSError as error:
        return _unreadable(observed, type(error).__name__, str(error))

    fixed_children = discover_fixed_children(
        physical_directory=physical_path,
        project_relative_directory=project_relative_path,
        project_root=project_root,
        schema=schema,
        ancestors=ancestors,
    )

    repeated_children = discover_repeated_children(
        entries=entries,
        project_relative_directory=project_relative_path,
        project_root=project_root,
        schema=schema,
        ancestors=ancestors,
    )

    claimed_names = claimed_child_names(
        fixed_children=fixed_children,
        repeated_children=repeated_children,
    )

    unclaimed_resources = discover_unknown_children(
        entries=entries,
        claimed_names=claimed_names,
        project_root=project_root,
    )

    # NOTE: @C-Achard 2026-09-03 - this kind of update is not ideal. Consider alternatives.
    return observed.model_copy(
        update={
            "schema_children": fixed_children + repeated_children,
            "unclaimed_resources": unclaimed_resources,
        }
    )


def _unreadable(observed: ObservedNode, error_type: str, error_message: str) -> ObservedNode:
    """Mark a present directory unreadable and say why; its probe, the entry's own observation, is unchanged."""

    return observed.model_copy(
        update={
            "status": NodeStatus.UNREADABLE,
            "traversal_error_type": error_type,
            "traversal_error_message": error_message,
        }
    )


def _directory_identity(path: Path) -> tuple[int, int]:
    """Return the device and inode of the directory a path names, links followed."""

    stat = os.stat(path)

    return stat.st_dev, stat.st_ino


def discover_project(project_root: Path) -> ObservedNode:
    """Observe a project against the canonical DLC layout."""

    return _discover_project_with_schema(
        project_root,
        schema=PROJECT,
    )


def _discover_project_with_schema(
    project_root: Path,
    *,
    schema: DirectorySchema,
) -> ObservedNode:
    return discover_directory_node(
        physical_path=project_root,
        project_relative_path=Path(),
        project_root=project_root,
        schema=schema,
    )
