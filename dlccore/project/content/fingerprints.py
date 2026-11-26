"""File fingerprints of a snapshot's file resources, requested from ``dlc_octopus.io`` and reported as findings."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated

from pydantic import Field

from dlc_octopus.content.reads import FailedRead, covered, present_files_in_scope
from dlc_octopus.io import UnreadableFileError, fingerprint_file
from dlc_octopus.schemas.base.core import FrozenModel, ProjectRelativePath, SchemaId, Sha256Hex
from dlc_octopus.schemas.base.filesystem import FileSchema, LayoutIndex
from dlc_octopus.schemas.base.finding_codes import FingerprintFindingCode
from dlc_octopus.schemas.base.findings import Finding
from dlc_octopus.schemas.base.observations import NodeStatus, walk_observed_nodes
from dlc_octopus.schemas.locations import node_location
from dlc_octopus.schemas.project_layout.project import PROJECT_INDEX
from dlc_octopus.schemas.scope import Scope
from dlc_octopus.schemas.snapshots import ProjectSnapshot
from dlc_octopus.validation.results import ValidationResult


class FileFingerprint(FrozenModel):
    """The file fingerprint of one file resource: the SHA-256 of its full bytes.

    Attributes:
        schema_id: The layout node fingerprinted; ``node_location`` gives its finding anchor.
        project_relative_path: The path as the snapshot observed it.
        sha256: Hex SHA-256 of the bytes read.
        size_bytes: Bytes read.
    """

    schema_id: SchemaId
    project_relative_path: ProjectRelativePath
    sha256: Sha256Hex
    size_bytes: Annotated[int, Field(strict=True, ge=0)]

    @property
    def posix_path(self) -> str:
        """The project-relative path with ``/`` separators, the same on every platform."""

        return self.project_relative_path.as_posix()


class ResourcesFileFingerprint(FrozenModel):
    """The requested file fingerprints, and the snapshot with one finding per resource that could not be read.

    Attributes:
        snapshot: The input snapshot with ``FILE_FINGERPRINT_FAILED`` findings merged in.
        fingerprints: One file fingerprint per present file node in scope that could be read.
        failed: One entry per present file node in scope that could not be read.
    """

    snapshot: ProjectSnapshot
    fingerprints: tuple[FileFingerprint, ...] = ()
    failed: tuple[FailedRead, ...] = ()


def resources_file_fingerprint(
    snapshot: ProjectSnapshot,
    scope: Scope,
    *,
    index: LayoutIndex = PROJECT_INDEX,
) -> ResourcesFileFingerprint:
    """File-fingerprint every present file node ``scope`` covers in ``snapshot``, without rediscovering the project.

    Nodes not ``PRESENT`` are skipped: structure validation already reports them. Symlinks are hashed through;
    paths are never resolved. Each such file is in ``fingerprints`` or in ``failed``, never both. A file that cannot
    be read yields a ``FILE_FINGERPRINT_FAILED`` finding, carried by its ``FailedRead`` and merged into the returned
    snapshot. The input snapshot is not modified.

    Args:
        snapshot: A snapshot from ``inspect_project``.
        scope: The nodes whose files to fingerprint, each within its instance prefix; see ``Scope.covers``.
        index: The layout the snapshot was observed against.

    Raises:
        ValueError: If ``scope`` names a node the layout does not declare.
    """

    fingerprints: list[FileFingerprint] = []
    failed: list[FailedRead] = []

    for node in present_files_in_scope(snapshot, scope, scope.file_schema_ids(index=index), index=index):
        path = snapshot.project_root / node.project_relative_path

        try:
            read = fingerprint_file(path)
        except UnreadableFileError as error:
            finding = Finding(
                code=FingerprintFindingCode.FILE_FINGERPRINT_FAILED,
                **node_location(node.schema_id, node.project_relative_path, index=index),
                path=path,
                evidence={
                    "resource_kind": node.kind.value,
                    "error_type": type(error.__cause__ or error).__name__,
                    "error_message": str(error),
                },
            )
            failed.append(FailedRead(node.schema_id, node.project_relative_path, (finding,)))
            continue

        fingerprints.append(
            FileFingerprint(
                schema_id=node.schema_id,
                project_relative_path=node.project_relative_path,
                sha256=read.sha256,
                size_bytes=read.size_bytes,
            )
        )

    findings = tuple(finding for failure in failed for finding in failure.findings)
    return ResourcesFileFingerprint(
        snapshot=snapshot.with_validation_result(ValidationResult(findings=findings)),
        fingerprints=tuple(fingerprints),
        failed=tuple(failed),
    )


def compare_file_fingerprints(
    snapshot: ProjectSnapshot,
    scope: Scope,
    recorded: Iterable[FileFingerprint],
    *,
    index: LayoutIndex = PROJECT_INDEX,
) -> ResourcesFileFingerprint:
    """File-fingerprint ``scope`` as ``resources_file_fingerprint`` does, and compare recorded file fingerprints.

    Matched by schema ID and path. A recorded file whose bytes differ, or that is not fingerprinted because it is
    gone, yields a ``FILE_FINGERPRINT_CHANGED`` finding merged into the returned snapshot. A recorded file that
    could not be read carries ``FILE_FINGERPRINT_FAILED`` only. One inspection found ``UNREADABLE``, or that sits
    in an ``UNREADABLE`` directory, carries nothing here: structure validation reports it. A file fingerprinted but
    not recorded is not reported.

    A change is a ``WARNING``: a consumer that relies on the recorded bytes refuses it by passing the code to
    ``get_blocking_findings(..., block=...)``.

    Raises:
        ValueError: If a record's node is not a file the layout declares or its path does not match the layout, if a
            file is recorded more than once, if ``scope`` does not cover a recorded file, or if ``scope`` names a node
            the layout does not declare.
    """

    recorded = tuple(recorded)
    malformed = [r.posix_path for r in recorded if not _matches_layout(r, index)]
    if malformed:
        raise ValueError(f"{malformed} are recorded under a node that is not a declared file, or not matching it.")

    counts = Counter((r.schema_id, r.project_relative_path) for r in recorded)
    repeated = sorted({r.posix_path for r in recorded if counts[r.schema_id, r.project_relative_path] > 1})
    if repeated:
        raise ValueError(f"{repeated} are recorded more than once.")

    outside = [r.posix_path for r in recorded if not covered(scope, r.schema_id, r.project_relative_path, index)]
    if outside:
        raise ValueError(f"{outside} are recorded but outside the scope compared.")

    fingerprinted = resources_file_fingerprint(snapshot, scope, index=index)
    observed = {(f.schema_id, f.project_relative_path): f.sha256 for f in fingerprinted.fingerprints}
    failed = {(failure.schema_id, failure.project_relative_path) for failure in fingerprinted.failed}
    unreadable = [
        node.project_relative_path
        for node in walk_observed_nodes(snapshot.project)
        if node.status is NodeStatus.UNREADABLE
    ]
    findings = tuple(
        Finding(
            code=FingerprintFindingCode.FILE_FINGERPRINT_CHANGED,
            **node_location(record.schema_id, record.project_relative_path, index=index),
            path=snapshot.project_root / record.project_relative_path,
            evidence={"recorded_sha256": record.sha256, "observed_sha256": observed.get(key)},
        )
        for record in recorded
        if (key := (record.schema_id, record.project_relative_path)) not in failed
        and observed.get(key) != record.sha256
        and not any(record.project_relative_path.is_relative_to(path) for path in unreadable)
    )
    return fingerprinted.model_copy(
        update={"snapshot": fingerprinted.snapshot.with_validation_result(ValidationResult(findings=findings))}
    )


def _matches_layout(record: FileFingerprint, index: LayoutIndex) -> bool:
    """Return whether the record's node is a declared file and each part of its path matches the node at that depth."""

    if record.schema_id not in index.schema_ids_of(FileSchema):
        return False
    nodes = [index.schemas[schema_id] for schema_id in index.ancestry(record.schema_id)[:-1]]
    parts = record.project_relative_path.parts[::-1]
    return len(parts) == len(nodes) and all(
        node.name.matches(Path(part)) for node, part in zip(nodes, parts, strict=True)
    )
