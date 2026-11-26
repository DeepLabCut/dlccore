import hashlib
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from dlc_octopus.api import findings_in_scope, inspect_project
from dlc_octopus.api.helpers import get_blocking_findings
from dlc_octopus.content import FailedRead, FileFingerprint, compare_file_fingerprints, resources_file_fingerprint
from dlc_octopus.schemas.base.finding_codes import FingerprintFindingCode, Severity, StructureFindingCode
from dlc_octopus.schemas.base.findings import Reach
from dlc_octopus.schemas.base.observations import NodeStatus
from dlc_octopus.schemas.base.resources import FileResourceKind
from dlc_octopus.schemas.project_layout.project import PROJECT_CONFIG
from dlc_octopus.schemas.project_layout.videos import SOURCE_VIDEO, VIDEOS
from dlc_octopus.schemas.scope import Scope

FAILED = FingerprintFindingCode.FILE_FINGERPRINT_FAILED
CHANGED = FingerprintFindingCode.FILE_FINGERPRINT_CHANGED
ALL_VIDEOS = Scope(SOURCE_VIDEO)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _codes(findings: tuple) -> list:
    return [finding.code for finding in findings]


def test_inspection_never_file_fingerprints(valid_project: Path) -> None:
    snapshot = inspect_project(valid_project)

    assert FAILED not in _codes(snapshot.findings)
    assert not [name for name in type(snapshot).model_fields if "fingerprint" in name]


def test_a_node_the_layout_does_not_declare_is_refused_even_when_nothing_is_present(tmp_path: Path) -> None:
    impostor = SOURCE_VIDEO.model_copy(update={"description": "not the layout's own"})

    with pytest.raises(ValueError, match="does not declare"):
        resources_file_fingerprint(inspect_project(tmp_path), Scope(impostor))


def test_only_the_files_in_scope_are_file_fingerprinted(valid_project: Path) -> None:
    (valid_project / "videos" / "session-01.mp4").write_bytes(b"frames")
    snapshot = inspect_project(valid_project)

    result = resources_file_fingerprint(snapshot, Scope(PROJECT_CONFIG))

    assert result.fingerprints == (
        FileFingerprint(
            schema_id="dlc.project.config",
            project_relative_path=Path("config.yaml"),
            sha256=_sha256(valid_project / "config.yaml"),
            size_bytes=(valid_project / "config.yaml").stat().st_size,
        ),
    )
    assert result.snapshot == snapshot


def test_the_snapshot_is_not_rediscovered(valid_project: Path) -> None:
    snapshot = inspect_project(valid_project)
    (valid_project / "videos" / "added-later.mp4").write_bytes(b"frames")

    result = resources_file_fingerprint(snapshot, Scope(SOURCE_VIDEO))

    assert (result.fingerprints, result.snapshot) == ((), snapshot)


def test_a_node_that_is_not_present_is_neither_file_fingerprinted_nor_reported(valid_project: Path) -> None:
    (valid_project / "config.yaml").unlink()
    snapshot = inspect_project(valid_project)
    statuses = [(node.kind, node.status) for node in snapshot.project.schema_children]
    assert (FileResourceKind.PROJECT_CONFIG, NodeStatus.MISSING) in statuses  # control: the node was observed

    result = resources_file_fingerprint(snapshot, Scope(PROJECT_CONFIG))

    assert (result.fingerprints, result.snapshot) == ((), snapshot)


def test_a_file_that_vanished_is_one_merged_error_finding_and_the_rest_are_file_fingerprinted(
    valid_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (valid_project / "videos" / "session-01.mp4").write_bytes(b"frames 1")
    (valid_project / "videos" / "session-02.mp4").write_bytes(b"frames 2")
    monkeypatch.chdir(valid_project.parent)
    snapshot = inspect_project(Path(valid_project.name))
    (valid_project / "videos" / "session-01.mp4").unlink()

    result = resources_file_fingerprint(snapshot, Scope(SOURCE_VIDEO))

    assert [f.project_relative_path for f in result.fingerprints] == [Path("videos/session-02.mp4")]
    (finding,) = result.snapshot.findings[len(snapshot.findings) :]
    assert result.snapshot.findings[: len(snapshot.findings)] == snapshot.findings
    assert (finding.code, finding.severity, finding.reach, finding.instance) == (
        FAILED,
        Severity.ERROR,
        Reach.ONE_OF,
        ("videos", "session-01.mp4"),
    )
    assert finding.path == valid_project / "videos" / "session-01.mp4"  # absolute, never resolved
    assert finding.evidence["resource_kind"] == "source_video"
    assert finding.evidence["error_type"] == "FileNotFoundError"
    assert "session-01.mp4" in finding.evidence["error_message"]
    assert result.failed == (FailedRead("dlc.videos.source_video", Path("videos/session-01.mp4"), (finding,)),)


def test_every_requested_file_is_either_file_fingerprinted_or_failed(valid_project: Path) -> None:
    for name in ("session-01.mp4", "session-02.mp4", "session-03.mp4"):
        (valid_project / "videos" / name).write_bytes(b"frames")
    snapshot = inspect_project(valid_project)
    (valid_project / "videos" / "session-02.mp4").unlink()

    result = resources_file_fingerprint(snapshot, Scope(SOURCE_VIDEO))

    fingerprinted = [f.project_relative_path.name for f in result.fingerprints]
    failed = [failure.project_relative_path.name for failure in result.failed]
    assert (fingerprinted, failed) == (["session-01.mp4", "session-03.mp4"], ["session-02.mp4"])


def test_a_merged_finding_is_seen_by_the_scope_query(valid_project: Path) -> None:
    (valid_project / "videos" / "session-01.mp4").write_bytes(b"frames")
    snapshot = inspect_project(valid_project)
    (valid_project / "videos" / "session-01.mp4").unlink()

    result = resources_file_fingerprint(snapshot, Scope(SOURCE_VIDEO))

    assert FAILED in _codes(
        findings_in_scope(result.snapshot, Scope(SOURCE_VIDEO, within=("videos", "session-01.mp4")))
    )


def test_a_scope_within_one_instance_file_fingerprints_that_instance_only(valid_project: Path) -> None:
    (valid_project / "videos" / "session-01.mp4").write_bytes(b"frames 1")
    (valid_project / "videos" / "session-02.mp4").write_bytes(b"frames 2")
    snapshot = inspect_project(valid_project)
    assert len(resources_file_fingerprint(snapshot, Scope(SOURCE_VIDEO)).fingerprints) == 2  # control

    result = resources_file_fingerprint(snapshot, Scope(SOURCE_VIDEO, within=("videos", "session-02.mp4")))

    assert [f.project_relative_path for f in result.fingerprints] == [Path("videos/session-02.mp4")]


def test_a_directory_scope_file_fingerprints_the_files_beneath_it(valid_project: Path) -> None:
    (valid_project / "videos" / "session-01.mp4").write_bytes(b"frames")

    result = resources_file_fingerprint(inspect_project(valid_project), Scope(VIDEOS))

    assert [f.schema_id for f in result.fingerprints] == ["dlc.videos.source_video"]


def test_a_symlinked_resource_is_file_fingerprinted_through_the_link(
    valid_project: Path, tmp_path_factory: pytest.TempPathFactory, create_symlink_or_skip: Callable[..., None]
) -> None:
    target = tmp_path_factory.mktemp("elsewhere") / "session-01.mp4"
    target.write_bytes(b"frames")
    create_symlink_or_skip(valid_project / "videos" / "session-01.mp4", target)

    result = resources_file_fingerprint(inspect_project(valid_project), Scope(SOURCE_VIDEO))

    assert [(f.project_relative_path, f.sha256) for f in result.fingerprints] == [
        (Path("videos/session-01.mp4"), _sha256(target))
    ]


def test_a_file_fingerprint_spells_its_path_the_same_on_every_platform(valid_project: Path) -> None:
    (valid_project / "videos" / "session-01.mp4").write_bytes(b"frames")

    (fingerprint,) = resources_file_fingerprint(inspect_project(valid_project), Scope(SOURCE_VIDEO)).fingerprints

    assert fingerprint.project_relative_path == Path("videos", "session-01.mp4")  # the host's Path
    assert fingerprint.posix_path == "videos/session-01.mp4"
    assert "posix_path" not in fingerprint.model_dump()  # not part of the serialized form


@pytest.mark.parametrize("sha256", ["A" * 64, "a" * 63, ""])
def test_a_malformed_sha256_is_refused_when_loaded(sha256: str) -> None:
    with pytest.raises(ValidationError):
        FileFingerprint.model_validate(
            {
                "schema_id": "dlc.project.config",
                "project_relative_path": "config.yaml",
                "sha256": sha256,
                "size_bytes": 1,
            }
        )


def _fingerprint_record(size_bytes: object) -> dict[str, object]:
    return {
        "schema_id": "dlc.project.config",
        "project_relative_path": "config.yaml",
        "sha256": "a" * 64,
        "size_bytes": size_bytes,
    }


@pytest.mark.parametrize("size_bytes", [-1, True, 3.0, "7"], ids=repr)
def test_a_size_that_is_not_a_byte_count_is_refused_when_loaded(size_bytes: object) -> None:
    with pytest.raises(ValidationError):
        FileFingerprint.model_validate(_fingerprint_record(size_bytes))


def test_a_byte_count_survives_a_json_round_trip() -> None:
    # Control: strict integers must still load from JSON, where a consumer stores them.
    fingerprint = FileFingerprint.model_validate(_fingerprint_record(0))

    assert FileFingerprint.model_validate_json(fingerprint.model_dump_json()) == fingerprint


def test_the_public_api_reexports_file_fingerprinting() -> None:
    import dlc_octopus.api as api
    import dlc_octopus.content as content

    assert (
        api.resources_file_fingerprint,
        api.compare_file_fingerprints,
        api.FileFingerprint,
        api.ResourcesFileFingerprint,
    ) == (
        content.resources_file_fingerprint,
        content.compare_file_fingerprints,
        content.FileFingerprint,
        content.ResourcesFileFingerprint,
    )
    assert api.FileResourceKind is FileResourceKind
    assert api.helpers.Scope is Scope


def _videos(project: Path, *names: str) -> tuple[FileFingerprint, ...]:
    for name in names:
        (project / "videos" / name).write_bytes(name.encode())
    return resources_file_fingerprint(inspect_project(project), Scope(SOURCE_VIDEO)).fingerprints


def _compared(project: Path, recorded: tuple[FileFingerprint, ...], scope: Scope = ALL_VIDEOS) -> tuple:
    snapshot = inspect_project(project)
    return compare_file_fingerprints(snapshot, scope, recorded).snapshot.findings[len(snapshot.findings) :]


def test_unchanged_files_compare_without_findings(valid_project: Path) -> None:
    recorded = _videos(valid_project, "session-01.mp4", "session-02.mp4")

    assert _compared(valid_project, recorded) == ()


def test_a_changed_file_is_one_finding_with_both_fingerprints_blocking_only_on_request(valid_project: Path) -> None:
    recorded = _videos(valid_project, "session-01.mp4", "session-02.mp4")
    (valid_project / "videos" / "session-02.mp4").write_bytes(b"relabeled")

    (finding,) = _compared(valid_project, recorded)

    assert (finding.code, finding.severity, finding.instance) == (
        CHANGED,
        Severity.WARNING,
        ("videos", "session-02.mp4"),
    )
    assert finding.evidence == {
        "recorded_sha256": recorded[1].sha256,
        "observed_sha256": _sha256(valid_project / "videos" / "session-02.mp4"),
    }
    assert (get_blocking_findings([finding]), get_blocking_findings([finding], block={CHANGED})) == ((), (finding,))


def test_a_recorded_file_that_is_gone_is_reported_with_no_observed_fingerprint(valid_project: Path) -> None:
    recorded = _videos(valid_project, "session-01.mp4")
    (valid_project / "videos" / "session-01.mp4").unlink()

    (finding,) = _compared(valid_project, recorded)

    assert (finding.code, finding.evidence["observed_sha256"]) == (CHANGED, None)


def test_a_recorded_file_that_cannot_be_read_is_reported_once_as_failed(valid_project: Path) -> None:
    recorded = _videos(valid_project, "session-01.mp4")
    snapshot = inspect_project(valid_project)
    (valid_project / "videos" / "session-01.mp4").unlink()

    result = compare_file_fingerprints(snapshot, Scope(SOURCE_VIDEO), recorded)

    assert _codes(result.snapshot.findings) == [FAILED]


def test_a_file_fingerprinted_but_never_recorded_is_not_reported(valid_project: Path) -> None:
    recorded = _videos(valid_project, "session-01.mp4")
    _videos(valid_project, "session-02.mp4")

    assert _compared(valid_project, recorded) == ()


def test_a_scope_within_one_instance_compares_that_instance_only(valid_project: Path) -> None:
    recorded = _videos(valid_project, "session-01.mp4", "session-02.mp4")
    (valid_project / "videos" / "session-01.mp4").unlink()
    assert _codes(_compared(valid_project, recorded)) == [CHANGED]  # control: the whole scope sees it gone

    within = Scope(SOURCE_VIDEO, within=("videos", "session-02.mp4"))

    assert _compared(valid_project, recorded[1:], within) == ()


def test_a_recorded_file_outside_the_scope_is_refused(valid_project: Path) -> None:
    recorded = _videos(valid_project, "session-01.mp4", "session-02.mp4")

    with pytest.raises(ValueError, match=r"session-01\.mp4"):
        _compared(valid_project, recorded, Scope(SOURCE_VIDEO, within=("videos", "session-02.mp4")))


@pytest.mark.parametrize(
    ("schema_id", "path"),
    [
        (SOURCE_VIDEO.schema_id, "videos/notes.txt"),
        (SOURCE_VIDEO.schema_id, "labeled-data/mouse1/clip.mp4"),
        (VIDEOS.schema_id, "videos"),
    ],
    ids=["name the node does not match", "a directory the node does not sit in", "a directory node"],
)
def test_a_record_that_does_not_fit_its_node_is_refused(valid_project: Path, schema_id: str, path: str) -> None:
    record = FileFingerprint(schema_id=schema_id, project_relative_path=Path(path), sha256="0" * 64, size_bytes=0)

    with pytest.raises(ValueError, match="not a declared file"):
        _compared(valid_project, (record,))


@pytest.mark.parametrize("sha256", ["same", "conflicting"])
def test_a_file_recorded_twice_is_refused(valid_project: Path, sha256: str) -> None:
    (record,) = _videos(valid_project, "session-01.mp4")
    twice = record if sha256 == "same" else record.model_copy(update={"sha256": "0" * 64})

    with pytest.raises(ValueError, match=r"session-01\.mp4.*more than once"):
        _compared(valid_project, (record, twice))


def _deny(monkeypatch: pytest.MonkeyPatch, method: str, denied: Path) -> None:
    real = getattr(Path, method)

    def refuse(self: Path, *args: object, **kwargs: object) -> object:
        if self == denied:
            raise PermissionError("Access is denied")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, method, refuse)


@pytest.mark.parametrize(
    ("method", "denied"),
    [("lstat", "videos/session-02.mp4"), ("iterdir", "videos")],
    ids=["the file", "its directory"],
)
def test_a_recorded_file_unreadable_at_inspection_is_left_to_structure_validation(
    valid_project: Path, monkeypatch: pytest.MonkeyPatch, method: str, denied: str
) -> None:
    recorded = _videos(valid_project, "session-01.mp4", "session-02.mp4")
    _deny(monkeypatch, method, valid_project / denied)
    snapshot = inspect_project(valid_project)
    assert StructureFindingCode.PROJECT_NODE_UNREADABLE in _codes(snapshot.findings)  # control: inspection saw it

    result = compare_file_fingerprints(snapshot, ALL_VIDEOS, recorded)

    assert result.snapshot.findings[len(snapshot.findings) :] == ()
