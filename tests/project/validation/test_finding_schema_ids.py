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

from collections.abc import Callable
from pathlib import Path

import pytest

from dlccore.config.utils import write_config
from dlccore.project.api import inspect_project
from dlccore.project.discovery.probe import probe_path
from dlccore.project.schemas.base.filesystem import walk_schema
from dlccore.project.schemas.base.finding_codes import (
    ConfigFindingCode,
    ConfiguredVideoFindingCode,
    StructureFindingCode,
)
from dlccore.project.schemas.base.findings import Finding, Reach
from dlccore.project.schemas.base.probes import FilesystemEntryKind, PathProbe, PathProbeStatus
from dlccore.project.schemas.base.resources import DirectoryResourceKind
from dlccore.project.schemas.base.symlinks import SymlinkObservation, SymlinkTargetKind
from dlccore.project.schemas.base.validation import ValidationDepth
from dlccore.project.schemas.locations import all_of, one_of, one_per_project
from dlccore.project.schemas.project_layout.project import PROJECT, PROJECT_INDEX
from dlccore.project.schemas.snapshots import ProjectSnapshot
from dlccore.project.validation import ProjectValidationRequest, ResourceValidationRequest

_DECLARED = frozenset(schema.schema_id for schema, _ in walk_schema(PROJECT))
_NODES = PROJECT_INDEX.schemas

_VIDEOS = ProjectValidationRequest(
    resources=(
        ResourceValidationRequest(
            kind=DirectoryResourceKind.VIDEOS_DIRECTORY,
            depth=ValidationDepth.FILENAME_ONLY,
        ),
    )
)

# Moving a code to another node changes every consumer's scope, so each non-structural code's
# anchor is pinned here. Structure codes anchor at the node they are about.
_PINNED_ANCHORS = {
    **dict.fromkeys(ConfigFindingCode, "dlc.project.config"),
    **dict.fromkeys(ConfiguredVideoFindingCode, "dlc.project.config"),
    StructureFindingCode.PROJECT_RESOURCE_PATH_COLLISION: "dlc.videos.source_video",
}


def _moved_anchors(findings: tuple[Finding, ...]) -> set[tuple[str, str]]:
    return {
        (finding.code.value, finding.schema_id)
        for finding in findings
        if finding.code in _PINNED_ANCHORS and finding.schema_id != _PINNED_ANCHORS[finding.code]
    }


def _undeclared_schema_ids(findings: tuple[Finding, ...]) -> set[str]:
    return {finding.schema_id for finding in findings} - _DECLARED


def _misfit_reaches(findings: tuple[Finding, ...]) -> set[tuple[str, Reach]]:
    """Return the declared nodes whose emitted reach the layout location helpers would refuse."""

    misfits: set[tuple[str, Reach]] = set()

    for finding in findings:
        node = _NODES.get(finding.schema_id)

        if node is None:
            continue

        try:
            if finding.reach is Reach.ONE_PER_PROJECT:
                one_per_project(node)
            elif finding.reach is Reach.ONE_OF:
                one_of(node, finding.instance)
            else:
                all_of(node, finding.instance)
        except ValueError:
            misfits.add((finding.schema_id, finding.reach))

    return misfits


def _empty_directory(root: Path, config: dict) -> None:
    del root, config


def _unparseable_config(root: Path, config: dict) -> None:
    del config
    (root / "config.yaml").write_text("a:\n  b: [\n", encoding="utf-8")


def _invalid_config(root: Path, config: dict) -> None:
    write_config(root / "config.yaml", {**config, "pcutoff": "not-a-number"})


def _moved_project_with_a_missing_video(root: Path, config: dict) -> None:
    (root / "videos").mkdir()
    write_config(
        root / "config.yaml",
        {
            **config,
            "project_path": str(root / "declared-elsewhere"),
            "video_sets": {str(root / "videos" / "missing.mp4"): {"crop": "0, 10, 0, 10"}},
        },
    )


def _videos_is_a_file(root: Path, config: dict) -> None:
    (root / "videos").touch()
    write_config(root / "config.yaml", config)


@pytest.mark.parametrize(
    "build",
    [
        _empty_directory,
        _unparseable_config,
        _invalid_config,
        _moved_project_with_a_missing_video,
        _videos_is_a_file,
    ],
    ids=lambda build: build.__name__.strip("_"),
)
def test_every_emitted_finding_has_a_declared_schema_id(
    tmp_path: Path,
    valid_config_data: dict,
    build: Callable[[Path, dict], None],
) -> None:
    build(tmp_path, dict(valid_config_data, project_path=str(tmp_path)))

    snapshot: ProjectSnapshot = inspect_project(tmp_path, validation=_VIDEOS)

    # Guards the guard: a scenario that emits nothing would pass vacuously.
    assert snapshot.findings
    assert _undeclared_schema_ids(snapshot.findings) == set()
    assert _misfit_reaches(snapshot.findings) == set()
    assert _moved_anchors(snapshot.findings) == set()


def test_a_finding_on_a_repeated_node_is_emitted_with_a_fitting_reach(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / "session-01.mp4").touch()

    def fake_probe_path(path: Path) -> PathProbe:
        if path.name != "session-01.mp4":
            return probe_path(path)

        return PathProbe(
            status=PathProbeStatus.OBSERVED,
            entry_kind=FilesystemEntryKind.SYMLINK,
            symlink=SymlinkObservation(
                raw_target=tmp_path / "gone.mp4",
                absolute_target=tmp_path / "gone.mp4",
                target_kind=SymlinkTargetKind.MISSING,
            ),
        )

    monkeypatch.setattr("dlccore.project.discovery.observer.probe_path", fake_probe_path)

    findings = inspect_project(tmp_path).findings

    # Guards the guard: without a ONE_OF finding, only one_per_project would be exercised.
    assert {finding.reach for finding in findings} >= {Reach.ONE_OF}
    assert _misfit_reaches(findings) == set()


def test_an_undeclared_schema_id_is_detected() -> None:
    code = StructureFindingCode.PROJECT_NODE_UNREADABLE
    planted = Finding(
        code=code,
        schema_id="dlc.not_declared_anywhere",
        reach=Reach.ONE_PER_PROJECT,
        evidence=dict.fromkeys(code.evidence_keys, "placeholder"),
    )

    assert _undeclared_schema_ids((planted,)) == {"dlc.not_declared_anywhere"}


def test_a_moved_anchor_is_detected() -> None:
    code = ConfigFindingCode.CONFIG_PROJECT_PATH_MISMATCH
    moved = Finding(
        code=code,
        schema_id="dlc.project",
        reach=Reach.ONE_PER_PROJECT,
        evidence=dict.fromkeys(code.evidence_keys, "placeholder"),
    )

    assert _moved_anchors((moved,)) == {("CONFIG_PROJECT_PATH_MISMATCH", "dlc.project")}


def test_a_reach_that_does_not_fit_the_layout_is_detected() -> None:
    code = StructureFindingCode.PROJECT_NODE_UNREADABLE
    # The model accepts this, since it does not know the layout: only the helpers would refuse it.
    planted = Finding(
        code=code,
        schema_id="dlc.videos.source_video",
        reach=Reach.ONE_PER_PROJECT,
        evidence=dict.fromkeys(code.evidence_keys, "placeholder"),
    )

    assert _misfit_reaches((planted,)) == {("dlc.videos.source_video", Reach.ONE_PER_PROJECT)}


def test_a_finding_about_something_missing_anchors_at_its_declaration(
    tmp_path: Path,
    valid_config_data: dict,
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    (config_missing,) = (
        finding
        for finding in inspect_project(empty).findings
        if finding.code == StructureFindingCode.PROJECT_REQUIRED_NODE_MISSING
    )

    assert (config_missing.schema_id, config_missing.reach) == ("dlc.project.config", Reach.ONE_PER_PROJECT)

    declared = tmp_path / "declared"
    (declared / "videos").mkdir(parents=True)
    missing = declared / "videos" / "missing.mp4"
    write_config(
        declared / "config.yaml",
        {**valid_config_data, "project_path": str(declared), "video_sets": {str(missing): {"crop": "0, 10, 0, 10"}}},
    )

    (video_missing,) = (
        finding
        for finding in inspect_project(declared, validation=_VIDEOS).findings
        if finding.code == "CONFIGURED_VIDEO_PATH_MISSING"
    )

    assert (video_missing.schema_id, video_missing.reach, video_missing.field_location) == (
        "dlc.project.config",
        Reach.ONE_PER_PROJECT,
        ("video_sets", str(missing)),
    )
