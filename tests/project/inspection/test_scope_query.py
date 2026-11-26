from __future__ import annotations

import re
from pathlib import Path

import pytest

from dlc_octopus import api
from dlc_octopus.api import helpers
from dlc_octopus.schemas.base.filesystem import NodeSchema
from dlc_octopus.schemas.base.finding_codes import StructureFindingCode
from dlc_octopus.schemas.base.findings import Finding, Reach
from dlc_octopus.schemas.project_layout.labeled_data import LABELED_DATASET, LABELED_FRAME
from dlc_octopus.schemas.project_layout.project import PROJECT_CONFIG
from dlc_octopus.schemas.project_layout.videos import VIDEOS

_DS1 = ("labeled-data", "ds1")


def _planted(
    schema_id: str,
    reach: Reach,
    instance: tuple[str, ...] = (),
    *,
    field_location: tuple[str, ...] = (),
    code: StructureFindingCode = StructureFindingCode.PROJECT_NODE_UNREADABLE,
) -> Finding:
    return Finding(
        code=code,
        schema_id=schema_id,
        reach=reach,
        instance=instance,
        field_location=field_location,
        evidence=dict.fromkeys(code.evidence_keys, "placeholder"),
    )


def _scoped(tmp_path: Path, findings: tuple[Finding, ...], scope: helpers.Scope) -> list[Finding]:
    snapshot = api.inspect_project(tmp_path).model_copy(update={"findings": findings})

    return list(api.findings_in_scope(snapshot, scope))


def _in_scope(
    tmp_path: Path,
    findings: tuple[Finding, ...],
    node: NodeSchema,
    within: tuple[str, ...] = (),
) -> list[Finding]:
    return _scoped(tmp_path, findings, helpers.Scope(node, within=within))


def test_octopus_provides_the_scope_query() -> None:
    assert hasattr(api, "findings_in_scope")


def test_scope_raises_on_finding_not_declared_by_schema_id(tmp_path: Path) -> None:
    """Rather than dropping it."""
    not_found = _planted("dlc.not_declared_anywhere", Reach.ONE_PER_PROJECT)

    with pytest.raises(ValueError, match=re.escape(not_found.schema_id)):
        _in_scope(tmp_path, (not_found,), PROJECT_CONFIG)


def test_the_scope_holds_the_subtree_and_the_ancestors(tmp_path: Path) -> None:
    inside = (
        _planted("dlc.labeled_data.dataset", Reach.ONE_OF, _DS1),
        _planted("dlc.labeled_data.frame", Reach.ONE_OF, (*_DS1, "img0.png")),
        _planted("dlc.labeled_data", Reach.ONE_PER_PROJECT),
        _planted("dlc.project", Reach.ONE_PER_PROJECT),
    )
    outside = (
        _planted("dlc.videos.source_video", Reach.ONE_OF, ("videos", "a.mp4")),
        _planted("dlc.labeled_data.frame", Reach.ONE_OF, ("labeled-data", "ds2", "img0.png")),
    )

    scoped = _in_scope(tmp_path, inside + outside, LABELED_DATASET, _DS1)

    assert scoped == list(inside)


def test_an_unverified_node_depends_on_the_whole_config(tmp_path: Path) -> None:
    unused_field = _planted(
        "dlc.project.config",
        Reach.ONE_PER_PROJECT,
        field_location=("video_sets", "/videos/a.mp4"),
    )

    assert _in_scope(tmp_path, (unused_field,), LABELED_DATASET, _DS1) == [unused_field]


def test_the_scope_reads_the_location_only(tmp_path: Path) -> None:
    # Same locations, different codes, severities and evidence: membership must not change.
    unreadable = _planted("dlc.labeled_data.frame", Reach.ONE_OF, (*_DS1, "img0.png"))
    wrong_type = _planted(
        "dlc.labeled_data.frame",
        Reach.ONE_OF,
        (*_DS1, "img0.png"),
        code=StructureFindingCode.PROJECT_NODE_WRONG_TYPE,
    )
    elsewhere = _planted(
        "dlc.labeled_data.frame",
        Reach.ONE_OF,
        ("labeled-data", "ds2", "img0.png"),
        code=StructureFindingCode.PROJECT_NODE_WRONG_TYPE,
    )

    assert _in_scope(tmp_path, (unreadable, wrong_type, elsewhere), LABELED_DATASET, _DS1) == [unreadable, wrong_type]


def test_an_ancestor_instance_is_in_scope_for_its_own_descendants_only(tmp_path: Path) -> None:
    own_dataset = _planted("dlc.labeled_data.dataset", Reach.ONE_OF, _DS1)
    other_dataset = _planted("dlc.labeled_data.dataset", Reach.ONE_OF, ("labeled-data", "ds2"))

    scoped = _in_scope(tmp_path, (own_dataset, other_dataset), LABELED_FRAME, (*_DS1, "img0.png"))

    assert scoped == [own_dataset]


def test_an_all_of_finding_with_a_shorter_prefix_is_in_scope(tmp_path: Path) -> None:
    every_frame = _planted("dlc.labeled_data.frame", Reach.ALL_OF)
    every_frame_of_ds1 = _planted("dlc.labeled_data.frame", Reach.ALL_OF, _DS1)

    scoped = _in_scope(tmp_path, (every_frame, every_frame_of_ds1), LABELED_DATASET, _DS1)

    assert scoped == [every_frame, every_frame_of_ds1]


def test_another_instance_is_out_of_scope(tmp_path: Path) -> None:
    every_frame_of_ds2 = _planted("dlc.labeled_data.frame", Reach.ALL_OF, ("labeled-data", "ds2"))

    assert _in_scope(tmp_path, (every_frame_of_ds2,), LABELED_DATASET, _DS1) == []


def test_a_union_of_scopes_returns_each_finding_once_in_snapshot_order(tmp_path: Path) -> None:
    video = _planted("dlc.videos.source_video", Reach.ONE_OF, ("videos", "a.mp4"))
    config = _planted("dlc.project.config", Reach.ONE_PER_PROJECT)
    frame = _planted("dlc.labeled_data.frame", Reach.ONE_OF, (*_DS1, "img0.png"))

    union = helpers.Scope(LABELED_DATASET, within=_DS1) | helpers.Scope(VIDEOS)

    # The config finding concerns both parts, and comes back once.
    assert _scoped(tmp_path, (video, config, frame), union) == [video, config, frame]


def test_a_union_of_scopes_only_widens(tmp_path: Path) -> None:
    findings = (
        _planted("dlc.videos.source_video", Reach.ONE_OF, ("videos", "a.mp4")),
        _planted("dlc.project.config", Reach.ONE_PER_PROJECT),
        _planted("dlc.labeled_data.frame", Reach.ONE_OF, (*_DS1, "img0.png")),
        _planted("dlc.labeled_data.frame", Reach.ONE_OF, ("labeled-data", "ds2", "img0.png")),
    )
    datasets = helpers.Scope(LABELED_DATASET, within=_DS1)
    videos = helpers.Scope(VIDEOS)

    combined = _scoped(tmp_path, findings, datasets | videos)

    for part in (datasets, videos):
        assert all(finding in combined for finding in _scoped(tmp_path, findings, part))


@pytest.mark.xfail(
    reason="field-level dependencies are not declared yet: once a node's config fields are verified against "
    "DeepLabCut, a finding on another config field leaves its scope, and a consumer widens by naming fields. "
    "When this passes, test_an_unverified_node_depends_on_the_whole_config must move to a node still unverified",
    strict=True,
    raises=AssertionError,
)
def test_a_verified_node_depends_only_on_its_declared_config_fields(tmp_path: Path) -> None:
    video_sets_finding = _planted(
        "dlc.project.config",
        Reach.ONE_PER_PROJECT,
        field_location=("video_sets", "/videos/a.mp4"),
    )

    assert _in_scope(tmp_path, (video_sets_finding,), LABELED_DATASET, _DS1) == []


def test_a_scope_covers_the_file_nodes_of_its_subtrees_in_walk_order() -> None:
    scope = helpers.Scope(VIDEOS) | helpers.Scope(LABELED_DATASET, within=_DS1)

    assert scope.file_schema_ids() == (
        "dlc.videos.source_video",
        "dlc.labeled_data.napari_metadata",
        "dlc.labeled_data.frame",
        "dlc.labeled_data.human_labels",
        "dlc.labeled_data.machine_labels",
        "dlc.labeled_data.machine_labels_csv",
    )


def test_a_scope_does_not_cover_files_at_its_ancestors() -> None:
    """Findings there are in scope; the files are not."""

    assert "dlc.project.config" not in helpers.Scope(LABELED_DATASET).file_schema_ids()


def test_covered_files_refuse_a_node_the_layout_does_not_declare() -> None:
    impostor = LABELED_FRAME.model_copy(update={"description": "not the layout's own"})

    with pytest.raises(ValueError, match="does not declare"):
        helpers.Scope(impostor).file_schema_ids()
