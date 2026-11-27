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

from dlccore.project.schemas.base.filesystem import walk_schema
from dlccore.project.schemas.base.finding_codes import (
    ConfigFindingCode,
    ConfiguredVideoFindingCode,
    StructureFindingCode,
)
from dlccore.project.schemas.base.findings import Finding, Reach
from dlccore.project.schemas.locations import all_of, one_of, one_per_project
from dlccore.project.schemas.project_layout.project import PROJECT, PROJECT_INDEX

_DECLARED = frozenset(schema.schema_id for schema, _ in walk_schema(PROJECT))
_NODES = PROJECT_INDEX.schemas


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
