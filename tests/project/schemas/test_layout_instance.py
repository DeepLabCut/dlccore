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

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from dlccore.project.schemas.base.core import instance_from_path
from dlccore.project.schemas.base.finding_codes import FindingCode
from dlccore.project.schemas.base.findings import Finding, Reach

_SCHEMA_ID = "dlc.labeled_data.dataset"


@pytest.fixture
def code(example_codes: type[FindingCode]) -> FindingCode:
    return example_codes.TEST_NODE_UNREADABLE


@pytest.fixture
def evidence(code: FindingCode) -> dict[str, str]:
    return dict.fromkeys(code.evidence_keys, "placeholder")


def test_a_finding_has_no_instance_by_default(code: FindingCode, evidence: dict[str, str]) -> None:
    assert Finding(code=code, schema_id=_SCHEMA_ID, reach=Reach.ONE_PER_PROJECT, evidence=evidence).instance == ()


def test_an_instance_serializes_as_a_plain_list(code: FindingCode, evidence: dict[str, str]) -> None:
    finding = Finding(
        code=code, schema_id=_SCHEMA_ID, reach=Reach.ONE_OF, instance=("labeled-data", "ds1"), evidence=evidence
    )

    assert json.loads(finding.model_dump_json())["instance"] == ["labeled-data", "ds1"]


@pytest.mark.parametrize("segment", ["", "labeled-data/ds1"], ids=["empty", "joined"])
def test_an_instance_refuses_a_segment_only_a_caller_can_produce(
    segment: str, code: FindingCode, evidence: dict[str, str]
) -> None:
    with pytest.raises(ValidationError, match="project-relative path segment"):
        Finding(
            code=code, schema_id=_SCHEMA_ID, reach=Reach.ONE_OF, instance=("labeled-data", segment), evidence=evidence
        )


@pytest.mark.parametrize("segment", ["..", "12:30.mp4", "a\\b.mp4"])
def test_an_instance_keeps_user_data_segments(segment: str, code: FindingCode, evidence: dict[str, str]) -> None:
    assert Finding(
        code=code, schema_id=_SCHEMA_ID, reach=Reach.ONE_OF, instance=("videos", segment), evidence=evidence
    ).instance == (
        "videos",
        segment,
    )


@pytest.mark.parametrize(
    "instance",
    [("C:\\", "x.mp4"), ("\\\\server\\share\\", "x.mp4"), ("C:rel", "x.mp4")],
    ids=["drive-root", "unc-share", "drive-relative"],
)
def test_an_instance_refuses_segments_that_carry_a_drive_or_root(
    instance: tuple[str, ...], code: FindingCode, evidence: dict[str, str]
) -> None:
    with pytest.raises(ValidationError, match="project-relative"):
        Finding(code=code, schema_id=_SCHEMA_ID, reach=Reach.ONE_OF, instance=instance, evidence=evidence)


def test_instance_from_path_splits_a_project_relative_path() -> None:
    assert instance_from_path(Path("labeled-data/ds1/img001.png")) == ("labeled-data", "ds1", "img001.png")


def test_instance_from_path_refuses_an_anchored_path() -> None:
    with pytest.raises(ValueError, match="project-relative"):
        instance_from_path(Path("Z:/shared/DLC/videos/a.mp4"))
