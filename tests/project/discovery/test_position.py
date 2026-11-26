from pathlib import Path

import pytest

from dlc_octopus.api import inspect_project
from dlc_octopus.discovery.position import CandidateNode, locate_in_layout
from dlc_octopus.schemas.base.finding_codes import LabeledDataFindingCode
from dlc_octopus.schemas.base.findings import Finding, Reach
from dlc_octopus.schemas.project_layout.labeled_data import HUMAN_LABELS, LABELED_DATASET
from dlc_octopus.validation.results import ValidationResult

NAME = "CollectedData_Sc.h5"
INSTANCE = ("labeled-data", "Sample1", NAME)


def test_a_file_in_a_project_is_anchored_from_the_project_root(valid_project: Path) -> None:
    position = locate_in_layout(valid_project / "labeled-data" / "Sample1" / NAME)

    assert position.location == {"schema_id": HUMAN_LABELS.schema_id, "reach": Reach.ONE_OF, "instance": INSTANCE}
    assert (position.project_root, position.detached) == (valid_project.absolute(), ())


def test_a_dataset_outside_a_project_keeps_the_instance_it_has_inside_one(tmp_path: Path) -> None:
    position = locate_in_layout(tmp_path / "backup" / "labeled-data" / "Sample1" / NAME)

    assert position.location == {"schema_id": HUMAN_LABELS.schema_id, "reach": Reach.ONE_OF, "instance": INSTANCE}
    assert (position.project_root, position.detached) == (None, ())


def test_a_folder_that_fits_a_dataset_by_position_only_is_a_detached_candidate(tmp_path: Path) -> None:
    position = locate_in_layout(tmp_path / "Downloads" / "Sample1" / NAME)

    assert position.location == {"schema_id": HUMAN_LABELS.schema_id, "reach": Reach.UNROOTED, "instance": (NAME,)}
    assert position.detached == (CandidateNode("Sample1", LABELED_DATASET.schema_id),)


def test_labeled_data_spelled_otherwise_is_not_trusted(tmp_path: Path) -> None:
    # napari-deeplabcut v0.4.0 matches "labeled-data" anywhere, in any case; the layout matches it at its position.
    position = locate_in_layout(tmp_path / "Labeled-Data" / "x" / "Sample1" / NAME)

    assert position.location["reach"] is Reach.UNROOTED
    assert position.detached == (CandidateNode("Sample1", LABELED_DATASET.schema_id),)


def test_a_name_no_file_node_matches_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="matches 0 file nodes"):
        locate_in_layout(tmp_path / "labels.h5")


def test_a_snapshot_refuses_an_unrooted_finding(valid_project: Path) -> None:
    snapshot = inspect_project(valid_project)
    evidence = {"resource_kind": "human_labels", "error_type": "OSError", "error_message": "unreadable"}
    finding = Finding(
        code=LabeledDataFindingCode.LABELED_DATA_UNREADABLE,
        schema_id=HUMAN_LABELS.schema_id,
        reach=Reach.UNROOTED,
        instance=(NAME,),
        evidence=evidence,
    )

    with pytest.raises(ValueError, match="UNROOTED"):
        snapshot.with_validation_result(ValidationResult(findings=(finding,)))
    # Control: the same finding anchored in the project is accepted.
    rooted = finding.model_copy(update={"reach": Reach.ONE_OF, "instance": INSTANCE})
    assert snapshot.with_validation_result(ValidationResult(findings=(rooted,))).findings[-1] == rooted
