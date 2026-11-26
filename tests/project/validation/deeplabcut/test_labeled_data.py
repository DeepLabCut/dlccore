from __future__ import annotations

from pathlib import Path

import pytest

from dlc_octopus.api import inspect_project
from dlc_octopus.schemas.base.resources import DirectoryResourceKind
from dlc_octopus.schemas.base.validation import ValidationDepth
from dlc_octopus.validation import ProjectValidationRequest, ResourceValidationRequest


@pytest.mark.xfail(
    reason="Human labels are CollectedData_<config scorer>.h5 and .csv only, as DeepLabCut reads them; the layout "
    "cannot know the scorer, so a copy, a backup or another scorer's file is discovered as human labels and is "
    "to be a finding at that file. No labeled-data suite is registered yet, so requesting one raises",
    strict=True,
    raises=(ValueError, AssertionError),
)
def test_a_collected_data_file_not_named_for_the_config_scorer_is_reported(valid_project: Path) -> None:
    """Both extra names were found in real projects, beside the pair DeepLabCut wrote."""

    dataset = valid_project / "labeled-data" / "session-01"
    dataset.mkdir()
    labels = ("CollectedData_test-scorer.h5", "CollectedData_test-scorer.csv")
    extras = (
        "CollectedData_test-scorer - Copy.csv",
        "CollectedData_test-scorer.h5.bak_20260410_101324.h5",
        "CollectedData_someone-else.h5",
    )
    for name in labels + extras:
        (dataset / name).touch()

    snapshot = inspect_project(
        valid_project,
        validation=ProjectValidationRequest(
            resources=(
                ResourceValidationRequest(
                    kind=DirectoryResourceKind.LABELED_DATA_DIRECTORY, depth=ValidationDepth.FILENAME_ONLY
                ),
            )
        ),
    )

    assert sorted(
        finding.instance[-1] for finding in snapshot.findings if finding.schema_id == "dlc.labeled_data.human_labels"
    ) == sorted(extras)
