from pathlib import Path

import pytest

from dlc_octopus.api import inspect_project
from dlc_octopus.schemas.base.resources import FileResourceKind


def _kinds_in_a_dataset(project_root: Path, names: tuple[str, ...]) -> dict[str, FileResourceKind]:
    dataset = project_root / "labeled-data" / "session-01"
    dataset.mkdir()
    for name in names:
        (dataset / name).touch()

    resources = inspect_project(project_root).resources_by_path

    return {name: resources[Path("labeled-data/session-01") / name].kind for name in names}


@pytest.mark.parametrize(
    "name",
    [
        # DeepLabCut v3.0.2 extract_outlier_frames and the tracklet toolbox; napari-deeplabcut v0.4.0 ui/cropping.py.
        "machinelabels-iter0.h5",
        "machinelabels-iter12.h5",
        # DeepLabCut v3.0.2 extract_outlier_frames and the tracklet toolbox, beside the H5.
        "machinelabels.csv",
    ],
)
def test_a_machine_labels_file_a_producer_writes_is_discovered(valid_project: Path, name: str) -> None:
    assert _kinds_in_a_dataset(valid_project, (name,)) == {name: FileResourceKind.MACHINE_LABELS}


def test_machine_and_human_labels_are_told_apart_in_one_dataset(valid_project: Path) -> None:
    kinds = _kinds_in_a_dataset(
        valid_project, ("CollectedData_me.h5", "CollectedData_me.csv", "machinelabels-iter0.h5", "machinelabels.csv")
    )

    assert kinds == {
        "CollectedData_me.h5": FileResourceKind.HUMAN_LABELS,
        "CollectedData_me.csv": FileResourceKind.HUMAN_LABELS,
        "machinelabels-iter0.h5": FileResourceKind.MACHINE_LABELS,
        "machinelabels.csv": FileResourceKind.MACHINE_LABELS,
    }


@pytest.mark.parametrize(
    "name",
    [
        # napari-deeplabcut v0.4.0 opens any case (core/discovery.py); no producer writes it.
        "MachineLabels-iter0.h5",
        # DeepLabCut v3.0.2 merge_datasets checks it exists; nothing in DeepLabCut's history writes it.
        "MachineLabelsRefine.h5",
        "machinelabels-iter0.csv",
    ],
)
def test_a_machine_labels_name_no_producer_writes_is_not_machine_labels(valid_project: Path, name: str) -> None:
    assert _kinds_in_a_dataset(valid_project, (name,)) == {name: FileResourceKind.UNKNOWN_FILE}


def test_a_non_numeric_iteration_is_still_machine_labels(valid_project: Path) -> None:
    """The prefix does not check that the iteration is a number; no producer writes one that is not."""

    name = "machinelabels-iterX.h5"

    assert _kinds_in_a_dataset(valid_project, (name,)) == {name: FileResourceKind.MACHINE_LABELS}


@pytest.mark.parametrize(
    "name",
    ["CollectedData_me - Copy.csv", "CollectedData_me.h5.bak_20260410_101324.h5", "CollectedData_someone-else.h5"],
)
def test_a_copy_backup_or_other_scorer_is_still_human_labels(valid_project: Path, name: str) -> None:
    """The layout cannot know the config's scorer; telling these apart is owed to validation."""

    assert _kinds_in_a_dataset(valid_project, (name,)) == {name: FileResourceKind.HUMAN_LABELS}
