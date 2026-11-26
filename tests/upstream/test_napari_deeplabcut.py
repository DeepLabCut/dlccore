"""Tests whose home is napari-deeplabcut. Imports ``napari_deeplabcut`` only."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dlc_octopus.discovery.position import locate_in_layout
from dlc_octopus.io.labeled_data import LabeledData, read_labeled_data, write_labeled_data
from dlc_octopus.schemas.base.findings import Reach

pytestmark = pytest.mark.destination("napari-deeplabcut")

FIXTURES_DIR = Path(__file__).parents[1] / "io" / "fixtures" / "labeled_data"
H5_FIXTURES = sorted(path.name for path in FIXTURES_DIR.glob("*.h5"))

Point = tuple[str, str, str, float, float, float]


def _napari_points(path: Path) -> list[Point]:
    """napari-deeplabcut's reading of one file, as (frame, individual, bodypart, x, y, likelihood)."""

    io = pytest.importorskip("napari_deeplabcut.core.io")
    ((data, attributes, _kind),) = io.read_hdf_single(path)
    paths = attributes["metadata"]["paths"]
    properties = attributes["properties"]

    return sorted(
        (paths[int(frame)], str(individual), str(bodypart), float(x), float(y), float(likelihood))
        for (frame, y, x), bodypart, individual, likelihood in zip(
            data, properties["label"], properties["id"], properties["likelihood"], strict=True
        )
    )


def _octopus_points(path: Path) -> list[Point]:
    """Our labeled points, mapped onto napari's conventions where they deliberately differ.

    Three mappings, each a difference rather than a disagreement: napari writes a missing individual as ``""``,
    a frame as one ``/``-joined string, and the likelihood of human labels as ``1.0``.
    """

    long = read_labeled_data(path).labeled_data.to_long_df()
    labeled = long[long[["x", "y"]].notna().all(axis=1)]
    individuals = labeled["individual"].astype(str) if "individual" in labeled else [""] * len(labeled)
    likelihoods = labeled["likelihood"] if "likelihood" in labeled else [1.0] * len(labeled)

    return sorted(
        ("/".join((folder, dataset, image)), individual, bodypart, float(x), float(y), float(likelihood))
        for folder, dataset, image, individual, bodypart, x, y, likelihood in zip(
            labeled["folder"].astype(str),
            labeled["dataset"].astype(str),
            labeled["image"].astype(str),
            individuals,
            labeled["bodypart"].astype(str),
            labeled["x"],
            labeled["y"],
            likelihoods,
            strict=True,
        )
    )


def test_every_h5_fixture_is_found() -> None:
    # Control: an empty glob would let the equivalence test pass by running nothing.
    assert len(H5_FIXTURES) == 8


@pytest.mark.parametrize("filename", H5_FIXTURES)
def test_napari_deeplabcut_reads_the_same_labeled_points(filename: str) -> None:
    ours = _octopus_points(FIXTURES_DIR / filename)

    assert ours, "a fixture without labeled points would make the comparison vacuous"
    assert _napari_points(FIXTURES_DIR / filename) == ours


def _rewrite(table: LabeledData, tmp_path: Path) -> Path:
    # Human labels never carry a likelihood, so labels that have one are rewritten as machine labels.
    target = tmp_path / ("machinelabels-iter0.h5" if table.has_likelihood else "CollectedData_Sc.h5")
    write_labeled_data(table, target, replaces=None)
    return target


def _table(filename: str) -> LabeledData:
    return read_labeled_data(FIXTURES_DIR / filename).labeled_data


@pytest.mark.parametrize("filename", H5_FIXTURES)
def test_napari_deeplabcut_reads_a_rewritten_file_as_the_original(tmp_path: Path, filename: str) -> None:
    original = _napari_points(FIXTURES_DIR / filename)

    assert original, "a fixture without labeled points would make the comparison vacuous"
    assert _napari_points(_rewrite(_table(filename), tmp_path)) == original


def test_napari_deeplabcut_sees_one_moved_point_in_a_rewritten_file(tmp_path: Path) -> None:
    # Control: the comparison above can fail.
    long = _table("single_animal_human.h5").to_long_df()
    long.loc[np.flatnonzero(long["x"].notna())[0], "x"] += 1.0

    rewritten = _rewrite(LabeledData.from_long_df(long), tmp_path)

    assert _napari_points(rewritten) != _napari_points(FIXTURES_DIR / "single_animal_human.h5")


def _hdf_reads_by_shared_io(monkeypatch: pytest.MonkeyPatch, read: object) -> list[bool]:
    """Call ``read`` and return, for each ``pd.read_hdf`` it makes, whether ``dlc_octopus.io`` made it."""

    calls: list[bool] = []
    read_hdf = pd.read_hdf

    def spy(*args: object, **kwargs: object) -> object:
        frame = sys._getframe(1)
        while frame is not None and not frame.f_globals.get("__name__", "").startswith("dlc_octopus.io."):
            frame = frame.f_back
        calls.append(frame is not None)
        return read_hdf(*args, **kwargs)

    monkeypatch.setattr(pd, "read_hdf", spy)
    read()
    return calls


def test_the_shared_reader_is_recognised_as_shared(monkeypatch: pytest.MonkeyPatch) -> None:
    from dlc_octopus.io.labeled_data.wide.h5 import read_labeled_data_h5

    calls = _hdf_reads_by_shared_io(monkeypatch, lambda: read_labeled_data_h5(FIXTURES_DIR / "single_animal_human.h5"))

    assert calls == [True]


@pytest.mark.xfail(
    reason="napari-deeplabcut reads labeled-data H5 files with pandas itself; adopting the shared labeled-data "
    "I/O in dlc_octopus.io.labeled_data finishes when every read goes through it",
    strict=True,
    raises=AssertionError,
)
def test_napari_deeplabcut_reads_labeled_data_only_through_the_shared_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = pytest.importorskip("napari_deeplabcut._reader")
    path = str(FIXTURES_DIR / "single_animal_human.h5")

    calls = _hdf_reads_by_shared_io(monkeypatch, lambda: reader.get_hdf_reader(path)(path))

    assert calls
    assert all(calls)


@pytest.mark.xfail(
    reason="napari-deeplabcut v0.4.0 takes a folder as a dataset when 'labeled-data' appears anywhere above it, in "
    "any case; the layout places a dataset only under labeled-data itself (dlc_octopus.discovery.position). "
    "Adopting the layout position in napari-deeplabcut makes them agree",
    strict=True,
    raises=AssertionError,
)
def test_napari_deeplabcut_takes_a_folder_as_a_dataset_only_where_the_layout_places_one(tmp_path: Path) -> None:
    project_paths = pytest.importorskip("napari_deeplabcut.core.project_paths")
    folder = tmp_path / "Labeled-Data" / "x" / "Sample1"
    folder.mkdir(parents=True)

    layout_says_dataset = locate_in_layout(folder / "CollectedData_Sc.h5").location["reach"] is not Reach.UNROOTED
    napari_says_dataset = project_paths.infer_labeled_data_folder_from_paths([], fallback_root=folder) is not None

    assert napari_says_dataset == layout_says_dataset
