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

from pathlib import Path

import pytest

from dlccore.project.schemas.base.filesystem import FileSchema, NamePattern
from dlccore.project.schemas.base.findings import Reach
from dlccore.project.schemas.base.resources import FileResourceKind
from dlccore.project.schemas.locations import all_of, node_location, one_of, one_per_project
from dlccore.project.schemas.project_layout.labeled_data import LABELED_DATASET, LABELED_FRAME, NAPARI_DLC_METADATA
from dlccore.project.schemas.project_layout.project import PROJECT_CONFIG
from dlccore.project.schemas.project_layout.videos import SOURCE_VIDEO


def test_the_only_node_of_its_kind_is_one_per_project() -> None:
    assert one_per_project(PROJECT_CONFIG) == {
        "schema_id": "dlc.project.config",
        "reach": Reach.ONE_PER_PROJECT,
        "instance": (),
    }


def test_a_repeated_node_is_not_one_per_project() -> None:
    with pytest.raises(ValueError, match=r"dlc\.videos\.source_video"):
        one_per_project(SOURCE_VIDEO)


def test_one_instance_of_a_repeated_node_is_one_of() -> None:
    assert one_of(SOURCE_VIDEO, ("videos", "a.mp4")) == {
        "schema_id": "dlc.videos.source_video",
        "reach": Reach.ONE_OF,
        "instance": ("videos", "a.mp4"),
    }


def test_a_node_inside_a_repeated_node_is_one_of() -> None:
    location = one_of(NAPARI_DLC_METADATA, ("labeled-data", "m1", ".napari-deeplabcut.json"))

    assert location["reach"] is Reach.ONE_OF


def test_one_of_refuses_an_empty_instance() -> None:
    with pytest.raises(ValueError, match="got none"):
        one_of(SOURCE_VIDEO, ())


def test_the_only_node_of_its_kind_is_not_one_of() -> None:
    with pytest.raises(ValueError, match=r"dlc\.project\.config"):
        one_of(PROJECT_CONFIG, ("config.yaml",))


def test_every_instance_within_a_prefix_is_all_of() -> None:
    assert all_of(LABELED_FRAME, ("labeled-data", "m1")) == {
        "schema_id": "dlc.labeled_data.frame",
        "reach": Reach.ALL_OF,
        "instance": ("labeled-data", "m1"),
    }
    assert all_of(LABELED_DATASET)["instance"] == ()


def test_the_only_node_of_its_kind_is_not_all_of() -> None:
    with pytest.raises(ValueError, match=r"dlc\.project\.config"):
        all_of(PROJECT_CONFIG)


def test_a_node_the_layout_does_not_declare_is_refused() -> None:
    elsewhere = FileSchema(
        schema_id="dlc.elsewhere",
        kind=FileResourceKind.PROJECT_CONFIG,
        name=NamePattern(exact="elsewhere.yaml"),
    )

    with pytest.raises(ValueError, match="does not declare"):
        one_per_project(elsewhere)


def test_a_node_reusing_a_layout_id_with_another_definition_is_refused() -> None:
    impostor = PROJECT_CONFIG.model_copy(update={"description": "Not the declared config node."})

    with pytest.raises(ValueError, match="does not declare"):
        one_per_project(impostor)


def test_an_observed_node_is_anchored_by_its_schema_id_and_path() -> None:
    assert node_location("dlc.project.config", Path("config.yaml")) == one_per_project(PROJECT_CONFIG)
    assert node_location("dlc.videos.source_video", Path("videos/a.mp4")) == one_of(SOURCE_VIDEO, ("videos", "a.mp4"))


@pytest.mark.parametrize("schema_id", ["dlc.project.config", "dlc.videos.source_video"])
@pytest.mark.parametrize("path", [Path("/abs/config.yaml"), Path("C:/abs/config.yaml")], ids=["posix", "windows"])
def test_an_anchored_path_is_refused_for_every_node(schema_id: str, path: Path) -> None:
    with pytest.raises(ValueError):
        node_location(schema_id, path)


def test_an_observed_node_the_layout_does_not_declare_is_refused() -> None:
    with pytest.raises(ValueError, match=r"dlc\.nowhere"):
        node_location("dlc.nowhere", Path("x"))
