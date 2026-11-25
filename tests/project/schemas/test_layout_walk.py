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

import pytest
from pydantic import ValidationError

from dlccore.project.schemas.base.filesystem import (
    DirectorySchema,
    FileSchema,
    NamePattern,
    RepeatedNodeSchema,
    child_schemas,
    index_layout,
    walk_schema,
)
from dlccore.project.schemas.base.resources import DirectoryResourceKind, FileResourceKind, ResourceManagement
from dlccore.project.schemas.project_layout.labeled_data import LABELED_FRAME
from dlccore.project.schemas.project_layout.project import PROJECT


def test_the_walk_flags_exactly_the_repeated_children() -> None:
    flags = {schema.schema_id: repeated for schema, repeated in walk_schema(PROJECT)}

    assert flags["dlc.project"] is False
    assert flags["dlc.labeled_data"] is False
    assert flags["dlc.labeled_data.dataset"] is True
    assert flags["dlc.labeled_data.napari_metadata"] is False
    assert flags["dlc.videos.source_video"] is True


def test_the_walk_visits_a_parent_before_its_children() -> None:
    order = [schema.schema_id for schema, _ in walk_schema(PROJECT)]

    assert order[0] == "dlc.project"
    assert order.index("dlc.labeled_data") < order.index("dlc.labeled_data.dataset")
    assert order.index("dlc.labeled_data.dataset") < order.index("dlc.labeled_data.frame")


def test_a_file_has_no_child_schemas() -> None:
    assert child_schemas(LABELED_FRAME) == ()


def test_the_project_layout_indexes_every_node_once() -> None:
    assert list(index_layout(PROJECT).schemas) == [schema.schema_id for schema, _ in walk_schema(PROJECT)]


def test_the_layout_index_records_each_parent() -> None:
    parents = index_layout(PROJECT).parents

    assert parents["dlc.project"] is None
    assert parents["dlc.project.config"] == "dlc.project"
    assert parents["dlc.videos.source_video"] == "dlc.videos"
    assert parents["dlc.labeled_data.napari_metadata"] == "dlc.labeled_data.dataset"
    assert parents["dlc.labeled_data.frame"] == "dlc.labeled_data.dataset"


def test_the_layout_index_flags_every_node_inside_a_repeated_node() -> None:
    # The sidecar is a plain child, but of a repeated dataset, so it lies inside one.
    assert index_layout(PROJECT).inside_repeated == {
        "dlc.videos.source_video",
        "dlc.labeled_data.dataset",
        "dlc.labeled_data.napari_metadata",
        "dlc.labeled_data.frame",
        "dlc.labeled_data.human_labels",
        "dlc.labeled_data.machine_labels",
        "dlc.labeled_data.machine_labels_csv",
    }


def test_the_layout_index_is_read_only() -> None:
    index = index_layout(PROJECT)

    with pytest.raises(TypeError):
        index.schemas["dlc.elsewhere"] = PROJECT  # type: ignore[index]


def test_a_schema_id_names_one_position() -> None:
    assert "schema_id" not in RepeatedNodeSchema.model_fields

    readme = FileSchema(
        schema_id="dlc.readme",
        kind=FileResourceKind.NAPARI_DLC_METADATA,
        name=NamePattern(exact="README.txt"),
    )
    dataset = DirectorySchema(
        schema_id="dlc.dataset",
        kind=DirectoryResourceKind.LABELED_DATASET,
        name=NamePattern(),
        children=(readme,),
    )
    layout = DirectorySchema(
        schema_id="dlc.project",
        kind=DirectoryResourceKind.PROJECT_ROOT,
        name=NamePattern(),
        children=(readme,),
        repeated_children=(RepeatedNodeSchema(child=dataset),),
    )

    with pytest.raises(ValueError, match=r"dlc\.readme"):
        index_layout(layout)


@pytest.mark.parametrize(
    ("schema_type", "kind"),
    [
        (FileSchema, FileResourceKind.NAPARI_DLC_METADATA),
        (DirectorySchema, DirectoryResourceKind.LABELED_DATASET),
    ],
)
def test_a_known_unmanaged_node_cannot_be_required(
    schema_type: type[FileSchema | DirectorySchema],
    kind: FileResourceKind | DirectoryResourceKind,
) -> None:
    with pytest.raises(ValidationError, match="cannot be required"):
        schema_type(
            schema_id="dlc.unmanaged",
            kind=kind,
            name=NamePattern(exact="unmanaged"),
            management=ResourceManagement.KNOWN_UNMANAGED,
            required=True,
        )


def test_a_known_unmanaged_node_may_be_optional_and_a_managed_node_required() -> None:
    sidecar = FileSchema(
        schema_id="dlc.sidecar",
        kind=FileResourceKind.NAPARI_DLC_METADATA,
        name=NamePattern(exact="sidecar.json"),
        management=ResourceManagement.KNOWN_UNMANAGED,
    )
    config = FileSchema(
        schema_id="dlc.config",
        kind=FileResourceKind.PROJECT_CONFIG,
        name=NamePattern(exact="config.yaml"),
        required=True,
    )

    assert (sidecar.required, config.required) == (False, True)


def test_the_ancestry_runs_from_a_node_up_to_the_root() -> None:
    index = index_layout(PROJECT)

    assert index.ancestry("dlc.labeled_data.frame") == (
        "dlc.labeled_data.frame",
        "dlc.labeled_data.dataset",
        "dlc.labeled_data",
        "dlc.project",
    )
    assert index.ancestry("dlc.project") == ("dlc.project",)


def test_the_subtree_holds_the_node_and_its_descendants_only() -> None:
    index = index_layout(PROJECT)

    assert index.subtree("dlc.labeled_data.dataset") == (
        "dlc.labeled_data.dataset",
        "dlc.labeled_data.napari_metadata",
        "dlc.labeled_data.frame",
        "dlc.labeled_data.human_labels",
        "dlc.labeled_data.machine_labels",
        "dlc.labeled_data.machine_labels_csv",
    )
    assert index.subtree("dlc.project.config") == ("dlc.project.config",)


@pytest.mark.parametrize("query", ["ancestry", "subtree"])
def test_an_undeclared_schema_id_is_refused(query: str) -> None:
    with pytest.raises(KeyError):
        getattr(index_layout(PROJECT), query)("dlc.not_declared")


def test_the_layout_lists_its_nodes_of_one_type_in_walk_order() -> None:
    index = index_layout(PROJECT)
    files = index.schema_ids_of(FileSchema)

    assert files == tuple(schema_id for schema_id in index.schemas if schema_id in files)
    assert "dlc.project.config" in files
    assert "dlc.labeled_data" not in files
    assert set(files) | set(index.schema_ids_of(DirectorySchema)) == set(index.schemas)
