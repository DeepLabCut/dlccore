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

from dlccore.project import api
from dlccore.project.api import helpers
from dlccore.project.schemas.project_layout.labeled_data import LABELED_DATASET, LABELED_FRAME
from dlccore.project.schemas.project_layout.videos import VIDEOS

_DS1 = ("labeled-data", "ds1")


def test_octopus_provides_the_scope_query() -> None:
    assert hasattr(api, "findings_in_scope")


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
