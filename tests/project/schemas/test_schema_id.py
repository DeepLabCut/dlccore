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

import pytest
from pydantic import ValidationError

from dlccore.project.capabilities import get_project_manager_capabilities
from dlccore.project.schemas.base.filesystem import FileSchema, NamePattern
from dlccore.project.schemas.base.resources import FileResourceKind


def _file_schema(schema_id: str) -> FileSchema:
    return FileSchema(schema_id=schema_id, kind=FileResourceKind.UNKNOWN_FILE, name=NamePattern())


@pytest.mark.parametrize("schema_id", ["dlc.Videos", "dlc.videos ", "videos", "dlc.", "dlc..videos"])
def test_a_malformed_schema_id_is_refused_where_it_is_declared(schema_id: str) -> None:
    with pytest.raises(ValidationError, match="layout schema ID"):
        _file_schema(schema_id)


def test_a_well_formed_schema_id_is_accepted_even_if_no_node_declares_it() -> None:
    # Membership is not this type's job; only the layout knows which IDs exist.
    assert _file_schema("dlc.not_declared_anywhere").schema_id == "dlc.not_declared_anywhere"


def test_a_schema_id_serializes_as_a_plain_string() -> None:
    layout = json.loads(get_project_manager_capabilities().model_dump_json())["layout"]

    assert layout["schema_id"] == "dlc.project"
