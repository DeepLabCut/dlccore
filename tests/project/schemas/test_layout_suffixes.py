from pathlib import Path

import pytest

from dlc_octopus.io.suffixes import LABELED_DATA_TABLE_SUFFIXES, VIDEO_SUFFIXES
from dlc_octopus.schemas.base.filesystem import FileSchema
from dlc_octopus.schemas.project_layout.labeled_data import HUMAN_LABELS, MACHINE_LABELS
from dlc_octopus.schemas.project_layout.videos import SOURCE_VIDEO


@pytest.mark.parametrize(
    ("schema", "suffixes"),
    [(HUMAN_LABELS, LABELED_DATA_TABLE_SUFFIXES), (MACHINE_LABELS, {".h5"}), (SOURCE_VIDEO, VIDEO_SUFFIXES)],
)
def test_the_layout_claims_the_suffixes_io_reads(schema: FileSchema, suffixes: frozenset[str]) -> None:
    assert schema.name.suffixes == suffixes


@pytest.mark.parametrize("name", ["session.flv", "session.MPV", "session.qt", "session.yuv"])
def test_every_video_deeplabcut_reads_is_claimed_as_a_video(name: str) -> None:
    assert SOURCE_VIDEO.name.matches(Path(name))
