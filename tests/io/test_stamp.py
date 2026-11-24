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

from importlib.metadata import version

import pytest

from dlccore.io.stamp import WRITER_KEY, WRITER_VERSION_KEY, WriterStamp


def test_the_current_stamp_records_the_installed_version() -> None:
    assert WriterStamp.current("some.writer") == WriterStamp("some.writer", version("dlccore"))


def test_a_stamp_survives_its_metadata_form() -> None:
    stamp = WriterStamp("some.writer", "1.2.3")

    metadata = stamp.as_metadata()

    assert metadata == {WRITER_KEY: "some.writer", WRITER_VERSION_KEY: "1.2.3"}
    assert all(type(value) is str for value in metadata.values())
    assert WriterStamp.from_metadata(metadata) == stamp


def test_metadata_without_either_key_is_no_stamp() -> None:
    assert WriterStamp.from_metadata({"other": "x"}) is None


@pytest.mark.parametrize(
    "metadata",
    [
        {WRITER_KEY: "w"},
        {WRITER_VERSION_KEY: "1"},
        {WRITER_KEY: "w", WRITER_VERSION_KEY: 1},
        {WRITER_KEY: b"w", WRITER_VERSION_KEY: "1"},
    ],
    ids=["writer_only", "version_only", "version_not_text", "writer_bytes"],
)
def test_a_present_stamp_that_is_not_two_plain_strings_raises(metadata: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="malformed writer stamp"):
        WriterStamp.from_metadata(metadata)
