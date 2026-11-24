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

from pathlib import Path

import pytest

from dlccore.io.suffixes import LABELED_DATA_TABLE_SUFFIXES, VIDEO_SUFFIXES, suffix_in


@pytest.mark.parametrize("name", ["labels.h5", "labels.H5", "labels.Csv"])
def test_a_suffix_is_matched_in_any_case_and_returned_in_lower_case(name: str) -> None:
    assert suffix_in(Path(name), LABELED_DATA_TABLE_SUFFIXES) == Path(name).suffix.lower()


@pytest.mark.parametrize("name", ["labels.txt", "labels", "labels.h5.bak"])
def test_another_suffix_is_not_matched(name: str) -> None:
    assert suffix_in(Path(name), LABELED_DATA_TABLE_SUFFIXES) is None


@pytest.mark.parametrize("suffixes", [LABELED_DATA_TABLE_SUFFIXES, VIDEO_SUFFIXES])
def test_every_suffix_is_lower_case_with_its_dot(suffixes: frozenset[str]) -> None:
    assert suffixes  # control: an empty set would pass vacuously
    assert all(suffix.startswith(".") and suffix == suffix.lower() for suffix in suffixes)


def test_the_video_suffixes_are_deeplabcuts() -> None:
    # DeepLabCut 3.0.2's SUPPORTED_VIDEOS, in deeplabcut/utils/auxfun_videos.py.
    supported = "avi", "mp4", "mov", "mpeg", "mpg", "mpv", "mkv", "flv", "qt", "yuv"

    assert {f".{extension}" for extension in supported} == VIDEO_SUFFIXES
