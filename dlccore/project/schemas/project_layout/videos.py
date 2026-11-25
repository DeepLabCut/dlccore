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

from dlccore.io.suffixes import VIDEO_SUFFIXES
from dlccore.project.schemas.base.filesystem import (
    DirectorySchema,
    FileSchema,
    NamePattern,
    RepeatedNodeSchema,
)
from dlccore.project.schemas.base.resources import DirectoryResourceKind, FileResourceKind

SOURCE_VIDEO = FileSchema(
    schema_id="dlc.videos.source_video",
    kind=FileResourceKind.SOURCE_VIDEO,
    name=NamePattern(
        suffixes=VIDEO_SUFFIXES,
    ),
    description="A source video registered with the project.",
)


VIDEOS = DirectorySchema(
    schema_id="dlc.videos",
    kind=DirectoryResourceKind.VIDEOS_DIRECTORY,
    name=NamePattern(exact="videos"),
    description="Root directory for project source videos.",
    repeated_children=(RepeatedNodeSchema(child=SOURCE_VIDEO),),
)
