# dlc_octopus/schemas/project_layout/videos.py

from dlc_octopus.io.suffixes import VIDEO_SUFFIXES
from dlc_octopus.schemas.base.filesystem import (
    DirectorySchema,
    FileSchema,
    NamePattern,
    RepeatedNodeSchema,
)
from dlc_octopus.schemas.base.resources import DirectoryResourceKind, FileResourceKind

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
