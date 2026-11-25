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

from dlccore.project.schemas.base.filesystem import (
    DirectorySchema,
    FileSchema,
    NamePattern,
    index_layout,
)
from dlccore.project.schemas.base.resources import DirectoryResourceKind, FileResourceKind
from dlccore.project.schemas.project_layout.labeled_data import LABELED_DATA
from dlccore.project.schemas.project_layout.videos import VIDEOS

PROJECT_CONFIG = FileSchema(
    schema_id="dlc.project.config",
    kind=FileResourceKind.PROJECT_CONFIG,
    name=NamePattern(exact="config.yaml"),
    required=True,
    description="The authoritative DeepLabCut project configuration.",
)


PROJECT = DirectorySchema(
    schema_id="dlc.project",
    kind=DirectoryResourceKind.PROJECT_ROOT,
    name=NamePattern(),
    required=True,
    description="A canonical DeepLabCut project root.",
    children=(
        # Config file
        PROJECT_CONFIG,
        ## Input and labeled data
        VIDEOS,
        LABELED_DATA,
        ## Training datasets
        ### Extend later...
    ),
)

PROJECT_INDEX = index_layout(PROJECT)
