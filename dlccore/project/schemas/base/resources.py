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

"""Basic project resource types.

Vocabulary only.
"""

from __future__ import annotations

from enum import Enum

from pydantic import model_validator
from typing_extensions import Self

from dlccore.project.schemas.base.core import FrozenModel, ProjectRelativePath
from dlccore.project.schemas.base.symlinks import SymlinkObservation


class ResourceManagement(str, Enum):
    """Describe whether DeepLabCut manages a resource or not."""

    # DLC manages the resource (e.g., config.yaml)
    MANAGED = "managed"
    # DLC does not manage the resource, but knows about it (e.g. .napari-deeplabcut JSON metadata)
    KNOWN_UNMANAGED = "known_unmanaged"
    # DLC does not manage the resource and does not know about it
    UNDETERMINED = "undetermined"


class DirectoryResourceKind(str, Enum):
    # Project folder
    PROJECT_ROOT = "project_root"

    # Videos
    VIDEOS_DIRECTORY = "videos_directory"

    # Labeled data
    LABELED_DATA_DIRECTORY = "labeled_data_directory"  # hosts all labeled datasets
    LABELED_DATASET = "labeled_dataset"  # frames + labels

    # Extend later...

    # Unknown resources
    UNKNOWN_DIRECTORY = "unknown_directory"


class FileResourceKind(str, Enum):
    # Project config file
    PROJECT_CONFIG = "project_config"

    # Videos
    SOURCE_VIDEO = "source_video"

    # Labeled data
    LABELED_FRAME = "labeled_frame"  # image file
    HUMAN_LABELS = "human_labels"  # keypoints
    MACHINE_LABELS = "machine_labels"  # keypoints
    NAPARI_DLC_METADATA = "napari_dlc_metadata"  # JSON metadata for napari plugin

    # Unknown resources
    UNKNOWN_FILE = "unknown_file"


class UnknownResourceKind(str, Enum):
    """An entry with an unknown resource kind."""

    UNKNOWN = "unknown"


ResourceKind = DirectoryResourceKind | FileResourceKind | UnknownResourceKind


class ProjectResource(FrozenModel):
    """A single project resource, as observed on disk."""

    project_relative_path: ProjectRelativePath
    symlink: SymlinkObservation | None = None
    kind: ResourceKind
    management: ResourceManagement = ResourceManagement.MANAGED

    @model_validator(mode="after")
    def validate_kind_and_management(
        self,
    ) -> Self:
        """Ensure semantic kind and management status agree."""

        unknown_kinds = {
            FileResourceKind.UNKNOWN_FILE,
            DirectoryResourceKind.UNKNOWN_DIRECTORY,
            UnknownResourceKind.UNKNOWN,
        }
        has_unknown_kind = self.kind in unknown_kinds
        has_undetermined_management = self.management == ResourceManagement.UNDETERMINED

        if has_unknown_kind != has_undetermined_management:
            raise ValueError(
                "Unknown resource kinds must use undetermined "
                "management, and undetermined management may "
                "only be used with an unknown resource kind."
            )

        return self
