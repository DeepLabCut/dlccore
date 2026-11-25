# schemas/labeled_data.py

from dlc_octopus.io.suffixes import H5_SUFFIX, HUMAN_LABELS_PREFIX, LABELED_DATA_TABLE_SUFFIXES
from dlc_octopus.schemas.base.filesystem import (
    DirectorySchema,
    FileSchema,
    NamePattern,
    RepeatedNodeSchema,
)
from dlc_octopus.schemas.base.resources import (
    DirectoryResourceKind,
    FileResourceKind,
    ResourceManagement,
)

# FIXME: @C-Achard review valid file types
LABELED_FRAME = FileSchema(
    schema_id="dlc.labeled_data.frame",
    kind=FileResourceKind.LABELED_FRAME,
    name=NamePattern(
        suffixes=frozenset(
            {
                ".png",
                ".jpg",
                ".jpeg",
                ".bmp",
                ".tif",
                ".tiff",
            }
        )
    ),
    description="An extracted frame available for labeling.",
)
HUMAN_LABELS = FileSchema(
    schema_id="dlc.labeled_data.human_labels",
    kind=FileResourceKind.HUMAN_LABELS,
    name=NamePattern(
        prefix=HUMAN_LABELS_PREFIX,
        suffixes=LABELED_DATA_TABLE_SUFFIXES,
    ),
    description="Human keypoint labels for a labeled dataset.",
)

MACHINE_LABELS = FileSchema(
    schema_id="dlc.labeled_data.machine_labels",
    kind=FileResourceKind.MACHINE_LABELS,
    name=NamePattern(
        prefix="machinelabels-iter",
        suffixes=frozenset({H5_SUFFIX}),
    ),
    description="Machine keypoint labels for a labeled dataset, per training iteration.",
)

MACHINE_LABELS_CSV = FileSchema(
    schema_id="dlc.labeled_data.machine_labels_csv",
    kind=FileResourceKind.MACHINE_LABELS,
    name=NamePattern(exact="machinelabels.csv"),
    description="Machine keypoint labels as CSV, written beside the latest iteration's H5.",
)

NAPARI_DLC_METADATA = FileSchema(
    schema_id="dlc.labeled_data.napari_metadata",
    kind=FileResourceKind.NAPARI_DLC_METADATA,
    name=NamePattern(
        exact=".napari-deeplabcut.json",
    ),
    required=False,
    description="JSON metadata for the napari DeepLabCut plugin.",
    management=ResourceManagement.KNOWN_UNMANAGED,
)


# ``name`` is deliberately unconstrained: a directory here cannot be told apart from
# DeepLabCut ``check_labels`` plot output, or from a dataset not labeled yet, by structure
# alone. Discovery declines to guess; the judgement belongs to validation.
LABELED_DATASET = DirectorySchema(
    schema_id="dlc.labeled_data.dataset",
    kind=DirectoryResourceKind.LABELED_DATASET,
    name=NamePattern(),
    description=("A labeled dataset associated with one source video or another config-defined dataset identity."),
    children=(NAPARI_DLC_METADATA,),
    repeated_children=(
        RepeatedNodeSchema(child=LABELED_FRAME),
        RepeatedNodeSchema(child=HUMAN_LABELS),
        RepeatedNodeSchema(child=MACHINE_LABELS),
        RepeatedNodeSchema(child=MACHINE_LABELS_CSV),
    ),
)


LABELED_DATA = DirectorySchema(
    schema_id="dlc.labeled_data",
    kind=DirectoryResourceKind.LABELED_DATA_DIRECTORY,
    name=NamePattern(exact="labeled-data"),
    description="Root directory containing labeled datasets.",
    repeated_children=(RepeatedNodeSchema(child=LABELED_DATASET),),
)
