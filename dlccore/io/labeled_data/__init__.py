"""Labeled-data files: ``model`` holds what is true in any format, and each table format is a subpackage.

Imports no format eagerly, so no format's dependencies load until a file is read or written, or that format is
imported.
"""

from dlc_octopus.io.labeled_data.arrays.from_long import LabeledDataArrays
from dlc_octopus.io.labeled_data.file import LabeledDataFile, read_labeled_data, write_labeled_data
from dlc_octopus.io.labeled_data.long.model import LABELED_DATA_LONG_VERSION, LabeledData
from dlc_octopus.io.labeled_data.model import BodypartScope, FramePath, Keypoint

__all__ = [
    "LABELED_DATA_LONG_VERSION",
    "BodypartScope",
    "FramePath",
    "Keypoint",
    "LabeledData",
    "LabeledDataArrays",
    "LabeledDataFile",
    "read_labeled_data",
    "write_labeled_data",
]
