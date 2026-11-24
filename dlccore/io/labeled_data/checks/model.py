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

"""Checks on the labels a file holds, using only ``LabeledData``'s public methods."""

from __future__ import annotations

from pathlib import PurePath
from typing import TYPE_CHECKING

from dlccore.io.labeled_data.model import BodypartScope, Keypoint
from dlccore.io.suffixes import HUMAN_LABELS_PREFIX

if TYPE_CHECKING:
    from dlccore.io.labeled_data.long.model import LabeledData


def likelihood_in_human_labels(labels: LabeledData, path: PurePath) -> bool:
    """Return whether labels store a likelihood although their file is a human-labels file.

    Only machine labels have a likelihood. Human labels are stored in ``CollectedData_*`` files.

    Args:
        labels: The labels.
        path: The file they were read from, or are about to be written to.
    """

    return labels.has_likelihood and path.name.startswith(HUMAN_LABELS_PREFIX)


def missing_keypoints(labels: LabeledData) -> tuple[Keypoint, ...]:
    """Return the bodyparts an individual lacks although another individual of the same scorer has them.

    Unique bodyparts (individual ``"single"``) are not compared with individuals' bodyparts.

    Returns:
        One keypoint per missing (individual, bodypart), ordered by scorer, then individual, then bodypart, each in
        order of first appearance.
    """

    keypoints = [k for k in labels.to_arrays().keypoints if k.scope is BodypartScope.INDIVIDUAL]
    present = {(k.scorer, k.individual, k.bodypart) for k in keypoints}
    missing = []
    for scorer in dict.fromkeys(k.scorer for k in keypoints):
        individuals = dict.fromkeys(k.individual for k in keypoints if k.scorer == scorer)
        bodyparts = dict.fromkeys(k.bodypart for k in keypoints if k.scorer == scorer)
        missing += [
            Keypoint(scorer, individual, bodypart)
            for individual in individuals
            for bodypart in bodyparts
            if (scorer, individual, bodypart) not in present
        ]
    return tuple(missing)


def keypoints_with_empty_individual(labels: LabeledData) -> tuple[Keypoint, ...]:
    """Return any keypoint with individual ``""``, in keypoint order.

    DeepLabCut never writes this, the header either names the individual or has no individuals level.
    napari-deeplabcut uses ``""`` for "no individual", but only in memory.
    Any data that shows this should be considered malformed.
    """

    return tuple(k for k in labels.to_arrays().keypoints if k.individual == "")
