"""Facts about the labels a file holds, read through the model's public surface."""

from __future__ import annotations

from pathlib import PurePath
from typing import TYPE_CHECKING

from dlc_octopus.io.labeled_data.model import BodypartScope, Keypoint
from dlc_octopus.io.suffixes import HUMAN_LABELS_PREFIX

if TYPE_CHECKING:
    from dlc_octopus.io.labeled_data.long.model import LabeledData


def likelihood_in_human_labels(labels: LabeledData, path: PurePath) -> bool:
    """Whether labels read from, or bound for, a human-labels (``CollectedData_*``) file store a likelihood."""

    return labels.has_likelihood and path.name.startswith(HUMAN_LABELS_PREFIX)


def missing_keypoints(labels: LabeledData) -> tuple[Keypoint, ...]:
    """The (individual, bodypart) pairs absent where another individual of the same scorer has the bodypart.

    Unique bodyparts, under ``"single"``, are their own set and never compared with an individual's. In order of
    scorer, individual, then bodypart, each by first appearance.
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
    """The keypoints whose individual is named ``""``, in keypoint order.

    DeepLabCut's header either names an individual or has no individuals level; napari-deeplabcut spells "none" as
    ``""`` in memory only.
    """

    return tuple(k for k in labels.to_arrays().keypoints if k.individual == "")
