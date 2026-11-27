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

"""Findings about the labels a file holds, one report per code; run only on a file that has a model."""

from __future__ import annotations

from pydantic import JsonValue

from dlccore.io.labeled_data import LabeledData, LabeledDataFile
from dlccore.io.labeled_data.checks.model import (
    keypoints_with_empty_individual,
    likelihood_in_human_labels,
    missing_keypoints,
)
from dlccore.project.content.labeled_data.file import keypoint_evidence
from dlccore.project.content.labeled_data.report import Report, finding
from dlccore.project.schemas.base.finding_codes import LabeledDataFindingCode as Code
from dlccore.project.schemas.base.findings import Finding
from dlccore.project.schemas.locations import Location


def _human_likelihood(data: LabeledDataFile, labels: LabeledData, at: Location) -> tuple[Finding, ...]:
    return (
        finding(Code.LABELED_DATA_HUMAN_LIKELIHOOD, data, at) if likelihood_in_human_labels(labels, data.path) else ()
    )


def _missing(data: LabeledDataFile, labels: LabeledData, at: Location) -> tuple[Finding, ...]:
    missing: list[JsonValue] = [keypoint_evidence(keypoint) for keypoint in missing_keypoints(labels)]
    return finding(Code.LABELED_DATA_KEYPOINT_MISSING, data, at, missing=missing) if missing else ()


def _empty_individual(data: LabeledDataFile, labels: LabeledData, at: Location) -> tuple[Finding, ...]:
    keypoints: list[JsonValue] = [keypoint_evidence(keypoint) for keypoint in keypoints_with_empty_individual(labels)]
    return finding(Code.LABELED_DATA_INDIVIDUAL_EMPTY, data, at, keypoints=keypoints) if keypoints else ()


# Each emits as ``emit(data: LabeledDataFile, labels: LabeledData, at: Location)``.
REPORTS: tuple[Report, ...] = (
    Report(likelihood_in_human_labels, Code.LABELED_DATA_HUMAN_LIKELIHOOD, _human_likelihood),
    Report(missing_keypoints, Code.LABELED_DATA_KEYPOINT_MISSING, _missing),
    Report(keypoints_with_empty_individual, Code.LABELED_DATA_INDIVIDUAL_EMPTY, _empty_individual),
)
