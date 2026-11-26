"""Findings about the labels a file holds, one report per code; run only on a file that has a model."""

from __future__ import annotations

from pydantic import JsonValue

from dlc_octopus.content.labeled_data.file import keypoint_evidence
from dlc_octopus.content.labeled_data.report import Report, finding
from dlc_octopus.io.labeled_data import LabeledData, LabeledDataFile
from dlc_octopus.io.labeled_data.checks.model import (
    keypoints_with_empty_individual,
    likelihood_in_human_labels,
    missing_keypoints,
)
from dlc_octopus.schemas.base.finding_codes import LabeledDataFindingCode as Code
from dlc_octopus.schemas.base.findings import Finding
from dlc_octopus.schemas.locations import Location


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
