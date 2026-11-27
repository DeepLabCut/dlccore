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

"""Findings about a labeled-data file as stored, one report per code, each turning one check's fact into findings."""

from __future__ import annotations

from pydantic import JsonValue

from dlccore.io.labeled_data import Keypoint, LabeledDataFile
from dlccore.io.labeled_data.checks.file import (
    frame_absorbed_into_index_names,
    frames_of_other_depth,
    frames_with_unnamed_parts,
    has_no_frames,
    header_is_unrecognised,
    keypoints_with_unrecognised_coords,
    keypoints_without_the_likelihood_others_store,
    legacy_h5_key,
    non_numeric_columns,
    row_index_is_unrecognised,
    unknown_h5_key,
    unread_h5_keys,
)
from dlccore.project.content.labeled_data.report import Report, finding
from dlccore.project.schemas.base.finding_codes import LabeledDataFindingCode as Code
from dlccore.project.schemas.base.findings import Finding
from dlccore.project.schemas.locations import Location


def keypoint_evidence(keypoint: Keypoint) -> dict[str, JsonValue]:
    """A keypoint as evidence: its scorer, individual (``None`` without an individuals level) and bodypart."""

    return {"scorer": keypoint.scorer, "individual": keypoint.individual, "bodypart": keypoint.bodypart}


def _multi_key(data: LabeledDataFile, at: Location) -> tuple[Finding, ...]:
    selection = data.key_selection
    if selection is None or not unread_h5_keys(selection):
        return ()
    keys_found: list[JsonValue] = list(selection.keys_found)
    return finding(Code.LABELED_DATA_H5_MULTI_KEY, data, at, keys_found=keys_found, selected_key=selection.selected_key)


def _legacy_key(data: LabeledDataFile, at: Location) -> tuple[Finding, ...]:
    key = legacy_h5_key(data.key_selection)
    return finding(Code.LABELED_DATA_H5_LEGACY_KEY, data, at, key=key) if key else ()


def _unknown_key(data: LabeledDataFile, at: Location) -> tuple[Finding, ...]:
    key = unknown_h5_key(data.key_selection)
    return finding(Code.LABELED_DATA_H5_UNKNOWN_KEY, data, at, key=key) if key else ()


def _header(data: LabeledDataFile, at: Location) -> tuple[Finding, ...]:
    if not header_is_unrecognised(data.raw_header):
        return ()
    level_names: list[JsonValue] = [None if name is None else str(name) for name in data.raw_header.level_names]
    return finding(Code.LABELED_DATA_HEADER_UNRECOGNISED, data, at, level_names=level_names)


def _coords(data: LabeledDataFile, at: Location) -> tuple[Finding, ...]:
    found = keypoints_with_unrecognised_coords(data.raw_header)
    keypoints: list[JsonValue] = [{**keypoint_evidence(c.keypoint), "coords": list(c.coords)} for c in found]
    return finding(Code.LABELED_DATA_COORDS_UNRECOGNISED, data, at, keypoints=keypoints) if found else ()


def _partial_likelihood(data: LabeledDataFile, at: Location) -> tuple[Finding, ...]:
    found = keypoints_without_the_likelihood_others_store(data.raw_header)
    keypoints: list[JsonValue] = [keypoint_evidence(columns.keypoint) for columns in found]
    return finding(Code.LABELED_DATA_LIKELIHOOD_PARTIAL, data, at, keypoints=keypoints) if found else ()


def _not_numeric(data: LabeledDataFile, at: Location) -> tuple[Finding, ...]:
    found = non_numeric_columns(data.to_raw_wide_df())
    columns: list[JsonValue] = [
        [str(level) for level in label] if isinstance(label, tuple) else str(label) for label in found
    ]
    return finding(Code.LABELED_DATA_VALUES_NOT_NUMERIC, data, at, columns=columns) if found else ()


def _no_frames(data: LabeledDataFile, at: Location) -> tuple[Finding, ...]:
    return finding(Code.LABELED_DATA_NO_FRAMES, data, at) if has_no_frames(data.raw_row_index) else ()


def _row_index(data: LabeledDataFile, at: Location) -> tuple[Finding, ...]:
    row_index = data.raw_row_index
    if not row_index_is_unrecognised(row_index):
        return ()
    first = next(entry for entry in row_index.entries if not isinstance(entry, str))
    return finding(Code.LABELED_DATA_ROW_INDEX_UNRECOGNISED, data, at, first_entry=repr(first))


def _frame_rows(code: Code, rows: tuple[int, ...], data: LabeledDataFile, at: Location) -> tuple[Finding, ...]:
    if not rows:
        return ()
    first_frame: list[JsonValue] = [repr(part) for part in data.raw_row_index.frames[rows[0]]]
    return finding(code, data, at, row_count=len(rows), first_frame=first_frame)


def _frame_depth(data: LabeledDataFile, at: Location) -> tuple[Finding, ...]:
    return _frame_rows(Code.LABELED_DATA_FRAME_DEPTH, frames_of_other_depth(data.raw_row_index), data, at)


def _frame_part(data: LabeledDataFile, at: Location) -> tuple[Finding, ...]:
    return _frame_rows(Code.LABELED_DATA_FRAME_PART_UNNAMED, frames_with_unnamed_parts(data.raw_row_index), data, at)


def _absorbed(data: LabeledDataFile, at: Location) -> tuple[Finding, ...]:
    frame: list[JsonValue] = list(frame_absorbed_into_index_names(data.raw_row_index))
    return finding(Code.LABELED_DATA_CSV_FRAME_ABSORBED, data, at, frame=frame) if frame else ()


def _stamp(data: LabeledDataFile, at: Location) -> tuple[Finding, ...]:
    error = data.writer_stamp_error
    return finding(Code.LABELED_DATA_WRITER_STAMP_MALFORMED, data, at, error_message=error) if error else ()


# Each emits as ``emit(data: LabeledDataFile, at: Location)``.
REPORTS: tuple[Report, ...] = (
    Report(unread_h5_keys, Code.LABELED_DATA_H5_MULTI_KEY, _multi_key),
    Report(legacy_h5_key, Code.LABELED_DATA_H5_LEGACY_KEY, _legacy_key),
    Report(unknown_h5_key, Code.LABELED_DATA_H5_UNKNOWN_KEY, _unknown_key),
    Report(header_is_unrecognised, Code.LABELED_DATA_HEADER_UNRECOGNISED, _header),
    Report(keypoints_with_unrecognised_coords, Code.LABELED_DATA_COORDS_UNRECOGNISED, _coords),
    Report(keypoints_without_the_likelihood_others_store, Code.LABELED_DATA_LIKELIHOOD_PARTIAL, _partial_likelihood),
    Report(non_numeric_columns, Code.LABELED_DATA_VALUES_NOT_NUMERIC, _not_numeric),
    Report(has_no_frames, Code.LABELED_DATA_NO_FRAMES, _no_frames),
    Report(row_index_is_unrecognised, Code.LABELED_DATA_ROW_INDEX_UNRECOGNISED, _row_index),
    Report(frames_of_other_depth, Code.LABELED_DATA_FRAME_DEPTH, _frame_depth),
    Report(frames_with_unnamed_parts, Code.LABELED_DATA_FRAME_PART_UNNAMED, _frame_part),
    Report(frame_absorbed_into_index_names, Code.LABELED_DATA_CSV_FRAME_ABSORBED, _absorbed),
    Report(None, Code.LABELED_DATA_WRITER_STAMP_MALFORMED, _stamp),
)
