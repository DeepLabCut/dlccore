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

"""Finding vocabulary: severities, and the stable codes to emit.

Validation-check, baseline-check and validator-suite identifiers share the
same naming style but are separate namespaces.

Codes are grouped by subject rather than by emitting module, following the
``<SUBJECT>_<PROBLEM>`` convention.
"""

from __future__ import annotations

from enum import Enum
from itertools import chain


class Severity(str, Enum):
    NOTE = "note"
    WARNING = "warning"
    ERROR = "error"


class FindingCode(str, Enum):
    """Base for every ``Finding.code`` namespace; declares no codes itself."""

    # Render as the bare code, as ``enum.StrEnum`` would; it is 3.11+.
    __str__ = str.__str__

    description: str
    severity: Severity
    evidence_keys: tuple[str, ...]

    def __new__(
        cls,
        code: str,
        description: str,
        severity: Severity,
        evidence_keys: tuple[str, ...],
    ) -> FindingCode:
        member = str.__new__(cls, code)
        member._value_ = code
        member.description = description
        member.severity = severity
        member.evidence_keys = tuple(evidence_keys)

        return member


class StructureFindingCode(FindingCode):
    """Findings about the project's schema-defined layout."""

    PROJECT_REQUIRED_NODE_MISSING = (
        "PROJECT_REQUIRED_NODE_MISSING",
        "A required project-layout node is missing.",
        Severity.ERROR,
        ("resource_kind",),
    )
    PROJECT_SYMLINK_TARGET_MISSING = (
        "PROJECT_SYMLINK_TARGET_MISSING",
        "A project resource is a symlink whose target does not exist.",
        Severity.ERROR,
        ("resource_kind", "raw_target", "absolute_target"),
    )
    PROJECT_SYMLINK_TARGET_WRONG_TYPE = (
        "PROJECT_SYMLINK_TARGET_WRONG_TYPE",
        "The symbolic-link target has an unexpected filesystem type.",
        Severity.ERROR,
        ("resource_kind", "raw_target", "absolute_target", "target_kind"),
    )
    PROJECT_NODE_WRONG_TYPE = (
        "PROJECT_NODE_WRONG_TYPE",
        "A project resource exists at the expected location but has the wrong filesystem type.",
        Severity.ERROR,
        ("resource_kind", "expected_type"),
    )
    PROJECT_NODE_UNREADABLE = (
        "PROJECT_NODE_UNREADABLE",
        "A project-layout node could not be read.",
        Severity.ERROR,
        ("resource_kind", "error_type", "error_message"),
    )
    PROJECT_RESOURCE_PATH_COLLISION = (
        "PROJECT_RESOURCE_PATH_COLLISION",
        "This project resource resolves to the same filesystem path as another; each one is reported.",
        Severity.ERROR,
        ("resource_kind", "collision_path", "collision_group"),
    )


class ConfigFindingCode(FindingCode):
    """Findings about reading, parsing and interpreting ``config.yaml``."""

    CONFIG_READ_FAILED = (
        "CONFIG_READ_FAILED",
        "The config file could not be read.",
        Severity.ERROR,
        ("error_type", "error_message"),
    )
    CONFIG_PARSE_FAILED = (
        "CONFIG_PARSE_FAILED",
        "The config file could not be parsed.",
        Severity.ERROR,
        ("error_type", "error_message"),
    )
    CONFIG_VALIDATION_ERROR = (
        "CONFIG_VALIDATION_ERROR",
        "The config file failed canonical schema validation.",
        Severity.ERROR,
        ("error_type", "error_message"),
    )
    CONFIG_LOAD_FAILED = (
        "CONFIG_LOAD_FAILED",
        "The config file could not be loaded through the canonical schema, for a reason other than a field "
        "failing validation, such as a config version with no migration path.",
        Severity.ERROR,
        ("error_type", "error_message"),
    )
    CONFIG_VERSION_UPGRADED = (
        "CONFIG_VERSION_UPGRADED",
        "The config declares an older config version; the config schema migrated it in memory to the current one.",
        Severity.NOTE,
        ("declared_version", "loaded_version"),
    )
    CONFIG_VERSION_DOWNGRADED = (
        "CONFIG_VERSION_DOWNGRADED",
        "The config declares a newer config version than this code reads; the config schema migrated it down in "
        "memory, which can drop fields.",
        Severity.WARNING,
        ("declared_version", "loaded_version"),
    )
    CONFIG_NO_KEYPOINTS = (
        "CONFIG_NO_KEYPOINTS",
        "The config loads but declares no keypoints.",
        Severity.WARNING,
        ("multianimal_project", "declared"),
    )
    CONFIG_NULL_KEY_DEFAULTED = (
        "CONFIG_NULL_KEY_DEFAULTED",
        "A config key is written with no value, so the config schema's field default was used.",
        Severity.NOTE,
        ("default",),
    )
    CONFIG_PROJECT_PATH_MISMATCH = (
        "CONFIG_PROJECT_PATH_MISMATCH",
        "The project path declared in config.yaml does not match the project root observed. A declared path "
        "probed as missing may also be an unreachable network location or a path from another platform.",
        Severity.NOTE,
        ("declared_project_path", "observed_project_root", "declared_path_probe", "error_type", "error_message"),
    )
    CONFIG_PROJECT_PATH_NOT_ABSOLUTE = (
        "CONFIG_PROJECT_PATH_NOT_ABSOLUTE",
        "The project path in config.yaml is missing, empty or relative, and cannot be compared with the project root.",
        Severity.NOTE,
        ("declared_project_path",),
    )


class ConfiguredVideoFindingCode(FindingCode):
    """Findings about video paths declared in ``config.yaml``."""

    CONFIGURED_VIDEO_PATH_MISSING = (
        "CONFIGURED_VIDEO_PATH_MISSING",
        "The configured video path does not exist.",
        Severity.ERROR,
        (),
    )
    CONFIGURED_VIDEO_PATH_SYMLINK_TARGET_MISSING = (
        "CONFIGURED_VIDEO_PATH_SYMLINK_TARGET_MISSING",
        "The configured video path is a symlink whose target does not exist.",
        Severity.ERROR,
        ("raw_target", "absolute_target"),
    )
    CONFIGURED_VIDEO_PATH_UNREADABLE = (
        "CONFIGURED_VIDEO_PATH_UNREADABLE",
        "The configured video path could not be read.",
        Severity.ERROR,
        ("error_type", "error_message"),
    )
    CONFIGURED_VIDEO_PATH_NOT_FILE = (
        "CONFIGURED_VIDEO_PATH_NOT_FILE",
        "The configured video path exists but is not a file.",
        Severity.ERROR,
        (),
    )
    CONFIGURED_VIDEOS_UNDER_MISSING_ROOT = (
        "CONFIGURED_VIDEOS_UNDER_MISSING_ROOT",
        "Configured video paths end in videos/<name>, the video exists in this project, but the paths "
        "themselves do not exist. Was the project moved or copied?",
        Severity.WARNING,
        ("declared_paths", "matched_paths"),
    )
    CONFIGURED_VIDEO_IN_OTHER_COPY = (
        "CONFIGURED_VIDEO_IN_OTHER_COPY",
        "The configured video path ends in videos/<name> and exists, but is a different file from this "
        "project's video with the same name. DeepLabCut will read the configured one.",
        Severity.WARNING,
        ("matched_path",),
    )
    CONFIGURED_VIDEO_OUTSIDE_PROJECT = (
        "CONFIGURED_VIDEO_OUTSIDE_PROJECT",
        "The configured video path exists but is not in the project's videos directory.",
        Severity.ERROR,
        (),
    )
    VIDEO_NOT_IN_VIDEO_SETS = (
        "VIDEO_NOT_IN_VIDEO_SETS",
        "A video in the project's videos directory has no corresponding path in the configuration.",
        Severity.NOTE,
        ("resource_kind",),
    )


class FingerprintFindingCode(FindingCode):
    """Findings about file and frame fingerprints of requested project resources."""

    FILE_FINGERPRINT_FAILED = (
        "FILE_FINGERPRINT_FAILED",
        "A requested resource could not be file-fingerprinted.",
        Severity.ERROR,
        ("resource_kind", "error_type", "error_message"),
    )
    FILE_FINGERPRINT_CHANGED = (
        "FILE_FINGERPRINT_CHANGED",
        "A resource's file fingerprint differs from the one recorded for it, or it was not fingerprinted.",
        Severity.WARNING,
        ("recorded_sha256", "observed_sha256"),
    )


class LabeledDataFindingCode(FindingCode):
    """Findings about the content of labeled-data files."""

    LABELED_DATA_UNREADABLE = (
        "LABELED_DATA_UNREADABLE",
        "A labeled-data file could not be read.",
        Severity.ERROR,
        ("resource_kind", "error_type", "error_message"),
    )
    LABELED_DATA_H5_MULTI_KEY = (
        "LABELED_DATA_H5_MULTI_KEY",
        "An H5 labels file holds several tables; one was read by declared precedence and the others were not.",
        Severity.ERROR,
        ("keys_found", "selected_key"),
    )
    LABELED_DATA_H5_LEGACY_KEY = (
        "LABELED_DATA_H5_LEGACY_KEY",
        "An H5 labels file stores its table under a key older DeepLabCut releases wrote.",
        Severity.WARNING,
        ("key",),
    )
    LABELED_DATA_H5_UNKNOWN_KEY = (
        "LABELED_DATA_H5_UNKNOWN_KEY",
        "An H5 labels file stores its table under a key no DeepLabCut release is known to write.",
        Severity.WARNING,
        ("key",),
    )
    LABELED_DATA_HEADER_UNRECOGNISED = (
        "LABELED_DATA_HEADER_UNRECOGNISED",
        "A labels file's column levels are neither of the headers DeepLabCut writes, so no labels can be read from it.",
        Severity.ERROR,
        ("level_names",),
    )
    LABELED_DATA_COORDS_UNRECOGNISED = (
        "LABELED_DATA_COORDS_UNRECOGNISED",
        "Keypoints whose coordinates are not exactly x, y and optionally likelihood, so no labels can be read from "
        "the file.",
        Severity.ERROR,
        ("keypoints",),
    )
    LABELED_DATA_LIKELIHOOD_PARTIAL = (
        "LABELED_DATA_LIKELIHOOD_PARTIAL",
        "Some keypoints store a likelihood and these do not, so no labels can be read from the file.",
        Severity.ERROR,
        ("keypoints",),
    )
    LABELED_DATA_VALUES_NOT_NUMERIC = (
        "LABELED_DATA_VALUES_NOT_NUMERIC",
        "Value columns hold something other than real numbers, so no labels can be read from the file.",
        Severity.ERROR,
        ("columns",),
    )
    LABELED_DATA_NO_FRAMES = (
        "LABELED_DATA_NO_FRAMES",
        "A labels file has a header and no frames, so it holds no labels.",
        Severity.WARNING,
        (),
    )
    LABELED_DATA_ROW_INDEX_UNRECOGNISED = (
        "LABELED_DATA_ROW_INDEX_UNRECOGNISED",
        "A labels file's row index is neither frame path parts nor path strings, so no labels can be read from it.",
        Severity.ERROR,
        ("first_entry",),
    )
    LABELED_DATA_FRAME_DEPTH = (
        "LABELED_DATA_FRAME_DEPTH",
        "Frames whose path has other than three parts, so no labels can be read from the file.",
        Severity.ERROR,
        ("row_count", "first_frame"),
    )
    LABELED_DATA_FRAME_PART_UNNAMED = (
        "LABELED_DATA_FRAME_PART_UNNAMED",
        "Frames with a path part that is neither text nor an integer, so no labels can be read from the file.",
        Severity.ERROR,
        ("row_count", "first_frame"),
    )
    LABELED_DATA_CSV_FRAME_ABSORBED = (
        "LABELED_DATA_CSV_FRAME_ABSORBED",
        "A CSV labels file's first frame has no values, so a CSV read takes it as the row index's names and the "
        "file reads back without that frame.",
        Severity.WARNING,
        ("frame",),
    )
    LABELED_DATA_WRITER_STAMP_MALFORMED = (
        "LABELED_DATA_WRITER_STAMP_MALFORMED",
        "A labels file carries a writer stamp that could not be read.",
        Severity.NOTE,
        ("error_message",),
    )
    LABELED_DATA_HUMAN_LIKELIHOOD = (
        "LABELED_DATA_HUMAN_LIKELIHOOD",
        "Human labels store a likelihood, which ground truth never carries.",
        Severity.WARNING,
        (),
    )
    LABELED_DATA_KEYPOINT_MISSING = (
        "LABELED_DATA_KEYPOINT_MISSING",
        "Individuals lack a bodypart another individual of the same scorer has.",
        Severity.WARNING,
        ("missing",),
    )
    LABELED_DATA_INDIVIDUAL_EMPTY = (
        "LABELED_DATA_INDIVIDUAL_EMPTY",
        "Keypoints whose individual is named with an empty string, which names no individual.",
        Severity.ERROR,
        ("keypoints",),
    )


def iter_finding_codes() -> tuple[FindingCode, ...]:
    """Return every declared finding code, namespace by namespace."""

    return tuple(chain.from_iterable(FindingCode.__subclasses__()))


def finding_code(value: str | FindingCode) -> FindingCode:
    """Return the declared code for one value, as read from serialized output.

    Raises:
        ValueError: If no namespace declares the value.
    """

    if isinstance(value, FindingCode):
        return value

    for namespace in FindingCode.__subclasses__():
        try:
            return namespace(value)
        except ValueError:
            continue

    declared = ", ".join(code.value for code in iter_finding_codes())

    raise ValueError(f"Unknown finding code {value!r}. Declared codes are: {declared or '<NONE>'}.")
