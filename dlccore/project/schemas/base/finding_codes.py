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
