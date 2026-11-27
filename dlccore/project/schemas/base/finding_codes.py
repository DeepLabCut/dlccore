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
