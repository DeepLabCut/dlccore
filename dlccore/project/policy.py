"""A consumer's policy over findings: Octopus's severity default, adjusted by the consumer's own finding codes.

The codes are the consumer's decision; these helpers only apply them.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable

from dlc_octopus.errors import BLOCKING_SEVERITIES
from dlc_octopus.schemas.base.finding_codes import FindingCode
from dlc_octopus.schemas.base.findings import Finding


def get_blocking_findings(
    findings: Iterable[Finding],
    *,
    block: Collection[FindingCode] = (),
    allow: Collection[FindingCode] = (),
) -> tuple[Finding, ...]:
    """Return the findings that block, in order: blocking severity by default, plus ``block``, minus ``allow``.

    Args:
        findings: The findings to filter, e.g. from ``findings_in_scope``.
        block: Codes that block whatever their severity.
        allow: Codes that never block, even at blocking severity.

    Raises:
        TypeError: If a code is not a ``FindingCode`` member, so that a renamed code fails where it is named.
        ValueError: If a code is both blocked and allowed.
    """

    _members(block)
    _members(allow)

    both = set(block) & set(allow)
    if both:
        raise ValueError(f"Codes both blocked and allowed: {sorted(both)}.")

    return tuple(
        finding
        for finding in findings
        if finding.code not in allow and (finding.severity in BLOCKING_SEVERITIES or finding.code in block)
    )


def unhandled(findings: Iterable[Finding], handled: Collection[FindingCode]) -> tuple[Finding, ...]:
    """Return the findings whose code is not in ``handled``, in order.

    Args:
        findings: The findings to check.
        handled: The codes the consumer maps; a mapping from each code to how it is handled also works, and only its
            keys are read.

    Raises:
        TypeError: If a code is not a ``FindingCode`` member, so that a renamed code fails where it is named.
    """

    _members(handled)

    return tuple(finding for finding in findings if finding.code not in handled)


def _members(codes: Collection[FindingCode]) -> None:
    strays = [code for code in codes if not isinstance(code, FindingCode)]
    if strays:
        raise TypeError(f"Name finding codes by their FindingCode members, not {strays!r}.")
