"""A report: which check it reads, which code it emits, and how. Every public check has one, or is not reported."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import JsonValue

from dlc_octopus.io.labeled_data import LabeledDataFile
from dlc_octopus.io.labeled_data.checks.file import frames_lost_to_csv
from dlc_octopus.schemas.base.finding_codes import LabeledDataFindingCode
from dlc_octopus.schemas.base.findings import Finding
from dlc_octopus.schemas.locations import Location


@dataclass(frozen=True)
class Report:
    """One code's report of one check.

    Attributes:
        check: The check whose fact it reports; ``None`` when it reports a fact the file exposes directly.
        code: The code it emits, and no other.
        emit: Returns its findings: ``emit(data, at)`` for a file report, ``emit(data, labels, at)`` for a model one.
    """

    check: Callable[..., object] | None
    code: LabeledDataFindingCode
    emit: Callable[..., tuple[Finding, ...]]


def finding(code: LabeledDataFindingCode, data: LabeledDataFile, at: Location, **evidence: JsonValue) -> tuple[Finding]:
    """One finding with ``code`` about ``data``, anchored at ``at``."""

    return (Finding(code=code, **at, path=data.path, evidence=evidence),)


# Checks a write acts on that Octopus never reports, because another check reports the same fact.
NOT_REPORTED: frozenset[Callable[..., object]] = frozenset(
    {
        # The writer's prediction; a CSV that did lose its frame reports frame_absorbed_into_index_names.
        frames_lost_to_csv,
    }
)
