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

"""A report: which check it reads, which code it emits, and how. Every public check has one, or is not reported."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import JsonValue

from dlccore.io.labeled_data import LabeledDataFile
from dlccore.io.labeled_data.checks.file import frames_lost_to_csv
from dlccore.project.schemas.base.finding_codes import LabeledDataFindingCode
from dlccore.project.schemas.base.findings import Finding
from dlccore.project.schemas.locations import Location


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


# Checks a write acts on that the project layer never reports, because another check reports the same fact.
NOT_REPORTED: frozenset[Callable[..., object]] = frozenset(
    {
        # The writer's prediction; a CSV that did lose its frame reports frame_absorbed_into_index_names.
        frames_lost_to_csv,
    }
)
