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

"""Simple diagnostic facts generated from walking through the project structure.

These are the main enablers for downstream reporting, consumption by GUI/CLI/LLM agents,
and offering actions based on the findings (e.g., "PROJECT_REQUIRED_NODE_MISSING" could trigger a prompt to
locate or create the missing file).
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field, JsonValue, model_validator
from typing_extensions import Self

from dlccore.project.schemas.base.core import FrozenModel, LayoutInstance, SchemaId
from dlccore.project.schemas.base.finding_codes import FindingCode, Severity, finding_code


class Reach(str, Enum):
    """Which instances of its layout node a finding is about."""

    ONE_PER_PROJECT = "one_per_project"
    ONE_OF = "one_of"
    ALL_OF = "all_of"
    # Outside any known project: the instance holds only the segments that are certain. Never in a snapshot.
    UNROOTED = "unrooted"


class Finding(FrozenModel):
    """A structured diagnostic produced while inspecting or validating a project item.

    Findings describe observable issues, warnings, or informational conditions
    without modifying project data.
    - Stable ``code`` values support programmatic handling, and carry the
      human-readable text as ``code.description``
    - ``schema_id``: the layout node it is about; consumers scope by it, never by ``path``
    - ``reach``: whether it is about the only node of its kind, one instance, or every instance
    - ``instance``: which instance of a repeated layout node it is about, as
      project-relative path segments; ``()`` for ``ONE_PER_PROJECT``, a prefix for ``ALL_OF``, the certain
      trailing segments for ``UNROOTED``
    - ``path`` and ``field_location``: where in the project it was observed
    - ``evidence``: the observations supporting it, e.g. "actual" vs "expected"
    """

    code: FindingCode
    severity: Severity
    schema_id: SchemaId
    reach: Reach
    instance: LayoutInstance = ()
    path: Path | None = None
    field_location: tuple[str | int, ...] = ()
    evidence: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _resolve_evidence(cls, data: object) -> object:
        """Resolve the code, take its severity, and require its declared evidence.

        Raises:
            ValueError: If the evidence the code declares is not all present.
        """

        if not isinstance(data, dict) or data.get("code") is None:
            return data

        code = finding_code(data["code"])
        evidence = data.get("evidence")
        present = evidence.keys() if isinstance(evidence, dict) else ()
        missing = tuple(key for key in code.evidence_keys if key not in present)

        if missing:
            raise ValueError(f"Finding {code.value} requires evidence keys {missing}.")

        return {**data, "code": code, "severity": code.severity}

    @model_validator(mode="after")
    def _check_reach(self) -> Self:
        """Refuse a reach the instance contradicts.

        Raises:
            ValueError: If ``ONE_PER_PROJECT`` names an instance, or ``ONE_OF`` or ``UNROOTED`` names none.
        """

        if self.reach is Reach.ONE_PER_PROJECT and self.instance:
            raise ValueError(f"Reach ONE_PER_PROJECT names no instance, but got {self.instance!r}.")

        if self.reach in {Reach.ONE_OF, Reach.UNROOTED} and not self.instance:
            raise ValueError(f"Reach {self.reach.name} names one instance, but got none.")

        return self
