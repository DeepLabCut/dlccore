from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from dlc_octopus.schemas.base.finding_codes import (
    ConfigFindingCode,
    ConfiguredVideoFindingCode,
    FindingCode,
    FingerprintFindingCode,
    LabeledDataFindingCode,
    Severity,
    StructureFindingCode,
    iter_finding_codes,
)
from dlc_octopus.schemas.base.findings import Finding, Reach

_UNREADABLE = StructureFindingCode.PROJECT_NODE_UNREADABLE
_SCHEMA_ID = "dlc.videos.source_video"


def _declared_evidence(code: FindingCode) -> dict[str, str]:
    """Return placeholder evidence satisfying one code's declared keys."""

    return dict.fromkeys(code.evidence_keys, "placeholder")


def test_finding_keeps_a_declared_code_as_its_enum_member() -> None:
    code = StructureFindingCode.PROJECT_NODE_WRONG_TYPE

    finding = Finding(
        code=code,
        schema_id=_SCHEMA_ID,
        reach=Reach.ONE_PER_PROJECT,
        evidence=_declared_evidence(code),
    )

    assert finding.code is code
    assert finding.code == "PROJECT_NODE_WRONG_TYPE"


def test_finding_code_renders_as_the_bare_code() -> None:
    code = StructureFindingCode.PROJECT_NODE_WRONG_TYPE

    assert str(code) == "PROJECT_NODE_WRONG_TYPE"
    assert f"{code}" == "PROJECT_NODE_WRONG_TYPE"
    assert ", ".join([code]) == "PROJECT_NODE_WRONG_TYPE"


def test_finding_rejects_an_undeclared_code() -> None:
    with pytest.raises(ValidationError):
        Finding(
            code="NOT_A_DECLARED_CODE",
            schema_id=_SCHEMA_ID,
            reach=Reach.ONE_PER_PROJECT,
        )


def test_finding_severity_comes_from_its_code() -> None:
    finding = Finding(
        code=_UNREADABLE,
        schema_id=_SCHEMA_ID,
        reach=Reach.ONE_PER_PROJECT,
        evidence=_declared_evidence(_UNREADABLE),
    )

    assert finding.severity is Severity.ERROR
    assert finding.model_dump()["severity"] == Severity.ERROR


def test_finding_ignores_a_caller_supplied_severity() -> None:
    finding = Finding(
        code=_UNREADABLE,
        schema_id=_SCHEMA_ID,
        reach=Reach.ONE_PER_PROJECT,
        severity=Severity.NOTE,
        evidence=_declared_evidence(_UNREADABLE),
    )

    assert finding.severity is Severity.ERROR


def test_finding_requires_the_evidence_its_code_declares() -> None:
    with pytest.raises(ValidationError, match="requires evidence keys"):
        Finding(
            code=_UNREADABLE,
            schema_id=_SCHEMA_ID,
            reach=Reach.ONE_PER_PROJECT,
        )


def test_finding_reports_only_the_missing_evidence_keys() -> None:
    partial = _declared_evidence(_UNREADABLE)
    del partial["error_message"]

    with pytest.raises(ValidationError, match=r"\('error_message',\)"):
        Finding(
            code=_UNREADABLE,
            schema_id=_SCHEMA_ID,
            reach=Reach.ONE_PER_PROJECT,
            evidence=partial,
        )


def test_finding_accepts_declared_evidence_that_was_not_observed() -> None:
    evidence: dict[str, str | None] = dict.fromkeys(_UNREADABLE.evidence_keys, None)

    finding = Finding(
        code=_UNREADABLE,
        schema_id=_SCHEMA_ID,
        reach=Reach.ONE_PER_PROJECT,
        evidence=evidence,
    )

    assert finding.evidence["error_message"] is None


def test_finding_round_trips_through_serialized_form() -> None:
    finding = Finding(
        code=_UNREADABLE,
        schema_id=_SCHEMA_ID,
        reach=Reach.ONE_PER_PROJECT,
        evidence=_declared_evidence(_UNREADABLE),
    )

    restored = Finding.model_validate(json.loads(finding.model_dump_json()))

    assert restored == finding
    assert restored.code is _UNREADABLE


def test_finding_rejects_an_undeclared_code_from_serialized_form() -> None:
    with pytest.raises(ValidationError, match="Unknown finding code"):
        Finding.model_validate({"code": "NOT_A_DECLARED_CODE"})


@pytest.mark.parametrize("code", iter_finding_codes(), ids=str)
def test_every_finding_code_is_described(code: FindingCode) -> None:
    assert code.value == code.name
    assert code.description
    assert isinstance(code.severity, Severity)


@pytest.mark.parametrize("code", iter_finding_codes(), ids=str)
def test_every_finding_code_declares_well_formed_evidence_keys(code: FindingCode) -> None:
    """A code may declare none: ``path`` and ``field_location`` carry the location, and evidence never repeats it."""

    assert len(set(code.evidence_keys)) == len(code.evidence_keys)
    assert all(key and key.islower() for key in code.evidence_keys)


def test_finding_code_values_are_unique_across_namespaces() -> None:
    codes = iter_finding_codes()

    assert len({code.value for code in codes}) == len(codes)


def test_iter_finding_codes_covers_every_namespace() -> None:
    namespaces = {type(code) for code in iter_finding_codes()}

    assert namespaces == set(FindingCode.__subclasses__())


@pytest.mark.xfail(
    reason="the train-side split reconciliation is not built; when it is, delete this and add "
    "a test under tests/upstream/ recording where ours intentionally differs from DeepLabCut's "
    "`_validate_dataframes`, which is disabled and known incomplete",
    strict=True,
    raises=AssertionError,
)
def test_training_split_reconciliation_declares_a_finding_code() -> None:
    assert "TRAINING_SPLIT_TABLE_MISMATCH" in {code.value for code in iter_finding_codes()}


def test_the_finding_code_catalogue_is_closed() -> None:
    """The catalogue is closed: adding a namespace is deliberate, so it fails here first."""

    assert set(FindingCode.__subclasses__()) == {
        StructureFindingCode,
        ConfigFindingCode,
        ConfiguredVideoFindingCode,
        FingerprintFindingCode,
        LabeledDataFindingCode,
    }


def test_severity_has_three_rungs() -> None:
    assert {member.name for member in Severity} == {"NOTE", "WARNING", "ERROR"}


def test_schema_id_is_a_required_finding_field() -> None:
    field = Finding.model_fields.get("schema_id")

    assert field is not None
    assert field.is_required()


def test_instance_is_a_finding_field() -> None:
    assert "instance" in Finding.model_fields


def test_a_finding_states_its_reach() -> None:
    evidence = _declared_evidence(_UNREADABLE)

    def finding(reach: Reach, instance: tuple[str, ...]) -> Finding:
        return Finding(code=_UNREADABLE, schema_id=_SCHEMA_ID, reach=reach, instance=instance, evidence=evidence)

    assert {member.name for member in Reach} == {"ONE_PER_PROJECT", "ONE_OF", "ALL_OF", "UNROOTED"}

    with pytest.raises(ValidationError):
        finding(Reach.ONE_PER_PROJECT, ("videos", "m1.mp4"))

    with pytest.raises(ValidationError):
        finding(Reach.ONE_OF, ())

    with pytest.raises(ValidationError):
        finding(Reach.UNROOTED, ())

    assert finding(Reach.ONE_OF, ("videos", "m1.mp4")).reach is Reach.ONE_OF
    assert finding(Reach.ALL_OF, ()).instance == ()
    assert finding(Reach.ALL_OF, ("labeled-data", "m2")).instance == ("labeled-data", "m2")


@pytest.mark.xfail(
    reason="a released code must never disappear silently: renamed, merged or split codes are to "
    "declare what they supersede; once they do, replace this with a tripwire over the released "
    "code list that also requires a version bump whenever the catalogue changes",
    strict=True,
    raises=AssertionError,
)
def test_a_released_code_cannot_disappear_without_a_successor() -> None:
    assert all(hasattr(code, "supersedes") for code in iter_finding_codes())
