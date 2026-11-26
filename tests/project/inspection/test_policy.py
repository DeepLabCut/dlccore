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

from __future__ import annotations

from collections.abc import Callable

import pytest

from dlccore.project import api, policy
from dlccore.project.policy import get_blocking_findings, unhandled
from dlccore.project.schemas.base.finding_codes import ConfigFindingCode, FindingCode
from dlccore.project.schemas.base.findings import Finding, Reach

ERROR = ConfigFindingCode.CONFIG_PARSE_FAILED
WARNING = ConfigFindingCode.CONFIG_NO_KEYPOINTS
NOTE = ConfigFindingCode.CONFIG_PROJECT_PATH_MISMATCH


def _finding(code: FindingCode) -> Finding:
    return Finding(
        code=code,
        schema_id="dlc.project.config",
        reach=Reach.ONE_PER_PROJECT,
        evidence=dict.fromkeys(code.evidence_keys, "placeholder"),
    )


FINDINGS = (_finding(NOTE), _finding(ERROR), _finding(WARNING))


def _codes(findings: tuple[Finding, ...]) -> list[FindingCode]:
    return [finding.code for finding in findings]


def test_by_default_only_blocking_severities_block() -> None:
    assert _codes(get_blocking_findings(FINDINGS)) == [ERROR]


def test_a_consumer_blocks_more_codes_and_keeps_snapshot_order() -> None:
    assert _codes(get_blocking_findings(FINDINGS, block={WARNING, NOTE})) == [NOTE, ERROR, WARNING]


def test_a_consumer_allows_a_code_the_default_blocks() -> None:
    assert _codes(get_blocking_findings(FINDINGS, allow={ERROR})) == []


def test_a_code_cannot_be_both_blocked_and_allowed() -> None:
    with pytest.raises(ValueError, match="both blocked and allowed"):
        get_blocking_findings(FINDINGS, block={WARNING}, allow={WARNING})


def test_unhandled_reads_the_keys_of_a_handling_table() -> None:
    handled = {NOTE: "project was moved: use the inspected folder"}

    assert _codes(unhandled(FINDINGS, handled)) == [ERROR, WARNING]


@pytest.mark.parametrize(
    "call",
    [
        lambda: get_blocking_findings(FINDINGS, block={"CONFIG_NO_KEYPOINTS"}),
        lambda: get_blocking_findings(FINDINGS, allow={"CONFIG_PARSE_FAILED"}),
        lambda: unhandled(FINDINGS, {"CONFIG_PROJECT_PATH_MISMATCH": "moved"}),
    ],
)
def test_a_code_named_by_string_raises(call: Callable[[], object]) -> None:
    """A string would still match today, and silently stop matching once the code is renamed."""

    with pytest.raises(TypeError, match="FindingCode members"):
        call()


def test_the_api_helpers_reexport_the_policy_helpers() -> None:
    assert (api.helpers.get_blocking_findings, api.helpers.unhandled) == (
        policy.get_blocking_findings,
        policy.unhandled,
    )


def test_the_api_helpers_hold_exactly_the_consumer_helpers() -> None:
    assert sorted(api.helpers.__all__) == ["Scope", "get_blocking_findings", "unhandled"]
    assert not {"get_blocking_findings", "unhandled"} & set(vars(api))
