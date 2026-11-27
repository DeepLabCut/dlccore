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

from dlccore.project import api, policy


def test_the_api_helpers_reexport_the_policy_helpers() -> None:
    assert (api.helpers.get_blocking_findings, api.helpers.unhandled) == (
        policy.get_blocking_findings,
        policy.unhandled,
    )


def test_the_api_helpers_hold_exactly_the_consumer_helpers() -> None:
    assert sorted(api.helpers.__all__) == ["Scope", "get_blocking_findings", "unhandled"]
    assert not {"get_blocking_findings", "unhandled"} & set(vars(api))
