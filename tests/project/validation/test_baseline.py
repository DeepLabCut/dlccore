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

from pathlib import Path

from dlccore.project.api import inspect_project
from dlccore.project.capabilities import get_project_manager_capabilities
from dlccore.project.validation.baseline import (
    iter_baseline_check_registrations,
    run_baseline_checks,
)
from dlccore.project.validation.context import (
    build_validation_context,
)


def test_baseline_check_registrations_are_well_formed() -> None:
    registrations = iter_baseline_check_registrations()
    codes = tuple(registration.code for registration in registrations)

    assert registrations
    assert len(codes) == len(set(codes))

    for registration in registrations:
        assert registration.code
        assert registration.description
        assert callable(registration.validator)


def test_baseline_checks_return_structural_findings(
    tmp_path: Path,
) -> None:
    snapshot = inspect_project(tmp_path)
    context = build_validation_context(snapshot)

    findings = run_baseline_checks(
        snapshot,
        context,
    )

    assert "PROJECT_REQUIRED_NODE_MISSING" in {finding.code for finding in findings}


def test_config_path_check_is_not_applicable_without_context(
    tmp_path: Path,
) -> None:
    snapshot = inspect_project(tmp_path)

    registration = next(
        registration
        for registration in iter_baseline_check_registrations()
        if registration.code == "PROJECT_CONFIG_PATH_RELATIONSHIP"
    )

    assert (
        registration.validator(
            snapshot,
            None,
        )
        == ()
    )


def test_baseline_capabilities_match_registrations() -> None:
    capabilities = get_project_manager_capabilities()

    capability_codes = tuple(capability.code for capability in capabilities.baseline_validations)
    registration_codes = tuple(registration.code for registration in iter_baseline_check_registrations())

    assert capability_codes == registration_codes
