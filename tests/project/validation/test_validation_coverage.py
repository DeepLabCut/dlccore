import pytest

from dlc_octopus.validation.registry import (
    iter_validator_registrations,
)


def _tested_validator_codes(
    items: list[pytest.Item],
) -> set[str]:
    """Return validator codes declared by collected tests."""

    return {str(marker.args[0]) for item in items for marker in item.iter_markers(name="data_validator") if marker.args}


def test_every_registered_validator_has_declared_unit_coverage(
    request: pytest.FixtureRequest,
) -> None:
    registered_codes = {registration.code for _, registration in iter_validator_registrations()}
    covered_codes = _tested_validator_codes(list(request.session.items))

    missing_coverage = registered_codes - covered_codes

    assert not missing_coverage, f"Registered validators without declared unit coverage: {sorted(missing_coverage)}"
