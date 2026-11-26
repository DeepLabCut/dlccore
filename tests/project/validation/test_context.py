from __future__ import annotations

from pathlib import Path

from dlc_octopus.api import inspect_project
from dlc_octopus.validation.context import (
    build_validation_context,
)


def test_build_validation_context_for_valid_config(
    valid_project: Path,
) -> None:
    snapshot = inspect_project(valid_project)

    context = build_validation_context(snapshot)

    assert context is not None
    assert context.snapshot is snapshot
    assert context.config is snapshot.config


def test_no_validation_context_without_config(
    tmp_path: Path,
) -> None:
    snapshot = inspect_project(tmp_path)

    context = build_validation_context(snapshot)

    assert context is None


def test_no_validation_context_for_invalid_config(
    invalid_project: Path,
) -> None:
    snapshot = inspect_project(invalid_project)

    context = build_validation_context(snapshot)

    assert context is None
