"""Inspect a DeepLabCut project into a snapshot, and read DeepLabCut's name for it."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from dlc_octopus.discovery.observer import discover_project
from dlc_octopus.errors import ProjectValidationError, _blocking_findings
from dlc_octopus.schemas.base.findings import Finding
from dlc_octopus.schemas.snapshots import ProjectSnapshot
from dlc_octopus.validation import ProjectValidationRequest, validate_project
from dlc_octopus.validation.config import validate_project_config

# DeepLabCut's create_new_project: config ``date`` is the 3-letter month and unpadded day, e.g. "Oct30".
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def inspect_project(
    path: str | Path,
    *,
    validation: ProjectValidationRequest | None = None,
    raise_on_validation_error: bool = False,
) -> ProjectSnapshot:
    """Inspect a DeepLabCut project without modifying its contents.

    Args:
        path: Path to a project root directory or its ``config.yaml`` file.
        validation: Resource validation to request, or ``None`` for baseline only.
        raise_on_validation_error: Whether to raise when a finding is of blocking severity.

    Raises:
        FileNotFoundError: If the supplied project root does not exist.
        ValueError: If the supplied existing file is not named ``config.yaml``.
        ProjectValidationError: If requested and a finding has ``ERROR`` severity.
    """

    project_root = _resolve_project_root(path)
    config_path = project_root / "config.yaml"

    config_observation = None
    config_findings: tuple[Finding, ...] = ()

    if config_path.is_file():
        config_observation, config_findings = validate_project_config(config_path)

    project = discover_project(project_root)

    snapshot = ProjectSnapshot.from_project(
        project_root=project_root,
        config_observation=config_observation,
        project=project,
        findings=config_findings,
    )
    validation_result = validate_project(
        snapshot,
        request=validation,
    )

    snapshot = snapshot.with_validation_result(validation_result)

    if raise_on_validation_error and _blocking_findings(snapshot):
        raise ProjectValidationError(snapshot)

    return snapshot


def project_name(snapshot: ProjectSnapshot) -> str | None:
    """Return DeepLabCut's name for the project, ``Task-scorer-YYYY-MM-DD``, as ``create_new_project`` built it.

    Read from the project folder's name when it is exactly that and agrees with the config's ``date``; ``None``
    otherwise, since ``date`` alone (``Oct30``) has no year. The config's ``project_path`` is not read: DeepLabCut
    rewrites it to the current location whenever it loads the config.

    Raises:
        ConfigUnavailableError: If the config was not loaded.
    """

    config = snapshot.config
    prefix = f"{config.Task}-{config.scorer}-"
    folder = snapshot.project_root.name

    if folder.startswith(prefix) and _agrees_with(folder.removeprefix(prefix), str(config.date)):
        return folder

    return None


def _agrees_with(stamp: str, config_date: str) -> bool:
    if not _ISO_DATE.fullmatch(stamp):
        return False
    try:
        created = date.fromisoformat(stamp)
    except ValueError:
        return False

    return f"{_MONTHS[created.month - 1]}{created.day}" == config_date


def _resolve_project_root(
    path: str | Path,
) -> Path:
    """Resolve an absolute project root from a directory or ``config.yaml`` path, without following links."""

    supplied_path = Path(path).absolute()

    if supplied_path.is_dir():
        return supplied_path

    if supplied_path.is_file():
        if supplied_path.name != "config.yaml":
            raise ValueError(f"A project file input must be named 'config.yaml': {supplied_path}")

        return supplied_path.parent

    if supplied_path.name == "config.yaml":
        return supplied_path.parent

    raise FileNotFoundError(f"Project path does not exist: {supplied_path}")
