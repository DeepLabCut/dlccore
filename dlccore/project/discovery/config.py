# dlc_octopus/discovery/config.py
"""Read a project's config file without interpreting or validating its content."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from deeplabcut.core.config.utils import read_config_as_dict


class RawConfigReadStatus(str, Enum):
    """Outcome of reading one config file from disk."""

    OK = "ok"
    READ_FAILED = "read_failed"
    PARSE_FAILED = "parse_failed"


@dataclass(frozen=True)
class RawConfigRead:
    """Internal result of reading one config file, before any validation."""

    status: RawConfigReadStatus
    raw_config: dict[str, Any] | None = None
    declared_project_path: str | None = None
    error_type: str | None = None
    error_message: str | None = None


def discover_config_file(
    config_path: Path,
) -> RawConfigRead:
    """Read a config file and record what was found, without validating it.

    The existing DeepLabCut config reader performs YAML I/O. This function
    does not migrate, validate, or interpret the loaded data beyond
    extracting the declared project path as written.
    """

    try:
        raw_config = read_config_as_dict(config_path)
    except OSError as exc:
        return RawConfigRead(
            status=RawConfigReadStatus.READ_FAILED,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
    except Exception as exc:
        return RawConfigRead(
            status=RawConfigReadStatus.PARSE_FAILED,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )

    return RawConfigRead(
        status=RawConfigReadStatus.OK,
        raw_config=dict(raw_config),
        declared_project_path=_declared_project_path(raw_config),
    )


def _declared_project_path(
    raw_config: object,
) -> str | None:
    """Return the project path as written; a ``Path`` would rewrite another platform's separators."""

    if not isinstance(raw_config, dict):
        return None

    value = raw_config.get("project_path")

    if not isinstance(value, (str, Path)) or not str(value):
        return None

    return str(value)
