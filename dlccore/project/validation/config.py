# dlc_octopus/validation/config.py
"""Validate a project's config file against the canonical DLC schema."""

from __future__ import annotations

from pathlib import Path

from deeplabcut.core.config import ProjectConfig
from pydantic import JsonValue, ValidationError

from dlc_octopus.discovery.config import (
    RawConfigRead,
    RawConfigReadStatus,
    discover_config_file,
)
from dlc_octopus.schemas.base.finding_codes import (
    ConfigFindingCode,
)
from dlc_octopus.schemas.base.findings import (
    Finding,
)
from dlc_octopus.schemas.locations import one_per_project
from dlc_octopus.schemas.project_layout.config.observations import (
    ConfigObservation,
    FailedConfigObservation,
    ValidConfigObservation,
)
from dlc_octopus.schemas.project_layout.project import PROJECT_CONFIG


def validate_project_config(
    config_path: Path,
) -> tuple[ConfigObservation, tuple[Finding, ...]]:
    """Read and validate a canonical DLC config without modifying it.

    The existing DeepLabCut config reader performs YAML I/O.
    ``ProjectConfig.from_dict`` performs validation against the canonical DLC schema.

    This function should not repair paths, modify the loaded model, or write
    any project data.
    """

    raw = discover_config_file(config_path)

    if raw.status == RawConfigReadStatus.READ_FAILED:
        return FailedConfigObservation(
            config_path=config_path,
            raw_config=None,
        ), (
            Finding(
                code=ConfigFindingCode.CONFIG_READ_FAILED,
                **one_per_project(PROJECT_CONFIG),
                path=config_path,
                evidence=_read_failure_evidence(raw),
            ),
        )

    if raw.status == RawConfigReadStatus.PARSE_FAILED:
        return FailedConfigObservation(
            config_path=config_path,
            raw_config=None,
        ), (
            Finding(
                code=ConfigFindingCode.CONFIG_PARSE_FAILED,
                **one_per_project(PROJECT_CONFIG),
                path=config_path,
                evidence=_read_failure_evidence(raw),
            ),
        )

    try:
        config = ProjectConfig.from_dict(dict(raw.raw_config))
    except ValidationError as exc:
        findings = tuple(
            Finding(
                code=ConfigFindingCode.CONFIG_VALIDATION_ERROR,
                **one_per_project(PROJECT_CONFIG),
                path=config_path,
                field_location=tuple(error["loc"]),
                evidence={
                    "error_type": str(error["type"]),
                    "error_message": str(error["msg"]),
                },
            )
            for error in exc.errors(
                include_url=False,
                include_input=False,
            )
        )

        return FailedConfigObservation(config_path=config_path, raw_config=raw.raw_config), findings
    except Exception as exc:
        # ProjectConfig.from_dict migrates before it validates, and a migration
        # re-raises with the original exception type, so the type is not knowable here.
        return FailedConfigObservation(config_path=config_path, raw_config=raw.raw_config), (
            Finding(
                code=ConfigFindingCode.CONFIG_LOAD_FAILED,
                **one_per_project(PROJECT_CONFIG),
                path=config_path,
                evidence={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            ),
        )

    return ValidConfigObservation(
        config_path=config_path,
        declared_project_path=raw.declared_project_path,
        value=config,
    ), (
        *_migration_findings(config, raw.raw_config, config_path),
        *_no_keypoints_findings(config, raw.raw_config, config_path),
        *_null_key_defaulted_findings(config, raw.raw_config, config_path),
    )


def _migration_findings(
    config: ProjectConfig,
    raw_config: dict,
    config_path: Path,
) -> tuple[Finding, ...]:
    """Report a config whose declared version DeepLabCut migrated in memory while loading it.

    A missing ``config_version`` is version 0, as DeepLabCut reads it.
    """

    declared = raw_config.get("config_version", 0)
    loaded = config.config_version

    if not isinstance(declared, int) or declared == loaded:
        return ()

    code = (
        ConfigFindingCode.CONFIG_VERSION_UPGRADED if declared < loaded else ConfigFindingCode.CONFIG_VERSION_DOWNGRADED
    )

    return (
        Finding(
            code=code,
            **one_per_project(PROJECT_CONFIG),
            path=config_path,
            field_location=("config_version",),
            evidence={"declared_version": declared, "loaded_version": loaded},
        ),
    )


def _no_keypoints_findings(
    config: ProjectConfig,
    raw_config: dict,
    config_path: Path,
) -> tuple[Finding, ...]:
    """Report a loaded config that declares no keypoints, with the keypoint fields as written.

    DeepLabCut normalises ``""``, ``[]`` and a missing key alike, so only the raw read tells them apart.
    """

    fields = ("multianimalbodyparts", "uniquebodyparts") if config.multianimalproject else ("bodyparts",)

    if any(getattr(config, field) for field in fields):
        return ()

    return (
        Finding(
            code=ConfigFindingCode.CONFIG_NO_KEYPOINTS,
            **one_per_project(PROJECT_CONFIG),
            path=config_path,
            field_location=fields[:1] if len(fields) == 1 else (),
            evidence={
                "multianimal_project": bool(config.multianimalproject),
                "declared": {field: raw_config[field] for field in fields if field in raw_config},
            },
        ),
    )


def _null_key_defaulted_findings(
    config: ProjectConfig,
    raw_config: dict,
    config_path: Path,
) -> tuple[Finding, ...]:
    """Report each key written with no value that DeepLabCut loaded as its field's default.

    A field that accepts ``None`` keeps it, so it is not reported.
    """

    loaded = config.model_dump(mode="json")

    return tuple(
        Finding(
            code=ConfigFindingCode.CONFIG_NULL_KEY_DEFAULTED,
            **one_per_project(PROJECT_CONFIG),
            path=config_path,
            field_location=(field,),
            evidence={"default": loaded[field]},
        )
        for field, value in raw_config.items()
        if value is None and loaded.get(field) is not None
    )


def _read_failure_evidence(
    raw: RawConfigRead,
) -> dict[str, JsonValue]:
    """Return the declared evidence for a failed config read or parse."""

    return {
        "error_type": raw.error_type,
        "error_message": raw.error_message,
    }
