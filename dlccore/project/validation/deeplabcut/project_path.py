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

import os
from pathlib import Path

from dlccore.project.discovery.probe import probe_path
from dlccore.project.schemas.base.core import is_absolute_path
from dlccore.project.schemas.base.finding_codes import ConfigFindingCode
from dlccore.project.schemas.base.findings import Finding
from dlccore.project.schemas.base.probes import PathProbeStatus
from dlccore.project.schemas.locations import one_per_project
from dlccore.project.schemas.project_layout.project import PROJECT_CONFIG
from dlccore.project.validation.context import ValidationContext


def validate_project_path(
    context: ValidationContext,
) -> tuple[Finding, ...]:
    """Compare the config's declared project_path with the inspected project root.

    A missing, empty or relative project_path is reported as ``CONFIG_PROJECT_PATH_NOT_ABSOLUTE``.
    A path counts as absolute if it is absolute on Windows or on POSIX; a relative one is never
    resolved against the working directory.

    An absolute path matches when it names the same directory as the root, following symlinks.
    Otherwise ``CONFIG_PROJECT_PATH_MISMATCH`` is reported with the declared path as written, the
    result of probing it (missing, observed or inaccessible) and any error. A path absolute only on
    the other platform is not probed, and those three are ``None``.

    This check is read-only and never writes the configuration.
    """

    observation = context.snapshot.config_observation
    declared_path = observation.declared_project_path

    if declared_path is None or not is_absolute_path(declared_path):
        return (
            Finding(
                code=ConfigFindingCode.CONFIG_PROJECT_PATH_NOT_ABSOLUTE,
                **one_per_project(PROJECT_CONFIG),
                path=observation.config_path,
                field_location=("project_path",),
                evidence={
                    "declared_project_path": declared_path,
                },
            ),
        )

    observed_path = context.snapshot.project_root.absolute()

    if Path(declared_path) == observed_path:
        return ()

    probe_status = error_type = error_message = None

    # Absolute only on the other platform: here it would resolve against the working directory.
    if Path(declared_path).is_absolute():
        probe = probe_path(Path(declared_path))
        probe_status, error_type, error_message = probe.status.value, probe.error_type, probe.error_message

        if probe.status == PathProbeStatus.OBSERVED:
            try:
                if os.path.samefile(declared_path, observed_path):
                    return ()
            except (OSError, ValueError) as error:
                # The entry exists but cannot be compared, such as a link whose target is gone.
                error_type, error_message = type(error).__name__, str(error)

    return (
        Finding(
            code=ConfigFindingCode.CONFIG_PROJECT_PATH_MISMATCH,
            **one_per_project(PROJECT_CONFIG),
            path=observation.config_path,
            field_location=("project_path",),
            evidence={
                "declared_project_path": declared_path,
                "observed_project_root": str(observed_path),
                "declared_path_probe": probe_status,
                "error_type": error_type,
                "error_message": error_message,
            },
        ),
    )
