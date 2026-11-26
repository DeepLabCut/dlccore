from __future__ import annotations

import ctypes
import os
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from deeplabcut.core.config.utils import read_config, write_config

from dlc_octopus.api import inspect_project
from dlc_octopus.schemas.base.finding_codes import ConfigFindingCode
from dlc_octopus.schemas.base.findings import Severity
from dlc_octopus.schemas.base.probes import PathProbe, PathProbeStatus
from dlc_octopus.validation.context import (
    build_validation_context,
)
from dlc_octopus.validation.deeplabcut import project_path as project_path_module
from dlc_octopus.validation.deeplabcut.project_path import (
    validate_project_path,
)


def test_matching_project_path_has_no_finding(
    valid_project: Path,
) -> None:
    snapshot = inspect_project(valid_project)
    context = build_validation_context(snapshot)

    assert context is not None
    assert validate_project_path(context) == ()


def test_mismatched_project_path_produces_note(
    tmp_path: Path,
    valid_config_data: dict,
) -> None:
    (tmp_path / "videos").mkdir()
    (tmp_path / "labeled-data").mkdir()

    declared_path = tmp_path / "declared-elsewhere"

    config_data = dict(valid_config_data)
    config_data["project_path"] = str(declared_path)

    write_config(
        tmp_path / "config.yaml",
        config_data,
    )

    snapshot = inspect_project(tmp_path)

    finding = next(finding for finding in snapshot.findings if finding.code == "CONFIG_PROJECT_PATH_MISMATCH")

    assert finding.severity == Severity.NOTE
    assert finding.schema_id == "dlc.project.config"
    assert finding.path == tmp_path / "config.yaml"
    assert finding.field_location == ("project_path",)
    assert finding.evidence == {
        "declared_project_path": str(declared_path.absolute()),
        "observed_project_root": str(tmp_path.absolute()),
        "declared_path_probe": "missing",
        "error_type": None,
        "error_message": None,
    }


def test_inspecting_a_moved_project_leaves_its_config_untouched(
    tmp_path: Path,
    valid_config_data: dict,
) -> None:
    """Octopus never repairs, although DeepLabCut's own reader rewrites this config on read."""

    (tmp_path / "videos").mkdir()
    (tmp_path / "labeled-data").mkdir()

    config_path = tmp_path / "config.yaml"
    config_data = dict(valid_config_data)
    config_data["project_path"] = str(tmp_path / "declared-elsewhere")

    write_config(
        config_path,
        config_data,
    )

    before = config_path.read_bytes()

    inspect_project(tmp_path)

    assert config_path.read_bytes() == before

    # Positive control: the scenario is one that DeepLabCut's reader repairs by writing.
    read_config(config_path)

    assert config_path.read_bytes() != before


_ABSENT = object()


def _project_with_declared_path(
    root: Path,
    config_data: dict,
    declared: object,
) -> Path:
    (root / "videos").mkdir(parents=True)
    (root / "labeled-data").mkdir()

    config_data = dict(config_data)

    if declared is _ABSENT:
        del config_data["project_path"]
    else:
        config_data["project_path"] = declared

    write_config(root / "config.yaml", config_data)

    return root


def _project_path_findings(project: Path) -> list:
    return [finding for finding in inspect_project(project).findings if finding.field_location == ("project_path",)]


@pytest.mark.parametrize(
    ("declared", "reported"),
    [
        (_ABSENT, None),
        (None, None),
        ("", None),
        (".", "."),
        ("mouse-project", "mouse-project"),
        ("C:mouse-project", "C:mouse-project"),
        ("\\mouse-project", "\\mouse-project"),
    ],
    ids=["absent", "null", "empty", "dot", "relative", "drive-relative", "root-relative"],
)
def test_a_project_path_naming_no_location_is_reported_as_not_absolute(
    tmp_path: Path,
    valid_config_data: dict,
    declared: object,
    reported: str | None,
) -> None:
    project = _project_with_declared_path(tmp_path, valid_config_data, declared)

    finding = next(
        finding
        for finding in inspect_project(project).findings
        if finding.code == ConfigFindingCode.CONFIG_PROJECT_PATH_NOT_ABSOLUTE
    )

    assert finding.severity == Severity.NOTE
    assert finding.field_location == ("project_path",)
    assert finding.evidence == {"declared_project_path": reported}
    assert not any(
        finding.code == ConfigFindingCode.CONFIG_PROJECT_PATH_MISMATCH for finding in inspect_project(project).findings
    )


@pytest.mark.parametrize("declared", [_ABSENT, "."], ids=["absent", "dot"])
def test_project_path_findings_do_not_depend_on_the_working_directory(
    tmp_path: Path,
    valid_config_data: dict,
    monkeypatch: pytest.MonkeyPatch,
    declared: object,
) -> None:
    project = _project_with_declared_path(tmp_path / "project", valid_config_data, declared)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    monkeypatch.chdir(project)
    from_root = _project_path_findings(project)
    monkeypatch.chdir(elsewhere)
    from_elsewhere = _project_path_findings(project)

    # Positive control: the condition is reported at all, so equality is not two empty lists.
    assert from_root
    assert [(f.code, f.evidence) for f in from_root] == [(f.code, f.evidence) for f in from_elsewhere]


@pytest.mark.parametrize(
    "declared",
    ["/home/someone/mouse-project", "D:/data/mouse-project"],
    ids=["posix", "windows-drive"],
)
def test_an_absolute_project_path_from_either_platform_is_compared(
    tmp_path: Path,
    valid_config_data: dict,
    declared: str,
) -> None:
    project = _project_with_declared_path(tmp_path, valid_config_data, declared)

    (finding,) = _project_path_findings(project)

    assert finding.code == ConfigFindingCode.CONFIG_PROJECT_PATH_MISMATCH
    assert finding.evidence["declared_project_path"] == declared


def test_a_path_absolute_only_on_another_platform_is_not_probed(
    tmp_path: Path,
    valid_config_data: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    declared = "/home/someone/mouse-project" if sys.platform == "win32" else "D:/data/mouse-project"
    project = _project_with_declared_path(tmp_path / "project", valid_config_data, declared)
    # On POSIX, a probe would resolve the declared path here and find this directory.
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / declared.lstrip("/")).mkdir(parents=True)
    monkeypatch.chdir(elsewhere)

    (finding,) = _project_path_findings(project)

    assert finding.code == ConfigFindingCode.CONFIG_PROJECT_PATH_MISMATCH
    assert (
        finding.evidence["declared_path_probe"],
        finding.evidence["error_type"],
        finding.evidence["error_message"],
    ) == (None, None, None)


def test_a_project_declared_through_a_linked_parent_matches(
    tmp_path: Path,
    valid_config_data: dict,
    create_symlink_or_skip: Callable[..., None],
) -> None:
    """``project_path`` names the project through a linked parent folder; the project is inspected by its real path."""

    real = tmp_path / "real"
    real.mkdir()
    create_symlink_or_skip(tmp_path / "alias", real, target_is_directory=True)
    project = _project_with_declared_path(real / "project", valid_config_data, str(tmp_path / "alias" / "project"))

    assert _project_path_findings(project) == []


@pytest.mark.skipif(sys.platform != "win32", reason="8.3 short names exist only on Windows")
def test_a_project_declared_by_its_windows_short_name_matches(
    tmp_path: Path,
    valid_config_data: dict,
) -> None:
    project = tmp_path / "long project name"
    buffer = ctypes.create_unicode_buffer(1024)
    project.mkdir()
    ctypes.windll.kernel32.GetShortPathNameW(str(project), buffer, len(buffer))

    if Path(buffer.value).name == project.name:
        pytest.skip("8.3 short names are disabled on this volume")

    _project_with_declared_path(project, valid_config_data, buffer.value)

    assert _project_path_findings(project) == []


def test_a_copied_project_whose_original_still_exists_is_a_mismatch(
    tmp_path: Path,
    valid_config_data: dict,
) -> None:
    original = tmp_path / "original"
    original.mkdir()
    copy = _project_with_declared_path(tmp_path / "copy", valid_config_data, str(original))

    (finding,) = _project_path_findings(copy)

    assert finding.code == ConfigFindingCode.CONFIG_PROJECT_PATH_MISMATCH
    assert finding.evidence["declared_path_probe"] == "observed"


def test_an_inaccessible_declared_path_is_reported_with_its_error(
    tmp_path: Path,
    valid_config_data: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project_with_declared_path(tmp_path, valid_config_data, "//unreachable-server/share/mouse-project")
    inaccessible = PathProbe(
        status=PathProbeStatus.INACCESSIBLE,
        error_type="PermissionError",
        error_message="access denied",
    )
    monkeypatch.setattr(project_path_module, "probe_path", lambda _path: inaccessible)

    (finding,) = _project_path_findings(project)

    assert finding.code == ConfigFindingCode.CONFIG_PROJECT_PATH_MISMATCH
    assert finding.evidence["declared_path_probe"] == "inaccessible"
    assert finding.evidence["error_type"] == "PermissionError"
    assert finding.evidence["error_message"] == "access denied"


def test_an_existing_declared_path_that_cannot_be_compared_keeps_the_error(
    tmp_path: Path,
    valid_config_data: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = tmp_path / "original"
    original.mkdir()
    copy = _project_with_declared_path(tmp_path / "copy", valid_config_data, str(original))

    def uncomparable(*_paths: object) -> bool:
        raise OSError("target vanished")

    monkeypatch.setattr(os.path, "samefile", uncomparable)

    (finding,) = _project_path_findings(copy)

    assert finding.evidence["declared_path_probe"] == "observed"
    assert finding.evidence["error_type"] == "OSError"
    assert finding.evidence["error_message"] == "target vanished"


def test_invalid_config_skips_project_path_validation(
    invalid_project: Path,
) -> None:
    snapshot = inspect_project(invalid_project)

    assert not any(finding.code == "CONFIG_PROJECT_PATH_MISMATCH" for finding in snapshot.findings)
