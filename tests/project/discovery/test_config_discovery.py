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

from pathlib import Path

import pytest

from dlccore.project.discovery.config import (
    RawConfigReadStatus,
    _declared_project_path,
    discover_config_file,
)


def test_discover_valid_config_reads_raw_data(
    valid_project: Path,
) -> None:
    config_path = valid_project / "config.yaml"

    raw = discover_config_file(config_path)

    assert raw.status == RawConfigReadStatus.OK
    assert raw.raw_config is not None
    assert raw.raw_config["scorer"] == "test-scorer"
    assert raw.declared_project_path == str(valid_project)
    assert raw.error_type is None
    assert raw.error_message is None


def test_discover_missing_config_returns_read_failure(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"

    raw = discover_config_file(config_path)

    assert raw.status == RawConfigReadStatus.READ_FAILED
    assert raw.raw_config is None
    assert raw.error_type is not None
    assert raw.error_message


def test_discover_empty_config_returns_parse_failure(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "",
        encoding="utf-8",
    )

    raw = discover_config_file(config_path)

    assert raw.status == RawConfigReadStatus.PARSE_FAILED
    assert raw.raw_config is None
    assert raw.error_type is not None
    assert raw.error_message


def test_discover_config_file_does_not_modify_file(
    valid_project: Path,
) -> None:
    config_path = valid_project / "config.yaml"

    before = (
        config_path.read_bytes(),
        config_path.stat().st_mtime_ns,
    )

    discover_config_file(config_path)

    after = (
        config_path.read_bytes(),
        config_path.stat().st_mtime_ns,
    )

    assert after == before


@pytest.mark.parametrize(
    ("raw_config", "expected"),
    [
        (
            {"project_path": "/tmp/project"},
            "/tmp/project",
        ),
        (
            {"project_path": "C:\\data\\project"},
            "C:\\data\\project",
        ),
        (
            {"project_path": "relative/project"},
            "relative/project",
        ),
        (
            {"project_path": ""},
            None,
        ),
        (
            {"project_path": None},
            None,
        ),
        (
            {},
            None,
        ),
        (
            [],
            None,
        ),
    ],
)
def test_declared_project_path(
    raw_config: object,
    expected: str | None,
) -> None:
    assert _declared_project_path(raw_config) == expected
