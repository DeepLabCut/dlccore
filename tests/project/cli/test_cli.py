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

from importlib.metadata import version
from pathlib import Path

import pytest

from dlccore.project import cli
from dlccore.project.cli.ascii_logo import DISTRIBUTION

LINE = f"{DISTRIBUTION} {version(DISTRIBUTION)}"


def _version_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], banner: object, *argv: str
) -> str:
    def fake_banner(**kwargs: object) -> str | None:
        if isinstance(banner, Exception):
            raise banner
        return None if banner is None else f"ART\n{kwargs['version_line']}"

    monkeypatch.setattr(cli, "logo_banner", fake_banner)
    assert cli.main(["--version", *argv]) == 0
    return capsys.readouterr().out


def test_the_version_is_printed_once_inside_the_banner(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _version_output(monkeypatch, capsys, "banner") == f"ART\n{LINE}\n"


@pytest.mark.parametrize(
    ("banner", "argv"),
    [(None, ()), ("banner", ("--no-art",)), (ImportError("no OpenCV"), ())],
    ids=["no-tty-or-too-narrow", "no-art", "logo-unavailable"],
)
def test_without_a_banner_the_version_is_printed_plain(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], banner: object, argv: tuple[str, ...]
) -> None:
    assert _version_output(monkeypatch, capsys, banner, *argv) == f"{LINE}\n"


def _inspect(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = cli.main(["inspect", *argv])
    out, err = capsys.readouterr()
    return code, out, err


def test_inspect_prints_the_tree_of_a_valid_project(valid_project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = _inspect(capsys, str(valid_project))

    assert code == 0
    assert "No findings were reported" in out
    assert "labeled-data [labeled_data_directory, present]" in out


def test_inspect_reports_findings_and_exits_zero_unless_asked_to_fail(
    invalid_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = _inspect(capsys, str(invalid_project))

    assert code == 0
    assert "ERROR CONFIG_VALIDATION_ERROR" in out


def test_inspect_fails_on_a_blocking_finding_when_asked(
    invalid_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _inspect(capsys, str(invalid_project), "--fail-on-error")[0] == 1


def test_inspect_passes_a_valid_project_when_asked_to_fail_on_errors(
    valid_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Control: --fail-on-error does not fail every project.
    assert _inspect(capsys, str(valid_project), "--fail-on-error")[0] == 0


def test_inspect_prints_findings_as_markdown_with_the_root_redacted(
    invalid_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = _inspect(capsys, str(invalid_project), "--markdown")

    assert code == 0
    assert "CONFIG_VALIDATION_ERROR" in out
    assert "|" in out
    assert str(invalid_project) not in out


def test_inspect_refuses_a_path_that_is_not_a_project(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = _inspect(capsys, str(tmp_path / "missing"))

    assert code == 2
    assert out == ""
    assert "dlc-octopus inspect:" in err
