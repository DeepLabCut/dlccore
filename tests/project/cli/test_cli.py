from importlib.metadata import version

import pytest

from dlc_octopus import cli
from dlc_octopus.cli.ascii_logo import DISTRIBUTION

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
