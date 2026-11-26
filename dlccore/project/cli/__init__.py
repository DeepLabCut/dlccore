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

"""Command-line entry point for `dlc-octopus`."""

from __future__ import annotations

import argparse
import sys
from importlib.metadata import version
from pathlib import Path

from dlccore.project.cli.ascii_logo import DISTRIBUTION, logo_banner


def build_parser() -> argparse.ArgumentParser:
    """Return the `dlc-octopus` argument parser."""
    parser = argparse.ArgumentParser(prog="dlc-octopus")
    parser.add_argument("--version", action="store_true", help="Print the logo and version, then exit.")
    parser.add_argument("--no-art", action="store_true", help="Disable ASCII art.")
    commands = parser.add_subparsers(dest="command")
    inspect = commands.add_parser("inspect", help="Inspect a project and print its tree and findings.")
    inspect.add_argument("project", type=Path, help="A project root, or its config.yaml.")
    inspect.add_argument(
        "--markdown", action="store_true", help="Print the findings as a Markdown table, project root redacted."
    )
    inspect.add_argument(
        "--fail-on-error", action="store_true", help="Exit with status 1 if a finding has blocking severity."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the `dlc-octopus` command.

    Args:
        argv: Arguments without the program name; defaults to `sys.argv[1:]`.

    Returns:
        Process exit code: 1 for blocking findings under ``--fail-on-error``, 2 for a path that is not a project.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "inspect":
        return _inspect(args.project, markdown=args.markdown, fail_on_error=args.fail_on_error)
    if not args.version:
        parser.print_help()
        return 0

    line = f"{DISTRIBUTION} {version(DISTRIBUTION)}"
    banner = None
    if not args.no_art:
        try:
            banner = logo_banner(version_line=line)
        except (ImportError, OSError, RuntimeError, ValueError):
            banner = None
    print(banner or line)
    return 0


def _inspect(project: Path, *, markdown: bool, fail_on_error: bool) -> int:
    """Print a project's report, or its findings as Markdown, and return the exit code."""
    from dlccore.project.api import inspect_project
    from dlccore.project.api.helpers import get_blocking_findings
    from dlccore.project.visualization import render_findings_markdown, render_project_report

    try:
        snapshot = inspect_project(project)
    except (FileNotFoundError, ValueError) as error:
        print(f"dlc-octopus inspect: {error}", file=sys.stderr)
        return 2

    if markdown:
        print(render_findings_markdown(snapshot.findings, project_root=snapshot.project_root))
    else:
        print(render_project_report(snapshot))

    return 1 if fail_on_error and get_blocking_findings(snapshot.findings) else 0
