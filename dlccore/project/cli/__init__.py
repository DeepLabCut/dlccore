"""Command-line entry point for `dlc-octopus`."""

from __future__ import annotations

import argparse
from importlib.metadata import version

from dlc_octopus.cli.ascii_logo import DISTRIBUTION, logo_banner


def build_parser() -> argparse.ArgumentParser:
    """Return the `dlc-octopus` argument parser."""
    parser = argparse.ArgumentParser(prog="dlc-octopus")
    parser.add_argument("--version", action="store_true", help="Print the logo and version, then exit.")
    parser.add_argument("--no-art", action="store_true", help="Disable ASCII art.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the `dlc-octopus` command.

    Args:
        argv: Arguments without the program name; defaults to `sys.argv[1:]`.

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
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
