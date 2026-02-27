"""CLI entry point for the conductor."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the conductor CLI."""
    parser = argparse.ArgumentParser(
        prog="conductor",
        description="Agent orchestrator — warm tmux pool, Pydantic contracts",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Generate .conductor/conductor.toml from source defaults",
    )

    _args = parser.parse_args(argv)

    if _args.init:
        print("conductor --init: not yet implemented")
        return 0

    parser.print_help()
    return 0


def _get_version() -> str:
    from conductor import __version__

    return __version__


if __name__ == "__main__":
    sys.exit(main())
