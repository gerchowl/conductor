"""CLI entry point for the conductor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


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

    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Start the orchestration loop")
    run_parser.add_argument(
        "--repo",
        default=None,
        help="GitHub repo in owner/repo format",
    )
    run_parser.add_argument(
        "--poll-interval",
        type=float,
        default=10.0,
        help="Seconds between poll cycles (default: 10)",
    )

    _args = parser.parse_args(argv)

    if _args.init:
        from conductor.config import init_config

        path = init_config(Path.cwd())
        print(f"Wrote {path}")
        return 0

    if _args.command == "run":
        from conductor.runner import ConductorRunner

        runner = ConductorRunner(
            project_root=Path.cwd(),
            repo=_args.repo,
        )
        runner.run(poll_interval=_args.poll_interval)
        return 0

    parser.print_help()
    return 0


def _get_version() -> str:
    from conductor import __version__

    return __version__


if __name__ == "__main__":
    sys.exit(main())
