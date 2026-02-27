"""Smoke tests for the conductor CLI."""

from __future__ import annotations

from conductor.cli import main


def test_version(capsys: object) -> None:
    """CLI --version prints version string."""
    import pytest

    with pytest.raises(SystemExit, match="0"):
        main(["--version"])


def test_help_default(capsys: object) -> None:
    """CLI with no args prints help and returns 0."""
    assert main([]) == 0
