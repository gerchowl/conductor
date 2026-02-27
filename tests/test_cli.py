"""Tests for the conductor CLI and runner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.table import Table

from conductor.cli import main


def test_version(capsys: object) -> None:
    """CLI --version prints version string."""
    with pytest.raises(SystemExit, match="0"):
        main(["--version"])


def test_help_default(capsys: object) -> None:
    """CLI with no args prints help and returns 0."""
    assert main([]) == 0


def test_run_help(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI 'run --help' shows run subcommand help."""
    with pytest.raises(SystemExit, match="0"):
        main(["run", "--help"])
    captured = capsys.readouterr()
    assert "--repo" in captured.out
    assert "--poll-interval" in captured.out


def test_init_calls_init_config(tmp_path: Path) -> None:
    with patch(
        "conductor.config.init_config",
        return_value=tmp_path / ".conductor" / "conductor.toml",
    ) as mock_init, patch("conductor.cli.Path") as mock_path_cls:
        mock_path_cls.cwd.return_value = tmp_path
        result = main(["--init"])

    assert result == 0
    mock_init.assert_called_once_with(tmp_path)


def test_run_creates_runner_and_calls_run(tmp_path: Path) -> None:
    mock_runner = MagicMock()
    with patch(
        "conductor.runner.ConductorRunner", return_value=mock_runner
    ) as mock_cls, patch("conductor.cli.Path") as mock_path_cls:
        mock_path_cls.cwd.return_value = tmp_path
        result = main(["run", "--repo", "owner/repo", "--poll-interval", "5"])

    assert result == 0
    mock_cls.assert_called_once_with(project_root=tmp_path, repo="owner/repo")
    mock_runner.run.assert_called_once_with(poll_interval=5.0)


def test_main_module_runnable() -> None:
    import runpy

    with (
        patch("conductor.cli.main", return_value=0) as mock_main,
        pytest.raises(SystemExit, match="0"),
    ):
        runpy.run_module("conductor", run_name="__main__")
    mock_main.assert_called_once()


class TestConductorRunner:
    def _make_runner(self, tmp_path: Path) -> object:
        from conductor.runner import ConductorRunner

        with patch("conductor.runner.load_config") as mock_cfg, patch(
            "conductor.runner.StateDB"
        ) as mock_db_cls, patch("conductor.runner.AgentPool") as mock_pool_cls:
            mock_cfg.return_value = MagicMock(
                pool=MagicMock(
                    max_sessions=3,
                    idle_ttl_seconds=60,
                    default_model="sonnet-4.5",
                )
            )
            mock_db_cls.return_value = MagicMock()
            mock_pool_cls.return_value = MagicMock()
            runner = ConductorRunner(tmp_path, repo="owner/repo")
        return runner

    def test_init_creates_config_db_pool(self, tmp_path: Path) -> None:
        from conductor.runner import ConductorRunner

        with patch("conductor.runner.load_config") as mock_cfg, patch(
            "conductor.runner.StateDB"
        ) as mock_db_cls, patch("conductor.runner.AgentPool") as mock_pool_cls:
            mock_cfg.return_value = MagicMock(
                pool=MagicMock(
                    max_sessions=3,
                    idle_ttl_seconds=60,
                    default_model="sonnet-4.5",
                )
            )
            mock_db_cls.return_value = MagicMock()
            mock_pool_cls.return_value = MagicMock()

            runner = ConductorRunner(tmp_path, repo="owner/repo")

        mock_cfg.assert_called_once_with(tmp_path)
        mock_db_cls.assert_called_once_with(tmp_path / ".conductor" / "state.db")
        mock_pool_cls.assert_called_once_with(
            max_sessions=3,
            idle_ttl_seconds=60,
            default_model="sonnet-4.5",
        )
        assert runner.repo == "owner/repo"
        assert runner._shutdown is False

    def test_render_dashboard_returns_table(self, tmp_path: Path) -> None:
        from conductor.dag import DAG

        runner = self._make_runner(tmp_path)
        dag = DAG()
        dag.add_node(1, "First issue", phase="design")
        dag.add_node(2, "Second issue", blocked_by=[1], phase="pending")

        table = runner._render_dashboard(dag)

        assert isinstance(table, Table)
        assert table.title == "Conductor Dashboard"
        assert len(table.columns) == 4
        assert table.row_count == 2

    def test_render_dashboard_empty(self, tmp_path: Path) -> None:
        runner = self._make_runner(tmp_path)

        table = runner._render_dashboard()

        assert isinstance(table, Table)
        assert table.row_count == 0

    def test_handle_shutdown_sets_flag(self, tmp_path: Path) -> None:
        runner = self._make_runner(tmp_path)
        assert runner._shutdown is False

        runner._handle_shutdown(2, None)

        assert runner._shutdown is True

    def test_completed_issues_returns_correct_set(self, tmp_path: Path) -> None:
        runner = self._make_runner(tmp_path)
        runner.db.list_issues.return_value = [
            {"number": 1, "phase": "merged"},
            {"number": 2, "phase": "design"},
            {"number": 3, "phase": "closed"},
            {"number": 4, "phase": "pending"},
        ]

        result = runner._completed_issues()

        assert result == {1, 3}

    def test_cleanup_calls_shutdown_and_close(self, tmp_path: Path) -> None:
        runner = self._make_runner(tmp_path)

        runner._cleanup()

        runner.pool.shutdown.assert_called_once()
        runner.db.close.assert_called_once()
