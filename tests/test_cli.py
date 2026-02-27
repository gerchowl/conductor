"""Tests for the conductor CLI and runner."""

from __future__ import annotations

import json
import subprocess
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
            mock_pool = MagicMock()
            mock_pool.pane_activity_age.return_value = None
            mock_pool_cls.return_value = mock_pool
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
        runner.db.list_issues.return_value = []
        dag = DAG()
        dag.add_node(1, "First issue", phase="design")
        dag.add_node(2, "Second issue", blocked_by=[1], phase="pending")

        table = runner._render_dashboard(dag)

        assert isinstance(table, Table)
        assert table.title == "Conductor Dashboard (milestone: all)"
        assert len(table.columns) == 7
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

    def test_refresh_dag_builds_from_open_issues(self, tmp_path: Path) -> None:
        runner = self._make_runner(tmp_path)
        fake_issues = [
            {
                "number": 1,
                "title": "First",
                "body": "",
                "labels": ["phase:design"],
                "milestone": None,
            },
            {
                "number": 2,
                "title": "Second",
                "body": "Blocked by: #1",
                "labels": [],
                "milestone": None,
            },
        ]
        with patch(
            "conductor.runner._list_open_issues", return_value=fake_issues
        ):
            dag = runner._refresh_dag()

        assert len(dag.nodes) == 2
        node2 = dag.get_node(2)
        assert node2 is not None
        assert node2.blocked_by == [1]
        node1 = dag.get_node(1)
        assert node1 is not None
        assert node1.phase == "pending"

    def test_refresh_dag_returns_empty_dag_on_failure(
        self, tmp_path: Path
    ) -> None:
        runner = self._make_runner(tmp_path)
        with patch(
            "conductor.runner._list_open_issues", return_value=[]
        ):
            dag = runner._refresh_dag()

        assert len(dag.nodes) == 0

    def test_sync_dag_to_db_inserts_new_issues(self, tmp_path: Path) -> None:
        from conductor.dag import DAG

        runner = self._make_runner(tmp_path)
        runner.db.get_issue.return_value = None

        dag = DAG()
        dag.add_node(1, "Issue A", phase="design")
        dag.add_node(2, "Issue B", blocked_by=[1], phase="pending")

        issues = [
            {"number": 1, "title": "Issue A", "body": "body A", "labels": ["bug"]},
            {"number": 2, "title": "Issue B", "body": "body B", "labels": []},
        ]
        runner._sync_dag_to_db(dag, issues)

        assert runner.db.upsert_issue.call_count == 2
        runner.db.upsert_issue.assert_any_call(
            number=1, title="Issue A", phase="pending",
            milestone=None, blocked_by="[]",
            body="body A", labels='["bug"]',
        )
        runner.db.upsert_issue.assert_any_call(
            number=2, title="Issue B", phase="pending",
            milestone=None, blocked_by="[1]",
            body="body B", labels="[]",
        )

    def test_sync_dag_to_db_preserves_existing_phase(
        self, tmp_path: Path
    ) -> None:
        from conductor.dag import DAG

        runner = self._make_runner(tmp_path)
        runner.db.get_issue.return_value = {
            "number": 1,
            "phase": "execute",
        }

        dag = DAG()
        dag.add_node(1, "Issue A")

        issues = [
            {"number": 1, "title": "Issue A", "body": "desc", "labels": []},
        ]
        runner._sync_dag_to_db(dag, issues)

        runner.db.upsert_issue.assert_not_called()
        runner.db.update_issue.assert_called_once_with(
            1, title="Issue A", milestone=None, blocked_by="[]",
            body="desc", labels="[]",
        )
        assert dag.get_node(1).phase == "execute"

    def test_tick_dispatches_ready_issues(self, tmp_path: Path) -> None:
        from conductor.dag import DAG

        runner = self._make_runner(tmp_path)
        runner.db.list_issues.return_value = []

        dag = DAG()
        dag.add_node(1, "Ready issue", phase="pending")
        dag.add_node(2, "Blocked issue", blocked_by=[1], phase="pending")

        with (
            patch.object(runner, "_dispatch_issue") as mock_dispatch,
            patch.object(runner, "_submit_dispatch") as mock_submit,
        ):
            runner._tick(dag)

        mock_submit.assert_called_once()
        call_args = mock_submit.call_args[0]
        assert call_args[0].number == 1
        assert call_args[1] == "design"

    def test_tick_skips_merged_and_pr_issues(self, tmp_path: Path) -> None:
        from conductor.dag import DAG

        runner = self._make_runner(tmp_path)
        runner.db.list_issues.return_value = []

        dag = DAG()
        dag.add_node(1, "Merged", phase="merged")
        dag.add_node(2, "In PR", phase="pr")
        dag.add_node(3, "Closed", phase="closed")

        with patch.object(runner, "_submit_dispatch") as mock_submit:
            runner._tick(dag)

        mock_submit.assert_not_called()

    def test_tick_uses_design_for_unknown_phase(self, tmp_path: Path) -> None:
        from conductor.dag import DAG

        runner = self._make_runner(tmp_path)
        runner.db.list_issues.return_value = []

        dag = DAG()
        dag.add_node(1, "Unknown phase", phase="pending")

        with patch.object(runner, "_submit_dispatch") as mock_submit:
            runner._tick(dag)

        mock_submit.assert_called_once()
        assert mock_submit.call_args[0][1] == "design"

    def test_tick_uses_existing_phase_when_known(
        self, tmp_path: Path
    ) -> None:
        from conductor.dag import DAG

        runner = self._make_runner(tmp_path)
        runner.db.list_issues.return_value = []

        dag = DAG()
        dag.add_node(1, "In plan phase", phase="plan")

        with patch.object(runner, "_submit_dispatch") as mock_submit:
            runner._tick(dag)

        mock_submit.assert_called_once()
        assert mock_submit.call_args[0][1] == "plan"

    def test_dispatch_issue_calls_run_phase(self, tmp_path: Path) -> None:
        from conductor.dag import DAGNode
        from conductor.phases import PhaseResult

        runner = self._make_runner(tmp_path)
        node = DAGNode(number=5, title="Test issue", phase="design")

        with patch(
            "conductor.runner.run_phase",
            return_value=PhaseResult(
                phase="design", success=True
            ),
        ) as mock_run:
            runner._dispatch_issue(node, "design")

        mock_run.assert_called_once()
        ctx = mock_run.call_args[0][0]
        assert ctx.issue_number == 5
        assert ctx.repo == "owner/repo"
        assert ctx.project_root == tmp_path

    def test_dispatch_issue_logs_failure(self, tmp_path: Path) -> None:
        from conductor.dag import DAGNode
        from conductor.phases import PhaseResult

        runner = self._make_runner(tmp_path)
        node = DAGNode(number=7, title="Failing", phase="verify")

        with patch(
            "conductor.runner.run_phase",
            return_value=PhaseResult(
                phase="verify", success=False, error="timeout"
            ),
        ), patch("conductor.runner.logger") as mock_logger:
            runner._dispatch_issue(node, "verify")

        mock_logger.warning.assert_called_once()
        assert "timeout" in mock_logger.warning.call_args[0][3]

    def test_render_dashboard_shows_blocked_style(
        self, tmp_path: Path
    ) -> None:
        from conductor.dag import DAG

        runner = self._make_runner(tmp_path)
        runner.db.list_issues.return_value = []

        dag = DAG()
        dag.add_node(1, "Dep", phase="design")
        dag.add_node(2, "Blocked", blocked_by=[1], phase="pending")

        table = runner._render_dashboard(dag)

        assert table.row_count == 2
        assert len(table.columns) == 7


class TestListOpenIssues:
    def test_success_with_repo_includes_milestone(self) -> None:
        from conductor.runner import _list_open_issues

        gh_output = json.dumps([
            {
                "number": 10,
                "title": "Issue ten",
                "body": "Blocked by: #5",
                "labels": [{"name": "bug"}, {"name": "phase:design"}],
                "state": "OPEN",
                "milestone": {"title": "0.1.0"},
            },
        ])
        with patch("conductor.runner.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=gh_output)
            result = _list_open_issues("owner/repo")

        assert len(result) == 1
        assert result[0]["number"] == 10
        assert result[0]["labels"] == ["bug", "phase:design"]
        assert result[0]["milestone"] == "0.1.0"
        cmd = mock_run.call_args[0][0]
        assert "--repo" in cmd
        assert "milestone" in " ".join(cmd)

    def test_success_without_repo(self) -> None:
        from conductor.runner import _list_open_issues

        gh_output = json.dumps([
            {
                "number": 1,
                "title": "Local",
                "body": "",
                "labels": [],
                "state": "OPEN",
                "milestone": None,
            },
        ])
        with patch("conductor.runner.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=gh_output)
            result = _list_open_issues(None)

        cmd = mock_run.call_args[0][0]
        assert "--repo" not in cmd
        assert result[0]["milestone"] is None

    def test_returns_empty_on_called_process_error(self) -> None:
        from conductor.runner import _list_open_issues

        with patch(
            "conductor.runner.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "gh"),
        ):
            result = _list_open_issues("owner/repo")

        assert result == []

    def test_returns_empty_on_file_not_found(self) -> None:
        from conductor.runner import _list_open_issues

        with patch(
            "conductor.runner.subprocess.run",
            side_effect=FileNotFoundError("gh not found"),
        ):
            result = _list_open_issues("owner/repo")

        assert result == []

    def test_normalizes_string_labels(self) -> None:
        from conductor.runner import _list_open_issues

        gh_output = json.dumps([
            {
                "number": 1,
                "title": "String labels",
                "body": "",
                "labels": ["bug", "phase:plan"],
                "state": "OPEN",
                "milestone": None,
            },
        ])
        with patch("conductor.runner.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=gh_output)
            result = _list_open_issues(None)

        assert result[0]["labels"] == ["bug", "phase:plan"]

    def test_handles_missing_body_and_milestone(self) -> None:
        from conductor.runner import _list_open_issues

        gh_output = json.dumps([
            {
                "number": 1,
                "title": "No body",
                "labels": [],
                "state": "OPEN",
            },
        ])
        with patch("conductor.runner.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=gh_output)
            result = _list_open_issues(None)

        assert result[0]["body"] == ""
        assert result[0]["milestone"] is None

    def test_milestone_dict_extraction(self) -> None:
        from conductor.runner import _list_open_issues

        gh_output = json.dumps([
            {
                "number": 1,
                "title": "With milestone",
                "body": "",
                "labels": [],
                "state": "OPEN",
                "milestone": {"title": "1.2.3", "number": 5},
            },
        ])
        with patch("conductor.runner.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=gh_output)
            result = _list_open_issues(None)

        assert result[0]["milestone"] == "1.2.3"


class TestIsEpic:
    def test_epic_prefix(self) -> None:
        from conductor.runner import _is_epic

        assert _is_epic("[EPIC] Phase 1 MVP") is True
        assert _is_epic("[epic] lowercase") is True
        assert _is_epic("  [EPIC] with spaces") is True

    def test_not_epic(self) -> None:
        from conductor.runner import _is_epic

        assert _is_epic("[FEATURE] Bootstrap Python") is False
        assert _is_epic("Regular issue") is False
        assert _is_epic("") is False


class TestSemverKey:
    def test_valid_semver(self) -> None:
        from conductor.runner import _semver_key

        assert _semver_key("0.1.0") == (0, 1, 0)
        assert _semver_key("1.2.3") == (1, 2, 3)
        assert _semver_key("10.20.30") == (10, 20, 30)

    def test_invalid_falls_back(self) -> None:
        from conductor.runner import _semver_key

        assert _semver_key("Phase 1: MVP") == (999, 999, 999)
        assert _semver_key("") == (999, 999, 999)

    def test_sorting(self) -> None:
        from conductor.runner import _semver_key

        titles = ["1.0.0", "0.2.0", "0.1.0", "Phase X"]
        sorted_titles = sorted(titles, key=_semver_key)
        assert sorted_titles == ["0.1.0", "0.2.0", "1.0.0", "Phase X"]


class TestResolveTargetMilestone:
    def test_returns_lowest_semver(self) -> None:
        from conductor.runner import _resolve_target_milestone

        with patch("conductor.runner.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="1.0.0\n0.2.0\n0.1.0\n"
            )
            result = _resolve_target_milestone("owner/repo")

        assert result == "0.1.0"

    def test_returns_none_on_empty(self) -> None:
        from conductor.runner import _resolve_target_milestone

        with patch("conductor.runner.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="\n")
            result = _resolve_target_milestone("owner/repo")

        assert result is None

    def test_returns_none_on_error(self) -> None:
        from conductor.runner import _resolve_target_milestone

        with patch(
            "conductor.runner.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "gh"),
        ):
            result = _resolve_target_milestone("owner/repo")

        assert result is None

    def test_single_milestone(self) -> None:
        from conductor.runner import _resolve_target_milestone

        with patch("conductor.runner.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="0.1.0\n")
            result = _resolve_target_milestone("owner/repo")

        assert result == "0.1.0"


class TestEpicAndMilestoneFiltering:
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
            mock_pool = MagicMock()
            mock_pool.pane_activity_age.return_value = None
            mock_pool_cls.return_value = mock_pool
            runner = ConductorRunner(tmp_path, repo="owner/repo")
        return runner

    def test_refresh_dag_excludes_epics(self, tmp_path: Path) -> None:
        runner = self._make_runner(tmp_path)
        runner._target_milestone = None
        fake_issues = [
            {
                "number": 1,
                "title": "[EPIC] Phase 1 MVP",
                "body": "",
                "labels": [],
                "milestone": None,
            },
            {
                "number": 2,
                "title": "[FEATURE] Bootstrap",
                "body": "",
                "labels": [],
                "milestone": None,
            },
        ]
        with patch(
            "conductor.runner._list_open_issues", return_value=fake_issues
        ):
            dag = runner._refresh_dag()

        assert len(dag.nodes) == 1
        assert dag.get_node(1) is None
        assert dag.get_node(2) is not None

    def test_refresh_dag_filters_by_milestone(self, tmp_path: Path) -> None:
        runner = self._make_runner(tmp_path)
        runner._target_milestone = "0.1.0"
        fake_issues = [
            {
                "number": 2,
                "title": "In milestone",
                "body": "",
                "labels": [],
                "milestone": "0.1.0",
            },
            {
                "number": 15,
                "title": "No milestone",
                "body": "",
                "labels": [],
                "milestone": None,
            },
        ]
        with patch(
            "conductor.runner._list_open_issues", return_value=fake_issues
        ):
            dag = runner._refresh_dag()

        assert len(dag.nodes) == 1
        assert dag.get_node(2) is not None
        assert dag.get_node(15) is None

    def test_refresh_dag_no_milestone_filter_includes_all(
        self, tmp_path: Path
    ) -> None:
        runner = self._make_runner(tmp_path)
        runner._target_milestone = None
        fake_issues = [
            {
                "number": 2,
                "title": "Has milestone",
                "body": "",
                "labels": [],
                "milestone": "0.1.0",
            },
            {
                "number": 15,
                "title": "No milestone",
                "body": "",
                "labels": [],
                "milestone": None,
            },
        ]
        with patch(
            "conductor.runner._list_open_issues", return_value=fake_issues
        ):
            dag = runner._refresh_dag()

        assert len(dag.nodes) == 2

    def test_tick_skips_already_dispatched(self, tmp_path: Path) -> None:
        from conductor.dag import DAG

        runner = self._make_runner(tmp_path)
        runner.db.list_issues.return_value = []
        runner._dispatches[1] = ("agent-1", 0.0, None)

        dag = DAG()
        dag.add_node(1, "Already dispatched", phase="design")
        dag.add_node(2, "Ready", phase="pending")

        with patch.object(runner, "_submit_dispatch") as mock_submit:
            runner._tick(dag)

        mock_submit.assert_called_once()
        assert mock_submit.call_args[0][0].number == 2

    def test_dispatch_tracks_and_clears_agent(self, tmp_path: Path) -> None:
        from conductor.dag import DAGNode
        from conductor.phases import PhaseResult

        runner = self._make_runner(tmp_path)
        node = DAGNode(number=5, title="Test", phase="design")

        with patch(
            "conductor.runner.run_phase",
            return_value=PhaseResult(phase="design", success=True),
        ):
            runner._dispatch_issue(node, "design")

        assert 5 not in runner._dispatches

    def test_dashboard_shows_milestone_in_title(self, tmp_path: Path) -> None:
        runner = self._make_runner(tmp_path)
        runner._target_milestone = "0.1.0"
        runner.db.list_issues.return_value = []

        table = runner._render_dashboard()

        assert "0.1.0" in table.title

    def test_dashboard_shows_agent_column(self, tmp_path: Path) -> None:
        from conductor.dag import DAG

        runner = self._make_runner(tmp_path)
        runner.db.list_issues.return_value = []
        runner._dispatches[1] = ("agent-0", 0.0, "conductor-agent-0")

        dag = DAG()
        dag.add_node(1, "Active issue", phase="execute")

        table = runner._render_dashboard(dag)

        assert len(table.columns) == 7
        assert table.columns[3].header == "Agent"
        assert table.columns[4].header == "Elapsed"
        assert table.columns[5].header == "Activity"

    def test_dashboard_has_pool_caption(self, tmp_path: Path) -> None:
        runner = self._make_runner(tmp_path)
        runner.db.list_issues.return_value = []
        runner.pool._sessions = {}
        runner.pool._max_sessions = 3

        table = runner._render_dashboard()

        assert "Pool:" in table.caption
        assert "0/3 busy" in table.caption

    def test_format_elapsed(self, tmp_path: Path) -> None:
        from conductor.runner import ConductorRunner

        assert ConductorRunner._format_elapsed(5) == "5s"
        assert ConductorRunner._format_elapsed(65) == "1m05s"
        assert ConductorRunner._format_elapsed(3661) == "1h01m"

    def test_dashboard_elapsed_and_activity(self, tmp_path: Path) -> None:
        import time

        from conductor.dag import DAG

        runner = self._make_runner(tmp_path)
        runner.db.list_issues.return_value = []
        runner.pool.pane_activity_age = MagicMock(return_value=2.0)

        dag = DAG()
        dag.add_node(1, "Active issue", phase="execute")
        runner._dispatches[1] = (
            "agent-1",
            time.monotonic() - 90,
            "conductor-agent-0",
        )

        table = runner._render_dashboard(dag)
        assert table.row_count == 1


class TestAsyncDispatch:
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
            mock_pool = MagicMock()
            mock_pool.pane_activity_age.return_value = None
            mock_pool_cls.return_value = mock_pool
            runner = ConductorRunner(tmp_path, repo="owner/repo")
        return runner

    def test_shutdown_event_property(self, tmp_path: Path) -> None:
        runner = self._make_runner(tmp_path)
        assert runner._shutdown is False
        runner._shutdown = True
        assert runner._shutdown is True
        assert runner._shutdown_event.is_set()
        runner._shutdown = False
        assert runner._shutdown is False

    def test_submit_dispatch_tracks_in_dispatches(self, tmp_path: Path) -> None:
        from conductor.dag import DAGNode

        runner = self._make_runner(tmp_path)
        node = DAGNode(number=5, title="Test", phase="design")

        with patch.object(runner, "_dispatch_issue"):
            runner._submit_dispatch(node, "design")

        assert 5 in runner._dispatches
        agent_name, started_at, session_name = runner._dispatches[5]
        assert agent_name == "agent-5"
        assert isinstance(started_at, float)
        assert 5 in runner._futures

    def test_submit_dispatch_skips_if_already_dispatched(self, tmp_path: Path) -> None:
        from conductor.dag import DAGNode

        runner = self._make_runner(tmp_path)
        runner._dispatches[5] = ("agent-5", 0.0, None)
        node = DAGNode(number=5, title="Test", phase="design")

        with patch.object(runner, "_dispatch_issue") as mock_d:
            runner._submit_dispatch(node, "design")

        mock_d.assert_not_called()
        assert 5 not in runner._futures

    def test_reap_futures_updates_phase_from_db(self, tmp_path: Path) -> None:
        from concurrent.futures import Future

        from conductor.dag import DAG

        runner = self._make_runner(tmp_path)
        dag = DAG()
        dag.add_node(5, "Test issue", phase="pending")

        future: Future[None] = Future()
        future.set_result(None)
        runner._futures[5] = future
        runner.db.get_issue.return_value = {"phase": "design"}

        runner._reap_futures(dag)

        assert 5 not in runner._futures
        assert dag.get_node(5).phase == "design"

    def test_cleanup_shuts_down_executor(self, tmp_path: Path) -> None:
        runner = self._make_runner(tmp_path)
        runner._cleanup()
        runner.pool.shutdown.assert_called_once()
        runner.db.close.assert_called_once()

    def test_handle_shutdown_sets_event(self, tmp_path: Path) -> None:
        runner = self._make_runner(tmp_path)
        runner._handle_shutdown(2, None)
        assert runner._shutdown_event.is_set()


class TestLoadIssueContextFromDB:
    def test_loads_from_db(self, tmp_path: Path) -> None:
        from conductor.phases import PhaseContext, _load_issue_context
        from conductor.state_db import StateDB

        db = StateDB(tmp_path / "test.db")
        db.upsert_issue(
            42, "Test feature",
            body="Build the thing",
            labels='["bug", "phase:design"]',
            blocked_by="[1, 2]",
            branch="feature/42",
        )
        ctx = PhaseContext(
            issue_number=42,
            config=MagicMock(),
            pool=MagicMock(),
            db=db,
            project_root=tmp_path,
            worktree=tmp_path / "wt",
        )
        ic = _load_issue_context(ctx, "plan")
        assert ic.number == 42
        assert ic.title == "Test feature"
        assert ic.body == "Build the thing"
        assert ic.labels == ["bug", "phase:design"]
        assert ic.phase == "plan"
        assert ic.branch == "feature/42"
        db.close()

    def test_raises_when_not_found(self, tmp_path: Path) -> None:
        from conductor.phases import PhaseContext, _load_issue_context
        from conductor.state_db import StateDB

        db = StateDB(tmp_path / "test2.db")
        ctx = PhaseContext(
            issue_number=999,
            config=MagicMock(),
            pool=MagicMock(),
            db=db,
            project_root=tmp_path,
            worktree=tmp_path / "wt",
        )
        with pytest.raises(ValueError, match="not found"):
            _load_issue_context(ctx, "design")
        db.close()


class TestStateDBBodyLabels:
    def test_body_and_labels_stored(self, tmp_path: Path) -> None:
        from conductor.state_db import StateDB

        db = StateDB(tmp_path / "test.db")
        db.upsert_issue(
            1, "Test", body="Issue body", labels='["bug"]'
        )
        issue = db.get_issue(1)
        assert issue["body"] == "Issue body"
        assert issue["labels"] == '["bug"]'
        db.close()

    def test_body_defaults_to_null(self, tmp_path: Path) -> None:
        from conductor.state_db import StateDB

        db = StateDB(tmp_path / "test.db")
        db.upsert_issue(1, "Test")
        issue = db.get_issue(1)
        assert issue["body"] is None
        assert issue["labels"] is None
        db.close()


class TestStateDBMilestone:
    def test_milestone_column_in_new_db(self, tmp_path: Path) -> None:
        from conductor.state_db import StateDB

        db = StateDB(tmp_path / "test.db")
        db.upsert_issue(
            number=1, title="Test", phase="pending", milestone="0.1.0"
        )
        issue = db.get_issue(1)
        assert issue is not None
        assert issue["milestone"] == "0.1.0"
        db.close()

    def test_milestone_null_default(self, tmp_path: Path) -> None:
        from conductor.state_db import StateDB

        db = StateDB(tmp_path / "test.db")
        db.upsert_issue(number=1, title="Test")
        issue = db.get_issue(1)
        assert issue is not None
        assert issue["milestone"] is None
        db.close()
