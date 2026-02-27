from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from conductor.health import (
    AgentState,
    check_agent_health,
    get_pane_activity_age,
    is_at_prompt,
    is_pane_alive,
    nudge,
    recover,
)
from conductor.pool import AgentPool, AgentSession


@pytest.fixture()
def session(tmp_path: Path) -> AgentSession:
    return AgentSession(name="test-agent-0", worktree=tmp_path / "repo")


@pytest.fixture()
def pool() -> AgentPool:
    p = AgentPool(max_sessions=3)
    p._run = MagicMock(return_value=MagicMock(returncode=0))
    return p


def _completed(
    stdout: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout)


class TestGetPaneActivityAge:
    @patch("conductor.health._run")
    def test_returns_age_for_valid_session(self, mock_run: MagicMock) -> None:
        import time

        now = int(time.time())
        mock_run.side_effect = [
            _completed(returncode=0),
            _completed(stdout=str(now - 15)),
        ]
        age = get_pane_activity_age("test-agent-0")
        assert age is not None
        assert 14.0 <= age <= 17.0

    @patch("conductor.health._run")
    def test_returns_none_for_missing_session(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _completed(returncode=1)
        assert get_pane_activity_age("nonexistent") is None

    @patch("conductor.health._run")
    def test_returns_none_when_display_fails(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = [
            _completed(returncode=0),
            _completed(returncode=1),
        ]
        assert get_pane_activity_age("test-agent-0") is None


class TestIsPaneAlive:
    @patch("conductor.health._run")
    def test_alive_when_pid_running(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = [
            _completed(stdout="12345"),
            _completed(returncode=0),
        ]
        assert is_pane_alive("test-agent-0") is True

    @patch("conductor.health._run")
    def test_dead_when_pid_not_running(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = [
            _completed(stdout="12345"),
            _completed(returncode=1),
        ]
        assert is_pane_alive("test-agent-0") is False

    @patch("conductor.health._run")
    def test_dead_when_display_fails(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _completed(returncode=1)
        assert is_pane_alive("test-agent-0") is False

    @patch("conductor.health._run")
    def test_dead_when_empty_pid(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _completed(stdout="")
        assert is_pane_alive("test-agent-0") is False


class TestIsAtPrompt:
    @patch("conductor.health._run")
    def test_detects_chevron_prompt(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _completed(stdout="some output\nagent >")
        assert is_at_prompt("test-agent-0") is True

    @patch("conductor.health._run")
    def test_detects_dollar_prompt(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _completed(stdout="some output\nuser@host:~ $")
        assert is_at_prompt("test-agent-0") is True

    @patch("conductor.health._run")
    def test_no_prompt_detected(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _completed(stdout="Processing files...\nRunning tests")
        assert is_at_prompt("test-agent-0") is False

    @patch("conductor.health._run")
    def test_false_when_capture_fails(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _completed(returncode=1)
        assert is_at_prompt("test-agent-0") is False

    @patch("conductor.health._run")
    def test_skips_blank_trailing_lines(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _completed(stdout="agent >\n\n\n")
        assert is_at_prompt("test-agent-0") is True

    @patch("conductor.health._run")
    def test_false_for_all_blank_output(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _completed(stdout="\n\n\n")
        assert is_at_prompt("test-agent-0") is False


class TestCheckAgentHealth:
    def test_done_when_output_exists(
        self, session: AgentSession, tmp_path: Path
    ) -> None:
        output = tmp_path / "output.md"
        output.write_text("result")
        assert check_agent_health(session, output) == AgentState.DONE

    @patch("conductor.health.is_pane_alive", return_value=False)
    def test_dead_when_pane_not_alive(
        self, _mock: MagicMock, session: AgentSession, tmp_path: Path
    ) -> None:
        assert check_agent_health(session, tmp_path / "missing.md") == AgentState.DEAD

    @patch("conductor.health.get_pane_activity_age", return_value=None)
    @patch("conductor.health.is_pane_alive", return_value=True)
    def test_dead_when_no_activity_info(
        self, _alive: MagicMock, _age: MagicMock, session: AgentSession, tmp_path: Path
    ) -> None:
        assert check_agent_health(session, tmp_path / "missing.md") == AgentState.DEAD

    @patch("conductor.health.get_pane_activity_age", return_value=5.0)
    @patch("conductor.health.is_pane_alive", return_value=True)
    def test_active_when_recent_activity(
        self, _alive: MagicMock, _age: MagicMock, session: AgentSession, tmp_path: Path
    ) -> None:
        result = check_agent_health(
            session, tmp_path / "missing.md", idle_threshold_seconds=30
        )
        assert result == AgentState.ACTIVE

    @patch("conductor.health.get_pane_activity_age", return_value=60.0)
    @patch("conductor.health.is_pane_alive", return_value=True)
    def test_idle_within_timeout(
        self, _alive: MagicMock, _age: MagicMock, session: AgentSession, tmp_path: Path
    ) -> None:
        assert (
            check_agent_health(
                session,
                tmp_path / "missing.md",
                idle_threshold_seconds=30,
                elapsed_seconds=100,
                timeout_seconds=300,
            )
            == AgentState.IDLE
        )

    @patch("conductor.health.is_at_prompt", return_value=True)
    @patch("conductor.health.get_pane_activity_age", return_value=60.0)
    @patch("conductor.health.is_pane_alive", return_value=True)
    def test_forgot_at_prompt_past_timeout(
        self,
        _alive: MagicMock,
        _age: MagicMock,
        _prompt: MagicMock,
        session: AgentSession,
        tmp_path: Path,
    ) -> None:
        assert (
            check_agent_health(
                session,
                tmp_path / "missing.md",
                idle_threshold_seconds=30,
                elapsed_seconds=400,
                timeout_seconds=300,
            )
            == AgentState.FORGOT
        )

    @patch("conductor.health.is_at_prompt", return_value=False)
    @patch("conductor.health.get_pane_activity_age", return_value=60.0)
    @patch("conductor.health.is_pane_alive", return_value=True)
    def test_hung_not_at_prompt_past_timeout(
        self,
        _alive: MagicMock,
        _age: MagicMock,
        _prompt: MagicMock,
        session: AgentSession,
        tmp_path: Path,
    ) -> None:
        assert (
            check_agent_health(
                session,
                tmp_path / "missing.md",
                idle_threshold_seconds=30,
                elapsed_seconds=400,
                timeout_seconds=300,
            )
            == AgentState.HUNG
        )


class TestNudge:
    @patch("conductor.health._run")
    def test_sends_tmux_keys(self, mock_run: MagicMock, session: AgentSession) -> None:
        nudge(session, "Please continue.")
        mock_run.assert_called_once_with(
            ["tmux", "send-keys", "-t", "test-agent-0", "Please continue.", "Enter"],
            check=True,
        )


class TestRecover:
    @patch("conductor.health.nudge")
    def test_nudge_path(
        self,
        mock_nudge: MagicMock,
        session: AgentSession,
        pool: AgentPool,
        tmp_path: Path,
    ) -> None:
        step_input = tmp_path / "input.md"
        step_input.write_text("do the thing")
        output = tmp_path / "output.md"

        result = recover(session, pool, step_input, output, max_nudges=2)
        assert result is session
        mock_nudge.assert_called_once()
        assert session._nudge_count == 1  # type: ignore[attr-defined]

    @patch("conductor.health.nudge")
    def test_second_nudge(
        self,
        mock_nudge: MagicMock,
        session: AgentSession,
        pool: AgentPool,
        tmp_path: Path,
    ) -> None:
        step_input = tmp_path / "input.md"
        step_input.write_text("do the thing")
        output = tmp_path / "output.md"
        session._nudge_count = 1  # type: ignore[attr-defined]

        result = recover(session, pool, step_input, output, max_nudges=2)
        assert result is session
        assert session._nudge_count == 2  # type: ignore[attr-defined]

    @patch("conductor.health.nudge")
    def test_retry_path_kills_and_reacquires(
        self,
        mock_nudge: MagicMock,
        session: AgentSession,
        pool: AgentPool,
        tmp_path: Path,
    ) -> None:
        step_input = tmp_path / "input.md"
        step_input.write_text("do the thing")
        output = tmp_path / "output.md"
        session._nudge_count = 2  # type: ignore[attr-defined]

        new_session = recover(
            session, pool, step_input, output, max_nudges=2, max_retries=1
        )
        assert new_session is not None
        assert new_session is not session
        assert new_session._retry_count == 1  # type: ignore[attr-defined]
        assert new_session._nudge_count == 0  # type: ignore[attr-defined]
        mock_nudge.assert_called_once_with(new_session, "do the thing")

    @patch("conductor.health.nudge")
    def test_escalation_when_retries_exhausted(
        self,
        mock_nudge: MagicMock,
        session: AgentSession,
        pool: AgentPool,
        tmp_path: Path,
    ) -> None:
        step_input = tmp_path / "input.md"
        step_input.write_text("do the thing")
        output = tmp_path / "output.md"
        session._nudge_count = 2  # type: ignore[attr-defined]
        session._retry_count = 1  # type: ignore[attr-defined]

        result = recover(session, pool, step_input, output, max_nudges=2, max_retries=1)
        assert result is None
        mock_nudge.assert_not_called()
