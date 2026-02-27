from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from conductor.pool import AgentPool


@pytest.fixture()
def pool() -> AgentPool:
    p = AgentPool(max_sessions=2, idle_ttl_seconds=10, default_model="sonnet-4.5")
    p._run = MagicMock(return_value=MagicMock(returncode=0))
    return p


@pytest.fixture()
def worktree(tmp_path: Path) -> Path:
    return tmp_path / "repo"


class TestAcquire:
    def test_creates_new_session(self, pool: AgentPool, worktree: Path) -> None:
        session = pool.acquire(worktree)
        assert session.name == "conductor-agent-0"
        assert session.worktree == worktree
        assert session.model == "sonnet-4.5"
        assert session.busy is True
        assert pool.active_count == 1

    def test_creates_with_custom_model(self, pool: AgentPool, worktree: Path) -> None:
        session = pool.acquire(worktree, model="opus-4.6")
        assert session.model == "opus-4.6"

    def test_reuses_idle_session(self, pool: AgentPool, worktree: Path) -> None:
        s1 = pool.acquire(worktree)
        pool.release(s1)
        s2 = pool.acquire(worktree)
        assert s2.name == s1.name
        assert s2.busy is True
        assert pool.active_count == 1

    def test_switches_model_on_reuse(self, pool: AgentPool, worktree: Path) -> None:
        s1 = pool.acquire(worktree, model="sonnet-4.5")
        pool.release(s1)
        s2 = pool.acquire(worktree, model="opus-4.6")
        assert s2.model == "opus-4.6"

    def test_raises_when_full(self, pool: AgentPool, worktree: Path) -> None:
        pool.acquire(worktree)
        pool.acquire(worktree)
        with pytest.raises(RuntimeError, match="max capacity"):
            pool.acquire(worktree)

    def test_increments_names(self, pool: AgentPool, worktree: Path) -> None:
        s0 = pool.acquire(worktree)
        s1 = pool.acquire(worktree)
        assert s0.name == "conductor-agent-0"
        assert s1.name == "conductor-agent-1"

    def test_calls_tmux_new_session(self, pool: AgentPool, worktree: Path) -> None:
        pool.acquire(worktree)
        pool._run.assert_any_call(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                "conductor-agent-0",
                "-c",
                str(worktree),
                "agent chat --yolo --trust --approve-mcps",
            ],
            check=True,
        )


class TestRelease:
    def test_marks_idle(self, pool: AgentPool, worktree: Path) -> None:
        s = pool.acquire(worktree)
        assert s.busy is True
        pool.release(s)
        assert s.busy is False

    def test_updates_last_used(self, pool: AgentPool, worktree: Path) -> None:
        s = pool.acquire(worktree)
        old_ts = s.last_used
        time.sleep(0.01)
        pool.release(s)
        assert s.last_used > old_ts


class TestSend:
    def test_send_keys(self, pool: AgentPool, worktree: Path) -> None:
        s = pool.acquire(worktree)
        pool._run.reset_mock()
        pool.send(s, "hello world")
        pool._run.assert_called_once_with(
            ["tmux", "send-keys", "-t", "conductor-agent-0", "hello world", "Enter"],
            check=True,
        )


class TestSwitchModel:
    def test_sends_model_command(self, pool: AgentPool, worktree: Path) -> None:
        s = pool.acquire(worktree)
        pool._run.reset_mock()
        pool.switch_model(s, "opus-4.6")
        pool._run.assert_called_once_with(
            [
                "tmux",
                "send-keys",
                "-t",
                "conductor-agent-0",
                "/model opus-4.6",
                "Enter",
            ],
            check=True,
        )
        assert s.model == "opus-4.6"


class TestClearContext:
    def test_sends_clear_command(self, pool: AgentPool, worktree: Path) -> None:
        s = pool.acquire(worktree)
        pool._run.reset_mock()
        pool.clear_context(s)
        pool._run.assert_called_once_with(
            ["tmux", "send-keys", "-t", "conductor-agent-0", "/clear", "Enter"],
            check=True,
        )


class TestDrainIdle:
    def test_kills_expired_sessions(self, pool: AgentPool, worktree: Path) -> None:
        s = pool.acquire(worktree)
        pool.release(s)
        s.last_used = time.time() - 20
        drained = pool.drain_idle()
        assert drained == 1
        assert pool.active_count == 0

    def test_keeps_fresh_sessions(self, pool: AgentPool, worktree: Path) -> None:
        s = pool.acquire(worktree)
        pool.release(s)
        drained = pool.drain_idle()
        assert drained == 0
        assert pool.active_count == 1

    def test_keeps_busy_sessions(self, pool: AgentPool, worktree: Path) -> None:
        s = pool.acquire(worktree)
        s.last_used = time.time() - 20
        drained = pool.drain_idle()
        assert drained == 0
        assert pool.active_count == 1

    def test_calls_kill_session(self, pool: AgentPool, worktree: Path) -> None:
        s = pool.acquire(worktree)
        pool.release(s)
        s.last_used = time.time() - 20
        pool._run.reset_mock()
        pool.drain_idle()
        pool._run.assert_called_once_with(
            ["tmux", "kill-session", "-t", "conductor-agent-0"],
            capture_output=True,
            timeout=5,
        )


class TestShutdown:
    def test_kills_all_sessions(self, pool: AgentPool, worktree: Path) -> None:
        pool.acquire(worktree)
        pool.acquire(worktree)
        assert pool.active_count == 2
        pool.shutdown()
        assert pool.active_count == 0

    def test_calls_kill_for_each(self, pool: AgentPool, worktree: Path) -> None:
        pool.acquire(worktree)
        pool.acquire(worktree)
        pool._run.reset_mock()
        pool.shutdown()
        kill_calls = [
            c
            for c in pool._run.call_args_list
            if c[0][0][1] == "kill-session"
        ]
        assert len(kill_calls) == 2


class TestProperties:
    def test_active_count(self, pool: AgentPool, worktree: Path) -> None:
        assert pool.active_count == 0
        pool.acquire(worktree)
        assert pool.active_count == 1
        pool.acquire(worktree)
        assert pool.active_count == 2

    def test_idle_sessions(self, pool: AgentPool, worktree: Path) -> None:
        s1 = pool.acquire(worktree)
        s2 = pool.acquire(worktree)
        assert pool.idle_sessions == []
        pool.release(s1)
        assert pool.idle_sessions == [s1]
        pool.release(s2)
        idle = pool.idle_sessions
        assert len(idle) == 2
        assert s1 in idle
        assert s2 in idle
