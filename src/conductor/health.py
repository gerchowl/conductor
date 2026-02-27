from __future__ import annotations

import subprocess
import time
from enum import Enum
from pathlib import Path

from conductor.pool import AgentPool, AgentSession


class AgentState(Enum):
    ACTIVE = "active"
    DONE = "done"
    IDLE = "idle"
    HUNG = "hung"
    FORGOT = "forgot"
    DEAD = "dead"


def _run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, **kwargs)


def get_pane_activity_age(session_name: str) -> float | None:
    """Seconds since last pane activity. None if session doesn't exist."""
    has = _run(
        ["tmux", "has-session", "-t", session_name],
        capture_output=True,
    )
    if has.returncode != 0:
        return None

    result = _run(
        ["tmux", "display-message", "-t", session_name, "-p", "#{pane_activity}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None

    epoch = int(result.stdout.strip())
    return time.time() - epoch


def is_pane_alive(session_name: str) -> bool:
    """Check if the tmux session/pane process is still running."""
    result = _run(
        ["tmux", "display-message", "-t", session_name, "-p", "#{pane_pid}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False

    pid = result.stdout.strip()
    if not pid:
        return False

    check = _run(["kill", "-0", pid], capture_output=True)
    return check.returncode == 0


def is_at_prompt(session_name: str) -> bool:
    """Check if agent is sitting at its input prompt (idle cursor)."""
    result = _run(
        ["tmux", "capture-pane", "-t", session_name, "-p", "-S", "-3"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False

    lines = result.stdout.rstrip("\n").split("\n")
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            continue
        return stripped.endswith(">") or stripped.endswith("$")
    return False


def check_agent_health(
    session: AgentSession,
    output_path: Path,
    idle_threshold_seconds: int = 30,
    elapsed_seconds: float = 0,
    timeout_seconds: float = 300,
) -> AgentState:
    """Determine current state of an agent session."""
    if output_path.exists():
        return AgentState.DONE

    if not is_pane_alive(session.name):
        return AgentState.DEAD

    activity_age = get_pane_activity_age(session.name)
    if activity_age is None:
        return AgentState.DEAD

    if activity_age < idle_threshold_seconds:
        return AgentState.ACTIVE

    if elapsed_seconds < timeout_seconds:
        return AgentState.IDLE

    if is_at_prompt(session.name):
        return AgentState.FORGOT

    return AgentState.HUNG


def nudge(session: AgentSession, message: str) -> None:
    """Send a follow-up nudge message to the agent."""
    _run(
        ["tmux", "send-keys", "-t", session.name, message, "Enter"],
        check=True,
    )


def recover(
    session: AgentSession,
    pool: AgentPool,
    step_input_path: Path,
    output_path: Path,
    max_nudges: int = 2,
    max_retries: int = 1,
) -> AgentSession | None:
    """Execute recovery protocol.

    Returns new session if retried, None if escalation needed.
    """
    nudge_count = getattr(session, "_nudge_count", 0)
    retry_count = getattr(session, "_retry_count", 0)

    if nudge_count < max_nudges:
        nudge(session, "You appear stuck. Please continue and write your output file.")
        session._nudge_count = nudge_count + 1  # type: ignore[attr-defined]
        return session

    if retry_count < max_retries:
        pool._kill_session(session.name)
        new_session = pool.acquire(session.worktree, model=session.model)
        new_session._retry_count = retry_count + 1  # type: ignore[attr-defined]
        new_session._nudge_count = 0  # type: ignore[attr-defined]
        task_text = step_input_path.read_text()
        nudge(new_session, task_text)
        return new_session

    return None
