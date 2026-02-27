from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_SUBPROCESS_TIMEOUT = 30


@dataclass
class AgentSession:
    name: str
    worktree: Path
    model: str | None = None
    busy: bool = False
    last_used: float = field(default_factory=time.time)


class AgentPool:
    def __init__(
        self,
        max_sessions: int = 3,
        idle_ttl_seconds: int = 60,
        default_model: str = "sonnet-4.5",
    ) -> None:
        self._max_sessions = max_sessions
        self._idle_ttl = idle_ttl_seconds
        self._default_model = default_model
        self._sessions: dict[str, AgentSession] = {}
        self._next_id = 0

    def _run(
        self, args: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        kwargs.setdefault("timeout", _SUBPROCESS_TIMEOUT)
        return subprocess.run(args, **kwargs)

    def _make_name(self) -> str:
        name = f"conductor-agent-{self._next_id}"
        self._next_id += 1
        return name

    def _session_exists(self, name: str) -> bool:
        try:
            result = self._run(
                ["tmux", "has-session", "-t", name],
                capture_output=True,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
        return result.returncode == 0

    def _create_session(self, name: str, worktree: Path) -> None:
        self._run(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                name,
                "-c",
                str(worktree),
                "agent chat --yolo --approve-mcps",
            ],
            check=True,
        )

    def _kill_session(self, name: str) -> None:
        try:
            self._run(
                ["tmux", "kill-session", "-t", name],
                capture_output=True,
                timeout=5,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("Failed to kill tmux session %s", name)

    def acquire(self, worktree: Path, model: str | None = None) -> AgentSession:
        target_model = model or self._default_model

        for session in self._sessions.values():
            if not session.busy:
                session.busy = True
                session.worktree = worktree
                session.last_used = time.time()
                if session.model != target_model:
                    self.switch_model(session, target_model)
                return session

        if len(self._sessions) >= self._max_sessions:
            msg = f"Pool at max capacity ({self._max_sessions}), all sessions busy"
            raise RuntimeError(msg)

        name = self._make_name()
        self._create_session(name, worktree)
        session = AgentSession(
            name=name,
            worktree=worktree,
            model=target_model,
            busy=True,
            last_used=time.time(),
        )
        self._sessions[name] = session
        return session

    def release(self, session: AgentSession) -> None:
        session.busy = False
        session.last_used = time.time()

    def send(self, session: AgentSession, text: str) -> None:
        self._run(
            ["tmux", "send-keys", "-t", session.name, text, "Enter"],
            check=True,
        )

    def switch_model(self, session: AgentSession, model: str) -> None:
        self.send(session, f"/model {model}")
        session.model = model

    def clear_context(self, session: AgentSession) -> None:
        self.send(session, "/clear")

    def drain_idle(self) -> int:
        now = time.time()
        expired = [
            name
            for name, s in self._sessions.items()
            if not s.busy and (now - s.last_used) >= self._idle_ttl
        ]
        for name in expired:
            self._kill_session(name)
            del self._sessions[name]
        return len(expired)

    def shutdown(self) -> None:
        for name in list(self._sessions):
            self._kill_session(name)
        self._sessions.clear()

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    def pane_activity_age(self, session_name: str) -> float | None:
        """Seconds since last tmux pane activity. None if unavailable."""
        try:
            result = self._run(
                [
                    "tmux", "display-message", "-t", session_name,
                    "-p", "#{pane_activity}",
                ],
                capture_output=True,
                text=True,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None
        if result.returncode != 0:
            return None
        try:
            epoch = int(result.stdout.strip())
        except ValueError:
            return None
        return time.time() - epoch

    @property
    def idle_sessions(self) -> list[AgentSession]:
        return [s for s in self._sessions.values() if not s.busy]
