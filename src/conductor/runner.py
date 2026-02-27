from __future__ import annotations

import signal
import time
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.table import Table

from conductor.config import load_config
from conductor.dag import DAG, DAGNode
from conductor.gh_sync import flush_sync_queue
from conductor.phases import PHASE_ORDER
from conductor.pool import AgentPool
from conductor.state_db import StateDB


class ConductorRunner:
    def __init__(self, project_root: Path, repo: str | None = None) -> None:
        self.project_root = project_root
        self.repo = repo
        self.config = load_config(project_root)
        self.db = StateDB(project_root / ".conductor" / "state.db")
        self.pool = AgentPool(
            max_sessions=self.config.pool.max_sessions,
            idle_ttl_seconds=self.config.pool.idle_ttl_seconds,
            default_model=self.config.pool.default_model,
        )
        self._shutdown = False

    def run(self, poll_interval: float = 10.0) -> None:
        signal.signal(signal.SIGINT, self._handle_shutdown)
        console = Console()

        with Live(
            self._render_dashboard(), console=console, refresh_per_second=1
        ) as live:
            while not self._shutdown:
                self._tick()
                live.update(self._render_dashboard())
                flush_sync_queue(self.db, self.repo)
                self.pool.drain_idle()
                time.sleep(poll_interval)

        self._cleanup()

    def _tick(self) -> None:
        dag = self._refresh_dag()
        completed = self._completed_issues()
        ready = dag.ready_issues(completed)

        for node in ready:
            if node.phase in ("merged", "pr"):
                continue
            current_phase = node.phase if node.phase in PHASE_ORDER else "design"
            self._dispatch_issue(node, current_phase)

    def _refresh_dag(self) -> DAG: ...

    def _completed_issues(self) -> set[int]:
        return {
            issue["number"]
            for issue in self.db.list_issues()
            if issue.get("phase") in ("merged", "closed")
        }

    def _dispatch_issue(self, node: DAGNode, phase: str) -> None: ...

    def _render_dashboard(self) -> Table:
        table = Table(title="Conductor Dashboard")
        table.add_column("Issue", style="cyan")
        table.add_column("Title")
        table.add_column("Phase", style="green")
        table.add_column("Status")
        table.add_column("Blocked By")

        for issue in self.db.list_issues():
            blocked = issue.get("blocked_by", "[]")
            table.add_row(
                f"#{issue['number']}",
                issue.get("title", ""),
                issue.get("phase", "pending"),
                issue.get("current_step", "-"),
                blocked,
            )
        return table

    def _handle_shutdown(self, signum: int, frame: object) -> None:
        self._shutdown = True

    def _cleanup(self) -> None:
        self.pool.shutdown()
        self.db.close()
