from __future__ import annotations

import json
import logging
import signal
import subprocess
import time
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.table import Table

from conductor.config import load_config
from conductor.dag import DAG, DAGNode, build_dag_from_issues
from conductor.gh_sync import flush_sync_queue
from conductor.phases import PHASE_ORDER, PhaseContext, run_phase
from conductor.pool import AgentPool
from conductor.state_db import StateDB

logger = logging.getLogger(__name__)


def _list_open_issues(repo: str | None) -> list[dict]:
    """Fetch all open issues from GitHub via gh CLI."""
    cmd = ["gh", "issue", "list", "--state", "open", "--limit", "200"]
    if repo:
        cmd.extend(["--repo", repo])
    cmd.extend(["--json", "number,title,body,labels,state"])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Failed to fetch issues from GitHub")
        return []
    raw = json.loads(result.stdout)
    issues: list[dict] = []
    for item in raw:
        labels = [
            lbl["name"] if isinstance(lbl, dict) else lbl
            for lbl in item.get("labels", [])
        ]
        issues.append(
            {
                "number": item["number"],
                "title": item["title"],
                "body": item.get("body", ""),
                "labels": labels,
            }
        )
    return issues


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

        console.print("[bold]Conductor starting...[/bold]")

        dag = self._refresh_dag()
        self._sync_dag_to_db(dag)

        with Live(
            self._render_dashboard(dag),
            console=console,
            refresh_per_second=1,
        ) as live:
            while not self._shutdown:
                self._tick(dag)
                live.update(self._render_dashboard(dag))
                flush_sync_queue(self.db, self.repo)
                self.pool.drain_idle()

                for _ in range(int(poll_interval)):
                    if self._shutdown:
                        break
                    time.sleep(1)

                dag = self._refresh_dag()
                self._sync_dag_to_db(dag)

        self._cleanup()
        console.print("[bold]Conductor stopped.[/bold]")

    def _tick(self, dag: DAG) -> None:
        completed = self._completed_issues()
        ready = dag.ready_issues(completed)

        for node in ready:
            if node.phase in ("merged", "pr", "closed"):
                continue
            current_phase = node.phase if node.phase in PHASE_ORDER else "design"
            self._dispatch_issue(node, current_phase)

    def _refresh_dag(self) -> DAG:
        issues = _list_open_issues(self.repo)
        return build_dag_from_issues(issues)

    def _sync_dag_to_db(self, dag: DAG) -> None:
        """Upsert DAG nodes into the state DB."""
        for node in dag.nodes:
            self.db.upsert_issue(
                number=node.number,
                title=node.title,
                phase=node.phase,
                blocked_by=json.dumps(node.blocked_by),
            )

    def _completed_issues(self) -> set[int]:
        return {
            issue["number"]
            for issue in self.db.list_issues()
            if issue.get("phase") in ("merged", "closed")
        }

    def _dispatch_issue(self, node: DAGNode, phase: str) -> None:
        logger.info("Dispatching #%d phase=%s", node.number, phase)
        worktree = self.project_root

        ctx = PhaseContext(
            issue_number=node.number,
            config=self.config,
            pool=self.pool,
            db=self.db,
            project_root=self.project_root,
            worktree=worktree,
            repo=self.repo,
        )
        result = run_phase(ctx, phase)
        if result.success:
            logger.info("Issue #%d phase %s succeeded", node.number, phase)
        else:
            logger.warning(
                "Issue #%d phase %s failed: %s",
                node.number,
                phase,
                result.error,
            )

    def _render_dashboard(self, dag: DAG | None = None) -> Table:
        table = Table(title="Conductor Dashboard")
        table.add_column("#", style="cyan", justify="right")
        table.add_column("Title", max_width=50)
        table.add_column("Phase", style="green")
        table.add_column("Blocked By", style="dim")

        nodes = dag.nodes if dag else []
        completed = self._completed_issues()

        for node in nodes:
            blocked_str = (
                ", ".join(f"#{b}" for b in node.blocked_by) if node.blocked_by else "-"
            )
            is_ready = not dag.is_blocked(node.number, completed)
            phase_style = "bold green" if is_ready else "dim"
            table.add_row(
                str(node.number),
                node.title,
                f"[{phase_style}]{node.phase}[/]",
                blocked_str,
            )
        return table

    def _handle_shutdown(self, signum: int, frame: object) -> None:
        self._shutdown = True

    def _cleanup(self) -> None:
        self.pool.shutdown()
        self.db.close()
