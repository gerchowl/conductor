from __future__ import annotations

import concurrent.futures
import json
import logging
import re
import signal
import subprocess
import threading
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

_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)")


def _list_open_issues(repo: str | None) -> list[dict]:
    """Fetch all open issues from GitHub via gh CLI."""
    cmd = ["gh", "issue", "list", "--state", "open", "--limit", "200"]
    if repo:
        cmd.extend(["--repo", repo])
    cmd.extend(["--json", "number,title,body,labels,state,milestone"])
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True
        )
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
        ms = item.get("milestone")
        milestone = ms.get("title") if isinstance(ms, dict) else None
        issues.append(
            {
                "number": item["number"],
                "title": item["title"],
                "body": item.get("body", ""),
                "labels": labels,
                "milestone": milestone,
            }
        )
    return issues


def _is_epic(title: str) -> bool:
    return title.strip().upper().startswith("[EPIC]")


def _semver_key(title: str) -> tuple[int, ...]:
    """Extract semver tuple from milestone title for sorting."""
    m = _SEMVER_RE.match(title)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return (999, 999, 999)


def _resolve_target_milestone(repo: str | None) -> str | None:
    """Fetch open milestones, return the one with the lowest semver."""
    cmd = ["gh", "api", "repos/{owner}/{repo}/milestones", "--jq",
           '.[] | select(.state=="open") | .title']
    if repo:
        cmd = [
            "gh", "api", f"repos/{repo}/milestones",
            "--jq", '.[] | select(.state=="open") | .title',
        ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Failed to fetch milestones from GitHub")
        return None
    titles = [t.strip() for t in result.stdout.strip().splitlines() if t.strip()]
    if not titles:
        return None
    titles.sort(key=_semver_key)
    return titles[0]


class ConductorRunner:
    def __init__(
        self, project_root: Path, repo: str | None = None
    ) -> None:
        self.project_root = project_root
        self.repo = repo
        self.config = load_config(project_root)
        self.db = StateDB(project_root / ".conductor" / "state.db")
        self.pool = AgentPool(
            max_sessions=self.config.pool.max_sessions,
            idle_ttl_seconds=self.config.pool.idle_ttl_seconds,
            default_model=self.config.pool.default_model,
        )
        self._shutdown_event = threading.Event()
        self._dispatches: dict[int, str] = {}
        self._dispatch_lock = threading.Lock()
        self._target_milestone: str | None = None
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.pool.max_sessions,
        )
        self._futures: dict[int, concurrent.futures.Future[None]] = {}

    @property
    def _shutdown(self) -> bool:
        return self._shutdown_event.is_set()

    @_shutdown.setter
    def _shutdown(self, value: bool) -> None:
        if value:
            self._shutdown_event.set()
        else:
            self._shutdown_event.clear()

    def run(self, poll_interval: float = 10.0) -> None:
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        console = Console()

        self._target_milestone = _resolve_target_milestone(self.repo)
        if self._target_milestone:
            console.print(
                f"[bold]Conductor starting (milestone: {self._target_milestone})[/bold]"
            )
        else:
            console.print("[bold]Conductor starting (no milestone filter)[/bold]")

        issues = _list_open_issues(self.repo)
        dag = self._refresh_dag(issues)
        self._sync_dag_to_db(dag, issues)

        with Live(
            self._render_dashboard(dag),
            console=console,
            refresh_per_second=1,
        ) as live:
            while not self._shutdown:
                self._tick(dag)
                self._reap_futures(dag)
                live.update(self._render_dashboard(dag))
                flush_sync_queue(self.db, self.repo)
                self.pool.drain_idle()

                if self._shutdown_event.wait(timeout=poll_interval):
                    break

                self._target_milestone = _resolve_target_milestone(self.repo)
                issues = _list_open_issues(self.repo)
                dag = self._refresh_dag(issues)
                self._sync_dag_to_db(dag, issues)

        self._cleanup()
        console.print("[bold]Conductor stopped.[/bold]")

    def _tick(self, dag: DAG) -> None:
        completed = self._completed_issues()
        ready = dag.ready_issues(completed)

        for node in ready:
            if node.phase in ("merged", "pr", "closed"):
                continue
            with self._dispatch_lock:
                if node.number in self._dispatches:
                    continue
            current_phase = (
                node.phase if node.phase in PHASE_ORDER else "design"
            )
            self._submit_dispatch(node, current_phase)

    def _submit_dispatch(self, node: DAGNode, phase: str) -> None:
        with self._dispatch_lock:
            if node.number in self._dispatches:
                return
            self._dispatches[node.number] = f"agent-{node.number}"
        future = self._executor.submit(self._dispatch_issue, node, phase)
        self._futures[node.number] = future

    def _reap_futures(self, dag: DAG) -> None:
        done = [n for n, f in self._futures.items() if f.done()]
        for number in done:
            future = self._futures.pop(number)
            exc = future.exception()
            if exc:
                logger.error("Dispatch #%d raised: %s", number, exc)
            db_issue = self.db.get_issue(number)
            if db_issue:
                node = dag.get_node(number)
                if node:
                    node.phase = db_issue["phase"]

    def _refresh_dag(self, issues: list[dict] | None = None) -> DAG:
        if issues is None:
            issues = _list_open_issues(self.repo)
        filtered = [
            i for i in issues
            if not _is_epic(i["title"])
            and (
                self._target_milestone is None
                or i.get("milestone") == self._target_milestone
            )
        ]
        return build_dag_from_issues(filtered)

    def _sync_dag_to_db(
        self, dag: DAG, issues: list[dict] | None = None
    ) -> None:
        """Sync DAG nodes to DB, preserving local phase and caching body/labels."""
        if issues is None:
            issues = []
        issues_by_number = {i["number"]: i for i in issues}
        for node in dag.nodes:
            issue_data = issues_by_number.get(node.number, {})
            body = issue_data.get("body", "")
            labels = json.dumps(issue_data.get("labels", []))
            existing = self.db.get_issue(node.number)
            if existing is None:
                self.db.upsert_issue(
                    number=node.number,
                    title=node.title,
                    phase="pending",
                    milestone=self._target_milestone,
                    blocked_by=json.dumps(node.blocked_by),
                    body=body,
                    labels=labels,
                )
            else:
                self.db.update_issue(
                    node.number,
                    title=node.title,
                    milestone=self._target_milestone,
                    blocked_by=json.dumps(node.blocked_by),
                    body=body,
                    labels=labels,
                )
            db_issue = self.db.get_issue(node.number)
            if db_issue:
                node.phase = db_issue["phase"]

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
            shutdown_event=self._shutdown_event,
        )
        try:
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
        finally:
            with self._dispatch_lock:
                self._dispatches.pop(node.number, None)

    def _render_dashboard(self, dag: DAG | None = None) -> Table:
        ms_label = self._target_milestone or "all"
        table = Table(title=f"Conductor Dashboard (milestone: {ms_label})")
        table.add_column("#", style="cyan", justify="right")
        table.add_column("Title", max_width=50)
        table.add_column("Phase", style="green")
        table.add_column("Agent", style="magenta")
        table.add_column("Blocked By", style="dim")

        nodes = dag.nodes if dag else []
        completed = self._completed_issues()

        for node in nodes:
            blocked_str = (
                ", ".join(f"#{b}" for b in node.blocked_by)
                if node.blocked_by
                else "-"
            )
            is_ready = not dag.is_blocked(node.number, completed)
            phase_style = "bold green" if is_ready else "dim"
            with self._dispatch_lock:
                agent = self._dispatches.get(node.number, "-")
            table.add_row(
                str(node.number),
                node.title,
                f"[{phase_style}]{node.phase}[/]",
                agent,
                blocked_str,
            )

        busy = sum(1 for s in self.pool._sessions.values() if s.busy)
        idle = sum(1 for s in self.pool._sessions.values() if not s.busy)
        total = self.pool._max_sessions
        table.caption = f"Pool: {busy}/{total} busy | Idle: {idle}"
        return table

    def _handle_shutdown(self, signum: int, frame: object) -> None:
        self._shutdown_event.set()

    def _cleanup(self) -> None:
        self._executor.shutdown(wait=False)
        self.pool.shutdown()
        self.db.close()
