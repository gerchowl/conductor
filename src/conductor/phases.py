from __future__ import annotations

import concurrent.futures
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from conductor.config import ConductorConfig
from conductor.dispatch import StepResult, dispatch_step
from conductor.gh_sync import add_label, post_comment, read_issue
from conductor.models import (
    FileOutput,
    IssueContext,
    StubManifest,
    TestMatrix,
)
from conductor.pool import AgentPool
from conductor.state_db import StateDB

PHASE_ORDER = ["design", "plan", "architect", "test", "implement", "verify", "pr"]


@dataclass
class PhaseContext:
    """Shared context passed through phases."""

    issue_number: int
    config: ConductorConfig
    pool: AgentPool
    db: StateDB
    project_root: Path
    worktree: Path
    repo: str | None = None


class PhaseResult:
    def __init__(self, phase: str, success: bool, error: str | None = None) -> None:
        self.phase = phase
        self.success = success
        self.error = error


def next_phase(current: str) -> str | None:
    """Return the next phase after current, or None if last."""
    try:
        idx = PHASE_ORDER.index(current)
    except ValueError:
        return None
    if idx + 1 >= len(PHASE_ORDER):
        return None
    return PHASE_ORDER[idx + 1]


def run_phase(ctx: PhaseContext, phase: str) -> PhaseResult:
    """Run a specific phase for an issue."""
    handlers: dict[str, Callable[[PhaseContext], PhaseResult]] = {
        "design": _run_design,
        "plan": _run_plan,
        "architect": _run_architect,
        "test": _run_test,
        "implement": _run_implement,
        "verify": _run_verify,
        "pr": _run_pr,
    }
    handler = handlers.get(phase)
    if handler is None:
        return PhaseResult(phase=phase, success=False, error=f"Unknown phase: {phase}")
    return handler(ctx)


def run_all_phases(ctx: PhaseContext, start_phase: str = "design") -> list[PhaseResult]:
    """Run all phases sequentially from start_phase. Stop on first failure."""
    results: list[PhaseResult] = []
    started = False
    for phase in PHASE_ORDER:
        if phase == start_phase:
            started = True
        if not started:
            continue
        result = run_phase(ctx, phase)
        results.append(result)
        if not result.success:
            break
    return results


# ---------------------------------------------------------------------------
# Swarm helper
# ---------------------------------------------------------------------------


def _dispatch_swarm(
    ctx: PhaseContext,
    step_prefix: str,
    inputs: list[tuple[str, object]],
    output_type: type,
) -> list[StepResult]:
    """Dispatch multiple agent steps concurrently and collect results.

    Each entry in *inputs* is ``(sub_id, input_model)`` producing step IDs
    like ``"{step_prefix}.{sub_id}"``.
    """
    results: list[StepResult] = []

    def _run_one(sub_id: str, input_data: object) -> StepResult:
        step_id = f"{step_prefix}.{sub_id}"
        return dispatch_step(
            issue_number=ctx.issue_number,
            step_id=step_id,
            input_data=input_data,  # type: ignore[arg-type]
            output_type=output_type,
            config=ctx.config,
            pool=ctx.pool,
            db=ctx.db,
            project_root=ctx.project_root,
            worktree=ctx.worktree,
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(len(inputs), 1),
    ) as executor:
        futures = {
            executor.submit(_run_one, sub_id, data): sub_id for sub_id, data in inputs
        }
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    return results


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_issue_context(ctx: PhaseContext, phase: str) -> IssueContext:
    issue = read_issue(ctx.issue_number, repo=ctx.repo)
    return IssueContext(
        number=issue.number,
        title=issue.title,
        body=issue.body,
        labels=issue.labels,
        phase=phase,
        blocked_by=[],
        branch="",
    )


def _phase_dispatch(
    ctx: PhaseContext,
    step_id: str,
    input_data: object,
    output_type: type,
) -> StepResult:
    return dispatch_step(
        issue_number=ctx.issue_number,
        step_id=step_id,
        input_data=input_data,  # type: ignore[arg-type]
        output_type=output_type,
        config=ctx.config,
        pool=ctx.pool,
        db=ctx.db,
        project_root=ctx.project_root,
        worktree=ctx.worktree,
    )


# ---------------------------------------------------------------------------
# Phase handlers
# ---------------------------------------------------------------------------


def _run_design(ctx: PhaseContext) -> PhaseResult:
    ctx.db.update_issue(ctx.issue_number, phase="design")
    try:
        issue_ctx = _load_issue_context(ctx, "design")
        result = _phase_dispatch(ctx, "1.2", issue_ctx, IssueContext)
        if not result.success:
            return PhaseResult("design", False, error=result.error)

        body = f"Design complete:\n{result.output}"
        post_comment(ctx.issue_number, body, repo=ctx.repo)
        add_label(ctx.issue_number, "phase:design", repo=ctx.repo)
        return PhaseResult("design", True)
    except Exception as exc:
        return PhaseResult("design", False, error=str(exc))


def _run_plan(ctx: PhaseContext) -> PhaseResult:
    ctx.db.update_issue(ctx.issue_number, phase="plan")
    try:
        issue_ctx = _load_issue_context(ctx, "plan")
        result = _phase_dispatch(ctx, "2.2", issue_ctx, IssueContext)
        if not result.success:
            return PhaseResult("plan", False, error=result.error)

        body = f"Plan complete:\n{result.output}"
        post_comment(ctx.issue_number, body, repo=ctx.repo)
        add_label(ctx.issue_number, "phase:plan", repo=ctx.repo)
        return PhaseResult("plan", True)
    except Exception as exc:
        return PhaseResult("plan", False, error=str(exc))


def _run_architect(ctx: PhaseContext) -> PhaseResult:
    ctx.db.update_issue(ctx.issue_number, phase="architect")
    try:
        issue_ctx = _load_issue_context(ctx, "architect")

        matrix_result = _phase_dispatch(ctx, "3.2", issue_ctx, TestMatrix)
        if not matrix_result.success:
            return PhaseResult("architect", False, error=matrix_result.error)

        stub_result = _phase_dispatch(ctx, "3.3", matrix_result.output, StubManifest)
        if not stub_result.success:
            return PhaseResult("architect", False, error=stub_result.error)

        add_label(ctx.issue_number, "phase:architect", repo=ctx.repo)
        return PhaseResult("architect", True)
    except Exception as exc:
        return PhaseResult("architect", False, error=str(exc))


def _run_test(ctx: PhaseContext) -> PhaseResult:
    ctx.db.update_issue(ctx.issue_number, phase="test")
    try:
        issue_ctx = _load_issue_context(ctx, "test")

        swarm_inputs: list[tuple[str, object]] = [
            ("1", issue_ctx),
            ("2", issue_ctx),
        ]
        swarm_results = _dispatch_swarm(ctx, "4.2", swarm_inputs, FileOutput)
        failures = [r for r in swarm_results if not r.success]
        if failures:
            errors = "; ".join(r.error or "unknown" for r in failures)
            return PhaseResult("test", False, error=errors)

        add_label(ctx.issue_number, "phase:test", repo=ctx.repo)
        return PhaseResult("test", True)
    except Exception as exc:
        return PhaseResult("test", False, error=str(exc))


def _run_implement(ctx: PhaseContext) -> PhaseResult:
    ctx.db.update_issue(ctx.issue_number, phase="implement")
    try:
        issue_ctx = _load_issue_context(ctx, "implement")

        swarm_inputs: list[tuple[str, object]] = [
            ("1", issue_ctx),
            ("2", issue_ctx),
        ]
        swarm_results = _dispatch_swarm(ctx, "5.2", swarm_inputs, FileOutput)
        failures = [r for r in swarm_results if not r.success]
        if failures:
            errors = "; ".join(r.error or "unknown" for r in failures)
            return PhaseResult("implement", False, error=errors)

        integrate = _phase_dispatch(ctx, "5.4", issue_ctx, FileOutput)
        if not integrate.success:
            return PhaseResult("implement", False, error=integrate.error)

        add_label(ctx.issue_number, "phase:implement", repo=ctx.repo)
        return PhaseResult("implement", True)
    except Exception as exc:
        return PhaseResult("implement", False, error=str(exc))


def _run_verify(ctx: PhaseContext) -> PhaseResult:
    ctx.db.update_issue(ctx.issue_number, phase="verify")
    try:
        issue_ctx = _load_issue_context(ctx, "verify")
        result = _phase_dispatch(ctx, "6.2", issue_ctx, FileOutput)
        if not result.success:
            return PhaseResult("verify", False, error=result.error)

        add_label(ctx.issue_number, "phase:verify", repo=ctx.repo)
        return PhaseResult("verify", True)
    except Exception as exc:
        return PhaseResult("verify", False, error=str(exc))


def _run_pr(ctx: PhaseContext) -> PhaseResult:
    ctx.db.update_issue(ctx.issue_number, phase="pr")
    try:
        issue_ctx = _load_issue_context(ctx, "pr")
        result = _phase_dispatch(ctx, "7.2", issue_ctx, FileOutput)
        if not result.success:
            return PhaseResult("pr", False, error=result.error)

        post_comment(ctx.issue_number, "PR created.", repo=ctx.repo)
        add_label(ctx.issue_number, "phase:pr", repo=ctx.repo)
        return PhaseResult("pr", True)
    except Exception as exc:
        return PhaseResult("pr", False, error=str(exc))
