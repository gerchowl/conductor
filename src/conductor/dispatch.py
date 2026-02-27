from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from pydantic import BaseModel, ValidationError

from conductor.config import ConductorConfig, _resolve_step_tier, resolve_step_model
from conductor.pool import AgentPool, AgentSession
from conductor.state_db import StateDB


class StepResult:
    """Result of a dispatched step."""

    def __init__(
        self,
        success: bool,
        output: BaseModel | None = None,
        error: str | None = None,
    ) -> None:
        self.success = success
        self.output = output
        self.error = error


def _write_input(
    project_root: Path, issue_number: int, step_id: str, data: BaseModel
) -> Path:
    """Write input JSON to .conductor/steps/{issue}/{step}.input.json"""
    step_dir = project_root / ".conductor" / "steps" / str(issue_number)
    step_dir.mkdir(parents=True, exist_ok=True)
    path = step_dir / f"{step_id}.input.json"
    path.write_text(data.model_dump_json(indent=2))
    return path


def _read_output(
    project_root: Path, issue_number: int, step_id: str
) -> str | None:
    """Read output JSON if file exists, return None if not yet written."""
    path = (
        project_root
        / ".conductor"
        / "steps"
        / str(issue_number)
        / f"{step_id}.output.json"
    )
    if not path.is_file():
        return None
    return path.read_text()


def _validate_output[T: BaseModel](raw_json: str, output_type: type[T]) -> T:
    """Parse and validate output JSON against Pydantic model. Raises ValidationError."""
    data = json.loads(raw_json)
    return output_type.model_validate(data)


def _build_prompt(input_path: Path, output_path: Path) -> str:
    """Build the prompt text sent to the agent.

    The input file contains a JSON object describing the task.
    The agent must write a valid JSON result matching the expected schema.
    """
    return (
        f"Read the task specification at {input_path}. "
        f"It contains a JSON object with the issue context, phase, and requirements. "
        f"Complete the task described in the specification. "
        f"Write ONLY valid JSON output (no markdown, no commentary) to {output_path}. "
        f"The output must match the schema expected by the conductor pipeline."
    )


def dispatch_step[T: BaseModel](
    issue_number: int,
    step_id: str,
    input_data: BaseModel,
    output_type: type[T],
    config: ConductorConfig,
    pool: AgentPool,
    db: StateDB,
    project_root: Path,
    worktree: Path,
    max_validation_retries: int = 2,
    poll_interval: float = 2.0,
    timeout: float | None = None,
    shutdown_event: threading.Event | None = None,
) -> StepResult:
    """Dispatch a single step to an agent and return validated output."""
    tier = _resolve_step_tier(config.steps, step_id)
    if tier == "python":
        msg = f"Step {step_id} is a python step and cannot be dispatched to an agent"
        raise ValueError(msg)
    model_id = resolve_step_model(config, step_id)

    step_row_id = db.insert_step(issue_number, step_id, model_id)
    db.update_step(step_row_id, status="dispatched")

    input_path = _write_input(project_root, issue_number, step_id, input_data)
    output_path = (
        project_root
        / ".conductor"
        / "steps"
        / str(issue_number)
        / f"{step_id}.output.json"
    )

    session: AgentSession | None = None
    try:
        session = pool.acquire(worktree, model=model_id)
        pool.clear_context(session)

        prompt = _build_prompt(input_path, output_path)
        pool.send(session, prompt)

        deadline = time.monotonic() + timeout if timeout is not None else None
        retries_left = max_validation_retries

        while True:
            if shutdown_event is not None and shutdown_event.is_set():
                db.update_step(step_row_id, status="cancelled", error="shutdown")
                return StepResult(success=False, error="shutdown")

            if deadline is not None and time.monotonic() >= deadline:
                db.update_step(step_row_id, status="failed", error="timeout")
                return StepResult(success=False, error="timeout")

            raw = _read_output(project_root, issue_number, step_id)
            if raw is None:
                if shutdown_event is not None:
                    shutdown_event.wait(timeout=poll_interval)
                else:
                    time.sleep(poll_interval)
                continue

            try:
                output = _validate_output(raw, output_type)
            except (ValidationError, json.JSONDecodeError) as exc:
                if retries_left <= 0:
                    error_msg = f"Validation failed after retries: {exc}"
                    db.update_step(
                        step_row_id, status="failed", error=error_msg
                    )
                    return StepResult(success=False, error=error_msg)
                retries_left -= 1
                output_path.unlink(missing_ok=True)
                retry_prompt = (
                    f"Your output had a validation error: {exc}. "
                    f"Please fix and rewrite {output_path}."
                )
                pool.send(session, retry_prompt)
                continue

            db.update_step(step_row_id, status="completed")
            return StepResult(success=True, output=output)
    finally:
        if session is not None:
            pool.release(session)
