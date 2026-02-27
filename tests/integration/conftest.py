from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from conductor.config import ConductorConfig, HealthConfig, PoolConfig
from conductor.gh_sync import IssueData
from conductor.phases import PhaseContext
from conductor.state_db import StateDB


@pytest.fixture()
def project_root(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    return root


@pytest.fixture()
def db(project_root: Path) -> StateDB:
    return StateDB(project_root / ".conductor" / "state.db")


@pytest.fixture()
def config() -> ConductorConfig:
    return ConductorConfig(
        pool=PoolConfig(
            max_sessions=3, idle_ttl_seconds=60, default_model="sonnet-4.5"
        ),
        models={
            "standard": "sonnet-4.5",
            "autonomous": "opus-4.6",
            "lightweight": "composer-1.5",
        },
        timeouts={
            "design": 300,
            "plan": 180,
            "architect": 300,
            "test": 120,
            "implement": 180,
            "verify": 60,
            "pr": 120,
        },
        health=HealthConfig(
            poll_interval_seconds=5,
            idle_threshold_seconds=30,
            max_nudges=2,
            max_retries=1,
        ),
        steps={
            "1.1": "python",
            "1.2": "autonomous",
            "1.3": "python",
            "2.1": "python",
            "2.2": "autonomous",
            "2.3": "python",
            "3.1": "python",
            "3.2": "autonomous",
            "3.3": "autonomous",
            "3.4": "python",
            "4.1": "python",
            "4.2.*": "standard",
            "4.3": "python",
            "5.1": "python",
            "5.2.*": "standard",
            "5.3": "python",
            "5.4": "standard",
            "6.1": "python",
            "6.2": "standard",
            "6.3": "python",
            "7.1": "python",
            "7.2": "standard",
            "7.3": "python",
            "7.4": "python",
        },
    )


@pytest.fixture()
def mock_pool() -> MagicMock:
    pool = MagicMock()
    session = MagicMock()
    session.name = "conductor-agent-0"
    pool.acquire.return_value = session
    return pool


@pytest.fixture()
def phase_ctx(
    project_root: Path, db: StateDB, config: ConductorConfig, mock_pool: MagicMock
) -> PhaseContext:
    wt = project_root / "worktrees" / "issue-42"
    wt.mkdir(parents=True)
    return PhaseContext(
        issue_number=42,
        config=config,
        pool=mock_pool,
        db=db,
        project_root=project_root,
        worktree=wt,
        repo="owner/repo",
    )


def make_issue_data(
    number: int = 42,
    title: str = "Test issue",
    body: str = "Implement feature X",
    labels: list[str] | None = None,
    state: str = "OPEN",
) -> IssueData:
    return IssueData(
        number=number,
        title=title,
        body=body,
        labels=labels or [],
        state=state,
    )


def write_output_file(
    project_root: Path, issue_number: int, step_id: str, data: dict
) -> Path:
    step_dir = project_root / ".conductor" / "steps" / str(issue_number)
    step_dir.mkdir(parents=True, exist_ok=True)
    path = step_dir / f"{step_id}.output.json"
    path.write_text(json.dumps(data))
    return path
