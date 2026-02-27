from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from conductor.config import ConductorConfig, HealthConfig, PoolConfig
from conductor.dispatch import (
    _build_prompt,
    _read_output,
    _validate_output,
    _write_input,
    dispatch_step,
)
from conductor.state_db import StateDB


class SampleOutput(BaseModel):
    answer: str
    score: int


class SampleInput(BaseModel):
    question: str


def _make_config(steps: dict[str, str] | None = None) -> ConductorConfig:
    return ConductorConfig(
        pool=PoolConfig(
            max_sessions=2, idle_ttl_seconds=60, default_model="sonnet-4.5"
        ),
        models={"standard": "sonnet-4.5", "autonomous": "opus-4.6"},
        timeouts={"design": 300},
        health=HealthConfig(
            poll_interval_seconds=5,
            idle_threshold_seconds=30,
            max_nudges=2,
            max_retries=1,
        ),
        steps=steps or {"2.2": "autonomous", "1.1": "python"},
    )


@pytest.fixture()
def db(tmp_path: Path) -> StateDB:
    state = StateDB(tmp_path / "state.db")
    state.upsert_issue(42, "Test issue")
    return state


@pytest.fixture()
def mock_pool() -> MagicMock:
    pool = MagicMock()
    session = MagicMock()
    session.name = "conductor-agent-0"
    pool.acquire.return_value = session
    return pool


class TestWriteInput:
    def test_creates_file(self, tmp_path: Path) -> None:
        data = SampleInput(question="What is 2+2?")
        path = _write_input(tmp_path, 42, "2.2", data)
        assert path.exists()
        assert path.name == "2.2.input.json"
        assert path.parent.name == "42"

    def test_content_is_valid_json(self, tmp_path: Path) -> None:
        data = SampleInput(question="What is 2+2?")
        path = _write_input(tmp_path, 42, "2.2", data)
        parsed = json.loads(path.read_text())
        assert parsed["question"] == "What is 2+2?"

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        data = SampleInput(question="test")
        path = _write_input(tmp_path, 99, "3.1", data)
        assert path.parent.is_dir()


class TestReadOutput:
    def test_returns_none_when_missing(self, tmp_path: Path) -> None:
        result = _read_output(tmp_path, 42, "2.2")
        assert result is None

    def test_returns_content_when_present(self, tmp_path: Path) -> None:
        step_dir = tmp_path / ".conductor" / "steps" / "42"
        step_dir.mkdir(parents=True)
        output_file = step_dir / "2.2.output.json"
        output_file.write_text('{"answer": "4", "score": 10}')
        result = _read_output(tmp_path, 42, "2.2")
        assert result == '{"answer": "4", "score": 10}'


class TestValidateOutput:
    def test_valid_json(self) -> None:
        raw = '{"answer": "4", "score": 10}'
        result = _validate_output(raw, SampleOutput)
        assert result.answer == "4"
        assert result.score == 10

    def test_invalid_json_raises(self) -> None:
        from pydantic import ValidationError

        raw = '{"answer": "4"}'
        with pytest.raises(ValidationError):
            _validate_output(raw, SampleOutput)

    def test_malformed_json_raises(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            _validate_output("not json at all", SampleOutput)


class TestBuildPrompt:
    def test_produces_expected_string(self) -> None:
        inp = Path("/project/.conductor/steps/42/2.2.input.json")
        out = Path("/project/.conductor/steps/42/2.2.output.json")
        prompt = _build_prompt(inp, out)
        assert str(inp) in prompt
        assert str(out) in prompt
        assert "Read" in prompt
        assert "JSON" in prompt


class TestDispatchStepRejectsPython:
    def test_raises_for_python_step(
        self, tmp_path: Path, db: StateDB, mock_pool: MagicMock
    ) -> None:
        config = _make_config()
        with pytest.raises(ValueError, match="python step"):
            dispatch_step(
                issue_number=42,
                step_id="1.1",
                input_data=SampleInput(question="test"),
                output_type=SampleOutput,
                config=config,
                pool=mock_pool,
                db=db,
                project_root=tmp_path,
                worktree=tmp_path / "wt",
            )


class TestDispatchStepHappyPath:
    def test_success(
        self, tmp_path: Path, db: StateDB, mock_pool: MagicMock
    ) -> None:
        config = _make_config()
        output_dir = tmp_path / ".conductor" / "steps" / "42"
        output_dir.mkdir(parents=True)
        output_file = output_dir / "2.2.output.json"
        output_file.write_text('{"answer": "hello", "score": 5}')

        result = dispatch_step(
            issue_number=42,
            step_id="2.2",
            input_data=SampleInput(question="greet"),
            output_type=SampleOutput,
            config=config,
            pool=mock_pool,
            db=db,
            project_root=tmp_path,
            worktree=tmp_path / "wt",
            poll_interval=0.01,
        )

        assert result.success is True
        assert result.output is not None
        assert result.output.answer == "hello"
        assert result.output.score == 5

        mock_pool.acquire.assert_called_once()
        mock_pool.clear_context.assert_called_once()
        mock_pool.send.assert_called_once()
        mock_pool.release.assert_called_once()

        steps = db.get_steps(42)
        assert len(steps) == 1
        assert steps[0]["status"] == "completed"


class TestDispatchStepTimeout:
    def test_timeout_returns_failure(
        self, tmp_path: Path, db: StateDB, mock_pool: MagicMock
    ) -> None:
        config = _make_config()

        result = dispatch_step(
            issue_number=42,
            step_id="2.2",
            input_data=SampleInput(question="slow"),
            output_type=SampleOutput,
            config=config,
            pool=mock_pool,
            db=db,
            project_root=tmp_path,
            worktree=tmp_path / "wt",
            poll_interval=0.01,
            timeout=0.03,
        )

        assert result.success is False
        assert result.error == "timeout"
        mock_pool.release.assert_called_once()

        steps = db.get_steps(42)
        assert steps[0]["status"] == "failed"
        assert steps[0]["error"] == "timeout"


class TestDispatchStepValidationRetry:
    def test_retries_then_succeeds(
        self, tmp_path: Path, db: StateDB, mock_pool: MagicMock
    ) -> None:
        config = _make_config()
        output_dir = tmp_path / ".conductor" / "steps" / "42"
        output_dir.mkdir(parents=True)
        output_file = output_dir / "2.2.output.json"

        call_count = 0

        def fake_send(session: object, text: str) -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                output_file.write_text('{"answer": "fixed", "score": 1}')

        mock_pool.send.side_effect = fake_send

        output_file.write_text('{"answer": "no score field whoops"}')

        result = dispatch_step(
            issue_number=42,
            step_id="2.2",
            input_data=SampleInput(question="retry me"),
            output_type=SampleOutput,
            config=config,
            pool=mock_pool,
            db=db,
            project_root=tmp_path,
            worktree=tmp_path / "wt",
            poll_interval=0.01,
            max_validation_retries=2,
        )

        assert result.success is True
        assert result.output is not None
        assert result.output.answer == "fixed"
        mock_pool.release.assert_called_once()

        steps = db.get_steps(42)
        assert steps[0]["status"] == "completed"


class TestDispatchShutdownEvent:
    def test_shutdown_cancels_dispatch(
        self, tmp_path: Path, db: StateDB, mock_pool: MagicMock
    ) -> None:
        config = _make_config()
        shutdown = threading.Event()
        shutdown.set()

        result = dispatch_step(
            issue_number=42,
            step_id="2.2",
            input_data=SampleInput(question="test"),
            output_type=SampleOutput,
            config=config,
            pool=mock_pool,
            db=db,
            project_root=tmp_path,
            worktree=tmp_path / "wt",
            poll_interval=0.01,
            shutdown_event=shutdown,
        )

        assert result.success is False
        assert result.error == "shutdown"
        steps = db.get_steps(42)
        assert steps[0]["status"] == "cancelled"


class TestDispatchTimeout:
    def test_timeout_returns_failure(
        self, tmp_path: Path, db: StateDB, mock_pool: MagicMock
    ) -> None:
        config = _make_config()

        result = dispatch_step(
            issue_number=42,
            step_id="2.2",
            input_data=SampleInput(question="test"),
            output_type=SampleOutput,
            config=config,
            pool=mock_pool,
            db=db,
            project_root=tmp_path,
            worktree=tmp_path / "wt",
            poll_interval=0.01,
            timeout=0.0,
        )

        assert result.success is False
        assert result.error == "timeout"
        steps = db.get_steps(42)
        assert steps[0]["status"] == "failed"
