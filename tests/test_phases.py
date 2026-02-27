from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from conductor.config import ConductorConfig, HealthConfig, PoolConfig
from conductor.dispatch import StepResult
from conductor.phases import (
    PHASE_ORDER,
    PhaseContext,
    PhaseResult,
    _dispatch_swarm,
    next_phase,
    run_all_phases,
    run_phase,
)
from conductor.state_db import StateDB


def _make_config() -> ConductorConfig:
    return ConductorConfig(
        pool=PoolConfig(
            max_sessions=2,
            idle_ttl_seconds=60,
            default_model="sonnet-4.5",
        ),
        models={"standard": "sonnet-4.5", "autonomous": "opus-4.6"},
        timeouts={"design": 300},
        health=HealthConfig(
            poll_interval_seconds=5,
            idle_threshold_seconds=30,
            max_nudges=2,
            max_retries=1,
        ),
        steps={"1.2": "autonomous", "2.2": "autonomous", "1.1": "python"},
    )


@pytest.fixture()
def db(tmp_path: Path) -> StateDB:
    state = StateDB(tmp_path / "state.db")
    state.upsert_issue(
        42, "Test issue", body="Implement feature X", labels=json.dumps([])
    )
    return state


@pytest.fixture()
def ctx(tmp_path: Path, db: StateDB) -> PhaseContext:
    return PhaseContext(
        issue_number=42,
        config=_make_config(),
        pool=MagicMock(),
        db=db,
        project_root=tmp_path,
        worktree=tmp_path / "wt",
        repo="owner/repo",
    )


class TestPhaseOrder:
    def test_has_seven_phases(self) -> None:
        assert len(PHASE_ORDER) == 7

    def test_correct_phases(self) -> None:
        assert PHASE_ORDER == [
            "design",
            "plan",
            "architect",
            "test",
            "implement",
            "verify",
            "pr",
        ]


class TestNextPhase:
    def test_design_to_plan(self) -> None:
        assert next_phase("design") == "plan"

    def test_plan_to_architect(self) -> None:
        assert next_phase("plan") == "architect"

    def test_architect_to_test(self) -> None:
        assert next_phase("architect") == "test"

    def test_test_to_implement(self) -> None:
        assert next_phase("test") == "implement"

    def test_implement_to_verify(self) -> None:
        assert next_phase("implement") == "verify"

    def test_verify_to_pr(self) -> None:
        assert next_phase("verify") == "pr"

    def test_pr_is_last(self) -> None:
        assert next_phase("pr") is None

    def test_unknown_phase_returns_none(self) -> None:
        assert next_phase("nonexistent") is None


class TestRunPhaseUnknown:
    def test_unknown_phase_returns_failure(self, ctx: PhaseContext) -> None:
        result = run_phase(ctx, "bogus")
        assert result.success is False
        assert result.error == "Unknown phase: bogus"
        assert result.phase == "bogus"


class TestRunPhaseDispatch:
    def test_dispatches_to_correct_handler(
        self, ctx: PhaseContext
    ) -> None:
        with patch(
            "conductor.phases._run_design"
        ) as mock_design:
            mock_design.return_value = PhaseResult("design", True)
            result = run_phase(ctx, "design")
            mock_design.assert_called_once_with(ctx)
            assert result.success is True

    def test_dispatches_plan(self, ctx: PhaseContext) -> None:
        with patch("conductor.phases._run_plan") as mock_plan:
            mock_plan.return_value = PhaseResult("plan", True)
            result = run_phase(ctx, "plan")
            mock_plan.assert_called_once_with(ctx)
            assert result.phase == "plan"

    def test_dispatches_pr(self, ctx: PhaseContext) -> None:
        with patch("conductor.phases._run_pr") as mock_pr:
            mock_pr.return_value = PhaseResult("pr", True)
            result = run_phase(ctx, "pr")
            mock_pr.assert_called_once_with(ctx)
            assert result.phase == "pr"


class TestRunAllPhases:
    def test_runs_all_when_all_succeed(
        self, ctx: PhaseContext
    ) -> None:
        with patch("conductor.phases.run_phase") as mock_run:
            mock_run.side_effect = [
                PhaseResult(p, True) for p in PHASE_ORDER
            ]
            results = run_all_phases(ctx)
            assert len(results) == 7
            assert all(r.success for r in results)
            assert [r.phase for r in results] == PHASE_ORDER

    def test_stops_on_first_failure(
        self, ctx: PhaseContext
    ) -> None:
        with patch("conductor.phases.run_phase") as mock_run:
            mock_run.side_effect = [
                PhaseResult("design", True),
                PhaseResult("plan", True),
                PhaseResult("architect", False, error="boom"),
            ]
            results = run_all_phases(ctx)
            assert len(results) == 3
            assert results[0].success is True
            assert results[1].success is True
            assert results[2].success is False
            assert results[2].error == "boom"

    def test_starts_from_given_phase(
        self, ctx: PhaseContext
    ) -> None:
        with patch("conductor.phases.run_phase") as mock_run:
            mock_run.side_effect = [
                PhaseResult(p, True)
                for p in PHASE_ORDER[3:]
            ]
            results = run_all_phases(ctx, start_phase="test")
            assert len(results) == 4
            assert [r.phase for r in results] == [
                "test",
                "implement",
                "verify",
                "pr",
            ]

    def test_empty_when_start_phase_not_found(
        self, ctx: PhaseContext
    ) -> None:
        results = run_all_phases(ctx, start_phase="nonexistent")
        assert results == []


class TestDesignPhaseHappyPath:
    @patch("conductor.phases.add_label")
    @patch("conductor.phases.post_comment")
    @patch("conductor.phases.dispatch_step")
    def test_design_succeeds(
        self,
        mock_dispatch: MagicMock,
        mock_comment: MagicMock,
        mock_label: MagicMock,
        ctx: PhaseContext,
    ) -> None:
        mock_dispatch.return_value = StepResult(
            success=True, output=MagicMock()
        )

        result = run_phase(ctx, "design")

        assert result.success is True
        assert result.phase == "design"
        mock_dispatch.assert_called_once()
        mock_comment.assert_called_once()
        mock_label.assert_called_once_with(
            42, "phase:design", repo="owner/repo"
        )

        issue = ctx.db.get_issue(42)
        assert issue is not None
        assert issue["phase"] == "design"


class TestDesignPhaseFailure:
    @patch("conductor.phases.dispatch_step")
    def test_design_dispatch_failure(
        self,
        mock_dispatch: MagicMock,
        ctx: PhaseContext,
    ) -> None:
        mock_dispatch.return_value = StepResult(
            success=False, error="agent died"
        )

        result = run_phase(ctx, "design")
        assert result.success is False
        assert result.error == "agent died"


class TestDesignPhaseException:
    @patch("conductor.phases.dispatch_step")
    def test_design_catches_exception(
        self,
        mock_dispatch: MagicMock,
        ctx: PhaseContext,
    ) -> None:
        mock_dispatch.side_effect = RuntimeError("dispatch failed")

        result = run_phase(ctx, "design")
        assert result.success is False
        assert "dispatch failed" in (result.error or "")


class TestDispatchSwarm:
    @patch("conductor.phases.dispatch_step")
    def test_dispatches_all_inputs(
        self,
        mock_dispatch: MagicMock,
        ctx: PhaseContext,
    ) -> None:
        mock_dispatch.return_value = StepResult(
            success=True, output=MagicMock()
        )
        inputs: list[tuple[str, object]] = [
            ("1", MagicMock()),
            ("2", MagicMock()),
            ("3", MagicMock()),
        ]
        results = _dispatch_swarm(ctx, "4.2", inputs, MagicMock)
        assert len(results) == 3
        assert all(r.success for r in results)
        assert mock_dispatch.call_count == 3

    @patch("conductor.phases.dispatch_step")
    def test_collects_failures(
        self,
        mock_dispatch: MagicMock,
        ctx: PhaseContext,
    ) -> None:
        mock_dispatch.side_effect = [
            StepResult(success=True, output=MagicMock()),
            StepResult(success=False, error="failed"),
        ]
        inputs: list[tuple[str, object]] = [
            ("1", MagicMock()),
            ("2", MagicMock()),
        ]
        results = _dispatch_swarm(ctx, "5.2", inputs, MagicMock)
        assert len(results) == 2
        failures = [r for r in results if not r.success]
        assert len(failures) == 1


class TestPhaseResultAttributes:
    def test_success_result(self) -> None:
        r = PhaseResult("design", True)
        assert r.phase == "design"
        assert r.success is True
        assert r.error is None

    def test_failure_result(self) -> None:
        r = PhaseResult("plan", False, error="timeout")
        assert r.phase == "plan"
        assert r.success is False
        assert r.error == "timeout"
