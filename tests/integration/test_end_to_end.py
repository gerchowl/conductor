from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from conductor.config import (
    ConductorConfig,
    init_config,
    load_config,
)
from conductor.dag import DAG, build_dag_from_issues
from conductor.defaults import (
    HEALTH_DEFAULTS,
    MODEL_DEFAULTS,
    POOL_DEFAULTS,
    STEP_DEFAULTS,
    TIMEOUT_DEFAULTS,
)
from conductor.dispatch import StepResult, dispatch_step
from conductor.models import FileOutput, IssueContext
from conductor.phases import PHASE_ORDER, PhaseContext, run_all_phases, run_phase
from conductor.state_db import StateDB

from .conftest import make_issue_data

# ---------------------------------------------------------------------------
# 1. Single-issue happy path
# ---------------------------------------------------------------------------


class TestSingleIssueHappyPath:
    """Issue #42 flows through all 7 phases with mocked dispatch and gh_sync."""

    @patch("conductor.phases.add_label")
    @patch("conductor.phases.post_comment")
    @patch("conductor.phases.read_issue")
    @patch("conductor.phases.dispatch_step")
    def test_full_pipeline(
        self,
        mock_dispatch: MagicMock,
        mock_read: MagicMock,
        mock_comment: MagicMock,
        mock_label: MagicMock,
        phase_ctx: PhaseContext,
    ) -> None:
        phase_ctx.db.upsert_issue(42, "Test issue", phase="pending")

        mock_read.return_value = make_issue_data(number=42, title="Test issue")
        mock_dispatch.return_value = StepResult(success=True, output=MagicMock())

        results = run_all_phases(phase_ctx, start_phase="design")

        assert len(results) == 7
        assert all(r.success for r in results)
        assert [r.phase for r in results] == PHASE_ORDER

        issue = phase_ctx.db.get_issue(42)
        assert issue is not None
        assert issue["phase"] == "pr"

        label_calls = [call.args for call in mock_label.call_args_list]
        phase_labels = {args[1] for args in label_calls}
        for phase in PHASE_ORDER:
            assert f"phase:{phase}" in phase_labels

    @patch("conductor.phases.add_label")
    @patch("conductor.phases.post_comment")
    @patch("conductor.phases.read_issue")
    @patch("conductor.phases.dispatch_step")
    def test_db_state_updates_per_phase(
        self,
        mock_dispatch: MagicMock,
        mock_read: MagicMock,
        mock_comment: MagicMock,
        mock_label: MagicMock,
        phase_ctx: PhaseContext,
    ) -> None:
        phase_ctx.db.upsert_issue(42, "Test issue", phase="pending")
        mock_read.return_value = make_issue_data(number=42)
        mock_dispatch.return_value = StepResult(success=True, output=MagicMock())

        for phase in PHASE_ORDER:
            result = run_phase(phase_ctx, phase)
            assert result.success is True

            issue = phase_ctx.db.get_issue(42)
            assert issue is not None
            assert issue["phase"] == phase

    @patch("conductor.phases.add_label")
    @patch("conductor.phases.post_comment")
    @patch("conductor.phases.read_issue")
    @patch("conductor.phases.dispatch_step")
    def test_sync_queue_gets_entries(
        self,
        mock_dispatch: MagicMock,
        mock_read: MagicMock,
        mock_comment: MagicMock,
        mock_label: MagicMock,
        phase_ctx: PhaseContext,
    ) -> None:
        phase_ctx.db.upsert_issue(42, "Test issue")
        mock_read.return_value = make_issue_data(number=42)
        mock_dispatch.return_value = StepResult(success=True, output=MagicMock())

        run_all_phases(phase_ctx)

        phase_ctx.db.enqueue_sync(
            42, "label_add", json.dumps({"label": "phase:design"})
        )
        phase_ctx.db.enqueue_sync(
            42, "comment_post", json.dumps({"body": "Design complete"})
        )

        pending = phase_ctx.db.pending_syncs()
        assert len(pending) == 2
        types = {s["sync_type"] for s in pending}
        assert "label_add" in types
        assert "comment_post" in types

    @patch("conductor.phases.add_label")
    @patch("conductor.phases.post_comment")
    @patch("conductor.phases.read_issue")
    @patch("conductor.phases.dispatch_step")
    def test_stops_on_failure(
        self,
        mock_dispatch: MagicMock,
        mock_read: MagicMock,
        mock_comment: MagicMock,
        mock_label: MagicMock,
        phase_ctx: PhaseContext,
    ) -> None:
        phase_ctx.db.upsert_issue(42, "Test issue")
        mock_read.return_value = make_issue_data(number=42)

        call_count = 0

        def _dispatch_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return StepResult(success=True, output=MagicMock())
            return StepResult(success=False, error="agent crashed")

        mock_dispatch.side_effect = _dispatch_side_effect

        results = run_all_phases(phase_ctx)

        failed = [r for r in results if not r.success]
        assert len(failed) == 1
        assert len(results) < 7


# ---------------------------------------------------------------------------
# 2. Multi-issue DAG ordering
# ---------------------------------------------------------------------------


class TestMultiIssueDAGOrdering:
    """Three issues with dependency chain: A(1) → B(2) → C(3)."""

    def _build_dag(self) -> DAG:
        dag = DAG()
        dag.add_node(1, "Issue A")
        dag.add_node(2, "Issue B", blocked_by=[1])
        dag.add_node(3, "Issue C", blocked_by=[1, 2])
        return dag

    def test_initially_only_a_is_ready(self) -> None:
        dag = self._build_dag()
        ready = dag.ready_issues(completed=set())
        assert [n.number for n in ready] == [1]

    def test_after_a_completes_b_is_ready(self) -> None:
        dag = self._build_dag()
        ready = dag.ready_issues(completed={1})
        ready_numbers = [n.number for n in ready]
        assert 2 in ready_numbers
        assert 3 not in ready_numbers

    def test_after_a_and_b_complete_c_is_ready(self) -> None:
        dag = self._build_dag()
        ready = dag.ready_issues(completed={1, 2})
        ready_numbers = [n.number for n in ready]
        assert 3 in ready_numbers

    def test_execution_tiers_groups_correctly(self) -> None:
        dag = self._build_dag()
        tiers = dag.execution_tiers()
        assert len(tiers) == 3
        assert tiers[0] == [1]
        assert tiers[1] == [2]
        assert tiers[2] == [3]

    def test_topological_sort_respects_order(self) -> None:
        dag = self._build_dag()
        order = dag.topological_sort()
        assert order.index(1) < order.index(2)
        assert order.index(2) < order.index(3)

    def test_build_dag_from_issue_bodies(self) -> None:
        issues = [
            {"number": 10, "title": "A", "body": "", "labels": []},
            {"number": 20, "title": "B", "body": "Blocked by: #10", "labels": []},
            {
                "number": 30,
                "title": "C",
                "body": "Blocked by: #10, #20",
                "labels": ["phase:design"],
            },
        ]
        dag = build_dag_from_issues(issues)
        ready = dag.ready_issues(completed=set())
        assert [n.number for n in ready] == [10]

        tiers = dag.execution_tiers()
        assert len(tiers) == 3
        assert tiers[0] == [10]
        assert tiers[1] == [20]
        assert tiers[2] == [30]

    def test_parallel_tier_with_independent_issues(self) -> None:
        dag = DAG()
        dag.add_node(1, "A")
        dag.add_node(2, "B")
        dag.add_node(3, "C", blocked_by=[1, 2])
        tiers = dag.execution_tiers()
        assert len(tiers) == 2
        assert sorted(tiers[0]) == [1, 2]
        assert tiers[1] == [3]


# ---------------------------------------------------------------------------
# 3. Config round-trip
# ---------------------------------------------------------------------------


class TestConfigRoundTrip:
    def test_init_then_load_matches_defaults(self, tmp_path: Path) -> None:
        config_path = init_config(tmp_path)
        assert config_path.exists()
        assert config_path.name == "conductor.toml"

        cfg = load_config(tmp_path)

        assert cfg.pool.max_sessions == POOL_DEFAULTS["max_sessions"]
        assert cfg.pool.idle_ttl_seconds == POOL_DEFAULTS["idle_ttl_seconds"]
        assert cfg.pool.default_model == POOL_DEFAULTS["default_model"]

        for key, value in MODEL_DEFAULTS.items():
            assert cfg.models[key] == value

        for key, value in TIMEOUT_DEFAULTS.items():
            assert cfg.timeouts[key] == value

        expected_poll = HEALTH_DEFAULTS["poll_interval_seconds"]
        assert cfg.health.poll_interval_seconds == expected_poll
        expected_idle = HEALTH_DEFAULTS["idle_threshold_seconds"]
        assert cfg.health.idle_threshold_seconds == expected_idle
        assert cfg.health.max_nudges == HEALTH_DEFAULTS["max_nudges"]
        assert cfg.health.max_retries == HEALTH_DEFAULTS["max_retries"]

        for key, value in STEP_DEFAULTS.items():
            assert cfg.steps[key] == value

    def test_init_creates_backup_on_second_call(self, tmp_path: Path) -> None:
        init_config(tmp_path)
        init_config(tmp_path)
        backup = tmp_path / ".conductor" / "conductor.toml.bak"
        assert backup.exists()

    def test_load_without_file_returns_defaults(self, tmp_path: Path) -> None:
        cfg = load_config(tmp_path)
        assert cfg.pool.max_sessions == POOL_DEFAULTS["max_sessions"]
        assert cfg.models == dict(MODEL_DEFAULTS)
        assert cfg.timeouts == dict(TIMEOUT_DEFAULTS)

    def test_toml_content_is_valid(self, tmp_path: Path) -> None:
        config_path = init_config(tmp_path)
        import tomllib

        data = tomllib.loads(config_path.read_text())
        assert "pool" in data
        assert "model" in data
        assert "timeout" in data
        assert "health" in data
        assert "step" in data


# ---------------------------------------------------------------------------
# 4. State DB lifecycle
# ---------------------------------------------------------------------------


class TestStateDBLifecycle:
    def test_issue_crud(self, db: StateDB) -> None:
        db.upsert_issue(1, "First issue", phase="pending")
        db.upsert_issue(2, "Second issue", phase="pending")

        issue = db.get_issue(1)
        assert issue is not None
        assert issue["title"] == "First issue"
        assert issue["phase"] == "pending"

        db.update_issue(1, phase="design", current_step="1.2")
        issue = db.get_issue(1)
        assert issue["phase"] == "design"
        assert issue["current_step"] == "1.2"

        all_issues = db.list_issues()
        assert len(all_issues) == 2

        pending = db.list_issues(phase="pending")
        assert len(pending) == 1
        assert pending[0]["number"] == 2

    def test_step_tracking_through_phases(self, db: StateDB) -> None:
        db.upsert_issue(10, "Pipeline issue")

        step1_id = db.insert_step(10, "1.2", "autonomous")
        db.update_step(step1_id, status="dispatched")
        db.update_step(step1_id, status="completed", duration_ms=5000)

        step2_id = db.insert_step(10, "2.2", "autonomous")
        db.update_step(step2_id, status="dispatched")
        db.update_step(step2_id, status="failed", error="timeout")

        steps = db.get_steps(10)
        assert len(steps) == 2
        assert steps[0]["step"] == "1.2"
        assert steps[0]["status"] == "completed"
        assert steps[0]["duration_ms"] == 5000
        assert steps[1]["step"] == "2.2"
        assert steps[1]["status"] == "failed"
        assert steps[1]["error"] == "timeout"

    def test_gh_sync_queue_operations(self, db: StateDB) -> None:
        db.upsert_issue(5, "Sync test issue")

        sid1 = db.enqueue_sync(5, "label_add", json.dumps({"label": "phase:design"}))
        sid2 = db.enqueue_sync(5, "comment_post", json.dumps({"body": "Done"}))
        sid3 = db.enqueue_sync(5, "label_remove", json.dumps({"label": "phase:plan"}))

        pending = db.pending_syncs()
        assert len(pending) == 3

        db.mark_synced(sid1)
        db.mark_sync_failed(sid2)

        pending = db.pending_syncs()
        assert len(pending) == 1
        assert pending[0]["id"] == sid3

    def test_upsert_updates_existing(self, db: StateDB) -> None:
        db.upsert_issue(7, "Original title", phase="pending")
        db.upsert_issue(7, "Updated title", phase="design")

        issue = db.get_issue(7)
        assert issue is not None
        assert issue["title"] == "Updated title"
        assert issue["phase"] == "design"

        all_issues = db.list_issues()
        assert len(all_issues) == 1

    def test_step_status_transitions(self, db: StateDB) -> None:
        db.upsert_issue(20, "Transition test")
        step_id = db.insert_step(20, "3.2", "autonomous")

        step = db.get_steps(20)[0]
        assert step["status"] == "pending"

        db.update_step(
            step_id, status="dispatched", dispatched_at="2026-02-27T10:00:00"
        )
        step = db.get_steps(20)[0]
        assert step["status"] == "dispatched"
        assert step["dispatched_at"] == "2026-02-27T10:00:00"

        db.update_step(
            step_id,
            status="completed",
            completed_at="2026-02-27T10:05:00",
            duration_ms=300000,
        )
        step = db.get_steps(20)[0]
        assert step["status"] == "completed"
        assert step["completed_at"] == "2026-02-27T10:05:00"
        assert step["duration_ms"] == 300000


# ---------------------------------------------------------------------------
# 5. Dispatch validation retry
# ---------------------------------------------------------------------------


class TestDispatchValidationRetry:
    def test_invalid_then_valid_output(
        self, project_root: Path, config: ConductorConfig, mock_pool: MagicMock
    ) -> None:
        db = StateDB(project_root / ".conductor" / "retry.db")
        db.upsert_issue(42, "Retry test")

        output_dir = project_root / ".conductor" / "steps" / "42"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "6.2.output.json"
        output_file.write_text("NOT VALID JSON {{{")

        call_count = 0

        def fake_send(session: object, text: str) -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                output_file.write_text(
                    json.dumps({"file": "result.py", "content": "print('hello')"})
                )

        mock_pool.send.side_effect = fake_send

        result = dispatch_step(
            issue_number=42,
            step_id="6.2",
            input_data=IssueContext(
                number=42,
                title="Retry test",
                body="body",
                labels=[],
                phase="verify",
                blocked_by=[],
                branch="feature/42",
            ),
            output_type=FileOutput,
            config=config,
            pool=mock_pool,
            db=db,
            project_root=project_root,
            worktree=project_root / "wt",
            poll_interval=0.01,
            max_validation_retries=2,
        )

        assert result.success is True
        assert result.output is not None
        assert result.output.file == "result.py"

        steps = db.get_steps(42)
        assert len(steps) == 1
        assert steps[0]["status"] == "completed"

        assert call_count >= 2

    def test_retry_exhaustion_fails(
        self, project_root: Path, config: ConductorConfig, mock_pool: MagicMock
    ) -> None:
        db = StateDB(project_root / ".conductor" / "retry-exhaust.db")
        db.upsert_issue(42, "Exhaustion test")

        output_dir = project_root / ".conductor" / "steps" / "42"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "6.2.output.json"
        output_file.write_text("BAD JSON")

        def always_bad(session: object, text: str) -> None:
            output_file.write_text("STILL BAD JSON")

        mock_pool.send.side_effect = always_bad

        result = dispatch_step(
            issue_number=42,
            step_id="6.2",
            input_data=IssueContext(
                number=42,
                title="Exhaust test",
                body="body",
                labels=[],
                phase="verify",
                blocked_by=[],
                branch="feature/42",
            ),
            output_type=FileOutput,
            config=config,
            pool=mock_pool,
            db=db,
            project_root=project_root,
            worktree=project_root / "wt",
            poll_interval=0.01,
            max_validation_retries=1,
        )

        assert result.success is False
        assert "Validation failed" in (result.error or "")

        steps = db.get_steps(42)
        assert steps[0]["status"] == "failed"

    def test_schema_mismatch_then_correct(
        self, project_root: Path, config: ConductorConfig, mock_pool: MagicMock
    ) -> None:
        db = StateDB(project_root / ".conductor" / "schema-retry.db")
        db.upsert_issue(42, "Schema test")

        output_dir = project_root / ".conductor" / "steps" / "42"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "6.2.output.json"
        output_file.write_text(json.dumps({"wrong_field": "value"}))

        call_count = 0

        def fix_on_retry(session: object, text: str) -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                output_file.write_text(
                    json.dumps({"file": "fixed.py", "content": "pass"})
                )

        mock_pool.send.side_effect = fix_on_retry

        result = dispatch_step(
            issue_number=42,
            step_id="6.2",
            input_data=IssueContext(
                number=42,
                title="Schema test",
                body="body",
                labels=[],
                phase="verify",
                blocked_by=[],
                branch="feature/42",
            ),
            output_type=FileOutput,
            config=config,
            pool=mock_pool,
            db=db,
            project_root=project_root,
            worktree=project_root / "wt",
            poll_interval=0.01,
            max_validation_retries=2,
        )

        assert result.success is True
        assert result.output.file == "fixed.py"

        mock_pool.release.assert_called_once()
