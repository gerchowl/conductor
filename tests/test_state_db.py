from __future__ import annotations

import json
from pathlib import Path
from sqlite3 import ProgrammingError

import pytest

from conductor.state_db import StateDB


@pytest.fixture()
def db(tmp_path: Path) -> StateDB:
    return StateDB(tmp_path / "state.db")


class TestSchemaCreation:
    def test_tables_exist(self, db: StateDB) -> None:
        rows = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        names = {r["name"] for r in rows}
        assert {"issues", "steps", "gh_sync"}.issubset(names)

    def test_wal_mode(self, db: StateDB) -> None:
        mode = db._conn.execute("PRAGMA journal_mode").fetchone()
        assert mode[0] == "wal"


class TestContextManager:
    def test_auto_close(self, tmp_path: Path) -> None:
        with StateDB(tmp_path / "ctx.db") as sdb:
            sdb.upsert_issue(1, "test")
        with pytest.raises(ProgrammingError):
            sdb._conn.execute("SELECT 1")

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c" / "state.db"
        with StateDB(nested) as sdb:
            sdb.upsert_issue(1, "nested")
            assert sdb.get_issue(1) is not None


class TestIssues:
    def test_upsert_insert(self, db: StateDB) -> None:
        db.upsert_issue(42, "Fix the widget")
        issue = db.get_issue(42)
        assert issue is not None
        assert issue["title"] == "Fix the widget"
        assert issue["phase"] == "pending"

    def test_upsert_update(self, db: StateDB) -> None:
        db.upsert_issue(42, "Fix the widget")
        db.upsert_issue(42, "Fix the widget v2", phase="active")
        issue = db.get_issue(42)
        assert issue is not None
        assert issue["title"] == "Fix the widget v2"
        assert issue["phase"] == "active"

    def test_get_issue_missing(self, db: StateDB) -> None:
        assert db.get_issue(999) is None

    def test_update_issue(self, db: StateDB) -> None:
        db.upsert_issue(1, "task")
        db.update_issue(1, phase="active", branch="feat/1")
        issue = db.get_issue(1)
        assert issue is not None
        assert issue["phase"] == "active"
        assert issue["branch"] == "feat/1"

    def test_update_issue_noop(self, db: StateDB) -> None:
        db.upsert_issue(1, "task")
        db.update_issue(1)
        assert db.get_issue(1)["title"] == "task"  # type: ignore[index]

    def test_list_issues_all(self, db: StateDB) -> None:
        db.upsert_issue(1, "a")
        db.upsert_issue(2, "b", phase="active")
        assert len(db.list_issues()) == 2

    def test_list_issues_phase_filter(self, db: StateDB) -> None:
        db.upsert_issue(1, "a")
        db.upsert_issue(2, "b", phase="active")
        db.upsert_issue(3, "c", phase="active")
        pending = db.list_issues(phase="pending")
        active = db.list_issues(phase="active")
        assert len(pending) == 1
        assert len(active) == 2
        assert pending[0]["number"] == 1

    def test_upsert_with_blocked_by(self, db: StateDB) -> None:
        blocked = json.dumps([10, 20])
        db.upsert_issue(5, "blocked issue", blocked_by=blocked)
        issue = db.get_issue(5)
        assert issue is not None
        assert json.loads(issue["blocked_by"]) == [10, 20]

    def test_upsert_with_all_fields(self, db: StateDB) -> None:
        db.upsert_issue(
            7,
            "full",
            phase="active",
            current_step="1.1",
            branch="feat/7",
            pr_number=99,
        )
        issue = db.get_issue(7)
        assert issue is not None
        assert issue["current_step"] == "1.1"
        assert issue["pr_number"] == 99


class TestSteps:
    def test_insert_and_get(self, db: StateDB) -> None:
        db.upsert_issue(1, "task")
        sid = db.insert_step(1, "1.1", "fast")
        assert isinstance(sid, int)
        steps = db.get_steps(1)
        assert len(steps) == 1
        assert steps[0]["step"] == "1.1"
        assert steps[0]["model_tier"] == "fast"
        assert steps[0]["status"] == "pending"

    def test_update_step(self, db: StateDB) -> None:
        db.upsert_issue(1, "task")
        sid = db.insert_step(1, "1.1", "fast")
        db.update_step(sid, status="completed", duration_ms=1234)
        steps = db.get_steps(1)
        assert steps[0]["status"] == "completed"
        assert steps[0]["duration_ms"] == 1234

    def test_update_step_noop(self, db: StateDB) -> None:
        db.upsert_issue(1, "task")
        sid = db.insert_step(1, "1.1", "fast")
        db.update_step(sid)
        assert db.get_steps(1)[0]["status"] == "pending"

    def test_multiple_steps_ordered(self, db: StateDB) -> None:
        db.upsert_issue(1, "task")
        db.insert_step(1, "1.1", "fast")
        db.insert_step(1, "1.2", "strong")
        db.insert_step(1, "2.1", "fast")
        steps = db.get_steps(1)
        assert [s["step"] for s in steps] == ["1.1", "1.2", "2.1"]

    def test_get_steps_empty(self, db: StateDB) -> None:
        db.upsert_issue(1, "task")
        assert db.get_steps(1) == []

    def test_step_error_field(self, db: StateDB) -> None:
        db.upsert_issue(1, "task")
        sid = db.insert_step(1, "1.1", "fast")
        db.update_step(sid, status="failed", error="timeout")
        steps = db.get_steps(1)
        assert steps[0]["error"] == "timeout"


class TestGhSync:
    def test_enqueue_and_pending(self, db: StateDB) -> None:
        db.upsert_issue(1, "task")
        sid = db.enqueue_sync(1, "label_add", '{"label":"bug"}')
        assert isinstance(sid, int)
        pending = db.pending_syncs()
        assert len(pending) == 1
        assert pending[0]["sync_type"] == "label_add"
        assert pending[0]["payload"] == '{"label":"bug"}'

    def test_mark_synced(self, db: StateDB) -> None:
        db.upsert_issue(1, "task")
        sid = db.enqueue_sync(1, "comment_post", '{"body":"hello"}')
        db.mark_synced(sid)
        assert db.pending_syncs() == []

    def test_mark_sync_failed(self, db: StateDB) -> None:
        db.upsert_issue(1, "task")
        sid = db.enqueue_sync(1, "label_remove", '{"label":"wip"}')
        db.mark_sync_failed(sid)
        assert db.pending_syncs() == []
        row = db._conn.execute(
            "SELECT status FROM gh_sync WHERE id=?", (sid,)
        ).fetchone()
        assert row["status"] == "failed"

    def test_pending_syncs_ordering(self, db: StateDB) -> None:
        db.upsert_issue(1, "task")
        db.enqueue_sync(1, "label_add", '{"label":"a"}')
        db.enqueue_sync(1, "label_add", '{"label":"b"}')
        db.enqueue_sync(1, "comment_post", '{"body":"c"}')
        pending = db.pending_syncs()
        assert len(pending) == 3
        payloads = [p["payload"] for p in pending]
        assert payloads == ['{"label":"a"}', '{"label":"b"}', '{"body":"c"}']

    def test_mixed_sync_statuses(self, db: StateDB) -> None:
        db.upsert_issue(1, "task")
        s1 = db.enqueue_sync(1, "label_add", '{"label":"a"}')
        db.enqueue_sync(1, "label_add", '{"label":"b"}')
        s3 = db.enqueue_sync(1, "label_add", '{"label":"c"}')
        db.mark_synced(s1)
        db.mark_sync_failed(s3)
        pending = db.pending_syncs()
        assert len(pending) == 1
        assert json.loads(pending[0]["payload"])["label"] == "b"
