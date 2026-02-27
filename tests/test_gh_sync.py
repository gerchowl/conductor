from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from conductor.gh_sync import (
    IssueData,
    add_label,
    detect_phase,
    flush_sync_queue,
    parse_blockers,
    post_comment,
    read_issue,
    remove_label,
)
from conductor.state_db import StateDB


@pytest.fixture()
def db(tmp_path: Path) -> StateDB:
    sdb = StateDB(tmp_path / "state.db")
    sdb.upsert_issue(1, "task")
    return sdb


def _gh_issue_json(
    number: int = 42,
    title: str = "Fix bug",
    body: str = "Details here",
    labels: list[dict] | None = None,
    state: str = "OPEN",
    comments: list[dict] | None = None,
) -> str:
    return json.dumps(
        {
            "number": number,
            "title": title,
            "body": body,
            "labels": labels or [],
            "state": state,
            "comments": comments or [],
        }
    )


class TestReadIssue:
    @patch("conductor.gh_sync.subprocess.run")
    def test_basic(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            stdout=_gh_issue_json(
                labels=[{"name": "bug"}, {"name": "phase:design"}],
                comments=[
                    {
                        "author": {"login": "alice"},
                        "body": "looks good",
                        "createdAt": "2025-01-01T00:00:00Z",
                    }
                ],
            )
        )
        issue = read_issue(42, repo="owner/repo")

        assert isinstance(issue, IssueData)
        assert issue.number == 42
        assert issue.title == "Fix bug"
        assert issue.labels == ["bug", "phase:design"]
        assert issue.state == "OPEN"
        assert len(issue.comments) == 1
        assert issue.comments[0]["author"] == "alice"

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[:3] == ["gh", "issue", "view"]
        assert "--repo" in cmd
        assert "owner/repo" in cmd

    @patch("conductor.gh_sync.subprocess.run")
    def test_no_repo(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(stdout=_gh_issue_json())
        read_issue(1)
        cmd = mock_run.call_args[0][0]
        assert "--repo" not in cmd


class TestParseBlockers:
    def test_single(self) -> None:
        assert parse_blockers("Blocked by: #5") == [5]

    def test_multiple(self) -> None:
        assert parse_blockers("Blocked by: #5, #6") == [5, 6]

    def test_no_colon(self) -> None:
        assert parse_blockers("Blocked by #5") == [5]

    def test_none(self) -> None:
        assert parse_blockers("No blockers here") == []

    def test_multiline(self) -> None:
        body = "Some text\nBlocked by: #10, #20\nMore text\n"
        assert parse_blockers(body) == [10, 20]

    def test_case_insensitive(self) -> None:
        assert parse_blockers("blocked by: #3") == [3]


class TestDetectPhase:
    def test_with_phase_label(self) -> None:
        assert detect_phase(["bug", "phase:design"]) == "design"

    def test_no_phase_label(self) -> None:
        assert detect_phase(["bug", "enhancement"]) == "pending"

    def test_multiple_phase_labels(self) -> None:
        assert detect_phase(["phase:plan", "phase:design"]) == "plan"

    def test_empty_labels(self) -> None:
        assert detect_phase([]) == "pending"


class TestAddLabel:
    @patch("conductor.gh_sync.subprocess.run")
    def test_with_repo(self, mock_run: MagicMock) -> None:
        add_label(42, "bug", repo="owner/repo")
        cmd = mock_run.call_args[0][0]
        assert cmd == [
            "gh",
            "issue",
            "edit",
            "42",
            "--repo",
            "owner/repo",
            "--add-label",
            "bug",
        ]

    @patch("conductor.gh_sync.subprocess.run")
    def test_without_repo(self, mock_run: MagicMock) -> None:
        add_label(42, "bug")
        cmd = mock_run.call_args[0][0]
        assert "--repo" not in cmd
        assert "--add-label" in cmd


class TestRemoveLabel:
    @patch("conductor.gh_sync.subprocess.run")
    def test_with_repo(self, mock_run: MagicMock) -> None:
        remove_label(10, "wip", repo="o/r")
        cmd = mock_run.call_args[0][0]
        assert cmd == [
            "gh",
            "issue",
            "edit",
            "10",
            "--repo",
            "o/r",
            "--remove-label",
            "wip",
        ]


class TestPostComment:
    @patch("conductor.gh_sync.subprocess.run")
    def test_with_repo(self, mock_run: MagicMock) -> None:
        post_comment(7, "hello world", repo="o/r")
        cmd = mock_run.call_args[0][0]
        assert cmd == [
            "gh",
            "issue",
            "comment",
            "7",
            "--repo",
            "o/r",
            "--body",
            "hello world",
        ]


class TestFlushSyncQueue:
    @patch("conductor.gh_sync.subprocess.run")
    def test_processes_all_types(self, mock_run: MagicMock, db: StateDB) -> None:
        db.enqueue_sync(1, "label_add", json.dumps({"label": "phase:design"}))
        db.enqueue_sync(1, "label_remove", json.dumps({"label": "phase:plan"}))
        db.enqueue_sync(1, "comment_post", json.dumps({"body": "hello"}))

        count = flush_sync_queue(db, repo="o/r")

        assert count == 3
        assert db.pending_syncs() == []
        assert mock_run.call_count == 3

    @patch("conductor.gh_sync.subprocess.run")
    def test_marks_synced_on_success(self, mock_run: MagicMock, db: StateDB) -> None:
        sid = db.enqueue_sync(1, "label_add", json.dumps({"label": "bug"}))
        flush_sync_queue(db)
        row = db._conn.execute(
            "SELECT status FROM gh_sync WHERE id=?", (sid,)
        ).fetchone()
        assert row["status"] == "synced"

    @patch("conductor.gh_sync.subprocess.run")
    def test_marks_failed_on_error(self, mock_run: MagicMock, db: StateDB) -> None:
        from subprocess import CalledProcessError

        mock_run.side_effect = CalledProcessError(1, "gh")
        sid = db.enqueue_sync(1, "label_add", json.dumps({"label": "bug"}))
        count = flush_sync_queue(db)
        assert count == 1
        row = db._conn.execute(
            "SELECT status FROM gh_sync WHERE id=?", (sid,)
        ).fetchone()
        assert row["status"] == "failed"

    @patch("conductor.gh_sync.subprocess.run")
    def test_unknown_sync_type_fails(self, mock_run: MagicMock, db: StateDB) -> None:
        sid = db.enqueue_sync(1, "unknown_op", json.dumps({"foo": "bar"}))
        count = flush_sync_queue(db)
        assert count == 1
        row = db._conn.execute(
            "SELECT status FROM gh_sync WHERE id=?", (sid,)
        ).fetchone()
        assert row["status"] == "failed"
        mock_run.assert_not_called()

    @patch("conductor.gh_sync.subprocess.run")
    def test_empty_queue(self, mock_run: MagicMock, db: StateDB) -> None:
        count = flush_sync_queue(db)
        assert count == 0
        mock_run.assert_not_called()

    @patch("conductor.gh_sync.subprocess.run")
    def test_no_repo_flag(self, mock_run: MagicMock, db: StateDB) -> None:
        db.enqueue_sync(1, "label_add", json.dumps({"label": "bug"}))
        flush_sync_queue(db)
        cmd = mock_run.call_args[0][0]
        assert "--repo" not in cmd
