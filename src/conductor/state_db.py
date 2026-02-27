from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS issues (
    number INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    phase TEXT NOT NULL DEFAULT 'pending',
    current_step TEXT,
    dispatched_at TEXT,
    completed_at TEXT,
    blocked_by TEXT,
    branch TEXT,
    pr_number INTEGER,
    stuck_reason TEXT
);

CREATE TABLE IF NOT EXISTS steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    issue_number INTEGER NOT NULL REFERENCES issues(number),
    step TEXT NOT NULL,
    model_tier TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    input_path TEXT,
    output_path TEXT,
    dispatched_at TEXT,
    completed_at TEXT,
    duration_ms INTEGER,
    error TEXT
);

CREATE TABLE IF NOT EXISTS gh_sync (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    issue_number INTEGER NOT NULL,
    sync_type TEXT NOT NULL,
    payload TEXT NOT NULL,
    synced_at TEXT,
    status TEXT NOT NULL DEFAULT 'pending'
);
"""

_ISSUE_COLUMNS = (
    "number",
    "title",
    "phase",
    "current_step",
    "dispatched_at",
    "completed_at",
    "blocked_by",
    "branch",
    "pr_number",
    "stuck_reason",
)


class StateDB:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> StateDB:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── Issues ───────────────────────────────────────────────────────

    def upsert_issue(self, number: int, title: str, **kwargs: Any) -> None:
        fields: dict[str, Any] = {"number": number, "title": title, **kwargs}
        cols = ", ".join(fields)
        placeholders = ", ".join(["?"] * len(fields))
        update_clause = ", ".join(f"{c}=excluded.{c}" for c in fields if c != "number")
        sql = (
            f"INSERT INTO issues ({cols}) VALUES ({placeholders}) "
            f"ON CONFLICT(number) DO UPDATE SET {update_clause}"
        )
        self._conn.execute(sql, list(fields.values()))
        self._conn.commit()

    def get_issue(self, number: int) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM issues WHERE number=?", (number,)
        ).fetchone()
        return dict(row) if row else None

    def list_issues(self, phase: str | None = None) -> list[dict[str, Any]]:
        if phase is None:
            rows = self._conn.execute("SELECT * FROM issues").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM issues WHERE phase=?", (phase,)
            ).fetchall()
        return [dict(r) for r in rows]

    def update_issue(self, number: int, **kwargs: Any) -> None:
        if not kwargs:
            return
        set_clause = ", ".join(f"{k}=?" for k in kwargs)
        sql = f"UPDATE issues SET {set_clause} WHERE number=?"
        self._conn.execute(sql, [*kwargs.values(), number])
        self._conn.commit()

    # ── Steps ────────────────────────────────────────────────────────

    def insert_step(self, issue_number: int, step: str, model_tier: str) -> int:
        cur = self._conn.execute(
            "INSERT INTO steps (issue_number, step, model_tier) VALUES (?, ?, ?)",
            (issue_number, step, model_tier),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def update_step(self, step_id: int, **kwargs: Any) -> None:
        if not kwargs:
            return
        set_clause = ", ".join(f"{k}=?" for k in kwargs)
        sql = f"UPDATE steps SET {set_clause} WHERE id=?"
        self._conn.execute(sql, [*kwargs.values(), step_id])
        self._conn.commit()

    def get_steps(self, issue_number: int) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM steps WHERE issue_number=? ORDER BY id",
            (issue_number,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── GH Sync queue ────────────────────────────────────────────────

    def enqueue_sync(self, issue_number: int, sync_type: str, payload: str) -> int:
        cur = self._conn.execute(
            "INSERT INTO gh_sync (issue_number, sync_type, payload) VALUES (?, ?, ?)",
            (issue_number, sync_type, payload),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def pending_syncs(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM gh_sync WHERE status='pending' ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_synced(self, sync_id: int) -> None:
        self._conn.execute("UPDATE gh_sync SET status='synced' WHERE id=?", (sync_id,))
        self._conn.commit()

    def mark_sync_failed(self, sync_id: int) -> None:
        self._conn.execute("UPDATE gh_sync SET status='failed' WHERE id=?", (sync_id,))
        self._conn.commit()
