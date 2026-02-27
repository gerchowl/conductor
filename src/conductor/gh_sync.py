from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field

from conductor.state_db import StateDB

_BLOCKER_RE = re.compile(r"Blocked by:?\s*(.+)", re.IGNORECASE)
_ISSUE_REF_RE = re.compile(r"#(\d+)")


@dataclass
class IssueData:
    number: int
    title: str
    body: str
    labels: list[str]
    state: str
    comments: list[dict] = field(default_factory=list)


def _repo_args(repo: str | None) -> list[str]:
    return ["--repo", repo] if repo else []


def read_issue(number: int, repo: str | None = None) -> IssueData:
    """Read issue metadata via gh CLI. repo format: 'owner/repo'."""
    result = subprocess.run(
        [
            "gh",
            "issue",
            "view",
            str(number),
            *_repo_args(repo),
            "--json",
            "number,title,body,labels,state,comments",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    labels = [lbl["name"] for lbl in data.get("labels", [])]
    comments = [
        {
            "author": c.get("author", {}).get("login", ""),
            "body": c.get("body", ""),
            "created_at": c.get("createdAt", ""),
        }
        for c in data.get("comments", [])
    ]
    return IssueData(
        number=data["number"],
        title=data["title"],
        body=data.get("body", ""),
        labels=labels,
        state=data.get("state", "OPEN"),
        comments=comments,
    )


def parse_blockers(body: str) -> list[int]:
    """Extract issue numbers from 'Blocked by: #N' lines."""
    blockers: list[int] = []
    for match in _BLOCKER_RE.finditer(body):
        refs = _ISSUE_REF_RE.findall(match.group(1))
        blockers.extend(int(n) for n in refs)
    return blockers


def detect_phase(labels: list[str]) -> str:
    """Detect current phase from phase:* labels. Returns 'pending' if none."""
    for label in labels:
        if label.startswith("phase:"):
            return label.removeprefix("phase:")
    return "pending"


def add_label(number: int, label: str, repo: str | None = None) -> None:
    """Add a label to an issue via gh CLI."""
    subprocess.run(
        [
            "gh",
            "issue",
            "edit",
            str(number),
            *_repo_args(repo),
            "--add-label",
            label,
        ],
        capture_output=True,
        text=True,
        check=True,
    )


def remove_label(number: int, label: str, repo: str | None = None) -> None:
    """Remove a label from an issue via gh CLI."""
    subprocess.run(
        [
            "gh",
            "issue",
            "edit",
            str(number),
            *_repo_args(repo),
            "--remove-label",
            label,
        ],
        capture_output=True,
        text=True,
        check=True,
    )


def post_comment(number: int, body: str, repo: str | None = None) -> None:
    """Post a comment on an issue via gh CLI."""
    subprocess.run(
        [
            "gh",
            "issue",
            "comment",
            str(number),
            *_repo_args(repo),
            "--body",
            body,
        ],
        capture_output=True,
        text=True,
        check=True,
    )


_DISPATCH = {
    "label_add": lambda num, payload, repo: add_label(num, payload["label"], repo),
    "label_remove": lambda num, payload, repo: remove_label(
        num, payload["label"], repo
    ),
    "comment_post": lambda num, payload, repo: post_comment(num, payload["body"], repo),
}


def flush_sync_queue(db: StateDB, repo: str | None = None) -> int:
    """Process all pending sync operations from the DB queue.

    Returns count processed.
    """
    pending = db.pending_syncs()
    count = 0
    for item in pending:
        handler = _DISPATCH.get(item["sync_type"])
        if handler is None:
            db.mark_sync_failed(item["id"])
            count += 1
            continue
        try:
            payload = json.loads(item["payload"])
            handler(item["issue_number"], payload, repo)
            db.mark_synced(item["id"])
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
            db.mark_sync_failed(item["id"])
        count += 1
    return count
