#!/usr/bin/env python3
"""Archive old done/superseded rows out of the central task queue.

Moves rows with status in (done, superseded_dup, superseded_oom) AND
ended_at older than 48h from scripts/queues/central_queue.json into
scripts/queues/archive_done_failed.jsonl (append, one JSON object per line,
each stamped with "archived_at").

The 48h threshold is IMPORTANT: recent done rows are read by in-flight
tooling (harvest scripts, ETA averages, agreement checks) — do not lower it.
Rows without an ended_at are never archived.

Locking + write pattern is copied verbatim from control/task_queue_daemon.py
(the one master of the queue): exclusive fcntl flock on central_queue.lock,
then tmp-file write + atomic rename. The queue file is backed up to
central_queue.json.bak_archive_<epoch> before the rewrite.

NOT wired into the daemon or the dashboard — invoked manually / by the
monitor loop:

    python3 scripts/archive_done_queue.py
"""
from __future__ import annotations

import fcntl
import json
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path("/home/ubuntu/tdmpc-glass")
QUEUE_FILE = REPO / "scripts" / "queues" / "central_queue.json"
ARCHIVE_FILE = REPO / "scripts" / "queues" / "archive_done_failed.jsonl"

ARCHIVE_STATUSES = {"done", "superseded_dup", "superseded_oom"}
MIN_AGE_HOURS = 48


# ── Queue I/O — same pattern as control/task_queue_daemon.py ─────────────────

def load_queue() -> list[dict]:
    if not QUEUE_FILE.exists():
        return []
    with open(QUEUE_FILE) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_queue(tasks: list[dict]):
    QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = QUEUE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(tasks, indent=2))
    tmp.replace(QUEUE_FILE)


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def main() -> int:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=MIN_AGE_HOURS)
    lock_path = QUEUE_FILE.with_suffix(".lock")
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            tasks = load_queue()
            keep, move = [], []
            for t in tasks:
                ended = _parse_iso(t.get("ended_at"))
                if t.get("status") in ARCHIVE_STATUSES and ended is not None and ended < cutoff:
                    move.append(t)
                else:
                    keep.append(t)

            if not move:
                print(f"[archive_done_queue] nothing to archive "
                      f"({len(tasks)} rows; none with status in {sorted(ARCHIVE_STATUSES)} "
                      f"older than {MIN_AGE_HOURS}h)")
                return 0

            # Back up the queue file BEFORE mutating anything.
            bak = QUEUE_FILE.with_name(f"{QUEUE_FILE.name}.bak_archive_{int(time.time())}")
            shutil.copy2(QUEUE_FILE, bak)
            print(f"[archive_done_queue] queue backed up to {bak}")

            # Append to the archive first (a crash here duplicates archive rows
            # but never loses queue rows), then truncate/rewrite the queue.
            archived_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
            with open(ARCHIVE_FILE, "a") as af:
                for t in move:
                    af.write(json.dumps({**t, "archived_at": archived_at}) + "\n")
            save_queue(keep)

            for t in move:
                print(f"[archive_done_queue] moved {t.get('id')} "
                      f"[{t.get('status')}] ended {t.get('ended_at')} — "
                      f"{(t.get('label') or '')[:90]}")
            print(f"[archive_done_queue] moved {len(move)} rows -> {ARCHIVE_FILE}; "
                  f"{len(keep)} rows remain in {QUEUE_FILE.name}")
            return 0
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


if __name__ == "__main__":
    raise SystemExit(main())
