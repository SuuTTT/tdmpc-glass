"""Central task-queue REST API for the dashboard.

A Flask Blueprint exposing /api/queue (GET/POST), /api/queue/<id> (DELETE),
priority/retry/log sub-routes, plus the ETA + promising-phase computation that
feeds the fleet summary. Reads/writes scripts/queues/central_queue.json under an
fcntl lock (same file the queue daemon manages).
"""
from __future__ import annotations

import fcntl
import json
import re
import subprocess
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

from flask import Blueprint, jsonify, request

from boxprobe import BOXES, SSH_KEY, _BOX_CACHE, _BOX_CACHE_LOCK
from data import (
    canonical_phase,
    discover_csvs,
    eval_summary,
    eval_is_countable,
    PHASE_NOTES,
)

REPO = Path("/home/ubuntu/tdmpc-glass")
QUEUE_DIR = REPO / "scripts" / "queues"
CENTRAL_QUEUE_FILE = QUEUE_DIR / "central_queue.json"

queue_bp = Blueprint("queue_api", __name__)

DEFAULT_TASK_DURATION_S = 14400  # 4-hour fallback when no SPS or history
_MAX_STEPS = 10_000_000
_PATIENCE_STEPS = 3_000_000


# ─── Queue file I/O (fcntl-locked, tmp+rename) ─────────────────────────────

def _load_central_queue() -> list[dict]:
    if not CENTRAL_QUEUE_FILE.exists():
        return []
    try:
        return json.loads(CENTRAL_QUEUE_FILE.read_text())
    except Exception:
        return []


def _save_central_queue(tasks: list[dict]):
    CENTRAL_QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CENTRAL_QUEUE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(tasks, indent=2))
    tmp.replace(CENTRAL_QUEUE_FILE)


def _with_queue_lock(fn):
    lock_path = CENTRAL_QUEUE_FILE.with_suffix(".lock")
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            tasks = _load_central_queue()
            result = fn(tasks)
            if result is not None:
                _save_central_queue(result)
            return tasks if result is None else result
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


# ─── ETA + promising-phase computation ─────────────────────────────────────

def _sps_remaining_s(task: dict) -> float | None:
    """Return remaining seconds for a running training task using cached SPS data."""
    box_tag = task.get("box")
    if not box_tag:
        return None
    with _BOX_CACHE_LOCK:
        box_data = _BOX_CACHE.get(box_tag, {})
    procs = box_data.get("procs", [])
    if not procs:
        return None
    seed_m = re.search(r"SEEDS?=(\S+)", task.get("env", ""))
    seed = seed_m.group(1) if seed_m else None
    matched = [p for p in procs if seed and str(p.get("seed")) == seed]
    proc = (matched or procs)[0]
    sps = proc.get("sps_avg")
    if not sps or sps <= 0:
        return None
    last_step = proc.get("last_step") or 0
    best_step = proc.get("best_any_step") or proc.get("best_mppi_step") or 0
    steps_since_best = max(0, last_step - best_step)
    patience_left = max(0, _PATIENCE_STEPS - steps_since_best)
    effective_target = min(_MAX_STEPS, last_step + patience_left)
    remaining_steps = max(0, effective_target - last_step)
    return remaining_steps / sps


def _phase_key_from_text(text: str) -> str:
    """Best-effort canonical phase key from task label/env/output tags."""
    text = text or ""
    for pat in (r"(phasei9[a-z])", r"(phase[a-z]+[a-z0-9_]*|phasex_ns1024)"):
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return canonical_phase(m.group(1))
    return ""


def _task_phase_key(task: dict) -> str:
    env = task.get("env", "")
    label = task.get("label", "")
    m = re.search(r"PROBE_ID=([^\s]+)", env)
    if m:
        key = _phase_key_from_text(m.group(1))
        if key:
            return key
    return _phase_key_from_text(f"{label} {env}")


def _compute_queue_etas(tasks: list[dict]) -> tuple[list[dict], str | None, list[dict]]:
    """Add ETA fields. Returns (annotated, queue_eta_iso, box_next_free)."""
    import heapq

    now = datetime.now(timezone.utc)

    def parse_iso(s):
        if not s:
            return None
        try:
            return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except Exception:
            return None

    def to_iso(dt: datetime) -> str:
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Per-launcher average duration from done tasks (fallback when no live SPS).
    launcher_durs: dict[str, list[float]] = {}
    for t in tasks:
        if t["status"] == "done" and t.get("started_at") and t.get("ended_at"):
            s, e = parse_iso(t["started_at"]), parse_iso(t["ended_at"])
            if s and e and e > s:
                launcher_durs.setdefault(t.get("launcher", ""), []).append((e - s).total_seconds())
    avg_dur: dict[str, float] = {l: sum(ds) / len(ds) for l, ds in launcher_durs.items()}

    def est_remaining_s(task) -> float:
        """Estimate remaining seconds for a task (running or pending)."""
        if task["status"] == "running":
            sps_rem = _sps_remaining_s(task)
            if sps_rem is not None:
                return sps_rem
        return avg_dur.get(task.get("launcher", ""), DEFAULT_TASK_DURATION_S)

    def est_total_dur(task) -> float:
        return avg_dur.get(task.get("launcher", ""), DEFAULT_TASK_DURATION_S)

    # Box free-at times: running tasks → now + remaining; idle boxes → now.
    all_tags = [b[0] for b in BOXES]
    box_free: dict[str, datetime] = {tag: now for tag in all_tags}
    box_task: dict[str, dict] = {}
    for t in tasks:
        # Guard: only fleet boxes (current daemon BOXES list) participate in
        # scheduling. A historical row whose box was destroyed/removed must not
        # inject a phantom slot into the ETA heap.
        if t["status"] == "running" and t.get("box") in box_free and t.get("started_at"):
            rem = est_remaining_s(t)
            box_free[t["box"]] = max(now + timedelta(seconds=rem), now)
            box_task[t["box"]] = t

    # Simulate pending task scheduling with a min-heap of (free_time, box_tag).
    heap = [(ts, tag) for tag, ts in box_free.items()]
    heapq.heapify(heap)
    pending_sorted = sorted(
        [t for t in tasks if t["status"] == "pending" and t.get("type") != "render"],
        key=lambda t: (t.get("priority", 10), t.get("created_at", ""))
    )
    sched: dict[str, tuple[datetime, datetime]] = {}
    for t in pending_sorted:
        if not heap:
            break
        free_ts, tag = heapq.heappop(heap)
        start = max(free_ts, now)
        finish = start + timedelta(seconds=est_total_dur(t))
        sched[t["id"]] = (start, finish)
        heapq.heappush(heap, (finish, tag))

    # Annotate tasks with ETA fields.
    result = []
    for t in tasks:
        t = dict(t)
        if t["status"] == "running" and t.get("started_at"):
            s = parse_iso(t["started_at"])
            if s:
                elapsed = (now - s).total_seconds()
                rem = est_remaining_s(t)
                eta = now + timedelta(seconds=rem)
                t["elapsed_s"] = int(elapsed)
                t["remaining_s"] = int(rem)
                t["eta_iso"] = to_iso(eta)
                box_tag = t.get("box", "")
                with _BOX_CACHE_LOCK:
                    box_data = _BOX_CACHE.get(box_tag, {})
                procs = box_data.get("procs", [])
                if procs:
                    seed_m = re.search(r"SEEDS?=(\S+)", t.get("env", ""))
                    seed = seed_m.group(1) if seed_m else None
                    matched = [p for p in procs if seed and str(p.get("seed")) == seed]
                    proc = (matched or procs)[0]
                    t["_live_sps"] = proc.get("sps_avg")
                    t["_live_last_step"] = proc.get("last_step")
                    t["_live_best_reward"] = proc.get("best_any") or proc.get("best_mppi")
        elif t["status"] == "pending" and t["id"] in sched:
            start, finish = sched[t["id"]]
            t["estimated_start_iso"] = to_iso(start)
            t["eta_iso"] = to_iso(finish)
        result.append(t)

    active_etas = [t["eta_iso"] for t in result
                   if t["status"] in ("running", "pending") and t.get("eta_iso")]
    queue_eta = max(active_etas) if active_etas else None
    box_next_free = []
    for tag in all_tags:
        free_at = box_free.get(tag, now)
        t = box_task.get(tag)
        box_next_free.append({
            "box": tag,
            "free_at_iso": to_iso(free_at),
            "free_in_s": max(0, int((free_at - now).total_seconds())),
            "idle_now": tag not in box_task,
            "task_id": t.get("id") if t else None,
            "label": t.get("label") if t else None,
            "phase": _task_phase_key(t) if t else None,
        })
    box_next_free.sort(key=lambda r: (r["free_in_s"], r["box"]))
    return result, queue_eta, box_next_free


def _promising_phases(tasks: list[dict], annotated_tasks: list[dict], limit: int = 8) -> list[dict]:
    """Rank phase families that look useful enough to watch right now."""
    csvs = discover_csvs()
    by_phase: dict[str, dict] = {}
    for c in csvs:
        phase = canonical_phase(c["phase"])
        entry = by_phase.setdefault(phase, {"bests": [], "variants": set()})
        if c["phase"] != phase:
            entry["variants"].add(c["phase"])
        summary = eval_summary(c["path"])
        if not eval_is_countable(summary):
            continue
        best = summary["best_any"]
        if best >= 0:
            entry["bests"].append(best)

    queue_counts: dict[str, dict[str, int]] = {}
    for t in tasks:
        phase = _task_phase_key(t)
        if not phase:
            continue
        counts = queue_counts.setdefault(phase, {"running": 0, "pending": 0, "failed": 0, "done": 0})
        status = t.get("status", "")
        if status in counts:
            counts[status] += 1

    rows = []
    for phase in sorted(set(by_phase) | set(queue_counts)):
        info = by_phase.get(phase, {"bests": [], "variants": set()})
        bests = info["bests"]
        n = len(bests)
        mean_b = sum(bests) / n if n else None
        max_b = max(bests) if bests else None
        n_g1 = sum(1 for b in bests if b >= 500)
        counts = queue_counts.get(phase, {"running": 0, "pending": 0, "failed": 0, "done": 0})
        notes = PHASE_NOTES.get(phase, "")
        lowered = notes.lower()
        if any(word in lowered for word in ("falsified", "collapsed", "smoke test", "hardware validation")):
            continue
        if n == 0 and not counts["running"] and not counts["pending"]:
            continue
        score = 0.0
        score += (max_b or 0) * 1.2
        score += (mean_b or 0) * 0.7
        score += n_g1 * 120
        score += counts["running"] * 60 + counts["pending"] * 25
        if max_b is not None and max_b >= 600:
            score += 250
        elif max_b is not None and max_b >= 500:
            score += 150
        elif max_b is not None and max_b >= 380:
            score += 60
        rows.append({
            "phase": phase,
            "n_with_data": n,
            "mean_best": round(mean_b, 1) if mean_b is not None else None,
            "max_best": round(max_b, 1) if max_b is not None else None,
            "n_g1": n_g1,
            "running": counts["running"],
            "pending": counts["pending"],
            "notes": notes,
            "score": round(score, 1),
        })
    rows.sort(key=lambda r: (-r["score"], r["phase"]))
    return rows[:limit]


# ─── Routes ────────────────────────────────────────────────────────────────

@queue_bp.route("/api/queue")
def api_queue_get():
    tasks = _load_central_queue()
    tasks_sorted = sorted(tasks, key=lambda t: (t.get("priority", 10), t.get("created_at", "")))
    annotated, queue_eta, box_next_free = _compute_queue_etas(tasks_sorted)
    return jsonify({
        "tasks": annotated,
        "queue_eta": queue_eta,
        "box_next_free": box_next_free,
        "promising_phases": _promising_phases(tasks_sorted, annotated),
    })


@queue_bp.route("/api/queue", methods=["POST"])
def api_queue_add():
    body = request.get_json(force=True)
    label = (body.get("label") or "").strip()
    launcher = (body.get("launcher") or "").strip()
    env = (body.get("env") or "").strip()
    priority = int(body.get("priority", 10))
    if not label or not launcher:
        return jsonify({"error": "label and launcher are required"}), 400
    task = {
        "id": "t" + uuid.uuid4().hex[:7],
        "label": label,
        "launcher": launcher,
        "env": env,
        "priority": priority,
        "status": "pending",
        "box": None,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "started_at": None,
        "ended_at": None,
    }

    def add(tasks):
        tasks.append(task)
        return tasks

    _with_queue_lock(add)
    return jsonify({"ok": True, "id": task["id"]})


@queue_bp.route("/api/queue/<task_id>", methods=["DELETE"])
def api_queue_delete(task_id):
    removed = [False]

    def delete(tasks):
        new = [t for t in tasks if t["id"] != task_id]
        removed[0] = len(new) < len(tasks)
        return new

    _with_queue_lock(delete)
    if not removed[0]:
        return jsonify({"error": "task not found"}), 404
    return jsonify({"ok": True})


@queue_bp.route("/api/queue/<task_id>/priority", methods=["POST"])
def api_queue_priority(task_id):
    body = request.get_json(force=True)
    delta = int(body.get("delta", -1))  # negative = higher priority (lower number)

    def bump(tasks):
        for t in tasks:
            if t["id"] == task_id and t["status"] == "pending":
                t["priority"] = max(1, t["priority"] + delta)
        return tasks

    _with_queue_lock(bump)
    return jsonify({"ok": True})


@queue_bp.route("/api/queue/<task_id>/retry", methods=["POST"])
def api_queue_retry(task_id):
    """Reset a running/failed/done task back to pending so the daemon re-runs it."""
    def retry(tasks):
        for t in tasks:
            if t["id"] == task_id and t["status"] in ("running", "failed", "done"):
                t["status"] = "pending"
                t["box"] = None
                t["started_at"] = None
                t["ended_at"] = None
        return tasks

    _with_queue_lock(retry)
    return jsonify({"ok": True})


@queue_bp.route("/api/queue/<task_id>/log")
def api_queue_task_log(task_id):
    """Return last 60 lines of the task's remote log (/tmp/tqd_<id>.log)."""
    tasks = _load_central_queue()
    task = next((t for t in tasks if t["id"] == task_id), None)
    if not task:
        return jsonify({"error": "task not found"}), 404
    box_tag = task.get("box")
    log_path = f"/tmp/tqd_{task_id}.log"
    box_info = next((b for b in BOXES if b[0] == box_tag), None)
    try:
        if box_tag == "local":
            r = subprocess.run(["tail", "-n", "60", log_path],
                               capture_output=True, text=True, timeout=5)
            lines = r.stdout.splitlines()
        elif box_info:
            tag, port, host, gpu_idx, label = box_info
            r = subprocess.run(
                ["ssh", "-p", str(port), "-i", SSH_KEY,
                 "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=8",
                 "-o", "BatchMode=yes", f"root@{host}",
                 f"tail -n 60 {log_path} 2>/dev/null || echo '(log not found at {log_path})'"],
                capture_output=True, text=True, timeout=15,
            )
            lines = r.stdout.splitlines()
        else:
            lines = [f"(box '{box_tag}' not in registry)"]
    except Exception as e:
        lines = [f"(error fetching log: {e})"]
    return jsonify({"lines": lines, "log_path": log_path, "box": box_tag})
