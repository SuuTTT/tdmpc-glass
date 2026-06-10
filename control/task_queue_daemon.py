#!/usr/bin/env python3
"""Central task queue daemon.

Polls every POLL_SECONDS. When a box is idle, claims the highest-priority
pending task from central_queue.json and SSH-launches it there.
Marks tasks done when their assigned box becomes free again.

Usage:
    nohup python3 scripts/task_queue_daemon.py \
        >> exp/tdmpc_glass/logs/daemons/tqd.log 2>&1 &
"""
import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/home/ubuntu/tdmpc-glass")
SSH_KEY = os.environ.get("SSH_IDENTITY_FILE", "/home/ubuntu/.ssh/vastai_id_ed25519")
QUEUE_FILE = REPO / "scripts" / "queues" / "central_queue.json"
POLL_SECONDS = 60
LOG_DIR = REPO / "exp" / "tdmpc_glass" / "logs" / "daemons"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Box registry — mirrors BOXES in web_dashboard.py
# (tag, port, host, gpu_idx)
# NOTE: EC2 control plane has NO GPU — there is intentionally NO "local" training
# slot. All training runs on remote vast.ai GPU workers. Do NOT add a local box.
# Fleet as of 2026-06-01 (glass-tdmpc boxes; ports are the real sshd ports).
BOXES = [
    ("ssh1_2080ti",   34217,  "ssh1.vast.ai",   0),
    ("ssh6_titanv",   31740,  "ssh6.vast.ai",   0),
    # ssh5_3060 (inst 34824701) REMOVED 2026-06-05: DISK FULL — was a failure sink
    # (fast-failed every task incl. all geoglass/behavglass arms). User reclaiming it
    # for another project; tdmpc files/env cleared. Do NOT re-add.
    # ("ssh5_3060",     24701,  "ssh5.vast.ai",   0),
    ("ssh1_a4000b",   16822,  "ssh1.vast.ai",   0),  # rented 2026-06-02; seed3=501 G1 (finishing)
    ("ssh8_a4000",    39560,  "ssh8.vast.ai",   0),  # rented 2026-06-02; seed4=550 G1 (finishing)
    # ── 1660 Super x2 (inst 38342607, ssh4.vast.ai:22607) — UNSTABLE/flaky, 6GB each.
    # 2026-06-07: silently killed 3/3 long runs (iter-15 seed1 @496k, both iter-17
    # BallInCup seeds @~200k — no traceback, GPUs empty). RE-ENABLED same day on user
    # instruction; the kills coincided with a now-finished se-bench workload on this
    # box (likely host-side competition), so risk may have passed. Prefer <=500k tasks;
    # any new silent death -> disable again and requeue via failed_box_died.
    # ASSIGNED TO LTSF 2026-06-10 (user, inst 38342607): do NOT touch. ("ssh4_1660s_g0", 22607, ...)
    # ASSIGNED TO LTSF 2026-06-10 (user, inst 38342607): do NOT touch. ("ssh4_1660s_g1", 22607, ...)
    # ── Former PBT pool, FOLDED BACK into the daemon 2026-06-04 (orchestrator stopped;
    # research pivoted off PBT). Daemon is now the single manager. In-flight PBT members
    # finish naturally (daemon sees the box busy, won't touch); when they finish the
    # daemon launches the next pending baseline task. ssh3 omitted (recycled/lost).
    # REMOVED 2026-06-08 (user): ssh1:24456 (inst 38664456) is the USER's ltsf/forecasting
    # box (a 'forecast' python job uses the GPU but not via run_benchmark, so is_box_idle
    # would falsely see it idle and clobber it). Do NOT schedule tdmpc here.
    # ("ssh1_a4000",    24456,  "ssh1.vast.ai",   0),
    # ASSIGNED TO LTSF 2026-06-10 (user, permanent): do NOT touch. ("ssh2_a4000", 18950, "ssh2.vast.ai", 0)
    ("ssh9_a4000",    16690,  "ssh9.vast.ai",   0),  # seed10 finished -> idle, ready for work
    ("ssh4_a4000",    29168,  "ssh4.vast.ai",   0),
    ("ssh4_a4000b",   10022,  "ssh4.vast.ai",   0),
    # Released from ltsf 2026-06-09 (user); 3060 12GB, env bootstrapped + verified (jax 0.10.1+cuda,
    # mujoco 3.8.0, mjx_playground OK). Proxy ssh6.vast.ai:11696 (distinct port from ssh6_titanv).
    ("ssh6_3060",     17241,  "91.150.160.38",  0),  # inst 40121696 (proxy ssh6.vast.ai:11696 refuses; direct IP works)
    # Added 2026-06-04 (user): 2 more A4000s. 38766691 == ssh9_a4000 above (same proxy
    # ssh9.vast.ai:16690), already covered. 38767427 below (proxy ssh3.vast.ai:17426 —
    # port churned earlier but maps to our instance now; *.vast.ai host-key tolerance +
    # is_box_idle fail-safe handle any re-churn). Env verified: jax 0.10.1 + repo present.
    # REMOVED 2026-06-08 (user): ssh3:17426 (inst 38767427) is the USER's mahjong box.
    # Same clobber risk (mahjong GPU job is run_benchmark-free). Do NOT schedule tdmpc here.
    ("ssh3b_a4000",   17426,  "ssh3.vast.ai",   0),  # inst 38767427 — RE-FREED 2026-06-10 (user): forecasting done
    # ssh9 4x2060 (inst 37457647) DEGRADED 2026-06-01: GPU2/GPU3 device-handle
    # "Unknown Error" (fell off the bus), JAX -> "Unknown backend cuda". Removed
    # from the fleet so the daemon stops churning fast-fails. Reboot the vast.ai
    # instance to recover, then un-comment.
    # ("ssh9_2060_gpu0", 17647, "ssh9.vast.ai",   0),
    # ("ssh9_2060_gpu1", 17647, "ssh9.vast.ai",   1),
    # ("ssh9_2060_gpu2", 17647, "ssh9.vast.ai",   2),
    # ("ssh9_2060_gpu3", 17647, "ssh9.vast.ai",   3),
]

# Per-box XLA_MEM override used when env doesn't already specify it.
DEFAULT_MEM = {
    "local":         "0.85",
    "ssh1_2080ti":   "0.75",
    "ssh1_a4000":    "0.75",
    "ssh6_3080":     "0.65",
    "ssh5_3060_bar":  "0.65",
    "ssh5_3060":      "0.6",
    "ssh1_a4000b":    "0.75",
    "ssh8_a4000":     "0.75",
    "ssh4_a4000":     "0.75",
    "ssh4_a4000b":    "0.75",
    "ssh6_3060":      "0.6",   # 12GB, released from ltsf
    "ssh3b_a4000":    "0.75",
    "ssh9_2060_gpu0": "0.35",
    "ssh9_2060_gpu1": "0.35",
    "ssh9_2060_gpu2": "0.35",
    "ssh9_2060_gpu3": "0.35",
    "ssh4_1660s_g0":  "0.5",   # 6GB; standard cfg uses only ~0.6GB, 0.5 cap is ample
    "ssh4_1660s_g1":  "0.5",
}
CUDA_MASK = {
    "ssh9_2060_gpu0": "CUDA_VISIBLE_DEVICES=0",
    "ssh9_2060_gpu1": "CUDA_VISIBLE_DEVICES=1",
    "ssh9_2060_gpu2": "CUDA_VISIBLE_DEVICES=2",
    "ssh9_2060_gpu3": "CUDA_VISIBLE_DEVICES=3",
    "ssh4_1660s_g0":  "CUDA_VISIBLE_DEVICES=0",  # two runs, one per GPU on inst 38342607
    "ssh4_1660s_g1":  "CUDA_VISIBLE_DEVICES=1",
}


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%SZ")


def log(msg: str):
    print(f"[tqd] {ts()} {msg}", flush=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Queue I/O with file lock ──────────────────────────────────────────────────

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


def with_queue_lock(fn):
    """Run fn(tasks) → tasks atomically with an exclusive file lock."""
    lock_path = QUEUE_FILE.with_suffix(".lock")
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            tasks = load_queue()
            result = fn(tasks)
            if result is not None:
                save_queue(result)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


# ── Box idle check ────────────────────────────────────────────────────────────

def is_box_idle(tag: str, port: int, host: str, gpu_idx: int) -> bool:
    """Return True if this GPU slot can accept a new task WITHOUT stacking onto an
    existing run.

    Hardened 2026-06-01 after a duplicate-launch incident: the original per-GPU
    memory check on multi-GPU boxes let the daemon launch a run on a free GPU
    while an identical phase+seed was already training on a SIBLING GPU, both
    writing the same CSV. The fix: never launch if (n_run_benchmark_procs >=
    n_gpus) on the box, AND for the targeted GPU index require it to be free.
    """
    if tag == "local":
        # EC2 control plane has no local training slot; never idle for training.
        return False

    # One SSH round-trip: total run procs, total GPUs, memory on THIS gpu_idx, disk%.
    probe = (
        "NP=$(ps -eo cmd | grep '[r]un_benchmark' | wc -l); "
        "NG=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l); "
        f"MU=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {gpu_idx} 2>/dev/null | head -1); "
        "DK=$(df / | tail -1 | awk '{print $5}' | tr -d '%'); "
        'echo "NP=$NP NG=$NG MU=$MU DK=$DK"'
    )
    cmd = ["ssh", "-p", str(port), "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no",
           "-o", "ConnectTimeout=8", "-o", "BatchMode=yes", f"root@{host}", probe]
    try:
        out = subprocess.check_output(cmd, timeout=14, stderr=subprocess.DEVNULL).decode()
        d = {}
        for tok in out.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                d[k] = v
        nproc = int(d.get("NP", "0") or 0)
        ngpu = max(1, int(d.get("NG", "1") or 1))
        mem_used = int(d.get("MU", "999999") or 999999)
        disk_pct = int(d.get("DK", "0") or 0)
    except Exception:
        return False  # SSH/probe failed → treat as busy (never launch blind)

    # DISK-FULL guard (2026-06-06, after ssh5_3060 silently fast-failed every task
    # at 100% disk and became a failure sink that ate a whole experiment batch):
    # never launch onto a box at >=93% — runs would crash at checkpoint/CSV writes.
    if disk_pct >= 93:
        log(f"{tag}: disk {disk_pct}% — refusing to launch (clean checkpoints!)")
        return False
    # Fleet-wide saturation guard: if as many run_benchmark procs as GPUs are
    # already alive on this box, every GPU is taken — do not stack.
    if nproc >= ngpu:
        return False
    # Targeted-GPU guard: the specific GPU we'd use must be genuinely free.
    return mem_used <= 100


# ── Task launch ───────────────────────────────────────────────────────────────

def repo_git_sha() -> str:
    """TRUE provenance of the code rsync_code is about to ship: the control-plane
    repo's short git SHA, with a '-dirty' suffix if scripts/ or src/ have
    uncommitted changes. Recorded per launch so a stale/typo'd CODE_SHA label in
    the task env can never silently misrepresent what actually ran."""
    try:
        sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if sha.returncode != 0:
            return "unknown"
        out = sha.stdout.strip()
        # Dirty = uncommitted CODE (src/ + launcher scripts). Exclude scripts/queues/
        # because that's live queue state the daemon itself rewrites every poll —
        # it must not count as a code change.
        dirty = subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain", "--",
             "src", "scripts", ":(exclude)scripts/queues"],
            capture_output=True, text=True, timeout=10)
        if dirty.stdout.strip():
            out += "-dirty"
        return out
    except Exception:
        return "unknown"


def rsync_code(port: int, host: str):
    """Rsync launcher and source code needed by queued experiment tasks."""
    mkdir_cmd = [
        "ssh", "-p", str(port), "-i", SSH_KEY,
        "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15",
        "-o", "BatchMode=yes", f"root@{host}",
        "mkdir -p /root/helios-rl/scripts /root/helios-rl/src",
    ]
    try:
        subprocess.run(mkdir_cmd, timeout=30, capture_output=True, check=True)
    except Exception as e:
        log(f"mkdir remote code dirs on {host}:{port} failed: {e}")
    for rel in ("scripts", "src"):
        cmd = [
            "rsync", "-az", "--delete",
            "-e", f"ssh -p {port} -i {SSH_KEY} -o StrictHostKeyChecking=no -o ConnectTimeout=15",
            str(REPO / rel) + "/",
            f"root@{host}:/root/helios-rl/{rel}/",
        ]
        try:
            subprocess.run(cmd, timeout=90, capture_output=True, check=True)
        except Exception as e:
            log(f"rsync {rel}/ to {host}:{port} failed: {e}")


def launch_task(task: dict, tag: str, port: int, host: str):
    """Launch a task on the given box. Fire-and-forget."""
    env = task["env"]
    # Inject CUDA mask for dual-GPU boxes if not already in env.
    mask = CUDA_MASK.get(tag, "")
    if mask and "CUDA_VISIBLE_DEVICES" not in env:
        env = f"{mask} {env}"
    # Inject default mem fraction if not already set.
    mem_key = "XLA_PYTHON_CLIENT_MEM_FRACTION"
    if mem_key not in env:
        env = f"{env} {mem_key}={DEFAULT_MEM.get(tag, '0.65')}"
    if "MUJOCO_GL" not in env:
        env = f"{env} MUJOCO_GL=egl"
    if "PYOPENGL_PLATFORM" not in env:
        env = f"{env} PYOPENGL_PLATFORM=egl"
    log(f"{tag} → launching task {task['id']}: {task['label']}")

    if tag == "local":
        log_local = f"/tmp/tqd_{task['id']}.log"
        try:
            try:
                Path(log_local).unlink(missing_ok=True)
            except Exception:
                pass
            proc_env = os.environ.copy()
            for item in shlex.split(env):
                if "=" in item:
                    k, v = item.split("=", 1)
                    proc_env[k] = v
            fh = open(log_local, "w")
            subprocess.Popen(
                ["bash", task["launcher"]],
                cwd=REPO,
                env=proc_env,
                stdin=subprocess.DEVNULL,
                stdout=fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        except Exception as e:
            log(f"local launch error: {e}")
        return

    # Remote: sync launcher + algorithm source first, then SSH-launch.
    rsync_code(port, host)

    log_remote = f"/tmp/tqd_{task['id']}.log"
    # Build the remote command as a Python string — passed as a SINGLE argument
    # to ssh so the local shell never word-splits the env vars.
    remote_cmd = (
        f"cd /root/helios-rl ; "
        f"{env} nohup setsid bash {task['launcher']} "
        f"> {log_remote} 2>&1 < /dev/null & disown ; sleep 1"
    )
    ssh_cmd = [
        "ssh", "-f", "-n", "-p", str(port), "-i", SSH_KEY,
        "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15",
        f"root@{host}",
        remote_cmd,
    ]
    try:
        subprocess.Popen(ssh_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        log(f"{tag} launch error: {e}")


def task_log_tail(task: dict, n: int = 80) -> str:
    """Best-effort tail of a queue task's launcher log."""
    task_id = task.get("id", "")
    box = task.get("box", "")
    log_path = f"/tmp/tqd_{task_id}.log"
    if box == "local":
        try:
            return subprocess.check_output(["tail", "-n", str(n), log_path], timeout=5).decode(errors="replace")
        except Exception:
            return ""
    box_info = next((b for b in BOXES if b[0] == box), None)
    if not box_info:
        return ""
    _tag, port, host, _gpu_idx = box_info
    cmd = [
        "ssh", "-p", str(port), "-i", SSH_KEY,
        "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=8",
        "-o", "BatchMode=yes", f"root@{host}",
        f"tail -n {int(n)} {shlex.quote(log_path)} 2>/dev/null || true",
    ]
    try:
        return subprocess.check_output(cmd, timeout=12).decode(errors="replace")
    except Exception:
        return ""


def task_eval_log(task: dict) -> str:
    """Best-effort eval-line extraction from a task log."""
    task_id = task.get("id", "")
    box = task.get("box", "")
    log_path = f"/tmp/tqd_{task_id}.log"
    pattern = r"step=.*pi_reward=.*MPPI="
    if box == "local":
        try:
            return subprocess.check_output(["grep", "-E", pattern, log_path], timeout=8).decode(errors="replace")
        except Exception:
            return ""
    box_info = next((b for b in BOXES if b[0] == box), None)
    if not box_info:
        return ""
    _tag, port, host, _gpu_idx = box_info
    cmd = [
        "ssh", "-p", str(port), "-i", SSH_KEY,
        "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=8",
        "-o", "BatchMode=yes", f"root@{host}",
        f"grep -E {shlex.quote(pattern)} {shlex.quote(log_path)} 2>/dev/null || true",
    ]
    try:
        return subprocess.check_output(cmd, timeout=15).decode(errors="replace")
    except Exception:
        return ""


def infer_finished_status(task: dict) -> str:
    """Return done/failed from the launcher log when possible."""
    tail = task_log_tail(task)
    if "No such file or directory" in tail:
        return "failed"
    if "ERROR in " in tail:
        return "failed"
    if "done status=" in tail and "done status=0" not in tail:
        return "failed"
    if "all done status=" in tail and "all done status=0" not in tail:
        return "failed"
    return "done"


def parse_env(env: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in shlex.split(env or ""):
        if "=" in item:
            k, v = item.split("=", 1)
            out[k] = v
    return out


def format_env(env_map: dict[str, str]) -> str:
    return " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env_map.items())


def normalized_family_env(env: str) -> tuple[tuple[str, str], ...]:
    skip = {"SEEDS", "PROBE_ID", "CUDA_VISIBLE_DEVICES"}
    return tuple(sorted((k, v) for k, v in parse_env(env).items() if k not in skip))


MIN_FAILED_PROMOTION_STEPS = 4_000_000


def best_any_and_last_step_from_log(task: dict) -> tuple[float | None, int]:
    text = task_eval_log(task)
    best = None
    last_step = -1
    for line in text.splitlines():
        sm = re.search(r"step=\s*([0-9,]+)", line)
        if sm:
            try:
                last_step = max(last_step, int(sm.group(1).replace(",", "")))
            except ValueError:
                pass
        m = re.search(r"pi_reward=\s*([-+]?\d+(?:\.\d+)?)\s+MPPI=\s*([-+]?\d+(?:\.\d+)?)", line)
        if not m:
            continue
        val = max(float(m.group(1)), float(m.group(2)))
        best = val if best is None else max(best, val)
    return best, last_step


def existing_family_seeds(tasks: list[dict], family_key: tuple[tuple[str, str], ...]) -> set[int]:
    seeds: set[int] = set()
    for task in tasks:
        if normalized_family_env(task.get("env", "")) != family_key:
            continue
        for token in parse_env(task.get("env", "")).get("SEEDS", "").split():
            try:
                seeds.add(int(token))
            except ValueError:
                pass
    return seeds


def auto_promote_task(tasks: list[dict], task: dict, status: str) -> None:
    """Append follow-up seed tasks according to Iteration 9 promotion thresholds.

    Kill-switch: DISABLED by default (iteration-11 uses manually-queued seed sets,
    so auto-promotion would only add noise/cost). Set AUTO_PROMOTE=1 in the daemon
    env to re-enable the original threshold-based seed fan-out."""
    if os.environ.get("AUTO_PROMOTE", "0") != "1":
        return
    if task.get("type") == "render" or task.get("auto_promoted"):
        return
    launcher = task.get("launcher", "")
    env = task.get("env", "")
    if "run_phasei9_glass_probe.sh" not in launcher or "SEEDS=" not in env:
        return
    best, last_step = best_any_and_last_step_from_log(task)
    if best is None:
        return
    if status == "failed" and last_step < MIN_FAILED_PROMOTION_STEPS:
        task["auto_promoted"] = {
            "best_any": best,
            "status": status,
            "added": 0,
            "at": now_iso(),
            "skip_reason": f"failed_before_{MIN_FAILED_PROMOTION_STEPS}_steps",
            "last_step": last_step,
        }
        log(f"skip auto-promote {task['id']}: failed at {last_step} < {MIN_FAILED_PROMOTION_STEPS}")
        return

    # Failed-but-informative runs get a 100-point lower bar. This covers SIGKILL
    # or infra failures after useful eval rows, without promoting header-only runs.
    discount = 100.0 if status == "failed" else 0.0
    if best >= 600.0 - discount:
        add_count = 5
    elif best >= 500.0 - discount:
        add_count = 2
    elif best >= 380.0 - discount:
        add_count = 1
    else:
        return

    family_key = normalized_family_env(env)
    existing = existing_family_seeds(tasks, family_key)
    env_map = parse_env(env)
    base_probe = env_map.get("PROBE_ID", task.get("id", "probe"))
    priority = int(task.get("priority", 7))
    added = 0
    for seed in range(1, 6):
        if seed in existing:
            continue
        new_env = dict(env_map)
        new_env["SEEDS"] = str(seed)
        new_env["PROBE_ID"] = f"{base_probe}_auto_s{seed}"
        new_task = {
            "id": "t" + uuid.uuid4().hex[:7],
            "label": f"auto-promote {base_probe} seed {seed} from {task['id']} best_any={best:.1f}",
            "launcher": launcher,
            "env": format_env(new_env),
            "priority": priority,
            "status": "pending",
            "box": None,
            "created_at": now_iso(),
            "started_at": None,
            "ended_at": None,
            "auto_parent": task["id"],
            "auto_reason": f"best_any={best:.1f} status={status} add_count={add_count}",
        }
        tasks.append(new_task)
        existing.add(seed)
        added += 1
        if added >= add_count:
            break
    task["auto_promoted"] = {
        "best_any": best,
        "status": status,
        "added": added,
        "at": now_iso(),
    }
    if added:
        log(f"auto-promoted {task['id']} best_any={best:.1f} status={status}: added {added} seed task(s)")


# ── Main loop ─────────────────────────────────────────────────────────────────

def poll_once():
    tasks = load_queue()
    if not tasks:
        return

    # Check which running tasks have finished (box became idle again).
    running = [t for t in tasks if t["status"] == "running"]
    idle_boxes: set[str] = set()

    for tag, port, host, gpu_idx in BOXES:
        busy = not is_box_idle(tag, port, host, gpu_idx)
        if not busy:
            idle_boxes.add(tag)

    # Mark done: running tasks whose assigned box is now idle.
    changed = False
    for t in running:
        if t.get("box") in idle_boxes:
            status = infer_finished_status(t)
            log(f"task {t['id']} ({t['label']}) {status} on {t['box']}")
            t["status"] = status
            t["ended_at"] = now_iso()
            auto_promote_task(tasks, t, status)
            changed = True

    if changed:
        save_queue(tasks)
        tasks = load_queue()  # reload after save

    # Assign pending tasks to idle boxes that have no running task assigned.
    busy_boxes = {t["box"] for t in tasks if t["status"] == "running" and t.get("box")}
    free_boxes = [b for b in BOXES if b[0] in idle_boxes and b[0] not in busy_boxes]
    pending = sorted(
        [t for t in tasks if t["status"] == "pending" and t.get("type") != "render"],
        key=lambda t: (t["priority"], t["created_at"])
    )

    for box_entry in free_boxes:
        if not pending:
            break
        tag, port, host, gpu_idx = box_entry
        task = pending.pop(0)
        sha = repo_git_sha()  # actual code rsync_code will ship to the worker

        def claim(tasks, _task=task, _tag=tag, _sha=sha):
            for t in tasks:
                if t["id"] == _task["id"] and t["status"] == "pending":
                    t["status"] = "running"
                    t["box"] = _tag
                    t["started_at"] = now_iso()
                    t["launched_git_sha"] = _sha  # TRUE provenance, not the env label
            return tasks

        with_queue_lock(claim)
        if sha.endswith("-dirty"):
            log(f"WARNING: launching task {task['id']} with DIRTY working tree "
                f"(git={sha}); shipped code may not match the CODE_SHA env label")
        # Re-read to get the claimed task's env/launcher.
        tasks = load_queue()
        claimed = next((t for t in tasks if t["id"] == task["id"]), None)
        if claimed and claimed["status"] == "running":
            launch_task(claimed, tag, port, host)
        else:
            log(f"task {task['id']} already claimed by another process, skipping")


def main():
    log(f"start — polling {len(BOXES)} boxes every {POLL_SECONDS}s")
    log(f"queue: {QUEUE_FILE}")
    while True:
        try:
            poll_once()
        except Exception as e:
            log(f"poll error: {e}")
        # Show summary every cycle.
        tasks = load_queue()
        pending = sum(1 for t in tasks if t["status"] == "pending")
        running = sum(1 for t in tasks if t["status"] == "running")
        if pending or running:
            log(f"queue: {pending} pending, {running} running, "
                f"{sum(1 for t in tasks if t['status']=='done')} done")
        else:
            log("queue: all idle or empty")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
