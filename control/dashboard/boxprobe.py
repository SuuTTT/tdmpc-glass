"""Box registry + SSH probing for the TD-MPC-Glass dashboard.

Single source of truth for the fleet: BOXES is imported directly from the
queue daemon (control/task_queue_daemon.py) so the dashboard and daemon can
never drift. The daemon guards its main loop behind `if __name__ ==
"__main__":`, so this import only loads module-level definitions — it does NOT
start a second daemon process.

The daemon's BOXES tuples are (tag, port, host, gpu_idx). The dashboard needs a
5th `label` field for display; we synthesize one here from the tuple so there's
no second hand-maintained list.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading

# ─── Single source of truth: import the daemon's fleet list ──────────────────
# Read-only import (daemon's main loop is __main__-guarded, so nothing runs).
_CONTROL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CONTROL_DIR not in sys.path:
    sys.path.insert(0, _CONTROL_DIR)
from task_queue_daemon import BOXES as _DAEMON_BOXES  # noqa: E402

# Optional human-readable labels keyed by tag. Any box without an entry falls
# back to a synthesized "host:port gpuN" label. Labels for boxes NOT in the
# daemon's BOXES list are inert (the panel renders only daemon-listed boxes);
# entries for destroyed instances were pruned 2026-06-12.
_BOX_LABELS = {
    "ssh1_2080ti":   "ssh1:34217 2080 Ti (22GB, destroy-pending)",
    "ssh6_titanv":   "ssh6:31740 Titan V (12GB, destroy-pending)",
    "ssh9_a4000":    "ssh9:16690 A4000 (16GB)",
    "ssh4_a4000":    "ssh4:29168 A4000 (16GB, rented)",
    "ssh4_a4000b":   "ssh4:10022 A4000 (16GB, rented)",
    "ssh6_3060":     "ssh6:11696 3060 (12GB)",
    "ssh3b_a4000":   "ssh3:17426 A4000 (16GB, inst 38767427)",
}


def _label_for(tag: str, port: int, host: str, gpu_idx: int) -> str:
    if tag in _BOX_LABELS:
        return _BOX_LABELS[tag]
    short = host.split(".")[0]
    return f"{short}:{port} {tag} g{gpu_idx}"


# Dashboard-facing 5-tuples: (tag, port, host, gpu_idx, label).
BOXES = [
    (tag, port, host, gpu_idx, _label_for(tag, port, host, gpu_idx))
    for (tag, port, host, gpu_idx) in _DAEMON_BOXES
]

# SSH key: remotes accept root@ login with the deployed pubkey.
SSH_KEY = os.environ.get("SSH_IDENTITY_FILE", "/home/ubuntu/.ssh/vastai_id_ed25519")

# Box probe cache — populated by the boxes payload builder, consumed by the
# queue ETA computation.
_BOX_CACHE: dict = {}
_BOX_CACHE_LOCK = threading.Lock()


def parse_etime_seconds(etime: str) -> int | None:
    """Parse `ps -o etime` formats: MM:SS, HH:MM:SS, D-HH:MM:SS. Returns seconds or None."""
    if not etime:
        return None
    try:
        days = 0
        if "-" in etime:
            d, etime = etime.split("-", 1)
            days = int(d)
        parts = [int(p) for p in etime.split(":")]
        if len(parts) == 2:
            h, m, s = 0, parts[0], parts[1]
        elif len(parts) == 3:
            h, m, s = parts
        else:
            return None
        return days * 86400 + h * 3600 + m * 60 + s
    except (ValueError, IndexError):
        return None


# Remote shell snippet — returns one-line ASCII parsable by parse_box_status.
# For each running run_benchmark PID, also reads TDMPC_GLASS_OUTPUT_TAG from
# /proc/<pid>/environ so the host can pin the proc to the right phase CSV.
REMOTE_PROBE = r'''
gpu_idx="$1"
gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits -i "$gpu_idx" 2>/dev/null | head -1)
cpu=$(top -bn1 2>/dev/null | grep -E "^%Cpu" | head -1 | awk '{printf "%.0f", 100-$8}')
printf "GPU=%s\nCPU=%s\n" "$gpu" "$cpu"
ps -eo pid,etime,cmd --no-headers 2>/dev/null | grep -E "run_benchmark" | grep -v grep \
  | awk '{cmd=""; for(i=3;i<=NF;i++) cmd=cmd" "$i; printf "%s\t%s\t%s\n", $1, $2, cmd}' \
  | while IFS=$'\t' read -r pid etime cmd; do
    tag=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null \
          | awk -F= '$1=="TDMPC_GLASS_OUTPUT_TAG"{print $2; exit}')
    cuda=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null \
          | awk -F= '$1=="CUDA_VISIBLE_DEVICES"{print $2; exit}')
    printf "PROC|%s|%s|%s|%s|%s\n" "$pid" "$etime" "$tag" "$cuda" "$cmd"
  done
'''


def parse_box_status(raw: str):
    """Parse REMOTE_PROBE output. Proc lines look like:
       PROC|<pid>|<etime>|<output_tag>|<cuda>|<full_cmd_line>
    """
    gpu_util = mem_used = mem_total = None
    cpu_util = None
    procs = []
    for line in raw.splitlines():
        line = line.rstrip()
        if line.startswith("GPU="):
            v = line[4:].strip()
            parts = [x.strip() for x in v.split(",")]
            if len(parts) >= 3:
                try:
                    gpu_util = int(parts[0].split()[0])
                    mem_used = int(parts[1].split()[0])
                    mem_total = int(parts[2].split()[0])
                except Exception:
                    pass
        elif line.startswith("CPU="):
            try:
                cpu_util = int(line[4:].strip())
            except Exception:
                pass
        elif line.startswith("PROC|"):
            try:
                _, pid, etime, output_tag, cuda, cmd = line.split("|", 5)
                m_seed = re.search(r"--seed\s+(\S+)", cmd)
                m_ns = re.search(r"--mppi_n_samples\s+(\S+)", cmd)
                m_algo = re.search(r"--algos\s+(\S+)", cmd)
                tags = []
                if "--knee_penalty_coef" in cmd:
                    tags.append("knee")
                if "--use_cluster_obs" in cmd:
                    tags.append("cobs")
                if "--glass_num_super_clusters" in cmd:
                    tags.append("hier")
                procs.append({
                    "pid": pid.strip(),
                    "etime": etime.strip(),
                    "seed": m_seed.group(1) if m_seed else "?",
                    "ns": m_ns.group(1) if m_ns else "512",
                    "algo": m_algo.group(1) if m_algo else "?",
                    "tag": "+".join(tags) if tags else "",
                    "output_tag": (output_tag or "").strip(),
                    "cuda_visible": (cuda or "").strip(),
                })
            except Exception:
                pass
    return {
        "gpu_util": gpu_util, "mem_used": mem_used, "mem_total": mem_total,
        "cpu_util": cpu_util, "procs": procs,
    }


def probe_box(tag, port, host, gpu_idx):
    """SSH to box, run REMOTE_PROBE, parse output. Local case: bash subprocess."""
    try:
        if tag == "local":
            res = subprocess.run(
                ["bash", "-c", REMOTE_PROBE, "_", str(gpu_idx)],
                capture_output=True, text=True, timeout=10,
            )
        else:
            res = subprocess.run(
                ["ssh", "-p", str(port), "-i", SSH_KEY,
                 "-o", "StrictHostKeyChecking=no",
                 "-o", "ConnectTimeout=8",
                 "-o", "BatchMode=yes",
                 f"root@{host}", "bash", "-s", "--", str(gpu_idx)],
                input=REMOTE_PROBE, capture_output=True, text=True, timeout=15,
            )
        status = parse_box_status(res.stdout)
        status["reachable"] = (res.returncode == 0)
        return status
    except subprocess.TimeoutExpired:
        return {"reachable": False, "error": "ssh-timeout", "procs": []}
    except Exception as e:
        return {"reachable": False, "error": str(e), "procs": []}
