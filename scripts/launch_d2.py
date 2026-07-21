#!/usr/bin/env python3
"""D2 suite orchestrator: vanilla vs jumpy TD-MPC2 across a diverse task set, 2-GPU job pool.

Runs each (arm, task, seed) as one run_benchmark.py process, balancing across GPUs by active
count, capping total concurrency. Detached-friendly (write a status file; survives via setsid).
Requeues a failed job once. Vanilla = no jumpy flags (harvest eval_type=mppi); jumpy =
--jumpy_k 8 --jumpy_plan --jumpy_n_macro 3 (harvest eval_type=jumpy).
"""
import subprocess, time, os, json
from pathlib import Path

REPO = "/root/helios-rl"
VENV_PY = f"{REPO}/.venv/bin/python3"
TASKS = os.environ.get("TASKS", "PandaPickCube,PandaPickCubeOrientation,PandaOpenCabinet,"
                        "CheetahRun,HopperHop,CartpoleSwingupSparse").split(",")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
ARMS = ["van", "jum"]
TOTAL = int(os.environ.get("TOTAL", "500000"))
PER_GPU = int(os.environ.get("PER_GPU", "3"))
NGPU = int(os.environ.get("NGPU", "2"))
MEMFRAC = os.environ.get("MEMFRAC", "0.30")
LOGDIR = Path(f"{REPO}/exp/tdmpc_glass/logs"); LOGDIR.mkdir(parents=True, exist_ok=True)
STATUS = Path(f"{REPO}/exp/tdmpc_glass/d2_status.json")

env_base = dict(os.environ)
env_base["PYTHONPATH"] = f"{REPO}/src:/root/mujoco_playground_repo"
env_base["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
env_base["XLA_PYTHON_CLIENT_MEM_FRACTION"] = MEMFRAC
env_base["MUJOCO_GL"] = "egl"

# seed-major ordering so every task gets seed-0 early (broad early signal)
jobs = [(arm, t, s) for s in SEEDS for t in TASKS for arm in ARMS]
# SHARD="idx/total" → this box runs every total-th job starting at idx (even split across boxes).
# Interleaved (jobs[idx::total]) so each box gets a balanced mix of slow-jumpy and fast-vanilla.
_shard = os.environ.get("SHARD", "").strip()
if _shard:
    _i, _n = (int(x) for x in _shard.split("/"))
    jobs = jobs[_i::_n]
    print(f"SHARD {_shard}: this box runs {len(jobs)} of the full job set", flush=True)
gpu_load = [0] * NGPU
active = []          # [proc, gpu, tag, (arm,task,seed), tries]
done = {}            # tag -> rc

def tag_of(arm, t, s): return f"d2_{arm}_{t}_s{s}"

SKIP_DONE = os.environ.get("SKIP_DONE", "") == "1"
def already_done(arm, t, s):
    """True if this job's eval CSV already has a row at >=98% of TOTAL steps."""
    import glob
    d = glob.glob(f"{REPO}/exp/tdmpc_glass/{t}_d2_{arm}_{t}_s{s}")
    if not d: return False
    csvs = [c for c in glob.glob(d[0] + "/seed_*.csv") if "_diag" not in c and "_arbitration" not in c]
    if not csvs: return False
    try:
        for line in open(csvs[0]):
            p = line.split(",")
            if p[0].isdigit() and int(p[0]) >= int(0.98 * TOTAL):
                return True
    except Exception:
        pass
    return False

def launch(job, tries=1):
    arm, t, s = job
    gpu = min(range(NGPU), key=lambda g: gpu_load[g])
    tag = tag_of(arm, t, s)
    e = dict(env_base); e["CUDA_VISIBLE_DEVICES"] = str(gpu); e["TDMPC_GLASS_OUTPUT_TAG"] = tag
    cmd = [VENV_PY, "-u", "scripts/run_benchmark.py", "--algos", "tdmpc2", "--tasks", t,
           "--total_steps", str(TOTAL), "--seed", str(s), "--no_plot"]
    if arm == "jum":
        # jumpy_k=8 needs buffer seq_len = mppi_horizon+1 >= k, and the anchor convention is
        # MPPI_H >= 2k (jumpy macro planning). Without this, act_b[:, :kk] reshape crashes.
        cmd += ["--jumpy_k", "8", "--jumpy_plan", "--jumpy_n_macro", "3", "--mppi_horizon", "16"]
    f = open(LOGDIR / f"{tag}.log", "a")
    p = subprocess.Popen(cmd, cwd=REPO, env=e, stdout=f, stderr=subprocess.STDOUT)
    gpu_load[gpu] += 1
    active.append([p, gpu, tag, job, tries])
    print(f"LAUNCH {tag} gpu{gpu} pid{p.pid} try{tries}", flush=True)

def write_status():
    STATUS.write_text(json.dumps({
        "total_jobs": len(jobs), "done": len(done), "active": [a[2] for a in active],
        "gpu_load": gpu_load, "failed": [k for k, v in done.items() if v != 0],
    }, indent=2))

qi = 0
while qi < len(jobs) or active:
    for a in active[:]:
        if a[0].poll() is not None:
            rc = a[0].returncode; gpu_load[a[1]] -= 1
            if rc != 0 and a[4] < 2:
                print(f"REQUEUE {a[2]} rc={rc}", flush=True); active.remove(a); launch(a[3], a[4] + 1)
            else:
                done[a[2]] = rc; print(f"DONE {a[2]} rc={rc} ({len(done)}/{len(jobs)})", flush=True)
                active.remove(a)
    while qi < len(jobs) and len(active) < NGPU * PER_GPU:
        job = jobs[qi]; qi += 1
        if SKIP_DONE and already_done(*job):
            done[tag_of(*job)] = 0; print(f"SKIP {tag_of(*job)} (already done)", flush=True); continue
        launch(job)
    write_status(); time.sleep(10)
write_status()
print("ALL_JOBS_DONE", flush=True)
