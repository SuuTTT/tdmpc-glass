#!/usr/bin/env python3
"""
pbt_orchestrator.py — Iteration 13: Population-Based Training for basin-entry robustness.

Iteration 12 showed restart-on-plateau rescues SOME stuck seeds (12.2->586, 277->550)
but ~half the re-rolls re-plateau, so it tops out ~2/5 G1 (mean 422) — best yet, but
short of >=3/5. PBT fixes the failure mode directly: instead of each laggard re-rolling
its OWN (often unlucky) actor, it INHERITS a basin-finder's full model (exploit) and
explores with a fresh seed. Since nearly every run produces a basin-finder, PBT should
propagate it and push the G1 rate toward >=3/5.

Truncation PBT, orchestrator-level (NO src/ changes — uses run_benchmark.py's existing
--resume_checkpoint, which restores params+target+opt+scale+glass_step+env_steps+buffer):
  - POOL of homogeneous boxes, each trains one member (tdmpc-glass off@1M) to TOTAL_STEPS.
  - Every INTERVAL_S, read each member's best_any. Rank.
  - EXPLOIT+EXPLORE: each bottom-quartile member whose best < (top best - MARGIN) and is
    past MIN_STEP copies a random top-quartile member's latest checkpoint (donor box ->
    EC2 relay -> laggard box), is killed, and relaunched with --resume_checkpoint + a NEW
    seed (explore). Top/mid members continue uninterrupted.
  - Stop when all members reach TOTAL_STEPS (or MAX_WALL).

This orchestrator OWNS its POOL boxes — they must be REMOVED from task_queue_daemon BOXES
so the queue daemon doesn't also launch on them. State in control/pbt_state.json; log to
exp/.../logs/daemons/pbt.log. Run detached: nohup setsid python3 control/pbt_orchestrator.py
"""
from __future__ import annotations
import json, os, subprocess, time
from pathlib import Path

REPO = Path("/home/ubuntu/tdmpc-glass")
SSH_KEY = os.environ.get("SSH_IDENTITY_FILE", "/home/ubuntu/.ssh/vastai_id_ed25519")
STATE = REPO / "control" / "pbt_state.json"
REMOTE = "/root/helios-rl"
CODE_SHA = "4d3b935"

# Homogeneous A4000 PBT pool (tag, host, sshd_port) — fair PBT comparison.
# 6 boxes = the iteration-12 NON-G1 (laggard/early) seeds, freed for a clean
# from-scratch PBT population. The two G1 basin-finders (ssh1_a4000b seed3=501,
# ssh8_a4000 seed4=550) are LEFT RUNNING in the daemon pool (productive G1 runs —
# guardrail) and fold in here after they finish. Exploit uses only IN-POPULATION
# top members (no external-checkpoint injection) to keep the from-scratch claim clean.
POOL = [
    ("pbt_ssh1_a4000",  "ssh1.vast.ai", 24456),
    ("pbt_ssh2_a4000",  "ssh2.vast.ai", 18950),
    # ssh3_a4000 (ssh3.vast.ai:17426) LOST 2026-06-03 ~21:53Z — vast.ai recycled the
    # instance (SSH host-key changed, publickey now denied). Dropped from the pool.
    ("pbt_ssh4_a4000",  "ssh4.vast.ai", 29168),
    ("pbt_ssh4_a4000b", "ssh4.vast.ai", 10022),
]
# SPARED (not in PBT pool, kept running): ssh9_a4000 seed10=442@1.75M climbing —
# a promising near-G1 run; folds into PBT (or becomes a donor) once it finishes.
TOTAL_STEPS = 10_000_000
INTERVAL_S = 3600            # PBT step cadence (wall-clock)
MIN_STEP_EXPLOIT = 1_500_000 # don't exploit a member before this (let it try first)
EXPLOIT_FRACTION = 0.30      # bottom fraction eligible to be overwritten
MARGIN = 80.0                # only exploit if laggard best < top best - MARGIN
G1 = 500.0
MAX_WALL_S = 30 * 3600

def log(m): print(f"[pbt {time.strftime('%H:%M:%S')}] {m}", flush=True)

def ssh(host, port, cmd, timeout=40):
    try:
        r = subprocess.run(["ssh", "-i", SSH_KEY, "-p", str(port), "-o", "StrictHostKeyChecking=no",
                            "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", f"root@{host}", cmd],
                           capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except Exception as e:
        return 1, "", str(e)

def load_state():
    try: return json.load(open(STATE))
    except Exception: return {"members": {}, "pbt_steps": 0, "started": time.time()}

def save_state(s):
    tmp = str(STATE) + ".tmp"; json.dump(s, open(tmp, "w"), indent=2); os.replace(tmp, STATE)

def member_tag(tag, gen):
    return f"phasei13pbt_{tag}_g{gen}_{CODE_SHA}"

def csv_path(tag, gen, seed):
    t = member_tag(tag, gen)
    return f"{REMOTE}/exp/tdmpc_glass/HopperHop_{t}/seed_{seed}.csv"

def read_best(host, port, tag, gen, seed):
    """Return (best_any, last_step) from the member's CSV on its box."""
    f = csv_path(tag, gen, seed)
    cmd = (f"awk -F, 'NR>1{{if($2>m)m=$2; if($1>s)s=$1}}END{{printf \"%.1f %d\", m, s}}' {f} 2>/dev/null")
    rc, out, _ = ssh(host, port, cmd, timeout=20)
    try:
        b, s = out.split(); return float(b), int(s)
    except Exception:
        return -1.0, -1

def alive(host, port):
    rc, out, _ = ssh(host, port, "pgrep -f '[r]un_benchmark' | wc -l", timeout=20)
    try: return int(out.strip()) > 0
    except Exception: return False

def launch(host, port, tag, gen, seed, resume_path=None):
    """SSH-launch a member: tdmpc-glass off@1M, optional --resume_checkpoint=<resume_path>.
    Pkill any existing run first (prevents the duplicate-launch CSV-corruption mode), but
    as a SEPARATE ssh round-trip — it must NOT share a shell with the launch command, whose
    cmdline contains "run_benchmark.py" and would itself be matched+killed by an in-line
    pkill -f run_benchmark (the [r] bracket trick only protects a command whose own text
    doesn't otherwise contain "run_benchmark"). resume_path is a full remote path or None."""
    otag = member_tag(tag, gen)
    ssh(host, port, "pkill -f '[r]un_benchmark'; sleep 2; exit 0", timeout=30)
    resume = f"--resume_checkpoint {resume_path} " if resume_path else ""
    cmd = (
        f"cd {REMOTE}; "
        f"source /root/venv/bin/activate 2>/dev/null; "
        f"export PYTHONPATH={REMOTE}/src:/root/mujoco_playground_repo MUJOCO_GL=egl "
        f"XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 "
        f"TDMPC_GLASS_OUTPUT_TAG={otag} TMPDIR={REMOTE}/tmp XLA_FLAGS=--xla_gpu_autotune_level=0; "
        f"mkdir -p {REMOTE}/tmp; "
        f"nohup python3 -u scripts/run_benchmark.py --algos tdmpc-glass --tasks HopperHop "
        f"--total_steps {TOTAL_STEPS} --seed {seed} --k_update 128 --mppi_n_samples 2048 "
        f"--expl_until 25000 --latent_action_smooth_coef 0.0 --latent_smooth_warmup_env_steps 0 "
        f"--glass_warmup_env_steps 100000 --glass_decay_steps 1000000 --glass_proto_temperature 0.7 "
        f"--glass_assign_logits_init_scale 0.5 --glass_stopgrad_graph true "
        f"--glass_num_prototypes 32 --glass_num_clusters 8 {resume}--no_plot "
        f"> {REMOTE}/exp/tdmpc_glass/logs/pbt_{tag}_g{gen}.log 2>&1 & disown; sleep 1; echo launched")
    rc, out, err = ssh(host, port, cmd, timeout=40)
    return "launched" in out

def copy_ckpt(donor, laggard):
    """Relay a donor member's latest checkpoint donor-box -> EC2 -> laggard-box as pbt_inherit.pkl."""
    dh, dp, dtag, dgen, dseed = donor
    lh, lp = laggard
    src = f"{REMOTE}/exp/tdmpc_glass/HopperHop_{member_tag(dtag,dgen)}/seed_{dseed}/checkpoints/best_any.pkl"
    local = "/tmp/pbt_inherit.pkl"
    r1 = subprocess.run(["scp", "-i", SSH_KEY, "-P", str(dp), "-o", "StrictHostKeyChecking=no",
                         f"root@{dh}:{src}", local], capture_output=True, text=True, timeout=180)
    if r1.returncode != 0: return False
    r2 = subprocess.run(["scp", "-i", SSH_KEY, "-P", str(lp), "-o", "StrictHostKeyChecking=no",
                         local, f"root@{lh}:{REMOTE}/pbt_inherit.pkl"], capture_output=True, text=True, timeout=180)
    return r2.returncode == 0

def main():
    s = load_state()
    if not s["members"]:
        # Gen 0: launch the whole population fresh (distinct seeds).
        for i, (tag, host, port) in enumerate(POOL):
            seed = 100 + i
            ok = launch(host, port, tag, 0, seed)
            s["members"][tag] = {"host": host, "port": port, "gen": 0, "seed": seed,
                                 "best": -1.0, "step": 0, "launched_ok": ok}
            log(f"gen0 launch {tag} seed={seed} ok={ok}")
        s["started"] = time.time(); save_state(s)

    while True:
        time.sleep(INTERVAL_S)
        if time.time() - s.get("started", time.time()) > MAX_WALL_S:
            log("MAX_WALL reached; orchestrator exiting (members keep running)"); return
        # 1) refresh each member's best/step
        for tag, m in s["members"].items():
            b, st = read_best(m["host"], m["port"], tag, m["gen"], m["seed"])
            if b >= 0: m["best"], m["step"] = max(m["best"], b), st
            m["running"] = alive(m["host"], m["port"])
        # 1b) resurrect CRASHED members so no GPU idles (conservative: 2 consecutive
        #     dead reads to ride out transient SSH failures; launch() pkills first so
        #     a stale process can't double-launch). Resume the member's own checkpoint.
        for tag, m in s["members"].items():
            if m.get("running") or m["step"] >= TOTAL_STEPS:
                m["dead_checks"] = 0; continue
            m["dead_checks"] = m.get("dead_checks", 0) + 1
            if m["dead_checks"] < 2:
                continue
            own_ckpt = (f"{REMOTE}/exp/tdmpc_glass/HopperHop_{member_tag(tag, m['gen'])}/"
                        f"seed_{m['seed']}/checkpoints/latest_eval.pkl")
            _, sout, _ = ssh(m["host"], m["port"], f"test -f {own_ckpt} && echo yes", timeout=20)
            resume = own_ckpt if "yes" in sout else None
            ok = launch(m["host"], m["port"], tag, m["gen"], m["seed"], resume_path=resume)
            log(f"RESURRECT {tag} (dead) resume={'own@'+str(m['step']//1000)+'k' if resume else 'fresh'} ok={ok}")
            m["dead_checks"] = 0; m["launched_ok"] = ok
        save_state(s)
        ranked = sorted(s["members"].items(), key=lambda kv: kv[1]["best"], reverse=True)
        g1 = sum(1 for _, m in s["members"].items() if m["best"] >= G1)
        log(f"PBT step {s['pbt_steps']}: G1={g1}/{len(s['members'])} | " +
            " ".join(f"{t.split('_')[-1]}={m['best']:.0f}@{m['step']/1e6:.1f}M" for t, m in ranked))
        # 2) exploit-explore: bottom fraction inherit a top member + fresh seed.
        #    Rank over LIVE members only (best>=0 AND running). A dead member sits at
        #    best=-1 and would otherwise monopolise the bottom rank, starving the real
        #    live laggard of exploitation (and the exploit-relaunch onto a dead box just
        #    fails). Dead members are handled by the resurrect pass (1b) instead.
        live = [(t, m) for t, m in ranked if m["best"] >= 0 and m.get("running")]
        n = len(live)
        if n < 2:
            log(f"  <2 live members ({n}); skip exploit this step")
            s["pbt_steps"] += 1; save_state(s); continue
        k = max(1, int(EXPLOIT_FRACTION * n))
        top = live[:k]; bottom = live[-k:]
        top_best = top[0][1]["best"]
        for ltag, lm in bottom:
            if lm["best"] >= top_best - MARGIN: continue       # already competitive
            if lm["step"] < MIN_STEP_EXPLOIT: continue          # give it a chance first
            # pick a top donor (deterministic: best, vary by pbt_step for diversity)
            dtag, dm = top[s["pbt_steps"] % len(top)]
            if dm["best"] < G1 - 50: continue                   # only exploit a near/basin donor
            donor = (dm["host"], dm["port"], dtag, dm["gen"], dm["seed"])
            log(f"EXPLOIT {ltag}(best={lm['best']:.0f}) <- {dtag}(best={dm['best']:.0f}) + explore")
            if not copy_ckpt(donor, (lm["host"], lm["port"])):
                log(f"  ckpt copy failed for {ltag}; skip"); continue
            lm["gen"] += 1; lm["seed"] = lm["seed"] + 1000      # explore: fresh RNG
            ok = launch(lm["host"], lm["port"], ltag, lm["gen"], lm["seed"],
                        resume_path=f"{REMOTE}/pbt_inherit.pkl")  # launch() pkills first
            lm["best"] = -1.0; lm["step"] = 0; lm["launched_ok"] = ok; lm["dead_checks"] = 0
            log(f"  relaunched {ltag} gen{lm['gen']} seed={lm['seed']} resume<-donor ok={ok}")
        s["pbt_steps"] += 1; save_state(s)
        if g1 >= max(3, int(0.6 * len(s["members"]))):
            log(f"*** PBT reached >=3/5-equivalent G1 ({g1}/{len(s['members'])}) — robustness target! ***")

if __name__ == "__main__":
    main()
