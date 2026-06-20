# HANDOFF — GHM-fix 8h autonomous run (2026-06-19)

## Done + published this session
- **InFOM cube reproduction CONFIRMED** — match/exceed paper on all 5 cube-single tasks
  (1.00/0.96, 0.94, 0.90, 0.96, 0.90 vs 92.5/78.4/56.4/91.5/70.0). Evidence: mirror CSVs.
- **CompPlan antmaze NOT reproduced** — base 0.30 → +CompPlan 0.10 (paper 0.49→0.85). Root cause
  (mechanism-check): GHM discounted-occupancy collapses to a near-identity (256 samples stay <0.5 of a
  20-unit maze, all γ). `exp/tdmpc_glass/ghm/occupancy_probe.json`.
- **Blog Part 15** published (repo) — the full diagnosed reproduction report.

## Running now (8h, all 4 b3060 GPUs)
- **GHM-fix agent** (anti-collapse search): wave-1 = {horizon_fourier_dim=64; discount=0.997+frac0.25;
  discount=0.999+frac0.5; combined}, ~500k steps each, then `occupancy_probe.py` GATE = does sampled
  occupancy reach far (frac>5 >0 / mean_dist grows from ~0.10)? Wave-2+ refines or tries code-level
  bootstrap fixes. If a fix spreads → full run + `eval_compplan.py` (base vs base+CompPlan). Progress
  log: `/root/ghm/logs/FIX_PROGRESS.md` on b3060.
- **Safety net**: `scripts/ghm_fix_safety.sh` (pid in `exp/tdmpc_glass/ghm/fix_safety.pid`) + cron —
  if all 4 GPUs idle 2 checks (~30min), launches default antmaze GHM runs so GPUs never idle.
- Dashboard: https://sales-court-farm-mba.trycloudflare.com (util + it/s).

## On return — what to read
1. `ssh b3060 cat /root/ghm/logs/FIX_PROGRESS.md` — the per-config occupancy-spread table + verdict.
2. If a config made occupancy spread: check `eval_compplan` base-vs-CompPlan success (the payoff).
3. If nothing spread: confirmed the InFOM bootstrap fundamentally collapses on antmaze → a faithful
   TD-Flow bootstrap is required (multi-day). That's a valid result; the Part-15 finding stands.

## ⚠️ Wave-search continuation (the fix agent EXITED early — I drive waves manually)
The fix agent set up Wave 1 + staged Wave 2 but returned (it expected a "monitor" that won't fire).
A background watcher pings when Wave-1 checkpoints land; then DRIVE each wave by hand:
- **Wave-1 runs:** `/root/ghm/exp_fix/{G0_fourier64,G1_disc997_frac025,G2_disc999_frac05,G3_combo}` —
  500k steps, checkpoint `params_500000.pkl` at the end (~50 min). All use `agents/ghm.py`.
- **GATE (probe each):** `ssh b3060 'cd /root/ghm/infom && . /root/ghm/.venv/bin/activate && PYTHONPATH=/root/ghm/infom MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.15 python planning/probe_one.py <ckpt_dir> 500000 ghm'`.
  SPREADS iff: 0.999 mean_dist > 1.5× the 0.95 mean_dist, OR 0.999 frac>2 > 0.05, OR 0.999 max_dist > 5
  (baseline collapse = mean_dist ~0.10, max ~0.48, frac>5 = 0).
- **If a config SPREADS:** full run (`launch_ghm.sh <gpu> full_<name> agents/ghm.py <its overrides>` with
  pretraining_steps=1000000 finetuning_steps=500000) → then `planning/eval_compplan.py` for base vs base+CompPlan.
- **If NONE spread → Wave 2 (code-fix):** `launch_ghm.sh <gpu> <name> agents/ghm_fix.py --agent.current_weight=<0.1..0.5> --agent.bootstrap_weight=<2..5>` (down-weight stay-put, up-weight propagation). Sweep e.g. cw0.1/bw3, cw0.25/bw2, cw0.05/bw5, cw0.5/bw3. Probe each (agent module = `ghm_fix`).
- launch helper: `launch_ghm.sh <GPU> <NAME> <agent_file> [--agent.* overrides]`. Progress log:
  `/root/ghm/logs/FIX_PROGRESS.md`.

## Fleet notes
- Usable: b3060 (4× 3060) only. b3070 disk-full (mahjong), a4 disk-full (SISA) — both unusable.
- Stop everything: remove the 2 crons (`crontab -e`: drop ghm_fix_safety), kill the agent's runs +
  the safety pid.
