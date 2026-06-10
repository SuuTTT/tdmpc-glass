# Handoff — TD-MPC-Glass (resume after Claude Code update / model switch)

*Written 2026-06-10 ~10:40Z, before user exits to update Claude Code + switch model.*
*Authoritative resume doc. Also see AGENT_HANDOFF_CONTEXT.md (live status) and CLAUDE.md (rules).*

## 0. FIRST THING TO DO ON RESUME
The previous session ran an **autonomous monitoring loop via `ScheduleWakeup`. That wakeup
is tied to the old session and is now DEAD.** The fleet keeps training (daemons are detached,
ppid 1), but nothing is harvesting/requeueing/reporting. **Re-arm the loop** (ScheduleWakeup,
~1800s) OR drive it manually. The loop's job each tick:
1. Harvest ti27 Panda peak+final per-task from CSVs only (never trust 1 snapshot; report BOTH peak and final).
2. Check the two arch smokes (see §3); if clean, queue the full arch A/B; if crashed, read log → fix → re-smoke.
3. Keep all non-ltsf boxes busy; reset dead `running`→`pending`.
4. Report at milestones.

## 1. What is RUNNING (survives session exit — do NOT restart unless needed)
- 3 control daemons on EC2, all ppid 1 (detached): `task_queue_daemon.py` (pid was 1462431),
  `web_dashboard.py` (port 5055), `iter5_stream_remotes.sh` (mirrors worker CSVs).
- `control/start_center.sh` restarts all three idempotently if any died.
- Fleet: ~10 vast.ai GPU workers. EC2 (this box, 1.9GB RAM, NO GPU) **never trains** — the one rule.

## 2. ti27 — Panda manipulation re-benchmark (the main in-flight experiment)
45 runs: {van, jumpy, value-equiv} × {PandaPickCube, …Cartesian, …Orientation, PandaRobotiqPushCube,
PandaOpenCabinet} × 3 seeds. CODE_SHA=i27a, TOTAL_STEPS=500000. As of handoff: first van seeds
running cleanly on all 5 new tasks (tasks load + train, no crash); ti26_ve DMC runs finishing.
**Pre-registered gates:** (a) does jumpy generalize across the Franka suite — jumpy>van CI-separated on ≥3/5?
(b) does value-equiv (JUMPY_VE_COEF=0.5) help on manipulation — ve≥jum? Configs: van=MPPI_H=3;
jum=JUMPY_K=4 JUMPY_PLAN=1 JUMPY_NMACRO=6 MPPI_H=8; ve=jum+JUMPY_VE_COEF=0.5. Harvest from
`exp/tdmpc_glass/remote_mirror/**/*.csv` (rliable IQM, paired-difference bootstrap is the correct head-to-head test).

## 3. Arch A/B — NEW this session, built & committed (cc72e4f), default-OFF
`--dyn_arch {mlp,attn,resmlp}` (env `DYN_ARCH`): `attn`=group-wise self-attention over SimNorm's 8
latent groups; `resmlp`=deeper gated-residual MLP. Backbone for enc/dyn/jumpy pre-norm latent;
SimNorm geometry unchanged → single-variable swap. Default `mlp` so ti27 is unaffected. py_compile clean.
Code: `src/helios/algorithms/tdmpc2.py` (`GroupAttn`, `ResGatedMLP`, `_arch_backbone`, `arch` field on
Encoder/Dynamics/JumpyDynamics); threaded through `run_benchmark.py` + `run_dmc_baseline.sh`.
**Cannot smoke on EC2 (no JAX/GPU).** Two smoke tasks queued at priority −1: `ti27smoke_attn`,
`ti27smoke_resmlp` (PandaPickCube 20k) — next freed box runs them. **On resume: check their worker
logs for NaN/crash + SPS BEFORE any multi-seed fanout.** If clean → queue van+jumpy × {attn,resmlp} ×
Panda suite × 3 seeds, single-variable vs mlp.

## 4. CURRENT TROUBLES / BLOCKERS (for the new model)
1. **Autonomous loop must be re-armed** (see §0) — the single most important resume action.
2. **The scientific wall:** every explicit-abstraction lever is NULL (Glass geometric+transition-SE,
   adaptive-k, SE-exploration SI2E/wmsi2e, Hermite-spline). Root cause: TD-MPC2's SimNorm +
   self-predictive latent is *already a sufficient abstraction* (measured: 53% SE community gap with
   no added objective). The ONLY win is the **jumpy world model** — and that's largely prior-art
   (Meta Farebrother 2026 is different in detail but the jumpy idea isn't novel). So the open question
   the new powerful model should attack: **is there ANY control-useful abstraction beyond what the
   self-predictive WM already encodes?** The two live probes are (a) value-equivalence loss (ti27 ve arm),
   (b) the arch A/B (attn/resmlp). If both null, the honest verdict is "no novel architecture-level beat;
   publish the mechanism-check methodology + the SimNorm-SE interpretability result instead."
3. **EC2 can't run JAX** → all smoke/validation must go through a worker via the queue (slow loop).
4. **n is small** (3 seeds on most arms) → discriminating Panda subset needs n=5 before any claim.
5. **DMC tasks are poor discriminators** (saturate ~1000 or floor 0) — focus manipulation/Panda.
6. **GitHub token `ghp_...` is in the transcript — user should ROTATE it.** Don't upload `*.pkl`.

## 5. OFF-LIMITS — never schedule tdmpc / never destroy (user-assigned to LTSF/Crossformer)
- **18950** (ssh2_a4000) — LTSF (permanent)
- **38342607** (GTX 1660S x2, port 22607) — LTSF (permanent)
- **38664456** (ssh1.vast.ai:24456) — user's Crossformer/ltsf running
If ssh2/ssh3b show `run.py` (forecasting resumed), remove from daemon BOXES. Never destroy instances
without explicit user confirmation; killing in-flight runs needs authorization.
Use TARGETED pkill (e.g. `_smoke`), never broad `pkill -f run_benchmark` (it kills daemon-launched runs).

## 6. Methodology (the project's core discipline — keep it)
Fair protocol: single-variable, compute-matched, pre-registered peak+final CI gates, mechanism-check
BEFORE fanout (cheap kill-test saved multiple multi-week campaigns), no procedure tricks (PBT/restarts).
Harvest from CSV/JSON only — this project fabricated numbers ~7× before the read-from-JSON discipline.
Report BOTH peak and final; distrust 1-snapshot effects; requeue dead seeds; commit/push progress.
