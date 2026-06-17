# HANDOFF — Paper-A discrimination + GHM repro (2026-06-17)

## What this session set up (two user asks)

### 1. Paper-A FIRST CLAIM: prove the redundancy criterion is PREDICTIVE
The criterion was over-claimed before: R²≈1 + 16 nulls is *consistent* with redundancy but never
*discriminated* (no low-R² case, so no proof R² predicts when abstraction helps). This experiment
supplies the missing discrimination + the causal chain the user asked for:
**R² low → add abstraction → R² high AND return high.**

- **Knobs (already in code):** `--distractor_dims` (inject value-irrelevant OU variance → make the
  latent value-INsufficient → low held-out value-decode R²) × `--bisim_coef` (bisimulation /
  value-equivalence abstraction; grad flows to encoder → should restore value-sufficiency).
- **Probe fixed:** `value_r2` in `scripts/run_benchmark.py` eval_pi now reports **held-out (80/20)
  test R²** of a linear probe latent→return-to-go (in-sample on a 512-d latent overfits to ~1).
  Logged per-eval to `*_phase.csv` (cols: step,seed,pi_return,mppi_return,phase_entropy,value_r2).
- **Matrix (CheetahRun, 500k, EVAL_NEPS=5):** clean/dist × bisim/no-bisim, 3 seeds, + a distractor
  dose-response (dim 32/64/128) hedge, + WalkerWalk 2×2 for a 2nd task.
- **Proof conditions (the criterion is real iff ALL hold):**
  1. clean: R² high, bisim ~NULL (redundant where already value-sufficient — explains the 16 nulls);
  2. distractor: R² LOW + return LOW (value-insufficient base exists);
  3. distractor+bisim: R² RISES and return RISES (abstraction restores sufficiency → the chain).
- **EARLY WATCH (noisy, 50–100k only — do not over-read):** dim=32 distractor R² ≈ 0.89–0.96, not
  dramatically below clean (~0.95). If at 500k the clean-vs-distractor R² gap is <~0.05, dim=32 is
  too weak → the dim=64/128 arms (already queued) are the fallback to find the low-R² regime.

### 2. GHM / CompPlan reproduction (arXiv 2602.19634; on TD-Flow 2503.09817)
- **No official code exists** (CompPlan, TD-Flow, or GHM). Scaffolds cloned to `ghm_repro/`:
  `infom` (JAX flow-occupancy on OGBench — best base), `value-flows` (Bellman flow-matching),
  `ogbench` (envs + base policies). Plan + recipe + task mapping in `docs/ghm_repro/PLAN.md`.
- **Not started training** — GPUs are on the discrimination matrix. Full repro = multi-day. 8h
  realistic = scaffold + env/data pipeline + 1 base policy + GHM draft, NOT reproduced numbers.

## Live infrastructure
- **Orchestrator:** `scripts/paperA_loop.sh` (EC2, detached). Fills slots g3:0, g3:1, g3090:0,
  g3090:1 (g3090:1 gpu-idle-guarded so it won't clobber the leftover jumpy job), detects done via
  "all done status=" marker, harvests `*_phase.csv` → `exp/tdmpc_glass/paperA/mirror/`, SELF-REFILLS
  CheetahRun seeds up to MAXSEED=9 so GPUs never idle. State: `exp/tdmpc_glass/paperA/state.tsv`,
  log: `exp/tdmpc_glass/paperA/loop.log`.
- **Boxes:** g3 = inst 41135265 (2×4090), g3090 = 2×3090. Reachable from EC2 (`ssh g3`, `ssh g3090`).
  User's local ssh to g3 was failing (local key/network) — box is UP; use vast.ai web terminal.
- **Speeds:** CheetahRun 500k ≈ 290 sps on 4090 (~30 min), ~117 sps on 3090 (~1.2 h).

## To resume / on return
1. `python3 scripts/paperA_aggregate.py` → the criterion table (R² + return per config, mean/seeds).
2. Read the CRITERION CHECK block: is the chain present (distractor R² low → bisim raises R²+return)?
3. If dim=32 too weak, look at dim=64/128 rows. If criterion holds → write Paper-A §1 with this table.
4. GHM: allocate a GPU (or new box), follow `docs/ghm_repro/PLAN.md` steps 1–5.
- Orchestrator restart if dead: `setsid bash scripts/paperA_loop.sh >/dev/null 2>&1 < /dev/null &`
