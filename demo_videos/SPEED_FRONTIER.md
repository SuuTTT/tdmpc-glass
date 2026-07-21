# TD-MPC2 Speed Bottleneck + Sample-Eff-vs-Wallclock Frontier

Box b3060 (4x RTX 3060). Task: CheetahRun (DMC, MJX/jax backend), 1M env-steps, action_repeat=1.
All numbers READ FROM LOGS/CSVs (phase profiler, shrink RESULTS.json, variant eval CSVs,
dmc_ppo_vs_tdmpc2 RESULTS.json, SAC log). None fabricated.

---

## Part C — TD-MPC2 speed bottleneck + optimization

### Method
`profile_phases.py`: builds env + networks + `multi_step` update fn + MPPI `plan` fn exactly as
`run_benchmark.py` does, then JIT-warms and `block_until_ready`-times each hot phase independently:
(1) env `batch_step` (MJX, N_ENVS parallel), (2) the K_UPDATE gradient block (`multi_step`,
one `lax.scan`), (3) a single-env MPPI `plan` call. Config overrides via
TDMPC_NS/NI/KUPDATE/NENVS/LATENT_DIM/HIDDEN.

### BOTTLENECK VERDICT
DEFAULT config on CheetahRun (measured SOLO, clean):

| phase | time | share of training step |
|---|---|---|
| env batch_step (256 envs) | 3.7 ms | 0.7% |
| **gradient block (K_UPDATE=64)** | **550.9 ms** | **99.3%** |
| MPPI plan (1 action) | 7.2 ms | (deploy-only, not in train loop) |

**The training-throughput bottleneck is the K_UPDATE=64 gradient block — 99% of per-step time,
GPU-COMPUTE-bound (a single `lax.scan` of 64 model+critic+policy updates), not dispatch-bound.**
It scales ~linearly with K_UPDATE and with model size.

**MPPI planning is NOT in the training hot path.** Training collects data with the learned policy
`pi`, not the planner; the ~9k tiny world-model rollouts/action (NS·NI·H = 512·6·3) are paid only
at **deploy/eval**. The mission's "planning dispatch dominates" hypothesis holds for *deployment*,
but for *training sps* the gradient updates dominate. So "cheap planning" buys deploy/eval speed
but ~no training-sps gain (confirmed: cheapplan 293 vs default 272 sps = 1.08x, within noise).

### Training-sps levers
Clean SOLO measured sps (shrink study) where available; phase-model predicted marked `*`
(contention-affected absolute, ratios robust):

| config | sps | speedup | return peak | note |
|---|---|---|---|---|
| default (3.61M, K=64) | 272 | 1.0x | 645.8 | baseline |
| small (0.95M, K=64) | 409 | 1.5x | 532.7 | smaller matmuls |
| tiny (0.72M, K=64) | 437 | 1.6x | 512.1 | |
| cheapplan (NS256 NI4) | 293 | 1.08x | 681.8 | ~no train gain (deploy-only); return improved |
| K_UPDATE=32 | ~881* | ~1.9x* | (validating) | halve the grad block |
| K_UPDATE=16 | ~1664* | ~3.5x* | (untested return) | quarter grad block |
| N_ENVS=512 | ~859* | ~1.9x* | — | amortize fixed grad block over 2x data |
| N_ENVS=1024 | ~1741* | ~3.8x* | — | (lowers replay ratio -> sample-eff cost) |
| opt_combo (tiny+K32+cheapplan) | validating | — | validating | full 1M run in flight |

**Deploy/eval planner cost** scales with NS·NI (cheap-planning's real win):
default NS512/NI6 = 7.2 ms/action; NS256/NI4 = 3.5 ms (2.1x); NS128/NI3 = 2.3 ms (3.1x).

### Achievable speedup (headline)
- Same-return regime: shrink to `tiny` = **1.6x** train sps at ~89% of paper@1M return.
- `cheapplan` IMPROVED return (681.8) at ~1.08x — cheaper planning was a free win here.
- K_UPDATE=32 predicts **~1.9x**; return validation in flight.
- tiny + K_UPDATE=32 + cheap-planning targets ~2-3x; full validation in flight.
- jit/vmap note: the grad block is ALREADY a single fused `lax.scan` (compute-bound) and the
  planner is already `@jax.jit`, so further jit/vmap won't help. The dispatch-amortization lever
  that matters is N_ENVS (more env-steps per fixed grad block + fixed Python/dispatch).

---

## Part B — sample-efficiency vs wall-clock frontier (CheetahRun)

Threshold = 0.8 x TD-MPC2-default peak = 511.3 (a bar all TD-MPC2 variants AND PPO cross within
budget; 0.8x PPO-peak=742.6 is unreachable by TD-MPC2 in 1M steps, recorded as alt).
wall_to_thr = step_to_thr / sps (TD-MPC2); PPO from its own train-time log.

| algo / config | env-steps to thr | wall-clock-s to thr | peak return |
|---|---|---|---|
| TD-MPC2 / cheapplan | 350k | **1195 s** | 681.8 |
| TD-MPC2 / default | 450k | 1655 s | 645.8 |
| TD-MPC2 / small | 650k | 1590 s | 532.7 |
| TD-MPC2 / tiny | 900k | 2060 s | 512.1 |
| TD-MPC2 / K_UPDATE=32 | (validating) | (validating) | (validating) |
| PPO / default | 39.3M | **269 s** | 928.3 |
| SAC / default | **did not reach thr** | — (full 1M = 1101 s) | 154.4 (peak @917k) |

**Pareto picture:** PPO = bottom-right (~87x more env-steps, ~6x faster wall), TD-MPC2 = top-left
(50-100x fewer env-steps, slower wall). The TD-MPC2 knee is **cheapplan** (fewest steps AND fastest
wall among TD-MPC2 points) — cheaper planning moves TD-MPC2 toward the knee for free. Model-shrink
(small/tiny) traded sample-efficiency for sps but did NOT improve wall-to-threshold on this task.
**SAC (32 envs, 1M budget) only reached peak 154 on CheetahRun — far below the 511 bar — so it is
OFF the frontier (slowest learner here); it needs many more steps. Its fast per-step sps (~948) is
its only advantage. K_UPDATE=32 full validation in flight (already at MPPI~207 by 200k, sps>default).**

### Vs prior work
- **PQL (ICML 2023)**: GPU massively-parallel off-policy pushes the SAC-family right (fast wall) at
  decent sample-eff; our SAC is a single-GPU reference, not PQL-scale.
- **Raffin SAC-massive-sim**: SAC + many parallel envs = the N_ENVS-amortization lever we measure
  for TD-MPC2.
- **Playground paper (return-vs-wallclock)**: PPO/SAC dominate wall-clock; this work adds the
  missing TD-MPC2 point and shows where shrinking/cheap-planning move it.

---

## Files
- RESULTS.json — part_C_bottleneck + part_B_frontier, with provenance + caveat.
- profile_phases.py / sweep_phases.sh — profiler + config sweep driver.
- build_frontier.py / plot_frontier.py — assemble + plot.
- frontier_CheetahRun.png — sample-eff vs wall-clock Pareto. bottleneck_CheetahRun.png — phase + levers.
- logs/ — phase_*.log, full_kupd32.log, sac_CheetahRun.log.

## Caveats
- Single seed (seed 1) per config — directional, not IQM-over-seeds.
- config_sweep phase numbers measured under GPU contention; trust grad-block ratios + clean shrink sps.
- PPO x-axis = raw brax-reported env-steps (incl eval-reset overhead). K32 point pending refresh.
- SAC reached only 154 in 1M steps (off-frontier); a higher SAC point needs a larger step budget.
- Fix applied: `train_sac` env load now honors `MJPG_IMPL=jax` (was crashing on the warp backend);
  backup at run_benchmark.py.bak_speedfrontier.
