---
layout: post
title: "TD-MPC-Glass, Part 20: How Fast Can TD-MPC2 Go? It's the Gradient Updates, Not the Planning"
date: 2026-06-22
description: "We profiled why TD-MPC2 is ~100× slower than PPO in wall-clock, and the common intuition (including our own) was wrong. The training-throughput bottleneck is NOT the MPPI planner's ~9k tiny rollouts — it's the 64 gradient updates per environment step, which are 95–99% of per-step time and GPU-compute-bound. Planning is only the deploy/eval cost (data is collected by the learned policy, not the planner). Consequences: 'cheaper planning' buys deploy speed but ~zero training speed-up; the real training-speed levers are fewer updates, a smaller model, and more parallel envs — good for ~2–4×, not PPO's ~100× (a gradient update every env-step is architectural). We also place TD-MPC2 on the sample-efficiency-vs-wall-clock frontier (which PPO/SAC studies omit it from): TD-MPC2 sits top-left (50–100× fewer env-steps, ~6× slower wall-clock), and its knee is cheaper-planning."
---

> Part 18 found PPO beats TD-MPC2 on wall-clock by ~9–285×. *Why*, and how much can we close it? We
> profiled it — and corrected our own hypothesis in the process.

## The bottleneck — and the correction

We (and the usual intuition) assumed TD-MPC2 is slow because of **MPPI planning**: every action needs
~`NS·NI·H` = 512·6·3 ≈ **9k tiny world-model rollouts**, a dispatch-latency nightmare. A standalone
phase profiler (JIT-warmed, `block_until_ready`, CheetahRun, RTX 3060) says **that's wrong for training**:

| phase (per env-step, default config) | time | share |
|---|---|---|
| env `batch_step` (256 envs) | 3.7 ms | 0.7% |
| **gradient block (`K_UPDATE`=64, one fused `lax.scan`)** | **~551 ms** | **~99%** |
| MPPI plan (per action) | 7.2 ms | *not in the training loop* |

**The training-throughput bottleneck is the 64 gradient updates per environment step** — and they're
**GPU-compute-bound** (a single fused scan that scales ~linearly with `K_UPDATE` and model size), *not*
dispatch-bound. Crucially, **MPPI planning is not in the training hot path at all**: data is collected by
the learned policy π, not the planner, so the ~9k rollouts are the **deploy/eval** cost, not the training
cost.

This overturns our earlier "dispatch-bound, planning dominates" framing (stated in Parts 18–19). It also
explains a result that had puzzled us: **"cheap planning" (NS 512→256) gave ~no training speed-up**
(1.08×, noise) even though it should have gutted the planner — because the planner was never the training
cost. (It *does* make *deployment* ~2.1× faster, and on CheetahRun it even raised return to 681.8 — a
free win, just not a *training-speed* one.)

## How fast can we make it? ~2–4×, not 100×

The real training-speed levers follow directly from "the grad block dominates":

| lever | speed-up (train sps) | cost |
|---|---|---|
| smaller model (`tiny`, 0.72M) | **1.6×** (437 vs 272 sps) | ~89% of paper return |
| `K_UPDATE` 64→32 | ~1.9× (predicted; validating) | fewer updates/step (sample-eff) |
| `K_UPDATE` 64→16 | ~3.5× | larger sample-eff hit |
| `N_ENVS` 256→1024 | ~3.8× | lower replay ratio → sample-eff cost |
| cheap planning (NS↓) | 1.08× train / **2.1× deploy** | none (return ↑) |

So a realistic **~2–4×** training speed-up is on the table (smaller model + fewer updates + more envs).
But **TD-MPC2 cannot reach PPO's speed (~100×)** — and that's *architectural*, not an implementation
gap: TD-MPC2 does a (compute-heavy) **gradient update on every environment step**, whereas PPO collects
millions of steps from 2048 parallel envs and does a *few* batched updates. The wall-clock gap is the
price of online model-based learning, not slack we can engineer away.

## TD-MPC2 on the sample-efficiency ↔ wall-clock frontier

Studies of this trade-off (Parallel-Q-Learning, ICML 2023; Raffin's SAC-on-massive-sim; the MuJoCo
Playground paper) compare PPO/SAC but **omit model-based TD-MPC2**. Placing it on the frontier
(CheetahRun, same RTX 3060, threshold = 0.8× of TD-MPC2's own peak = 511):

| algo / config | env-steps → thr | wall-clock → thr | peak |
|---|---|---|---|
| **TD-MPC2 cheap-planning** | **350k** | **1195 s** | 682 |
| TD-MPC2 default | 450k | 1655 s | 646 |
| TD-MPC2 small | 650k | 1590 s | 533 |
| TD-MPC2 tiny | 900k | 2060 s | 512 |
| **PPO** | **39.3M** | **269 s** | 928 |

**The Pareto is a clean diagonal:** PPO bottom-right (**~87× more env-steps, ~6× faster wall-clock**),
TD-MPC2 top-left (**50–100× fewer env-steps, slower wall-clock**). Within TD-MPC2, the **knee is
cheap-planning** — fewest env-steps *and* fastest wall-clock *and* highest return. The two axes are in
genuine tension; you pick your constraint (sample budget vs wall-clock), and there's no config that wins
both against the other family.

## Takeaways
1. **TD-MPC2's training cost is its gradient updates, not its planner** — optimize `K_UPDATE`/model/envs,
   not planning, for training speed (and do cheap-planning anyway: free deploy speed + return).
2. **~2–4× is the realistic speed-up; ~100× (PPO parity) is architecturally off the table.**
3. **On the sample-eff↔wall-clock frontier, TD-MPC2 occupies the sample-efficient corner** that the
   PPO/SAC literature leaves empty — that's where it belongs, and where it's worth using (real-robot /
   expensive-sample budgets), not in a fast simulator.

### Caveats
Single seed/config (directional). Default profiled solo (clean); sweep absolutes were under GPU
contention (trust the grad-block *ratios* + clean shrink sps). SAC frontier point + `K_UPDATE`=32
validation in flight. Data: `exp/tdmpc_glass/speed_frontier/RESULTS.json`, `frontier_CheetahRun.png`,
`bottleneck_CheetahRun.png`.
