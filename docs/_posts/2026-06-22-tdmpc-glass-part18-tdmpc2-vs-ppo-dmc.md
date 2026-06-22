---
layout: post
title: "TD-MPC-Glass, Part 18: Does TD-MPC2 Actually Beat PPO? Sample-Efficiency vs Wall-Clock on DMC"
date: 2026-06-22
description: "TD-MPC2's reputation is sample-efficiency — but on which axis does it actually beat model-free PPO? We ran TD-MPC2 (1M env-steps, dispatch-bound) against the official MuJoCo-Playground brax PPO (50–285M steps, massively parallel) on six DM-Control tasks, on the same hardware, measuring three things: return at matched env-steps, final return, and wall-clock time. The honest answer: TD-MPC2 wins sample-efficiency decisively (28–160× fewer steps to a given return) — that claim holds. But on final return it's a split decision (PPO matches or beats it on 3/6 tasks given its budget), PPO wins wall-clock (~9× faster to a competence threshold; 50–285× the throughput), and robustness is split — PPO flatlines on HopperHop while default TD-MPC2 diverges to NaN on HumanoidRun. 'TD-MPC2 beats PPO' is true only, and narrowly, per-env-step."
---

> The question behind this whole project — is a strong world model worth it? — has a concrete sub-question
> we'd never directly tested: **does TD-MPC2 actually beat PPO, and on what axis?** Its fame is
> *sample-efficiency*. But sample-efficiency (per env-step) and wall-clock (per second) are different
> things, and a benchmark suite's PPO can use 100× more steps in less time. So we measured both, on the
> same GPU, on six DM-Control tasks chosen to span easy → hard.

## Setup

- **TD-MPC2** (ours): 1M env-steps, default config, model-based (MPPI planning + learned world model),
  **dispatch-bound** (~250–340 steps/s on an RTX 3060).
- **PPO** (the official MuJoCo-Playground `brax_ppo_config`, arXiv 2502.08844): 50–285M env-steps (its
  tuned budget), 2048 parallel envs, **massively parallel** (orders of magnitude more steps/s).
- Same box (RTX 3060) for a fair **wall-clock** comparison. Success/return = the standard DMC return
  (max ~1000). For CheetahRun/HopperHop we corroborate TD-MPC2's return with our **5-seed** Part-2 "d2"
  runs (the table below is the same-box single-seed re-run, used so wall-clock is consistent across all
  six tasks; the 5-seed means agree — e.g. CheetahRun ~717). The paper reports PPO **and** SAC (not
  TD-MPC2) on DMC with both return-vs-step and return-vs-wallclock curves, "<10 min on an A100" — our
  contribution is putting TD-MPC2 *into* that comparison. (Our Glass variant ≈ TD-MPC2 on DMC, a Part-2
  null, so it isn't broken out here.)

## The numbers (read from logs/CSVs)

| task | TD-MPC2 @500k | TD-MPC2 final (~1M) | PPO @500k | PPO final (steps) | sample-eff winner | final winner |
|---|---|---|---|---|---|---|
| WalkerWalk | **964** | 970 | 34 | 970 (49M) | TD-MPC2 (~28×) | tie |
| CheetahRun | **541** | 639 *(5-seed 717)* | 3.4 | **928** (187M) | TD-MPC2 (~160×) | **PPO** |
| AcrobotSwingup | **320** | **415** | 5.0 | 268 (285M) | TD-MPC2 (~64×) | **TD-MPC2** |
| HopperHop | 0 | **367** | 0.3 | 33 (285M) | — (both slow) | **TD-MPC2** (PPO flatlines) |
| CartpoleSwingupSparse | 91 | 109 | 0 | **806** (246M) | TD-MPC2 early | **PPO** (TD-MPC2 stalls) |
| HumanoidRun | **NaN — diverged** (peak 12.6 @100k) | NaN | 1.5 | **≥159, still training** | — | **PPO** (TD-MPC2 fails) |

**Wall-clock:** TD-MPC2 takes ~3,000–3,700 s for its 1M steps; PPO takes ~1,200–3,000 s for **50–285M**
steps — i.e. PPO's throughput is ~**50–285×** TD-MPC2's. To reach a common competence threshold on
CheetahRun (80% of the better final): **TD-MPC2 ~1,235 s vs PPO ~134 s — PPO ~9× faster.**

## The honest, four-axis verdict

1. **Sample-efficiency (return @ matched env-steps): TD-MPC2 wins, decisively.** Wherever it's stable it
   reaches a given return in **28–160× fewer env-steps** (CheetahRun 541 vs 3.4; WalkerWalk 964 vs 34;
   Acrobot 320 vs 5). **This is its real, reproducible strength** — the reputation is earned.
2. **Final / asymptotic return: a split decision.** Given its 50–285M-step budget, PPO **matches or
   beats** TD-MPC2 on 3/6 (CheetahRun 928>717, CartpoleSparse 806>109, HumanoidRun ≥159 vs NaN), while
   TD-MPC2 wins 2/6 (HopperHop, Acrobot) and ties WalkerWalk. So **"TD-MPC2 beats PPO on final return"
   is false in general** — brute-force on-policy sampling often catches up or surpasses.
3. **Wall-clock: PPO wins** (~9× faster to competence; 50–285× the throughput). For a fixed *time*
   budget on commodity hardware, PPO is the practical choice.
4. **Robustness: split, and both have failure modes.** PPO **flatlines on HopperHop** (33) and is slow
   on sparse exploration; **default TD-MPC2 diverges to NaN on HumanoidRun** (high-dim, 67-obs/21-act)
   and **stalls on sparse CartpoleSwingupSparse** (109 vs PPO's 806).

## So — does TD-MPC2 beat PPO?

**Only, and narrowly, on sample-efficiency.** Per env-step it is 1–2 orders of magnitude ahead. But it
does *not* dominate on final return (task-dependent, PPO often wins), it loses on wall-clock (its
planning + per-step model updates are dispatch-bound, ~250 steps/s vs PPO's parallel firehose), and it
is *less robust* out-of-box on high-dim and sparse tasks. The "world model >> model-free" story is real
but **specifically about interaction budget** — which matters in the real world (sample cost) and not at
all in a fast simulator (where PPO's throughput wins the clock). That neatly explains why, across this
project's contact-manipulation work on commodity GPUs, **PPO was repeatedly the practical winner**: in a
simulator, wall-clock is the binding constraint, and that's PPO's axis.

### Caveats (kept honest)
- TD-MPC2 here is **default config, single-seed** for the same-box runs (5-seed d2 corroborates the
  CheetahRun/HopperHop returns). Its HumanoidRun divergence and sparse-Cartpole stall are real but
  **likely tunable** (the official TD-MPC2 uses larger models for high-dim tasks; more steps would help
  sparse) — read them as "default TD-MPC2 is fragile here," not "TD-MPC2 cannot do these."
- PPO is the paper's **tuned official config**; TD-MPC2 is our default. Hardware is a 3060 — on an A100
  PPO's wall-clock edge is even larger.
- HumanoidRun PPO was still training at writing (≥159, climbing); the qualitative verdict (PPO learns it,
  TD-MPC2 diverges) is settled.

### Pointers
**See Part 22** for the full budgets table (steps *and* wall-clock per task), the mechanistic *why* behind
every pro/con here, and the design rules for planning + abstraction that follow.
Raw: `exp/tdmpc_glass/dmc_ppo_vs_tdmpc2/{RESULTS.json,logs/}`; 5-seed d2 curves
`exp/tdmpc_glass/{CheetahRun_d2_van_*,HopperHop_d2_van_*}/seed_*.csv`. Baseline corroboration: MuJoCo
Playground (arXiv 2502.08844), PPO/SAC on DMC. Glass≈TD-MPC2 null: Part 2.
