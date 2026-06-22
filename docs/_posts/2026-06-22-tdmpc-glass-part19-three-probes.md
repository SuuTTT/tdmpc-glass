---
layout: post
title: "TD-MPC-Glass, Part 19: Three Probes — A 5× Smaller TD-MPC2, Jumpy Is Reward-Hacking, and OpenCabinet Hacks Too"
date: 2026-06-22
description: "Three parallel experiments on the 8-GPU fleet, all read from logs. (W4) TD-MPC2 shrinks ~5× (3.6M→0.72M params) and runs ~1.6× faster while keeping 85–100% of paper performance across four DM-Control tasks — and its default MPPI planning is over-provisioned (cheaper planning is free speed). (W2) The 'jumpy' k-step world model — the one abstraction in this project that ever showed a gain — is null-to-harmful on clean sparse-reward DMC: vanilla TD-MPC2 solves CartpoleSwingupSparse (736) while jumpy never exceeds 121, confirming its earlier '+60%' was reward-hacking a dense term, not abstraction. (W3) TD-MPC2 (vanilla and jumpy) reward-hacks PandaOpenCabinet to 0% real success, same as PickCube. A positive efficiency result and two clean negatives."
---

> After the PPO-vs-TD-MPC2 comparison (Part 18), we ran three probes in parallel: can we make TD-MPC2
> *smaller and faster*; is the *jumpy* abstraction real on a clean metric; and does TD-MPC2's world model
> *crack a multi-step skill* (OpenCabinet). One positive, two negatives — all from logs, peak+final.

## W4 — A smaller, faster TD-MPC2 that still matches the paper

The user's question: can we shrink TD-MPC2's params and speed it up without losing performance? We swept
config via env-var overrides (latent_dim, hidden, MPPI samples/iters) on four DM-Control tasks, 1M steps,
measuring **params, throughput, and return** vs the official TD-MPC2 paper (~575 on CheetahRun @1M):

| config | params | speed | CheetahRun | WalkerRun | AcrobotSwingup | FingerSpin |
|---|---|---|---|---|---|---|
| default (latent-512) | 3.6M | 1.0× (273 sps) | 614 | 669 | 420 | 972 |
| small (latent-256) | 0.95M | 1.5× (409) | 532 (87%) | 645 (96%) | 393 (94%) | 980 |
| **tiny (latent-128)** | **0.72M** | **1.6× (437)** | 511 (83%) | 638 (95%) | **418 (≈100%)** | sat. |
| cheap-planning (NS 512→256) | 3.6M | 1.1× | **654** | 670 | — | — |

**Result: yes.** A **tiny TD-MPC2 (0.72M params, 5× smaller, 1.6× faster)** keeps **83–100%** of full
performance across these tasks — and on AcrobotSwingup/WalkerRun/FingerSpin it's within noise of default.
Separately, **cheaper MPPI is free speed**: halving the planning samples (NS 512→256) cost nothing and
sometimes scored *higher* (CheetahRun 654 vs 614) — the default `NS=512, NI=6` planning is
over-provisioned for these tasks. Takeaway: most of TD-MPC2's parameters and planning budget are slack on
standard DMC; you can run a much lighter model for the same return. (Caveat: 1 seed/config; hard high-dim
tasks like HumanoidRun may need the capacity — default TD-MPC2 already *diverged* there, Part 18.)

## W2 — Is the jumpy abstraction real? No: it's reward-hacking

Back in Parts 12/16 the one method that beat vanilla TD-MPC2 was a **jumpy (k-step macro) world model**
— "+60%" on PandaPickCube. Part 17 showed that win was on a **reward-hacked** dense return. So we tested
jumpy where the metric *can't* be hacked: **clean sparse-reward DMC** (return only on success).

| task | vanilla TD-MPC2 (peak return) | jumpy TD-MPC2 |
|---|---|---|
| CartpoleSwingupSparse | **736 / 621** (2 of 3 seeds solve) | **121 / 0 / 0** — never |
| AcrobotSwingupSparse | 170 | 0 |

**Verdict: jumpy is null-to-harmful on clean metrics.** Vanilla TD-MPC2 learns the sparse swing-ups;
jumpy essentially never does — its macro-steps suppress the fine-grained exploration sparse tasks need.
So the "+60%" was the jumpy planner farming the dense `gripper_box` proximity term faster, **not a real
abstraction advantage**. This closes the loop: across this entire project, *no* abstraction we tried —
ours (structural-entropy "Glass") or prior-art (jumpy) — gives a real gain on a clean metric.

## W3 — Does TD-MPC2 crack OpenCabinet's multi-step skill? No: it reward-hacks

PandaOpenCabinet (reach → grasp handle → pull to target) has the *same dense reward shape* as PickCube.
We ran TD-MPC2 (vanilla and jumpy) to 2M steps, scoring real success (`box_target ≥ 0.9`):

| agent | peak box_target | real success |
|---|---|---|
| TD-MPC2 vanilla | 0.20 | **0%** |
| TD-MPC2 jumpy | ~0 | **0%** |
| (PPO, Part 17) | — | 98% @ ~33M |

**Same hover-hack as PickCube.** Neither the world model nor temporal abstraction crosses the multi-step
skill — they farm the proximity term and never pull. PPO solves it (98%) but needed ~33M steps; a
hand-coded open-loop controller couldn't either (the closed-loop reach-frontier wall, Part 17 §6a).

## The through-line
One genuinely useful positive (TD-MPC2 is heavily over-parameterized for DMC — shrink it 5×), and two
negatives that tighten the campaign's thesis: **the abstraction well is dry** (jumpy included, on clean
metrics), and **TD-MPC2 reward-hacks dense manipulation** at the budgets we can run (0% real success on
both PickCube and OpenCabinet). Where world models actually pay off is **sample-efficiency** (Part 18),
not abstraction tricks or wall-clock.

### Pointers
W4: `exp/tdmpc_glass/tdmpc2_shrink/` + `exp/benchmark/tdmpc2_{CheetahRun,WalkerRun,AcrobotSwingup,FingerSpin}_*.csv`.
W2: `tdmpc2_{CartpoleSwingupSparse,AcrobotSwingupSparse}_*.csv`. W3: `tdmpc2_PandaOpenCabinet_opencab_*_realsuccess.csv`. Glass≈TD-MPC2 null: Part 2.
