---
layout: post
title: "TD-MPC-Glass, Part 38: When Does Planning Help — and When Does It Hurt?"
date: 2026-06-25
description: "The tool for the beat-PPO plan: a controlled measurement of model-based planning's value across 10 tasks. For each task we compare TD-MPC2 acting with planning (MPPI) vs the reactive policy prior (pi) at convergence — planning advantage = (mppi-pi)/|pi|. It ranges from +39% (AcrobotSwingup) and +13% (PendulumSwingup) down to ~0 on near-saturated tasks (WalkerWalk, BallInCup, ReacherEasy, FingerSpin) and NEGATIVE on PandaPickCube (planning over a reward-hacked value) and CartpoleSwingup. The clean finding: planning's benefit is governed by TASK HEADROOM (is the reactive policy suboptimal — e.g. underactuated swingups needing multi-step lookahead?) and OBJECTIVE SPECIFICATION (is the value/reward right?), NOT by a single model-accuracy scalar — the held-out value-decode R2 does not predict the advantage. The actionable rule for beating PPO: deploy planning where the task is long-horizon/hard AND the objective is clean; distrust it where the value is mis-specified."
---

> Beating PPO on a headroom task means bringing a sharper instrument than a reactive policy. That instrument is
> planning — but planning isn't free, and Part 32 already showed it can *hurt* (PandaPickCube). Part 38 measures
> *when* it helps, across 10 tasks, so we can use it deliberately.

## The measurement

TD-MPC2 learns a model + value and, at action time, can either (a) act with the **reactive policy prior**
(`pi`, no planning) or (b) run **MPPI planning** through the learned model (`mppi`). Both are logged at every
eval. We take the converged numbers and define

$$\text{planning advantage} = \frac{R_{\text{mppi}} - R_{\pi}}{|R_{\pi}|}.$$

10 tasks: 8 DMC trained fresh to 1M (TD-MPC2) plus PandaPickCube and HopperHop from Part 32.

## The result

| task | pi return | mppi return | **planning advantage** | value_r2 |
|---|---|---|---|---|
| AcrobotSwingup | 301.3 | 419.7 | **+39.3%** | 0.92 |
| PendulumSwingup | 814.0 | 922.7 | **+13.4%** | 0.58 |
| HopperHop (Part 32) | 338.7 | 366.8 | **+8.3%** | — |
| CheetahRun | 646.5 | 673.7 | +4.2% | 0.07 |
| FingerSpin | 968.4 | 978.3 | +1.0% | −0.56 |
| ReacherEasy | 983.8 | 993.3 | +1.0% | 0.02 |
| BallInCup | 976.0 | 976.0 | 0.0% | 0.49 |
| WalkerWalk | 974.1 | 974.4 | 0.0% | −0.12 |
| CartpoleSwingup | 768.2 | 756.7 | **−1.5%** | 0.62 |
| PandaPickCube (Part 32) | — | — | **negative** (planning hurts) | — |

## What governs the advantage

Three patterns, and one non-result:

1. **Biggest wins on hard, underactuated, long-horizon tasks.** AcrobotSwingup (+39%) and PendulumSwingup
   (+13%) are swing-up problems where you must act *against* the immediate reward (build momentum) — a reactive
   policy is myopically suboptimal, and multi-step lookahead is exactly the fix. HopperHop (+8%) is the same
   flavor. This is where planning earns its cost.
2. **~0 on near-saturated tasks.** WalkerWalk, BallInCup, ReacherEasy, FingerSpin all sit at ceiling return
   (~970–990); the reactive policy is already near-optimal, so lookahead has nothing to add (0–1%).
3. **Negative when the objective is mis-specified.** On PandaPickCube the learned value is a *reward-hacked
   hover* (Parts 17, 32), so planning optimizes the wrong target and **hurts**. CartpoleSwingup (−1.5%, already
   near-solved) is a mild version.
4. **The non-result:** `value_r2` (held-out value-decode R²) **does not predict** the advantage — CheetahRun
   helps at r2 0.07, CartpoleSwingup *hurts* at r2 0.62, FingerSpin helps at r2 −0.56. A single "model/value
   accuracy scalar" is not the gate.

## The rule — and how it feeds the beat-PPO plan

**Planning helps when (i) the task has headroom for lookahead (the reactive policy is suboptimal — hard /
long-horizon / underactuated) AND (ii) the objective is well-specified; it adds nothing when the policy is
already optimal and *hurts* when the value is mis-specified.** It is not governed by a single accuracy number,
so a useful gate is behavioral (does planning *change* the action toward higher *true* return?) plus an
objective-sanity check, rather than a learned-model-error threshold.

For the campaign's open goal — *beating PPO on a headroom task* — this is the green light: the target task
(PandaOpenCabinet) is **long-horizon and contact-rich** (a drawer pull), exactly the regime where planning paid
off above; if its reward is clean (not reward-hackable like PickCube's dense shaping), then **abstraction prior
+ selective planning** should add real lookahead value where reactive PPO is weakest. That is the next
experiment.

### Caveats
Single seed per DMC task; planning advantage from the run's own pi/mppi eval columns at the converged
checkpoint; `value_r2` is a single final-row number (noisy) and is reported precisely because it *fails* as a
predictor. We did not compute the iterated k-step *dynamics* error per task — the existing tool's phase-binning
is PandaPickCube/HopperHop-specific and crashes on generic DMC obs; a task-agnostic dynamics-error axis is
future work, but the behavioral advantage above already gives the actionable rule. Data:
`exp/tdmpc_glass/<Task>_p38_<Task>/seed_1_phase.csv` (cols pi_return, mppi_return, value_r2); PandaPickCube /
HopperHop from Part 32. Prior: Part 32 (planning hurts PandaPick, helps HopperHop).
