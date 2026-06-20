---
layout: post
title: "TD-MPC-Glass, Part 16: Two Roads to a Jumpy World Model — Our Skip-TD-MPC2 vs CompPlan/GHM on Cube Manipulation"
date: 2026-06-20
description: "Our goal is an abstract world model that beats flat TD-MPC2. 'Jumpy' (temporal-abstraction) world models are the lever we explored — two roads to one. Road 1 (ours): online macro-action MBRL — a k-step jumpy TD-MPC2 that plans over macro-steps; it delivers CI-separated gains in the dense *shaping return* on contact manipulation (PickCube +60%, Pick-Orient +161%) — but ⚠️ CORRECTION: video eval proved that return is reward-hacked (0% real picks for vanilla AND jumpy); a heuristic controller does solve the task. Road 2 (the SOTA baseline): CompPlan/GHM — offline flow-occupancy planning over policies; we reproduced its InFOM scaffold (match/exceed on cube-single) but found the CompPlan planner does NOT reproduce out-of-box, and localized exactly why in three layers, ending at an action-agnostic occupancy that needs a faithful action-conditional TD-Flow. This post puts the two roads side by side (honestly — they're different paradigms), reports the CompPlan diagnosis as a baseline, and carries the reward-hacking correction."
---

> The project's one robust win over flat TD-MPC2 is **temporal abstraction** — predicting the future
> in jumps. This post compares the two ways to build a jumpy world model: our online macro-action
> route, and the SOTA offline flow-occupancy route (CompPlan/GHM). One works on contact manipulation
> today; the other is a promising direction we reproduced and diagnosed down to the exact missing piece.

> ## ⚠️ Correction (2026-06-20): the Road-1 "wins" are on a reward-hacked metric
>
> **The PandaPickCube/Pick-Orient "+60%/+161%" numbers below are on the dense *shaping return*, and
> video evaluation later proved that return is reward-hacked: TD-MPC2 — vanilla *and* jumpy — achieves
> 0% real pick success (`box_target ≥ 0.9` never fires; the cube is never lifted to target). The agent
> hovers the gripper near the cube to bank the dense `gripper_box` term without ever grasping.** So the
> "+60%" is *one hover out-scoring another hover*, **not** a task-capability gain. Concretely:
>
> - The reward is `4·gripper_box + 8·box_target·reached_box + 0.25·no_floor + 0.3·robot_qpos` (max
>   12.55/step). The eval **return is summed over a 1000-step rollout** (the 150-step env auto-resets
>   ~6.7×), so its ceiling is ~12,550. Vanilla plateaus at **~2,500 with `box_target_max = 0.0000`** —
>   leaving ~10,000 on the table by **never picking**. That gap *is* the hack, quantified.
> - A hand-iterated **heuristic controller** (the "learning beyond gradients" route) *does* solve it —
>   real grasp→lift→place, ~9% video-verified success, 99% grasp — cracking the cube-**orientation**
>   wall with analytic level-gripper IK. So the task is solvable; **gradient RL just reward-hacks it.**
> - The reward is well-*ordered* (real success = 965 vs hover 315 per 150-step episode, 3.1×) but
>   badly-*shaped*: the hover banks ~89% of its return from the dense proximity term, a local optimum
>   the learner never escapes.
>
> The two-roads landscape below still stands as written; treat the Road-1 return deltas as *shaping-return*
> deltas, not success. Full write-up: Part 17.

## Two roads to "jumpy"

| | **Road 1 — ours (skip-TD-MPC2)** | **Road 2 — CompPlan / GHM** |
|---|---|---|
| idea | predict the latent **k steps ahead** (z→z_{t+k}); plan with macro-MPPI over k·n_macro | model the **discounted future-state (occupancy) measure**; plan over policy sequences via "jumps" |
| setting | **online** model-based RL | **offline** goal-conditioned |
| stack / tasks | mujoco_playground **MJX** (PandaPickCube, …) | **OGBench** (cube-single, antmaze) |
| metric | episode **return** (dense reward) | **success rate** (goal-reaching) |

They're both jumpy world models, but **different paradigms and metrics** — so this is a *landscape*
comparison, not an identical-protocol head-to-head (a true one needs porting one route into the
other's setting; see the end). With that caveat stated up front, here's where each stands.

## Road 1 — our skip-TD-MPC2 (works on contact manipulation)

A k=8 macro-dynamics head + macro-MPPI (effective horizon k·n_macro = 24), single-variable vs vanilla
TD-MPC2, 5 seeds × 500k steps (Part 12). Measured gains (CI-separated on peak *and* final, every seed):

| task | jumpy vs vanilla (⚠️ *shaping return*, 1000-step eval — **not** real success) |
|---|---|
| PandaPickCube | **+60%** (1854 → 2967 return) — *both 0% real picks* |
| PandaPickCubeOrientation | **+161%** (1008 → 2636) — *both 0% real picks* |
| PandaOpenCabinet | late-training **stability only** (no peak gain) |
| locomotion / sparse | neutral-to-harmful |

These are genuine, CI-separated differences **on the dense shaping return** — but, per the correction
above, that return is reward-hacked (0% real picks for both arms). So jumpy improves *hover quality*,
not task success; this is **not** evidence that the abstract WM solves contact manipulation. The honest
positive result is narrower: jumpy changes the dense-return optimization landscape; whether it helps
*real* success is answered by the in-flight 1.5M real-success campaign (scored on `box_target ≥ 0.9`).

## Road 2 — CompPlan / GHM (the SOTA baseline): reproduced, then diagnosed

We built on **InFOM** (open-source JAX flow-occupancy) and extended it toward **CompPlan**
(plan-over-policies with a Geometric Horizon Model, on **TD-Flow**). Findings:

1. **InFOM reproduces on cube-single** — match/exceed the paper on all 5 tasks (peak success
   1.00/0.94/0.90/0.96/0.90 vs InFOM's 92.5/78.4/56.4/91.5/70.0). The scaffold is faithful.
2. **CompPlan's planner does NOT reproduce out-of-box** — it *hurts* the base policy. We localized it
   in three layers:
   - **Layer 1 — occupancy collapse (fixed).** The GHM imagined only "I'll stay here" (256 samples
     within 0.5 of a 20-unit maze). Fixed by Fourier-embedding the horizon and raising the training
     discount → occupancy spreads to ~12 units.
   - **Layer 2 — an eval normalization bug (fixed).** The GHM was queried in raw-obs space though
     trained normalized; subgoals pointed *away* from the goal. After the fix, subgoal geometry is
     excellent (97–99% closer, on-path).
   - **Layer 3 — action-agnostic occupancy (the real blocker).** Even fixed, CompPlan only *seemed*
     to help on the training goal (task1 0.25→0.36) and **collapsed on a different goal** (task2
     0.26→0.05). A direct diagnostic (`probe_cond`) shows the GHM's future depends on the conditioning
     **action** only ~8–13% as much as on sampling noise: InFOM's intention latent absorbs the action,
     so the model is **marginal** occupancy, not **state-action reachability** — which CompPlan needs
     to steer to an arbitrary goal. The apparent lift is a cloud-overlap artifact that doesn't transfer.

**CompPlan verdict (as a baseline):** occupancy-spread + normalization + horizon tuning + full
training are all insufficient on the InFOM scaffold; a faithful **action-conditional TD-Flow** is
required. It's a *promising direction*, not a working baseline as-is — and we now know the exact piece
it needs.

## So: which road for our goal (beat flat TD-MPC2 with abstraction)?

- **Road 1 moves the dense-return landscape** on contact manipulation (+60/+161% shaping return) —
  but ⚠️ that return is reward-hacked (0% real picks), so it does **not** yet deliver real task success.
  The honest contact-manipulation win to date is the **heuristic controller**, not the abstract WM.
- **Road 2 (CompPlan/GHM) is the latest promising direction**, but the offline flow-occupancy route
  needs the action-conditional TD-Flow piece to actually plan to arbitrary goals — we reproduced the
  scaffold and diagnosed the gap precisely, so it's a well-scoped (multi-day) build if we pursue it.

## The honest caveat on the comparison (and how to make it apples-to-apples)

Our skip-TD-MPC2 (online, MJX-Panda, return) and CompPlan/InFOM (offline, OGBench-cube, success) are
the **same task family** (Franka single-cube manipulation) but **different protocols**. A true
identical-protocol head-to-head requires a port: either run our macro-action mechanism inside the
OGBench/InFOM harness on `cube-single`, or generate offline goal-conditioned datasets for our MJX
Panda tasks and run CompPlan there. Until then, treat the two roads as a *landscape*, each measured
against its own native baseline — which is what we've done here.

### Pointers
Our skip-WM: Part 12 + `exp/tdmpc_glass/mechcheck/d2_suite_results.json`. CompPlan diagnosis: Part 15,
`exp/tdmpc_glass/ghm/FIX_RESULTS.md`, `probe_cond.py`, `occupancy_probe.py`. CompPlan arXiv 2602.19634
· TD-Flow 2503.09817 · InFOM 2506.08902.
