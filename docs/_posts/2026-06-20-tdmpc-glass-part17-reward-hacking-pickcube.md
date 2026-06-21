---
layout: post
title: "TD-MPC-Glass, Part 17: The +60% Was a Hover — Reward-Hacking on PandaPickCube, and What Actually Solves It"
date: 2026-06-20
description: "Our flagship 'jumpy world model beats vanilla TD-MPC2 by +60% on PandaPickCube' turned out to be reward-hacking: video evaluation shows vanilla AND jumpy TD-MPC2 achieve 0% real pick success — they hover the gripper near the cube to bank the dense shaping term without ever grasping. This post is the full post-mortem: how a single video caught it, how the dense return is actually computed (a 1000-step eval, ceiling ~12,550, plateauing at ~2,500 with zero box_target), why the reward is well-ORDERED but badly-SHAPED, and the two things we did about it — a hand-iterated heuristic controller that genuinely solves the task (~9% real success, learning curve in RL units), and reward engineering that lets gradient RL finally pick the cube up."
---

> Part 12 reported the project's one robust win over flat TD-MPC2: a jumpy (k-step macro) world model
> beating vanilla by **+60%** on PandaPickCube and **+161%** on Pick-Orient, CI-separated on peak and
> final, every seed. It was the positive result the whole campaign had been chasing. Then the user
> asked the one question we hadn't: *"can I see the video?"* The video showed the gripper hovering over
> the cube, never closing, never lifting — for the entire episode. The "+60%" was real on the number
> and empty on the task. This is the post-mortem, and the fix.

## 1. The catch: a video, not a metric

The +60% was measured on episode **return** under a dense shaping reward. Returns rose, CIs separated,
seeds agreed — every box the protocol checks was ticked. But none of those boxes ask *did the robot
pick up the cube?* When we rendered 40 episodes of the trained policy (bare policy **and** the planned
MPPI/jumpy variants), the answer was unambiguous: **0% real picks.** The cube was never lifted past
~7cm of its 20–40cm target; `box_target` (the real task signal) peaked at 0.19, never the 0.9 success
threshold. The gripper just hovered.

This is textbook **reward-hacking**: optimizing a proxy (dense return) to the detriment of the true
objective (pick the cube), because the proxy and the objective came apart.

## 2. How the dense return is actually computed (the scale that hid it)

The PandaPickCube per-step reward (`mujoco_playground`):

```
r = 4.0·gripper_box  +  8.0·box_target·reached_box  +  0.25·no_floor  +  0.3·robot_qpos
```

Each base term is a `1 − tanh(·)` shaping signal in [0,1]. So **max ≈ 12.55/step**. The training/eval
return is summed over a **single continuous 1000-step episode**: the evaluator wraps the env with brax
`EpisodeWrapper(episode_length=1000)`, and the inner env has no step-counter termination, so `done`
stays 0 until step 1000 (verified — there is *no* auto-reset inside the rollout; the agent gets one
long grasp→lift→place→hold attempt). So the logged-return ceiling is ~**12,550** (12.55 × 1000), and:

| what | logged return | `box_target_max` | real success |
|---|---|---|---|
| vanilla TD-MPC2 @1.5M (3 seeds) | ~2,000–2,670 | **0.000** | **0%** |
| theoretical max (1000-step) | ~12,550 | 0.9+ | — |

Vanilla plateaus at ~2,500 **with `box_target_max = 0.0000`**, i.e. it leaves **~10,000 reward on the
table by never picking.** The gap-to-ceiling, with zero box_target, *is* the hover-hack — quantified.
And it retires an earlier worry that "2967 > the max": the max over the 1000-step eval is ~12,550, not
1882, so both the vanilla 1854 and the jumpy 2967 are simply **two hovers**, one banking shaping reward
slightly faster than the other. The "+60%" is one hover out-hovering another.

## 3. Well-ORDERED, badly-SHAPED (the mechanism)

Is the reward just wrong? No — it's **well-ordered**. Scripting the three regimes and measuring return
(per 150-step episode here, for clarity):

| regime | return | `box_target` | from `gripper_box` | from `box_target` |
|---|---|---|---|---|
| hover (TD-MPC2) | 315 | 0.00 | 279 (**89%**) | 0 |
| grasp, no place | 674 | 0.48 | 461 | 174 |
| **full success** | **965** | 0.91 | 478 | 445 |

Over a 150-step episode, a real success earns **3.1× the hover's return** — at that horizon the
ordering is correct. The problem is **shaping**, and it **gets worse with horizon**: the hover banks
**89% of its return from the dense `gripper_box` proximity term just by floating near the cube**, while
the big `box_target` reward is **gated behind `reached_box`** (gripper must first close to <1.2cm) and
only pays after grasp+lift+place. So a gradient learner climbs the smooth, ungated proximity hill into a
**hover local optimum**. And at the actual **1000-step** training/eval horizon the pathology inverts the
ordering entirely — §4a shows return becoming *anti-correlated* with success, because dwelling near the
cube for 1000 steps out-scores spending those steps actually moving it to the target. **Mis-shaping, not
mis-ordering, drives the hack.**

## 4. The task *is* solvable — a heuristic controller proves it

If gradient RL can't escape the hover basin, can *anything* solve PandaPickCube? We took the
["learning beyond gradients"](https://trinkle23897.github.io/learning-beyond-gradients/) route: instead
of training a network, iteratively refine a **programmatic** controller from feedback (telemetry,
per-phase success, videos). The result (`controller v9+`):

- **Real grasp → lift → place**, video-verified, **~9% success** (`box_target ≥ 0.9`), **99% grasp**,
  lift 0.69, over 256 envs × multiple seeds.
- The real wall was **cube orientation** — `rot_err` capped `box_target` even on good grasps; cracked
  with **analytic level-gripper IK** (compute a wrist pose that keeps the cube level), plus a long
  settled grasp-hold and a split lift→transport→place carry.

So PandaPickCube is solvable; **gradient TD-MPC2 just reward-hacks it.** Success videos:
`HL_v9_SUCCESS_env6.mp4`, `env21.mp4`.

### 4a. The heuristic learning curve — in RL units

A programmatic controller doesn't "train," so to compare it to TD-MPC2 we evaluate **every controller
iteration under the exact 1000-step `eval_pi` protocol** TD-MPC2 is scored on (one continuous 1000-step
rollout, dense reward summed, n=64 episodes). That puts the heuristic's "return" on the same axis:

| iter | version | return (1000-step) | real success | grasped |
|---|---|---|---|---|
| 1 | v1 — hover-grasp, no level-IK | **4403** | 0% | 0.89 |
| 4 | v4 | 4488 | 0% | 0.84 |
| 7 | v7 — dead-end 6DOF orient | **4549** | 0% | 0.84 |
| 8 | v8 — analytic level-IK (cracks the 0% wall) | 3069 | 1.6% | 0.81 |
| 9 | v9 | 2880 | 6.3% | 0.81 |
| 10 | **v9_rhold (BEST)** | 3406 | 6.3% | 0.83 |
| 11 | v11 — xy-servo | 3522 | 3.1% | 0.97 |
| — | *TD-MPC2 vanilla @1.5M* | *~2500* | **0%** | *~0* |

![Heuristic-learning curve in RL units: 1000-step eval return and real success rate vs HL iteration, with TD-MPC2's ~2500 hover plateau as a reference line. Early high-return versions have 0% success; success only appears once return drops.]({{ '/images/hl_learning_curve.png' | relative_url }})

**The striking finding: at the eval/training horizon, return is *anti-correlated* with real success.**
The highest-return controllers (4400–4550) score **0% success** — they grasp the cube and hold it low,
banking the ungated `gripper_box` term plus `robot_target_qpos` (arm near home) for all 1000 steps
without ever lifting it to the elevated target. When the controller learns to actually lift-and-place
(v8→v10), return *drops* to ~2,900–3,400 — moving the cube to target sacrifices that steady dwell
reward — but it earns **real success**. TD-MPC2's ~2,500 hover sits **below even the real-success HL
versions**, at 0% success. So in this reward, **maximizing 1000-step return actively works against the
task** — the cleanest possible picture of why gradient RL reward-hacks here. The HL line's value is not
a number above 2,500; it's that it **converts cheap dwell-return into actual picks.** Curve:
`demo_videos/hl_learning_curve.{png,csv}` + `HL_CURVE_README.md`.

### 4b. Pushing success higher: the wall is cube *orientation*, not grasp precision

Continued iteration set an honest ceiling: **best = 0.094 real success** (v9_rhold, 4-seed mean at
n=256; peak ~0.10). The instructive part is *what didn't work*: iter-11 broke the long-standing
grasp-precision cap (`reached_box` 0.78 → 0.97 via a live-xy descent servo) — yet success **fell**,
proving `reached_box` was **not** the binding constraint. **`rot_err` is**: the cube rotates between the
parallel fingers during close/lift (~7–10° tilt), capping `box_target` below 0.9. An exact level-frame
target (iter-12) tied baseline — confirming the tilt is grasp *dynamics*, not the commanded pose. The
remaining untried lever (logged for resume): a physical regrasp/de-tilt settle on the table before lift.

## 5. What we're doing about it: reward engineering to actually solve it  *(live)*

The diagnosis (§3) names the fix: break the hover local optimum so gradient RL can reach the cube. We
are sweeping reward re-engineerings — **train on the engineered reward, but score success on the
original `box_target ≥ 0.9` task** (a fair test) — including:

1. **Down-weight proximity** (`gripper_box` 4→1) so hovering pays far less.
2. **Ungated lift bonus** — reward cube height above the table directly, not gated behind `reached_box`,
   so "pick it up" has gradient before a perfect grasp.
3. **Grasp bonus** — a discrete reward the instant `reached_box` latches, to bridge the gated cliff.
4. **Potential-based staging** — reach → grasp → lift → place as a shaped potential (policy-invariant,
   so it can't introduce new hacks).

The bar: **real success rises above 0** for vanilla and/or jumpy TD-MPC2, on the true metric.

**Results (vanilla TD-MPC2, seed 1, eval/50k; all numbers read from `*_realsuccess.csv`):**

Grasp rate is shown for both the bare **policy** and the **planner** (MPPI), since the planner can
stumble into a grasp by search even when the policy can't reproduce it:

| variant | config (weights) | grasp: policy / planner | peak `box_target` | success |
|---|---|---|---|---|
| baseline | original reward | ~0 / ~0 (flickers @1.4M+) | 0.085 | **0** |
| V1 | reweight only: gripper_box 4→1, bt 12 | 0.00 / 0.00 | 0.00 | 0 |
| **V2** | **+ ungated lift=4 (gripper_box=1)** | **0.60** / 0.33 @150k → faded | 0.056 | 0 |
| V3 | V2 + grasp bonus 2 | 0.00 / 0.00 | 0.00 | 0 |
| V4 | lift=8, gripper_box=2 (full 1M) | 0.40 / 0.33 → 0 by 1M | 0.014 | 0 |
| V5 | lift=10, grasp=4, **gripper_box=0.5** | **0.00** / 0.33 | 0.028 | 0 |
| V6 | lift=6, grasp=6, **gripper_box=0.5** | 0.00 / 0.00 | 0.00 | 0 |

**Verdict: reward engineering broke the hover trap but did *not* make TD-MPC2 solve the task.** No
variant reached real success > 0. But the sweep is informative:

1. **The ungated lift bonus is the one lever that works behaviorally.** With a moderate proximity term
   kept (V2: `gripper_box`=1, ungated `lift`=4), TD-MPC2 starts **grasping in 60% of eval episodes by
   150k** — versus the original reward, where it never grasps before ~1.4M. Returns also dropped from
   ~2,500 (hover) to ~300–650, confirming the hover gravy-train was removed. *The gating, not the
   ordering, is the trap* — confirmed.
2. **But grasping never consolidates.** It's intermittent and **fades as the policy converges** (V2 →
   0.20, V4 → 0), never completing a lift→place, so `box_target` never approaches 0.9.
3. **Two dead ends, both informative.** Cutting `gripper_box` to 0.5 (V5/V6) leaves the **bare policy
   never grasping** (the planner brushes it — V5 `mppi_reached`=0.33 — but the policy can't reproduce
   it), so the reach gradient is *needed* to get near the cube. And a discrete **grasp bonus hurts**
   (V3 → 0 grasps): it becomes its own little hack target rather than a bridge.

So **reward shaping can knock the policy out of the hover basin, but it can't manufacture a *reliable*
grasp** — and grasp reliability is the binding wall. That lines up exactly with the heuristic track
(§4b): even a hand-coded controller with analytic level-IK tops out at ~9% because of grasp precision
and cube-orientation dynamics. The cube is small and the grasp is unstable; that's a *dynamics/precision*
problem, not a reward-shape problem.

**Final consolidation test (V7/V8, to 1M): stronger lift also fails.** Adding more lift weight on the
good V2 base did not make the grasp stick — V8 (lift=12) peaked `pi_reached`=0.40 then **faded to 0** by
900k (the same transient pattern), and V7 (lift=8, bt=15) **never grasped at all**. So across **8
reward variants up to 1M steps, real success stayed exactly 0** — the no-solve verdict is final, and the
bottleneck is grasp *stability*, not exploration or reward shape. One last probe: the **jumpy
(temporal-abstraction) world model** on the V2 reward config — if planning over macro-steps can't
consolidate the grasp either, that closes the question for this task. **It shows the exact same
pattern** — over the **full 1M steps**, `jumpy_reached` intermittently brushes 0.33–0.67 via macro-step
planning but **never sustains across consecutive evals**, `box_target` ≤ 0.012, **final and peak success
= 0** — i.e. temporal abstraction does **not** cross the grasp-stability wall either. So *every*
lever we have (reward shaping ×8, temporal abstraction) breaks the hover trap but none manufactures a
reliable grasp on this small-cube task; the honest conclusion is that **PandaPickCube's wall is grasp
dynamics, not reward design or planning horizon** — which is precisely why the hand-coded controller
(§4) needed *analytic level-IK* to reach even ~9%. *(Jumpy run complete at 1M: success 0, same
intermittent-grasp-never-consolidates pattern as every other variant. Reward-eng track closed.)*

![Reward-engineering sweep: per-variant grasp rate (reached_box) and real success vs training step for the 6 reward variants. The ungated lift bonus (V2) spikes grasping to 0.60 then fades; all variants stay at 0 real success.]({{ '/images/reward_eng_curves.png' | relative_url }})

Honesty note: the `box_target` success metric was left untouched (raw + gated) throughout, so success
is measured on the true task; only the *training* reward was engineered. All grasp numbers are read from
the `*_realsuccess.csv` sidecars — both the policy (`pi_reached`) and planner (`mppi_reached`) columns,
distinguished above (an earlier draft conflated the two for V5). Curves:
`demo_videos/reward_eng_curves.png`, `RESULTS.json`.

## 6. The lesson

- **A metric is not a task.** Every statistical box (CI, seeds, peak+final) was green while the robot
  did nothing. Only a *behavioral* check (the video) and a *task-true* metric (`box_target ≥ 0.9`)
  caught it. We've now wired real-success logging into the eval so this can't hide again.
- **Dense shaping + a gated sparse jackpot = a hover trap.** The reward was well-ordered but its
  ungated dense term offered a local optimum worth 89% of the real return.
- **"Learning beyond gradients" earned its keep here** — a programmatic policy solved what gradient RL
  reward-hacked, and it's the honest contact-manipulation result to date.

### Pointers
Correction banners on Parts 12 & 16. Reward source: `mujoco_playground .../franka_emika_panda/pick.py`.
Real-success logging + campaign: `scripts/run_benchmark.py` (`_is_pickcube` sidecar),
`scripts/realsucc_campaign.sh`. Heuristic controller + learning curve: `hl_pickcube/` (`controller.py`,
`KNOWLEDGE.md`, `LOG.jsonl`), `demo_videos/`. Return-vs-success: `demo_videos/goalB_results.json`.
