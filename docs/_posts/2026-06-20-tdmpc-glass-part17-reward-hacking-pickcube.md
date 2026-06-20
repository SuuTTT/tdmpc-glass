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

Each base term is a `1 − tanh(·)` shaping signal in [0,1]. So **max ≈ 12.55/step**, and the env's
episode is **150 steps** → max ≈ **1882 per episode**. But the *training/eval* return you see logged
(1500–2500) is **not one episode**: the evaluator rolls out a flat **1000 steps**, and the 150-step
env **auto-resets ~6.7×** inside that window, summing the dense reward across all of them. So the
logged-return ceiling is ~**12,550**, and:

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

A real success earns **3.1× the hover's return** — the ordering is correct, success genuinely pays
most. The problem is **shaping**: the hover banks **89% of its return from the dense `gripper_box`
proximity term just by floating near the cube**, while the big `box_target` reward is **gated behind
`reached_box`** (gripper must first close to <1.2cm) and only pays after grasp+lift+place. So a gradient
learner climbs the smooth, ungated proximity hill into a **hover local optimum** and never explores far
enough to discover the gated jackpot. **Mis-shaping, not mis-ordering, drives the hack.**

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

### 4a. The heuristic learning curve — in RL units  *(live; updating over this session)*

A fair question: a programmatic controller doesn't "train," so how do you compare it to TD-MPC2's
learning curve? We evaluate **every controller iteration under the exact 1000-step eval protocol**
TD-MPC2 uses, so its return is on the same axis. Plotting **return vs HL-iteration** gives a genuine
"learning curve" for heuristic learning — and we overlay TD-MPC2's ~2,500 hover plateau as a reference.
*(Curve + table inline here once the sweep lands.)*

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
*(Results table + curves inline as runs complete.)*

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
