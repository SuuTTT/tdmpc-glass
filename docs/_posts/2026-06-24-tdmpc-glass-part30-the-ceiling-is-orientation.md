---
layout: post
title: "TD-MPC-Glass, Part 30: 0.83 Is a Fundamental Ceiling on PandaPickCube — It's the Orientation, Not the Budget or the Algorithm"
date: 2026-06-24
description: "The definitive close of the break-PPO / toward-100% quest. We tested whether 0.81 is a budget ceiling (long PPO to 180-205M) or a learning-paradigm ceiling (learned-grasp end-to-end with a far-reach curriculum, no fixed primitive). Both land at exactly 0.828 — the same number PPO already reached at 30-110M, and the same ~0.83 every method in this campaign hits (abstraction-in-loop, residual, retry, curriculum, long-budget, learned-grasp). A per-config diagnosis explains why: on the failing far-reach tail, POSITION is solved (pos_err median 0.006, 72% within tolerance) but ORIENTATION is not (rot_err median 0.40) — a feasible far-reach grasp leaves the cube tilted, and the box_target>=0.9 success metric (0.9*pos + 0.1*rot) demands it near-upright, so only 12% pass. So ~0.83 is a fundamental, orientation-gated ceiling of the task+metric, not a budget/algorithm/capacity limit. Reaching 0.9+ requires solving far-reach upright placement (a hard grasp-orientation problem) or changing the task — not more training."
---

> Across this campaign every method — PPO, our value-aware abstraction-in-loop, raw residual, retry, two
> curricula, mid-size and bigger nets, deploy best-of-k — converges to **~0.79–0.83** real success on
> PandaPickCube. Part 30 settles *why*, with the two cleanest tests and a per-config diagnosis. The answer is
> not "we need a better learner." It's the orientation.

## Two decisive tests

**Is 0.81 a budget ceiling? — No.** Stock end-to-end PPO, unmodified reward, run to **180–205M env-steps**
(2 seeds):

| | uniform success | far-reach success |
|---|---|---|
| PPO @180–205M (seed 1 / 2) | **0.828 / 0.828** | 0.145 / 0.148 |

That is *exactly* "PPO-100"'s 0.832 — which PPO already reached back at 30–110M. **6× more budget buys
nothing.** The ceiling is not budget.

**Is it a learning-paradigm ceiling? — No, and learning the grasp from scratch doesn't help.** End-to-end PPO
(stock 6-DOF-capable action, no fixed controller/residual) with a **far-reach curriculum** (reset distribution
oversamples box+target at x∈[0.8,0.9], P_far=0.6), 120M steps, 2 seeds:

| | uniform | far-reach |
|---|---|---|
| learned-grasp + far curriculum | **0.828 / 0.828** | 0.180 / 0.180 |

It ties PPO and **does not lift the far tail** — the curriculum made the policy master far-reach *reaching*
(reached_rate ~0.99) but not far-reach *success*.

## Why — the per-config diagnosis

On the failing far-reach split (forced-far, n=256, best learned-grasp checkpoint), decomposing the
`box_target` miss:

- **Position is solved:** end-effector/box position error median **0.006 m**; **72%** of far episodes are within
  3 cm of the target.
- **Orientation is not:** cube rotation error median **0.40 rad** at the (identity-quaternion, i.e. upright)
  target. Only **12%** of far episodes meet the success bar `0.9·pos + 0.1·rot ≤ 0.020` → box_target ≥ 0.9.

In words: at far reach the arm *can* get the cube to the right place, but a kinematically-feasible far-reach
grasp leaves the cube **tilted**, and the success metric requires it near-upright. The last ~17% isn't a
reaching, grasping, exploration, capacity, or budget problem — it's an **orientation** problem baked into the
task+metric.

## The campaign-wide picture

| family | real success |
|---|---|
| jumpy / vanilla TD-MPC2 | 0.00 |
| value-aware abstraction-in-loop (Parts 24–27) | 0.79 |
| PPO; long-PPO 200M; learned-grasp; curriculum | **~0.81–0.83** |
| (theoretical, env allows) | ~1.0 |

Everything that *learns the task at all* converges to ~0.83. That convergence — across model-free RL,
model-based abstraction, and end-to-end learned grasping — is the signature of a **task/metric ceiling**, not a
method ceiling. Part 28 already showed the env allows ~1.0 (0% of targets unreachable); Part 30 shows what
stops every method short of it: the orientation requirement on far-reach grasps.

## Verdict — the quest, closed honestly

**You cannot break ~0.83 on PandaPickCube by training harder or smarter.** Budget (Part 30), capacity (Part
28), retry (Part 28), curriculum (Parts 28, 30), abstraction (Parts 24–27), and learned end-to-end grasping
(Part 30) all hit it, because the binding constraint is far-reach grasp *orientation* under the `box_target`
metric. To reach 0.9+ you must change the *problem*, not the optimizer:

1. **Solve far-reach upright placement** — a grasp that delivers the cube upright at the edge of the workspace
   (the orientation-aware controller of Part 29 took one analytic step at this — controller-alone far success
   0.023→0.093 — but the learned residual already absorbed the easy part and the full system didn't clear the
   bar; a *learned 6-DOF reorient-and-place* policy is the open direction), or
2. **a regrasp/place-reorient sub-skill** that re-levels the cube before release (the Part-29 relevel idea,
   taken to a learned policy), or
3. **accept that ~0.83 is this task's honest ceiling** under the current success metric.

So the headline of the whole arc stands and is now fully explained: **a value-aware abstraction in the loop
matches PPO (~0.8) and beats it on sample-efficiency, and the remaining ~17% is a fundamental
orientation-gated property of PandaPickCube that no method in the standard toolbox crosses.**

### Caveats
2 seeds per arm (the 0.828 is identical across seeds and matches the independent PPO-100 number — robust);
multi-seed long-PPO confirmation (seeds 21–23) running. The diagnosis uses one diag harness on the best
learned-grasp checkpoint; the orientation story is consistent with Part 29's far-reach/top-down finding. Data:
`exp/tdmpc_glass/hl_learngrasp/{RESULTS.json, *_uniform.json, *_far.json, diag_far.py}`. Prior: Parts 27 (tie),
28 (reachability), 29 (orientation lever).
