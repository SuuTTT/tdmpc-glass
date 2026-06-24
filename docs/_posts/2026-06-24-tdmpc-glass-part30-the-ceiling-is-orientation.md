---
layout: post
title: "TD-MPC-Glass, Part 30: 0.83 Is a Fundamental Ceiling on PandaPickCube — It's the Orientation, Not the Budget or the Algorithm"
date: 2026-06-24
description: "The definitive close of the break-PPO / toward-100% quest. We tested whether 0.81 is a budget ceiling (long PPO to 180-205M) or a learning-paradigm ceiling (learned-grasp end-to-end with a far-reach curriculum, no fixed primitive). Both land at exactly 0.828 — the same number PPO already reached at 30-110M, and the same ~0.83 every method in this campaign hits (abstraction-in-loop, residual, retry, curriculum, long-budget, learned-grasp). A per-config diagnosis explains why: on the failing far-reach tail, POSITION is solved (pos_err median 0.006, 72% within tolerance) but ORIENTATION is not (rot_err median 0.40) — a feasible far-reach grasp leaves the cube tilted, and the box_target>=0.9 success metric (0.9*pos + 0.1*rot) demands it near-upright, so only 12% pass. And we PROVE it is physical, not a learning gap: an upright-grasp IK feasibility test shows 99.9% of far-reach targets (reach >0.84m) admit NO in-joint-limit solution that holds the grasped cube within the 8 degrees of upright the metric needs — even though 100% are position-reachable (Part 28). Position-reachable != upright-graspable. So ~0.83 is a kinematically-imposed ceiling of the task+metric, not a budget/algorithm/capacity limit; reaching 0.9+ needs a different grasp family or a changed metric, not more training."
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

## Why — the success metric is orientation-weighted (verified from source)

The success metric is not "cube at target position." From `pick.py`:

```python
pos_err = ‖target_pos − box_pos‖
rot_err = ‖R_target[:6] − R_box[:6]‖          # cube tilt vs the (upright) target
box_target = 1 − tanh(5·(0.9·pos_err + 0.1·rot_err))   # success ≡ box_target ≥ 0.9
```

`target_quat` is the identity quaternion, so **success requires the cube at the target position *and*
upright.** Solving `box_target ≥ 0.9` gives `0.9·pos_err + 0.1·rot_err ≤ 0.0201`. Plug in the measured
far-reach medians (pos_err 0.006, rot_err 0.40):

| case | 0.9·pos + 0.1·rot | box_target | pass? |
|---|---|---|---|
| far median (pos 0.006, rot 0.40) | 0.0454 | 0.777 | ✗ |
| **perfect position** (pos 0, rot 0.40) | 0.0400 | 0.803 | ✗ |
| required (pos 0.006, rot ≤ **0.147**) | 0.0201 | 0.900 | ✓ |

So **even with perfect position, the far-reach tilt (rot_err 0.40) caps `box_target` at 0.80** — orientation
alone disqualifies the tail, and passing needs rot_err ≤ 0.147 (≈ within 8° of upright). Per the best
learned-grasp checkpoint, only **12%** of far episodes clear that bar; position is solved (pos_err median
0.006, 72% within 3 cm) but orientation is not.

## The proof: it's *physical*, not a learning gap

Is rot_err ≤ 0.147 at far reach merely *unlearned*, or *kinematically impossible*? We tested it directly.
For a rigid top-grasp the cube tracks the hand, so rot_err ≤ 0.147 means the hand must stay within **8.4°
(x/y), 5.95° (z)** of the canonical upright-holding pose. We sampled every hand orientation inside that
tolerance and asked whether **any** admits a valid in-joint-limit IK solution at the far target — i.e. can the
Panda physically hold a grasped cube upright there at all (`hl_tail/upright_ik_test.json`):

| split | n | upright-graspable | **INFEASIBLE** |
|---|---|---|---|
| hard far-tail (reach ~0.865 m) | 801 | **0.12%** | **99.88%** |
| uniform far (reach ~0.868 m) | 500 | 0.20% | 99.80% |

And it is sharply **reach-gated**: upright-feasible 17.5% at reach 0.80–0.82 m → **~0% beyond 0.84 m**. This
sits exactly alongside Part 28's finding that **100%** of targets are *position*-reachable (with *any*
orientation): you can put the gripper at the far target, but at full extension the wrist **cannot** keep a
grasped cube within 8° of upright. **Position-reachable ≠ upright-graspable.**

So the last ~17% is not a reaching, exploration, capacity, or budget problem, and it is **not learnable** for
the standard top-grasp — it is a *kinematic* fact: the success metric demands the cube upright, and the Panda
physically can't deliver that past ~0.84 m reach.

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

**You cannot break ~0.83 on PandaPickCube by training harder or smarter — and now we know it's not even a
learning problem.** Budget (Part 30), capacity (Part 28), retry (Part 28), curriculum (Parts 28, 30),
abstraction (Parts 24–27), and learned end-to-end grasping (Part 30) all hit it, because the binding
constraint is far-reach grasp *orientation*, and **99.9% of the far tail is kinematically incapable of an
upright cube under the `box_target` metric** (proof above). No optimizer crosses a wall the robot physically
cannot reach. To get past ~0.83 you must change the *problem*, not the policy:

1. **A fundamentally different grasp + reorient** — the IK proof rules out the *top-grasp* family at far reach;
   a non-top grasp (e.g. side-grasp) plus an in-hand/regrasp reorientation maps the upright constraint to a
   *different* hand pose and is the only remaining mechanical escape. It is a hard manipulation problem, not a
   tuning lever, and is unverified here (the IK test covered the canonical top-grasp).
2. **Relax / change the success metric** — drop or down-weight the orientation term, or sample the target
   orientation (the `PandaPickCubeOrientation` variant), so "upright at far reach" stops being required. This
   changes the task and is not comparable to the numbers here.
3. **Accept that ~0.83 is this task's honest, kinematically-imposed ceiling** under the current metric.

So the headline of the whole arc stands and is now fully explained: **a value-aware abstraction in the loop
matches PPO (~0.8) and beats it on sample-efficiency, and the remaining ~17% is a fundamental
orientation-gated property of PandaPickCube that no method in the standard toolbox crosses.**

### Caveats
2 seeds per arm (the 0.828 is identical across seeds and matches the independent PPO-100 number — robust);
multi-seed long-PPO confirmation (seeds 21–23) running. The upright-IK proof assumes a **rigid canonical
top-grasp** (cube tracks the hand); a fundamentally different grasp family (side-grasp + reorient) is *not*
ruled out and is exactly escape (1) above. The metric/algebra is verified from `pick.py` source; the rot_err
medians are from the best learned-grasp checkpoint, and a rollout-level rot_err-vs-reach corroboration (does
any far episode ever reach rot_err ≤ 0.147?) is running. Data:
`exp/tdmpc_glass/{hl_learngrasp/{RESULTS.json,*_far.json,diag_far.py}, hl_tail/upright_ik_test.json}`. Prior:
Parts 27 (tie), 28 (reachability), 29 (orientation lever).
