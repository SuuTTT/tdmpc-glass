---
layout: post
title: "TD-MPC-Glass, Part 37: Constraining the Workspace Lifts PandaPickCube to 1.0 — Confirming Far-Reach Orientation Is the Ceiling"
date: 2026-06-25
description: "Part 30 proved that ~0.83 on PandaPickCube is a kinematic ceiling: the failing ~17% are far-reach configs (reach >0.84m) where holding the grasped cube near-upright — which the box_target>=0.9 metric demands — is ~99.9% IK-infeasible. Part 37 tests the converse, the cleanest possible falsification: if we simply REMOVE those far-reach configs from the spawn distribution, does success go to ~1.0? It does. An env-var-gated workspace constraint (PICKCUBE_MAX_REACH=0.84, clamping the 3D distance of box AND target from the robot base) was added to pick.py's reset. PPO trained and evaluated on this constrained distribution reaches success = 1.000 (peak and final, n=256) — versus 0.766 for a fresh full-range control at the same 35M-step budget, and the established 0.83 long-PPO ceiling. The same constrained policy evaluated on the FULL range drops to 0.758 (it never learned the far configs), which is the honest caveat: this solves an EASIER task, not PandaPickCube. Net: the far-reach orientation tail is exactly the ceiling — remove it and the task is perfectly solvable; the residual ~17% is not a learning gap but the kinematically-infeasible far workspace, now confirmed from both directions."
---

> Part 30 closed the break-PPO quest by *proving* the ~0.83 ceiling is kinematic: the failing far-reach tail
> (reach >0.84m) physically cannot hold a grasped cube within ~8° of upright, and the success metric
> (`box_target >= 0.9`, weighted 0.9·position + 0.1·orientation) requires exactly that. Part 37 runs the
> converse experiment — the cleanest falsification available. If far-reach orientation is *the* ceiling, then
> deleting the far-reach configs from the spawn distribution should send success to ~1.0. It does, exactly.

## The lever

We added an **env-var-gated workspace constraint** to `pick.py`'s `reset()` (backed up to `pick.py.bak`,
default OFF — identical to the original when the var is unset):

```python
# PICKCUBE_MAX_REACH = max 3D distance from the robot base (origin) to the box AND target.
# When a sampled point exceeds the radius, scale its XY (keeping Z) so its 3D distance == radius:
rxy   = sqrt(max_reach^2 - z^2)              # required horizontal radius at this height
scale = where(d > max_reach, rxy / |xy|, 1)  # deterministic radial projection (JAX-friendly)
```

This **does not touch the success metric** — only where the cube and target spawn. We measured the actual
spawn distribution from the env (the keyframe `home` qpos puts the box base at `[0.7, 0, 0.03]`, so box
x ∈ [0.5, 0.9]): the box's 3D distance from base reaches **0.92m** and the target's **1.01m**. Crucially,
**17.6% of box spawns exceed 0.84m** — matching the ~17% failure rate, and matching Part 30's
"upright-feasible ~0% beyond 0.84m" reach gate exactly. So `PICKCUBE_MAX_REACH=0.84` surgically removes the
kinematically-infeasible far tail the hypothesis names.

## The result (read from JSON, n=256, success = max `box_target` ≥ 0.9, 35M-step budget)

| arm | success (peak) | success (final) | reached |
|---|---|---|---|
| **constrained-train / constrained-eval** (reach ≤ 0.84) | **1.000** | **1.000** | 1.00 |
| constrained-train / **full**-eval (the tradeoff) | 0.758 | 0.758 | 0.96 |
| control: **full / full**, fresh @ 35M (matched budget) | 0.766 | 0.766 | 1.00 |
| control: full / full, established long-PPO (Part 30, @108M) | — | **0.832** | — |

The constrained learning curve is clean and monotone: 0 → 0.68 (16M) → 0.996 (19.7M) → **1.000** (32.8–36M).
Removing the far-reach configs makes the task **perfectly solvable** — no plateau below 1.0, so the tiny-cube
grasp dynamics flagged in Part 17 are not a separate wall once the kinematic wall is gone.

## Reading the three numbers honestly

**1.000 vs 0.766 (matched budget) and 0.83 (established).** At the same 35M-step budget, the full-range
control reaches only 0.766 and its curve is *still rising* at 36M (mean `box_target` 0.92) — it would climb
toward the established 0.832 with more steps (Part 30's PPO-100 hit 0.832 at 108M). Either way the gap is
decisive: **constraining the workspace takes success from ~0.77–0.83 to 1.0.** Far-reach orientation is the
ceiling; remove it and there is no ceiling.

**The tradeoff — 0.758 on full range.** The constrained policy evaluated on the *original* full distribution
scores 0.758, slightly *below* the full-range control. This is expected and important: it never saw far-reach
configs, so it cannot do them (full-eval reached_rate also drops to 0.96). **This is the honest caveat: the
constrained arm solves an *easier* task — it deletes the hard configs rather than conquering them.** It is a
root-cause confirmation and a practical lever (if your deployment workspace is bounded, you get ~1.0), not a
method that cracks PandaPickCube as posed.

## Verdict

**Confirmed, from both directions.** Part 30 showed the far tail (reach >0.84m) is ~99.9% upright-IK-infeasible
and that every method stalls at ~0.83 because of it. Part 37 shows the converse: **remove that tail and PPO
hits exactly 1.000.** The residual ~17% on full PandaPickCube is therefore not exploration, capacity, budget,
or grasp-dynamics — it is precisely the kinematically-infeasible far workspace under the orientation-weighted
metric. The honest framing: *you cannot train past ~0.83 on PandaPickCube, but you can make the task 100%
solvable by bounding the workspace to where an upright far-reach grasp is physically possible — which solves a
different, easier task.* To genuinely beat 0.83 on the full task you still need what Part 30 named: a different
grasp family (side-grasp + reorient) or a changed metric.

### Caveats
Single seed per arm (the constrained 1.000 is a clean, monotone curve, not a lucky checkpoint — 0.996 by
19.7M, 1.0 by 32.8M and held). The fresh full-range control (0.766 @ 35M) is **undertrained** vs the
established 0.832 @108M (its curve is still rising) — the matched-budget control is the conservative
comparison; the established 0.83 is the true ceiling. The constraint clamps the **3D distance** of box and
target from the base (a radial XY projection keeping Z), gated by `PICKCUBE_MAX_REACH`; default-off leaves the
env bit-identical. Success metric untouched. Numbers read directly from
`exp/tdmpc_glass/baselines_ppo_sac/part37/{eval_constrained_on_constrained,eval_constrained_on_full,eval_control_on_full,PART37_RESULTS}.json`
(`eval_ckpts.py`, n=256, steps=1000, seed=1000), never fabricated. Code: `pick.py` (constraint + `.bak`),
`part37/{launch_part37.sh,eval_part37.sh}`. Prior: Part 30 (the kinematic proof this confirms), Part 28
(reachability), Part 17 (grasp dynamics).
