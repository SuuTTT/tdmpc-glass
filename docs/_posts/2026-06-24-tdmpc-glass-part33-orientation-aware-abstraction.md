---
layout: post
title: "TD-MPC-Glass, Part 33: Is the Wall the Prior's FIXEDNESS? An Orientation-Aware Abstraction on PandaPickCubeOrientation"
date: 2026-06-24
description: "Part 31 showed the abstraction-in-loop LOSES on PandaPickCubeOrientation (residual 0.33 vs PPO 0.82) because its grasp prior is fixed top-down at a fixed yaw — wrong for a random-orientation target. Part 33 parametrizes the prior: the controller sets grasp/place rotation to R_target @ R_home and the target quaternion is added to the residual obs (77->81), alpha=1.0. Result (3 seeds): the orientation-aware residual peaks 0.621/0.695/0.688 (mean ~0.67), final ~0.64 — DOUBLE the fixed-prior 0.33, recovering ~70% of the gap to PPO 0.82. So prior-FIXEDNESS was the dominant limiter and parametrized priors are a real route to reusable abstractions — but honest-positive, not a clean win: it still trails PPO by ~0.15, so a residual correcting an analytic grasp can't fully match end-to-end learning. Bonus for the stability axis: the orientation-aware residual barely drops peak->final (~0.04), unlike the fragile fixed PickCube residual that collapsed — the right prior also stabilizes."
---

> Part 31 found the abstraction-in-loop residual **loses** on `PandaPickCubeOrientation` — peak success
> 0.320/0.344 vs PPO 0.805/0.836 — because its analytic grasp is a **fixed top-down grasp at a fixed yaw**, the
> wrong prior for a random-orientation target, and a residual that only corrects a *fixed* primitive cannot
> synthesize an arbitrary reorientation. Part 31 closed with the obvious follow-up: make the prior **right** for
> the task — feed the per-episode target orientation into the analytic controller — and see whether that recovers
> performance toward PPO. If it does, the limitation is the *fixedness* of the prior, not abstraction itself, and
> *parametrized* priors are the route to reusable abstractions. Part 33 runs exactly that experiment.

## The change — a parametrized (orientation-aware) prior

The Part 31 controller's analytic IK held a **fixed** desired end-effector rotation `R_home` (top-down, level)
through grasp→lift→place. The orientation-aware controller instead rotates that desired frame by the
**per-episode target orientation**:

```
R_des = R_target @ R_home     (applied from the GRASP phase onward)
```

where `R_target = quat_to_mat(data.mocap_quat[target])` is the random per-episode target rotation (the same
quantity the `box_target` reward's rot term scores against). Because the cube is grasped rigidly, rotating the
gripper to `R_des` carries the cube to the target orientation. Two more pieces keep it honest and Markov:

- **Markov conditioning:** the per-episode target quaternion (4 dims) is appended to the residual's observation
  (77 → **81** dims), exactly as the controller phase `z` was appended in Parts 26-27. The base 66-dim obs only
  carried the *relative* rot-error; the residual now also sees the *absolute* target it must achieve.
- **Everything else identical to Part 31:** α fixed at 1.0, the same reach→descend→grasp→lift→place machine,
  budget-matched (~56M env-steps, brax-rounded from a 35M nominal request), eval n=256, success =
  `box_target ≥ 0.9`.

## The result

Real success (box_target ≥ 0.9, n=256):

| arm | peak success | final success |
|---|---|---|
| PPO (Part 31) seed 1 / seed 2 | 0.805 / 0.836 | 0.785 / 0.836 |
| fixed-prior residual (Part 31) seed 1 / seed 2 | 0.320 / 0.344 | 0.320 / 0.320 |
| **orientation-aware residual (Part 33) seed 1 / 2 / 3** | **0.621 / 0.695 / 0.688** | 0.606 / 0.652 / 0.648 |
| orientation-aware controller-alone (α=0, sanity) | 0.023 | — |

**Mean peak ~0.67 — double the fixed-prior 0.33, and ~70% of the way to PPO's 0.82**
((0.67−0.33)/(0.82−0.33) ≈ 0.69). Making the prior orientation-aware recovers most of the gap that Part 31
opened. The controller-alone is only 0.023 — open-loop it does almost nothing — so the residual still does the
heavy lifting, but *only once the prior points the right way*.

## Why — prior-fixedness was the dominant limiter (but not the whole story)

The same in-loop-residual class that *failed at 0.33* with a fixed prior reaches *~0.67* once the prior is
parametrized by the task's degree of freedom (target yaw). That ~2× lift localizes the Part 31 failure: it was
the prior being **fixed**, not the use of abstraction. But the recovery stops ~0.15 short of PPO's 0.82 — a
residual correcting an analytic grasp cannot *fully* match end-to-end learning on the random-orientation task.
There is a real residual gap that the analytic-controller-plus-residual class does not close, even with the
right parametrization.

## What this means — parametrized vs fixed priors

- **Reusability axis (the headline):** abstractions become reusable when their priors are **parametrized to the
  task's degrees of freedom**, not hard-coded to one regime. A fixed prior is brittle (Part 31: 0.33, *worse*
  than learning-from-scratch); a parametrized prior is a genuine asset (0.67, recovering most of the gap).
- **Honest-positive, not a clean win:** parametrization gets ~70% of the way back, not 100%. The ceiling of
  "analytic prior + Markov residual" sits below end-to-end RL on hard tasks — consistent with the whole
  campaign (ties when the prior fits, trails when the task is hard for it).
- **Stability bonus:** the orientation-aware residual drops only ~0.03-0.04 peak→final, whereas the fixed
  PickCube residual peaked ~0.79 and **collapsed** (the "fragile optimum", Part 28). A prior matched to the
  task makes the learned correction *small and stable* instead of large and brittle — so the right prior buys
  both performance and stability.

## Verdict

**Parametrize the prior, and the abstraction recovers most of what the fixed prior lost (0.33 → ~0.67 of PPO's
0.82) and trains more stably — so the route to reusable abstractions is parametrized, not fixed, priors. But it
does not fully close to learning-from-scratch.** This is the cleanest statement of the reusability axis the
campaign has produced: the value of an abstraction is conditional on its prior matching the task's variation,
parametrization is how you make that match, and even then a residual on an analytic skill trails end-to-end RL
by a task-dependent margin.

### Caveats
3 seeds (orientation-aware residual) vs Part 31's 2 (PPO, fixed residual). Budget-matched at ~56M env-steps.
"Success" is the same `box_target ≥ 0.9` metric as Parts 27-31, with per-episode random target orientation. α
fixed at 1.0 (the variant that ties on PickCube). The orientation-aware controller rotates the desired EE frame
by `R_target` from the grasp phase onward and conditions the residual obs on the target quaternion (obs 77→81);
all else matches Part 31. Data:
`exp/tdmpc_glass/generalization_PandaPickCubeOrientation_oriaware/{residual_eval_s1,residual_eval_s2,residual_eval_s3,controller_alone_eval}.json`;
Part 31 baselines: `exp/tdmpc_glass/generalization_PandaPickCubeOrientation/{ppo_eval_s*,residual_eval_s*}.json`.
Prior: Part 31 (the fixed-prior loss + this follow-up's design), Parts 27/30 (the tie and the orientation ceiling).
