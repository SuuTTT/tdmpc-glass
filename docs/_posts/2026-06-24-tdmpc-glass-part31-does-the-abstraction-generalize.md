---
layout: post
title: "TD-MPC-Glass, Part 31: Does the Abstraction Generalize? A Second Panda Task Says No — and Says Why"
date: 2026-06-24
description: "The abstraction-in-loop ties PPO on PandaPickCube (Parts 27-30). Does that hold on a second task, or is it a PickCube artifact? We ran the same method (analytic top-down-grasp controller + Markov-conditioned alpha=1.0 residual) vs stock PPO on PandaPickCubeOrientation — same env family but the target now has a random yaw, so the cube must be placed at an arbitrary orientation. Budget-matched, 2 seeds each. The answer is a clean, honest LOSS: PPO reaches peak success 0.805/0.836, the abstraction only 0.320/0.344. The mechanism is the whole point: the abstraction's hand-coded grasp is fixed top-down at a fixed yaw, which is the WRONG prior for a random-yaw target, and a residual that only corrects a fixed primitive cannot supply an arbitrary reorientation — it plateaus at 0.33 while PPO, carrying no wrong prior, learns the reorientation and wins. A revealing nuance: early in training the residual LEADS (mean box_target 0.41-0.53 at 3.3M steps vs PPO's 0.06) because the prior is a head-start — but the prior also caps the ceiling. So the abstraction is reusable only WITHIN the regime its hand-coded prior covers; when the task violates that prior, the abstraction actively underperforms learning-from-scratch. This is the reusability axis made concrete, and it ties the orientation thread together: on PickCube orientation is the far-reach tail (a kinematic wall, Part 30); on PickCubeOrientation orientation is the whole task (a wrong-prior wall)."
---

> Parts 27-30 established that a value-aware abstraction *in the loop* ties PPO (~0.8) on PandaPickCube and
> beats it on sample-efficiency. The obvious next question — and the one that decides whether any of this is a
> general result or a single-task artifact — is whether it transfers to a *second* task. Part 31 answers it,
> and the answer is an instructive **no**.

## The test

Same method (analytic reach→descend→grasp→lift→place controller + Markov-conditioned residual at α=1.0),
budget-matched stock PPO baseline, 2 seeds each, on **`PandaPickCubeOrientation`** — the same env as PickCube
except `sample_orientation=True`, so the target cube has a **random yaw** and success requires placing the cube
at that arbitrary orientation (the `box_target` metric's rot term, Part 30, now varies per-episode instead of
being fixed upright).

## The result — a clean loss

Real success (box_target ≥ 0.9, n=256):

| arm | peak success | final success |
|---|---|---|
| PPO seed 1 / seed 2 | **0.805 / 0.836** | 0.785 / 0.836 |
| abstraction residual (α=1.0) seed 1 / seed 2 | **0.320 / 0.344** | 0.320 / 0.320 |

**PPO ~0.82; the abstraction ~0.33.** The method that *tied* PPO on PickCube *loses by ~0.5* on
PickCubeOrientation. This is not noise (2 seeds, tight) and not a reaching failure (residual reached_rate ≈
1.0, mean box_target ≈ 0.83 — it grasps and gets close, but misses the orientation bar).

## Why — the prior is wrong, and a residual can't fix a wrong prior

The abstraction's analytic grasp is **fixed top-down at a fixed yaw**. On PickCube that prior is *correct*
(target is always upright), so the residual only has to make small corrections → tie. On PickCubeOrientation
the target yaw is random, so the fixed-yaw grasp is *systematically wrong*, and a residual that corrects a
*fixed primitive* cannot synthesize an arbitrary reorientation. It plateaus at 0.33. PPO, carrying **no** prior,
simply learns the reorientation and reaches 0.82.

A revealing nuance in the curves: **early in training the residual leads** — at 3.3M steps its mean box_target
is 0.41-0.53 vs PPO's 0.06 — because the controller is a head-start. But the same prior that helps early
**caps the ceiling**: PPO crosses it by ~10-13M steps and climbs on while the residual stalls. The prior trades
ceiling for early speed.

## What this means — the reusability axis (and the orientation thread)

This is the **reusability** question from [Part 29-ish forward, the "what is abstraction actually good for"
reframe] made concrete:

- **Abstraction is reusable only within the regime its hand-coded prior covers.** Where the prior fits
  (PickCube, upright target) it ties strong RL and is more sample-efficient early. Where the task *violates*
  the prior (PickCubeOrientation, random yaw) the abstraction **underperforms learning-from-scratch** — the
  wrong primitive is a liability, not an asset.
- It ties the **orientation** thread together. Orientation is the recurring wall: on PickCube it is the
  far-reach *tail* and the wall is *kinematic* (Part 30 — 99.9% of far targets can't be grasped upright); on
  PickCubeOrientation it is the *whole task* and the wall is a *wrong prior* (this part). Same axis, two
  different failure modes.

## Verdict and the open follow-up

**The abstraction-in-loop does not generalize as-is** — its value is conditional on the prior matching the
task. The honest one-line summary of the whole arc: *a value-aware abstraction in the loop matches strong RL
and is more sample-efficient when its built-in skill is right for the task, and hurts when it isn't.*

The natural next experiment (queued as a design question, not yet run): make the abstraction
**orientation-aware** — feed the target yaw into the analytic controller so it grasps/places at the right yaw,
turning the wrong prior into a right one. If that recovers PickCubeOrientation toward PPO, it would show the
abstraction's limitation is the *fixedness* of the prior, not abstraction per se — and point at *parametrized*
priors as the route to reusable abstractions. That is the toward-reusability direction.

### Caveats
2 seeds per arm (tight; the 0.33-vs-0.82 gap is far outside seed noise). Budget-matched at ~56M env-steps;
both arms are converged (PPO plateaus ~0.82 by ~16M, residual ~0.33 by ~36M). "Success" is the same
box_target≥0.9 metric as Parts 27-30, now with per-episode target yaw. The residual is the α=1.0 PPO-trained
in-loop variant (the one that ties on PickCube); we did not test an orientation-aware controller here (the open
follow-up). Data: `exp/tdmpc_glass/generalization_PandaPickCubeOrientation/{ppo_eval_s1,ppo_eval_s2,residual_eval_s1,residual_eval_s2}.json`.
Prior: Parts 27 (the tie), 30 (the kinematic orientation ceiling).
