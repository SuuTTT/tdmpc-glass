---
layout: post
title: "TD-MPC-Glass, Part 35: What an Abstraction Is Actually Good For — Interpretable, Reusable, Stable (Not the Success Ceiling)"
date: 2026-06-24
description: "The campaign optimized SUCCESS RATE, and on that scoreboard the value-aware abstraction-in-loop merely TIES PPO on PandaPickCube (~0.8), hits a proven physical ceiling there (Part 30), and trails PPO on harder variation (PandaPickCubeOrientation: 0.67 vs 0.82 even parametrized, Parts 31/33/34). But success rate is the wrong scoreboard. This capstone synthesizes the campaign's data along the three axes where abstraction actually differentiates: REUSABILITY (the abstraction transfers — but only when its prior is parametrized to the task's DOF: fixed 0.33 < parametrized 0.67 on the orientation task), STABILITY (a prior matched to the task makes the learned correction small and stable — ori-aware residual peak=final 0.68, vs the fixed PickCube residual's fragile-optimum collapse from 0.79; PPO and the InFOM 0.16-outlier seed bound the comparison), and INTERPRETABILITY (a human-readable phase machine with phase-attributable failures and a residual-magnitude diagnostic, vs PPO's black-box MLP). The honest takeaway: an abstraction does not beat strong RL on asymptotic success; its value is sample-efficiency, stability, reusability-when-parametrized, and interpretability — which is a different and more defensible claim than 'abstraction wins'."
---

> Across Parts 23-34 the verdict on the *success-rate* scoreboard is clear and modest: the value-aware
> abstraction-in-loop **ties** PPO where the prior fits, hits a **physical** ceiling on PickCube, and **trails**
> PPO on harder variation. This capstone asks the better question — *what is the abstraction actually good for?*
> — along the three axes a user flagged: **interpretable, reusable, stable.**

## Why success rate is the wrong scoreboard

| task / regime | abstraction | PPO | note |
|---|---|---|---|
| PandaPickCube (prior fits) | ~0.79 | ~0.82 | tie; abstraction more sample-efficient (Part 27) |
| PickCube ceiling | ~0.83 | ~0.83 | **physical** (far-reach upright IK 99.9% infeasible, Part 30) |
| PickCubeOrientation, fixed prior | 0.33 | 0.82 | abstraction *loses* (wrong prior, Part 31) |
| PickCubeOrientation, parametrized | ~0.67 | 0.82 | recovers ~70% (Part 33) |
| PickCubeOrientation, fuller param | ~0.68 | 0.82 | class ceiling (Part 34) |

On asymptotic success the abstraction never *beats* PPO. If that were the whole story, the answer would be
"don't bother." But three other axes tell a different, more defensible story.

## 1. Reusability — real, but conditional on a *parametrized* prior

The abstraction transfers across tasks **only when its prior is parametrized to the task's degrees of freedom**:
- Fixed top-down/fixed-yaw prior on a random-yaw target → **0.33** (worse than learning-from-scratch, Part 31).
- Parametrize the prior by the target yaw → **0.67**, recovering ~70% of the gap (Part 33), with no further
  gain from richer parametrization (Part 34).

So reusability is not free and not automatic: a *fixed* abstraction is a liability off its design regime; a
*parametrized* abstraction is a genuine, transferable asset (and far more sample-efficient early — it carries a
head-start the from-scratch learner lacks). The design lesson: **parametrize priors to the task's variation, do
not hard-code them.**

## 2. Stability — a matched prior makes the correction small and stable

Stability tracks how *well the prior fits*:
- **Fixed PickCube residual:** fragile optimum — peaks ~0.79 then **collapses** under continued training (Part
  28). A large correction on a so-so prior is brittle.
- **Orientation-aware residual:** peak ≈ final (0.68, ~0.03 drop, Part 33/34) — a small correction on a matched
  prior is **stable**.
- **Over-complicated prior (v2 combo):** larger peak→final drop (0.63→0.55, Part 34) — stacking levers
  re-introduces instability.
- For external calibration: PPO is stable at its ceiling; and the InFOM reproduction shows the *other* face of
  instability — **1 of 3 seeds collapsed to 0.16** while two hit ~0.97 (seed-variance), a reminder that
  "stable" must be measured across seeds, not asserted.

The pattern: **the right (matched, parametrized) prior buys stability** — the learned part stays a small, robust
correction instead of a large, fragile one.

## 3. Interpretability — structural, and the campaign relied on it

This axis is structural rather than a single metric, but it did real work throughout:
- The controller is a **human-readable phase machine** (reach → descend → grasp → lift → place), not a
  black-box MLP. You can read *what the policy is trying to do*.
- Failures are **phase-attributable**: the ~17% PickCube tail was localized to the *grasp phase orientation*
  (Parts 28-29), then *proven* a physical far-reach-upright limit (Part 30) — a diagnosis you can only make
  when the policy's structure is legible.
- The **residual-magnitude diagnostic** told us *where the learned part was doing work*: "the residual was
  already compensating for the controller's orientation limit" (Part 29) — which is why hard-coding the tilt was
  redundant. That is an interpretability win with direct experimental consequences.
- PPO offers none of this: a 0.82 success rate and an opaque network.

## The honest takeaway

**An abstraction does not beat strong model-free RL on asymptotic success** — on PandaPickCube it ties and hits
a physical ceiling; on harder variation it trails even when parametrized. **Its value is elsewhere, and it is
real:** sample-efficiency (Part 27), **stability** when the prior is matched, **reusability** when the prior is
parametrized to the task's DOF, and **interpretability** (legible structure, phase-attributable failures, a
residual diagnostic). That is a narrower but far more defensible claim than "abstraction wins" — and it is what
the data actually supports. The route forward for *capability* is learning the skill (end-to-end RL); the route
forward for *trustworthy, transferable, debuggable* control is a parametrized abstraction with a small learned
residual on top.

### Caveats
A synthesis of Parts 23-34 (all numbers cited there, read from JSON). The interpretability axis is structural,
not a new quantitative metric — claims are tied to specific diagnoses the structure enabled (Parts 28-30),
not to a fresh number. Reusability/stability are quantified from the orientation-task arc (Parts 31/33/34) and
the PickCube fragile-optimum (Part 28); the InFOM seed-variance (0.16 vs ~0.97) is from the OGBench cube
reproduction. Data across `exp/tdmpc_glass/{generalization_PandaPickCubeOrientation*, hl_*}` and the InFOM
`repro_results/`. Prior: Parts 27 (tie + sample-efficiency), 28 (fragile optimum), 30 (physical ceiling), 31/33/34
(reusability arc).
