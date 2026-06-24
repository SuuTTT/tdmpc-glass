---
layout: post
title: "TD-MPC-Glass, Part 34: Fuller Parametrization Doesn't Close the Gap — ~0.7 Is the Analytic-Prior + Residual Ceiling"
date: 2026-06-24
description: "Part 33 showed that making the abstraction's grasp prior orientation-aware lifts PandaPickCubeOrientation from 0.33 (fixed prior) to ~0.67, recovering ~70% of the gap to PPO (0.82) but still trailing by ~0.15. Part 34 tests Part 33's open question: does FULLER parametrization close the rest? Four variants — apply the target rotation from the approach phase (earlyD), richer rot-error-to-target conditioning (roterr), a mid-size residual capacity bump (cap), and all combined (combo) — each at budget-matched ~56M. None exceeds ~0.68 (best: roterr 0.684, the same as Part 33's ~0.67); combo is WORSE (0.55-0.63). So fuller parametrization does NOT close the gap: ~0.7 is the ceiling of the analytic-controller + Markov-residual class on the random-orientation task, ~0.15 below end-to-end PPO. The honest conclusion of the whole orientation-aware thread: parametrizing the prior is necessary and recovers most of the gap (Part 33), but the last ~0.15 is a genuine capability ceiling of correcting an analytic skill — closing it needs learning the skill, not correcting one. (Bonus, consistent with the campaign: combining levers hurts — the residual wants a small, single correction, not a stacked one.)"
---

> Part 33 recovered most of the abstraction's lost ground on PandaPickCubeOrientation by parametrizing the prior
> (0.33 → ~0.67 of PPO's 0.82) and asked: does *fuller* parametrization close the remaining ~0.15? Part 34
> answers **no** — and that pins down the ceiling.

## Four fuller-parametrization variants

Each builds on Part 33's orientation-aware controller (`R_des = R_target @ R_home` from grasp + target quat in
obs), budget-matched ~56M, eval n=256, success box_target ≥ 0.9:

| variant | what it adds | peak | final |
|---|---|---|---|
| **roterr** | richer rot-error-to-target conditioning in the residual obs | **0.684** | 0.684 |
| earlyD | apply `R_target` from the **approach** phase (pre-align before grasp) | 0.668 | 0.645 |
| cap | mid-size residual capacity bump (~64-128×4, Part 28 sweet-spot) | 0.637 | 0.609 |
| combo | all three levers together | 0.629 | 0.551 |
| — | (Part 33 ori-aware) | ~0.67 | ~0.64 |
| — | (PPO, Part 31) | 0.82 | 0.81 |

## The result — a ceiling, not a step

**No variant beats ~0.68.** The best (roterr 0.684) merely matches Part 33's ~0.67; earlier rotation and
capacity do nothing; **combining levers makes it worse** (combo 0.55-0.63). So the ~0.15 gap to PPO is **not** a
parametrization-richness problem — it is the **ceiling of the analytic-controller + Markov-residual class** on
the random-orientation task.

Two campaign-consistent notes:
- **Combo hurts.** Stacking levers (earlier rotation + richer obs + more capacity) underperforms any single
  lever — the residual wants a *small, single* correction on a good prior, not a complicated one. This is the
  same "more isn't better for the residual" seen with bigger nets (Part 28) and the orientation controller
  (Part 29).
- **roterr is stable** (peak = final = 0.684, no collapse), reinforcing Part 33's point that a well-matched
  prior makes the learned correction small and stable.

## Verdict — the orientation-aware thread, closed

**Parametrizing the prior recovers most of the gap (Part 33: 0.33 → ~0.67), but no amount of *fuller*
parametrization closes the rest (Part 34: ~0.68 ceiling) — the last ~0.15 to PPO is a genuine capability
ceiling of *correcting* an analytic skill rather than *learning* one.** Combined with the whole arc, the honest
picture of the abstraction-in-loop is now complete:

- **Ties PPO** when the prior fits the task (PickCube, upright target: ~0.8, Parts 27-30) and is more
  sample-efficient and stable;
- **Loses badly** when the prior is fixed-wrong (PickCubeOrientation: 0.33, Part 31);
- **Recovers ~70%** when the prior is parametrized to the task's DOF (Part 33);
- **Plateaus ~0.15 below PPO** even with fuller parametrization (Part 34) — the analytic-prior + residual class
  ceiling.

To close that last gap you must *learn the skill*, not correct an analytic one — which is exactly end-to-end RL.
The abstraction's enduring advantages are elsewhere (sample-efficiency, stability, interpretability), not in the
asymptotic success ceiling on hard, high-DOF-variation tasks.

### Caveats
1 seed per v2 variant (the ~0.63-0.68 band is tight and clearly below PPO 0.82; Part 33's 3-seed ori-aware
~0.67 anchors it). Budget-matched ~56M, same box_target≥0.9 metric, per-episode target yaw. The levers are four
specific analytic/conditioning choices; we cannot rule out that some *other* parametrization closes more, but
four reasonable ones did not, and combining them hurt. Data:
`exp/tdmpc_glass/generalization_PandaPickCubeOrientation_oriaware/v2_eval_{roterr,earlyD,cap,combo}_s1.json`.
Prior: Parts 31 (fixed-prior loss), 33 (parametrized recovery).
