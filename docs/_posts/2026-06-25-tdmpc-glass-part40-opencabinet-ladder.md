---
layout: post
title: "TD-MPC-Glass, Part 40: The OpenCabinet Ladder — Can a Learned Abstraction Beat PPO on a Task Whose Prior Doesn't Fit? No."
date: 2026-06-25
description: "PPO solves PandaOpenCabinet at 0.98. We tried to beat it with the in-loop abstraction, escalating from hand-coded to fully-learned skills. The ladder: hand-coded controller 0% (reach-frontier IK wall) -> learned-reach abstraction peak ~0.40 (reach fixed to 0.98 reached, faster than PPO, but frozen grasp+pull cap it) -> learned reach+grasp+pull skills peak ~0.61 (best 0.77) -> PPO 0.98. Each learned phase removes the previous bottleneck (0->0.40->0.61), confirming the diagnosis, but the phase-structured abstraction caps ~0.6-0.8 below end-to-end PPO and never reaches 0.9. The honest conclusion: the in-loop abstraction does NOT beat PPO on OpenCabinet, because the structure that helps where the prior fits (PickCube: tie + 1.7x sample-efficiency) CONSTRAINS the policy below free end-to-end RL on a task that needs flexible whole-arm coordination. This is the negative boundary of the method, and it sharpens the positive claim: abstraction wins (sample-efficiency) when its prior fits the task, and loses when the task needs coordination the phase machine can't express."
---

> We asked whether a learned abstraction could beat PPO on a *second, harder* task — PandaOpenCabinet, which PPO
> solves at 0.98. We escalated from a hand-coded controller all the way to fully-learned phase skills. It does
> not beat PPO — and *why* it can't is the result.

## The ladder (real success, env drawer-open metric, box_target ≥ 0.9, n=256)

| approach | peak success | reached_rate | note |
|---|---|---|---|
| hand-coded analytic controller | **0.00** | 0.05 | reach-frontier IK wall — can't even latch the handle (Part 39) |
| learned **reach** only (+ analytic grasp/pull + residual) | **~0.40** (mean peak 0.39) | 0.98 | learned reach fixes the wall (reached 0.98 by 6.5M, *faster* than PPO's ~16M) — but frozen grasp+pull lose the handle on ~55-65% of resets |
| learned **reach + grasp + pull** skills (+ residual) | **0.51 / 0.77 / 0.59 / 0.51** (best 0.77, mean ~0.61) | ~1.0 | learning grasp+pull lifts it further, but still caps; **no seed reaches 0.9** |
| **end-to-end PPO** | **0.98** (peak 0.9883 @29.5M) | — | the bar |

## What the ladder shows

Each rung **removes the previous bottleneck** — 0 → 0.40 → 0.61 — which confirms the mechanism-checks at every
step: reach was the first wall, grasp+pull the second, and *learning* each one helps. The prior belief that
"grasp+pull work given reach" was an artifact of reach almost never happening; once reach is solved, grasp+pull
become the cap, and learning them helps but doesn't close it.

But it **plateaus ~0.6-0.8, well short of PPO's 0.98**, even with every phase learned and a full-authority
residual on top. The remaining gap is the **phase-machine structure itself**: the forced reach→grasp→pull
hand-offs and the per-phase factoring constrain the policy below the *free* end-to-end optimum that PPO finds.
OpenCabinet needs flexible whole-arm coordination (re-grip, mid-pull correction) that a staged abstraction does
not express; PPO, with no such structure, finds it.

## The honest verdict — and how it sharpens the positive claim

**A learned in-loop abstraction does not beat PPO on OpenCabinet.** This is the *negative boundary* of the
method, and it makes the positive result precise rather than weakening it:

- **Where the prior fits the task (PickCube, upright top-down pick):** the in-loop abstraction *ties* PPO on
  success and *beats it ~1.7× on sample-efficiency* (Part 27) — the real beat-PPO win, on the axis that matters
  for the prior-fits regime.
- **Where the task needs coordination the structure can't express (OpenCabinet, drawer-open):** the same
  structure *caps* the policy below end-to-end RL (0.6-0.8 vs 0.98), no matter how much of it is learned.

So the law is: **a structured prior is an asset exactly when its structure matches the task, and a liability
when the task needs flexibility the structure forbids** — the behavioral-abstraction analogue of the
reusability finding (Parts 31-34) and the when-planning-helps finding (Part 38). The honest one-liner: *the
abstraction beats PPO on sample-efficiency when its prior fits, and end-to-end RL wins when it doesn't.*

We stop the OpenCabinet beat-PPO grind here (this is the third and most-learned lever; further levers would just
approach a structureless PPO). The next step is the write-up, where OpenCabinet stands as the method's honest
scope boundary.

### Caveats
Per-seed peaks are the 65.5M (converged) checkpoint eval, n=256, env drawer-open success (NOT the shaped-reward
`box_target` in training logs — verification discipline). 4 seeds for skillD, 4 for learned-reach. PPO 0.98 is
the matched-protocol baseline (Part 46 eval). "Structure caps it" is inferred from the ladder + the >half
incomplete-pull diagnosis; a fundamentally different abstraction (e.g. learned options without forced hand-offs)
is not ruled out but is outside this method class. Data:
`/root/helios-rl/hl_opencabinet/beatppo_C/{EVALSKILL_s*.json, FINAL_s*.json, RESULTS.json}`. Prior: Part 39
(learned-reach), Part 37 (PickCube ceiling), Part 27 (the PickCube tie + sample-efficiency win).
