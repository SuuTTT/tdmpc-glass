---
layout: post
title: "TD-MPC-Glass, Part 39: A Learned-Reach Abstraction Solves the Wall — and Hits a New One"
date: 2026-06-25
description: "Direction C on PandaOpenCabinet: replace the broken analytic REACH phase with a small LEARNED policy whose only job is to drive the gripper into the 0.012 m handle-latch radius, then hand off to the analytic grasp+pull. The learned reach works beautifully — reached_rate ~0.98 by 6.5M env-steps vs PPO's ~16M. But end-to-end success caps at ~0.30-0.47 (best seed 0.47) and never approaches PPO's 0.98. The honest verdict: the abstraction does NOT beat PPO. It does something more useful for understanding — it LOCALIZES the failure. Reach was never the only wall; given reach, the frozen-config analytic grasp+pull completes only ~30-45% of the perturbed resets. The bottleneck moved, it did not disappear."
---

> Parts 34–35 established the analytic prior's ceiling on PandaOpenCabinet: the hand-coded controller fails
> *only at REACH* (per-step IK stalls ~10 cm short of the handle; the 0.012 m `reached_box` latch fires on ~5%
> of resets), so end-to-end success is 0. The grasp and pull, *given* reach, looked fine. Part 39 tests the
> obvious fix — learn exactly the reach — and reports what actually happened.

## The bar and the prior failure

The official brax PPO baseline solves PandaOpenCabinet to **success 0.9883 (peak @29.5M env-steps), 0.9844
final**, scored as `max box_target >= 0.9` over n=256 episodes (1000-step protocol). Success is 0 until ~20M,
then climbs to ~0.98.

Part 36-era work tried **Direction B**: an in-loop residual (`a = clip(a_ctrl + α·π, -1, 1)`, α=1.0) on the
analytic phase machine. Four seeds, trained to 62M steps, **success = 0.0 for all of them.** The residual never
latched `reached_box`: the analytic controller commits hard to a single frozen grasp config `_GRASP_Q` (which
matches the handle on only ~5% of the perturbed resets), and a clipped additive residual can't override that
commitment. The reach wall stood.

## Direction C: gate on the latch, learn only the reach

`PandaOpenCabinetGatedReach` makes the learned policy responsible for *exactly* the broken piece:

- **Phase 0 (REACH):** the policy has **full authority** (gripper forced open). Its job is to drive the gripper
  site into the 0.012 m handle-latch radius — the frontier reach the analytic controller stalls on.
- **Phase 1 (GRASP+PULL):** the instant `reached_box` latches, control hands off to the analytic
  controller — close on the bar + a committed leading pull (the part assumed to work given reach).
- **Reward:** the base env reward (so the *success metric is unchanged* and directly comparable to PPO) plus a
  reach-shaping term active only in phase 0 — `reach_shaping·exp(−30·d)` to sharpen the last centimetre where
  the env's own `gripper_box` term plateaus, plus a one-time `latch_bonus` when `reached_box` first fires.

Four seeds (3 at shaping 2.0/latch 50, one "hi" at 3.0/100), same PPO hyperparameters and budget as the
baseline, scored with the identical n=256/1000-step protocol on saved checkpoints.

## What happened: the reach is solved, fast

`reached_rate` climbs to **~0.98 by 6.5M env-steps** and stays there — versus PPO reaching ~0.98 reach at
~16M. The learned reach decisively fixes the analytic wall (which latched ~5%). The direction-C hypothesis,
*narrowly*, is confirmed: a learned policy is the right instrument for the frontier reach.

## What didn't: success caps at ~0.30–0.47

| env-steps | s1 | s2 | s3 | s4_hi | mean succ | mean reach |
|-----------|------|------|------|------|-----------|------------|
| 3.3M  | 0.242 | 0.062 | 0.000 | 0.281 | 0.146 | 0.735 |
| 6.6M  | 0.301 | 0.270 | 0.293 | 0.231 | 0.273 | 0.983 |
| 16.4M | 0.316 | 0.285 | 0.297 | 0.227 | 0.281 | 0.979 |
| 26.2M | 0.281 | 0.324 | 0.270 | 0.344 | 0.305 | 0.978 |
| 36.0M | 0.273 | 0.297 | 0.238 | 0.395 | 0.301 | 0.981 |
| 39.3M | 0.305 | 0.441 | 0.270 | 0.473 | 0.372 | 0.981 |
| 49.2M | 0.348 | 0.324 | 0.250 | 0.395 | 0.329 | 0.981 |
| 55.7M | 0.332 | 0.281 | 0.258 | 0.453 | 0.331 | 0.980 |
| 62.3M | 0.316 | 0.242 | 0.273 | 0.410 | 0.311 | 0.980 |

Per-seed **peak success: 0.35 / 0.44 / 0.30 / 0.47** (best 0.4727, mean of peaks 0.39). The success curve hovers
~0.30 and tops out under ~0.47 — it does not climb toward PPO's 0.98, at any budget through 62M steps.
`reached_rate ~0.98` the whole time. So **over half of successful reaches never complete the pull.**

## The verdict: does NOT beat PPO — but it localizes the failure

Honestly: **no.** Best success 0.47 ≪ 0.98, and because it never reaches 0.98 there is no sample-efficiency win
to claim on the full task either. The only efficiency win is on the *reach sub-metric* (0.98 reach by 6.5M vs
PPO 16M), which is not the task.

The valuable result is diagnostic. The prior belief — "the analytic grasp+pull work given reach
(grasp_ok ~0.95)" — was an artifact of reach almost never happening. Once the learned policy delivers reach on
98% of resets, we finally observe the grasp+pull at scale: the frozen single-config `_GRASP_Q` close and the
committed pull lose the marginal handle contact on ~55-65% of the perturbed reset distribution before
`|target − handle| ≤ 0.02`. The reach wall was real, but it was *masking a second wall*. The abstraction's
contribution is to have moved the bottleneck somewhere measurable: the next lever is a learned (or
per-reset-solved) grasp+pull, not another reach trick.

> Decomposition is a microscope, not a shortcut. Fixing the skill everyone agreed was broken did exactly what
> it promised — and revealed that "the rest already works" was a story we could only tell while the first skill
> kept failing.

*All numbers read from real eval JSON via a scorer identical to the PPO baseline's (n=256, 1000-step,
success = max box_target ≥ 0.9). Code + curves: `helios-rl/hl_opencabinet/beatppo_C/` (RESULTS.json,
gated_reach_env.py, FINAL_s*.json).*
