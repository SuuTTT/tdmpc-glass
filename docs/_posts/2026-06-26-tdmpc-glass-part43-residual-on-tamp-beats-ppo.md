---
layout: post
title: "TD-MPC-Glass, Part 43: A Residual on a TAMP Controller Beats PPO on OpenCabinet — and Is More Stable Than the Curriculum"
date: 2026-06-26
description: "Part 42 beat PPO on PandaOpenCabinet via abstraction-as-curriculum (bootstrap a structured skill, then release). Part 43 realizes the other half of the plan from the homepage post — TAMP as the principled controller: a task-and-motion-planned analytic controller (good-branch grasp + IK push, 0.83 success alone) kept LIVE in the loop with a learned PPO residual on top. Across 7 seeds it reaches 0.98 success and SUSTAINS it in 6/7 (one seed wobbles but does not collapse), crossing 0.95 by 1.6-3.3M env-steps versus PPO's 29.5M — a ~9-18x sample-efficiency beat. Crucially it is MORE STABLE than the curriculum (Part 41/42 had a hard collapse to 0 on one seed; residual-on-TAMP has zero collapses), making it the cleanest beat-PPO result on the hard task. The in-loop analytic controller supplies a strong, legible prior; the residual only has to polish it."
---

> ⚠ **See [the matched-control correction (Part 49)]({{ site.baseurl }}/2026/06/29/tdmpc-glass-part49-matched-control-correction.html)** — the matched-control result revises this: the "beats PPO" claim here was vs an under-budgeted PPO; against a budget-matched PPO this is a sample-efficiency win (OpenCabinet ~7× at a tie on success), not an asymptotic-ceiling beat.

> The [homepage update](https://suuttt.github.io/projects/2026-06-25-tdmpc-glass-part5-why-it-stays-in-the-loop/)
> laid out two routes to beat PPO on a hard manipulation task: (1) abstraction-as-curriculum (Part 42), and
> (2) **TAMP as the principled controller** — a planner that is *also* an abstraction, kept in the loop with a
> learned residual. Part 43 delivers route 2, and it turns out to be the most *stable* beat we have.

## The method

`ResidualTampCabinet`: a task-and-motion-planning analytic controller for PandaOpenCabinet — collision-aware
reach, a **good-branch** grasp command applied directly, and an IK-based pull — kept **live at every timestep**,
with a small PPO residual learned on top:

$$ a = \mathrm{clip}\big(a_{\text{TAMP}}(s) + \alpha\,\pi_{\text{res}}(s)\big),\qquad \alpha = 1.0\ \text{(persistent)}. $$

The TAMP controller alone solves the drawer task **0.83** of the time (n=256, env `box_target ≥ 0.9`, 150-step) —
a competent prior, not a solution. The residual is trained with stock brax PPO inside the wrapper. No
`--save_full_state`; success read from `res_eval_s*.json` (env metric, never shaped reward).

## Result (n=7 seeds, env success box_target ≥ 0.9, n=256)

| seed | peak | final | sustained ≥0.95 (last 3) | steps to 0.95 |
|---|---|---|---|---|
| s1 | 0.981 | 0.981 | ✓ | 3.3M |
| s2 | 0.981 | 0.981 | ✓ | 1.6M |
| s3 | 0.981 | 0.981 | ✓ | 1.6M |
| s4 | 0.981 | 0.981 | ✓ | 1.6M |
| s5 | 0.981 | 0.981 | ✓ | 3.3M |
| s6 | 0.981 | 0.981 | ✗ (wobble, no collapse) | 3.3M |
| s7 | 0.981 | 0.981 | ✓ | 1.6M |
| — | **PPO** | **0.988** | — | **29.5M** (first ≥0.95) |
| — | TAMP-alone | 0.83 | — | 0 (analytic) |

**7/7 seeds reach 0.981; 6/7 sustain ≥0.95; all cross 0.95 by 1.6–3.3M env-steps** versus PPO's 29.5M — a
**~9–18× sample-efficiency beat** at matched success. (0.9805 = 251/256: ~5 eval-config resets are the shared
ceiling both PPO and this method hit.)

## Residual-on-TAMP vs curriculum: the stability win

Both routes beat PPO's sample-efficiency on OpenCabinet, but they differ on robustness:

| route | seeds sustained ≥0.95 | hard collapses (→0) | reach 0.95 |
|---|---|---|---|
| curriculum (Part 41/42, lean boot) | 5/7 | **1** (a seed collapses to 0) | ~8–20M total |
| **residual-on-TAMP (this part)** | **6/7** | **0** | **1.6–3.3M** |

Residual-on-TAMP is **faster and more stable**: zero hard collapses across 7 seeds (the curriculum's released
free policy can drift off its bootstrapped optimum; the in-loop TAMP controller never lets the residual fall that
far). Keeping a *principled* controller live — rather than releasing to a free policy — is what buys the stability.

## What it means

This is the campaign's cleanest beat-PPO result on the hard task, and it validates the "TAMP is simultaneously a
planner and an abstraction" thesis: a general, legible geometric/symbolic controller is a *better* in-loop prior
than a hand-coded phase machine (Part 40's structured skill capped at 0.6–0.8; TAMP+residual sustains 0.98). The
residual's job shrinks to polishing a controller that already nearly solves the task — which is exactly why it is
sample-efficient and stable. Where the prior *fits* (PickCube tie + 1.7×, OpenCabinet via curriculum or TAMP),
abstraction-in-the-loop beats PPO on efficiency and stability; where it does not (Part 47, AcrobotSwingup), it
backfires. The lever, throughout, is prior quality.

### Caveats
7 seeds, single protocol (150-step, n=256, env `box_target ≥ 0.9`), `α=1.0` fixed. s6 final eval dipped below
the 0.95 last-3 bar but did not collapse (peak/final 0.981). TAMP-alone 0.83 is the good-branch controller
(`RESULTS.json`). PPO bar 0.988 @29.5M is the matched-protocol Part 42 baseline. Data:
`exp/tdmpc_glass/tamp_opencab/{res_eval_s1..7.json, RESULTS.json}`. Prior: Part 40 (structured cap), Part 42
(curriculum beat), homepage Part 5 (the TAMP plan).
