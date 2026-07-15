---
layout: post
title: "TD-MPC-Glass, Part 42: The Abstraction Beats PPO on OpenCabinet — 2.5× Sample-Efficiency via a Lean Bootstrap-then-Release Curriculum"
date: 2026-06-25
description: "Part 41 matched PPO on OpenCabinet by warm-starting from the structured skill then releasing the rigid hand-offs, but the 62M bootstrap meant no total-step win. Part 42 makes it a clean beat: a LEAN 6.5M structured bootstrap (the point where the skill already reaches the handle ~98%) + a released free-policy fine-tune with LR lowered to 5e-4 reaches sustained 0.98-1.0 success in only ~8-15M TOTAL env-steps (mean ~11M) across all 4 seeds — versus PPO's 29.5M to 0.9883. That is a ~2.5x sample-efficiency beat at matched-or-better success (s1=1.0), and the lower LR fixed the Part 41 late-collapse (all 4 seeds now sustained, vs 1/4 collapsing). The campaign's beat-PPO claim now holds on the hard task too: a structured abstraction used as a curriculum — bootstrap fast, then release the structure — beats end-to-end PPO on sample-efficiency on OpenCabinet, the task where the rigid abstraction capped at 0.77."
---

> ⚠ **See [the matched-control correction (Part 49)]({{ site.baseurl }}/2026/06/29/tdmpc-glass-part49-matched-control-correction.html)** — the matched-control result revises this: the "beats PPO" claim here was vs an under-budgeted PPO; against a budget-matched PPO the abstraction wins sample-efficiency only, not the asymptotic ceiling (OpenCabinet stays a ~7× sample-efficiency win at a tie on success).

> Part 40: the rigid structured abstraction caps at 0.77 on OpenCabinet (PPO 0.98). Part 41: warm-start then
> release uncaps it to PPO parity, but the 62M bootstrap meant no sample-efficiency win. Part 42 closes it: make
> the bootstrap *lean*, and it's a clean beat.

## The lean curriculum

- **Bootstrap:** restore from the **6.5M** skillD checkpoint — the leanest point where the structured skill
  already reaches the handle (~0.98 reached). (No retrain: the skillD checkpoints already exist at every 3.3M.)
- **Release:** continue as a free end-to-end policy (action = clip(policy(obs)), no hand-offs), **LR = 5e-4**
  (lowered from 1e-3) as a stabilizer against the Part-41 late-collapse.

## Result (true success, env metric box_target ≥ 0.9, n=256, sustained over last 3 checkpoints)

| seed | peak success | total steps to sustained ≥0.95 |
|---|---|---|
| s1 | **1.000** | 6.5M + 4.9M = **11.4M** |
| s2 | 0.984 | 6.5M + 1.6M = **8.1M** |
| s3 | 0.981 | 6.5M + 3.3M = **9.8M** |
| s4 | 0.981 | 6.5M + 8.2M = **14.7M** |
| — | **PPO** | 0.9883 @ **29.5M** |

**All 4 seeds reach 0.98–1.0 in ~8–15M total env-steps (mean ~11M) vs PPO's 29.5M — a ~2.5× sample-efficiency
beat at matched/better success.** The LR=5e-4 change fixed the late-collapse: all 4 seeds now *sustain* (vs
1/4 collapsing in Part 41).

## Why this is the real beat

It satisfies both axes honestly: **success parity** (0.98–1.0 ≥ PPO 0.9883, s1=1.0) **and** a **total-step win**
(bootstrap + release counted in full: ~11M ≪ 29.5M). The structured abstraction is the *accelerant* — a 6.5M
bootstrap that gets reach + a coarse skill — and the release removes the rigidity that capped the ceiling
(Part 40). Bootstrap fast, then release: the abstraction's structure buys the sample-efficiency, and the free
late phase buys the ceiling.

This extends the campaign's law to the hard task: on PickCube the in-loop abstraction ties PPO and beats it on
sample-efficiency (prior fits); on OpenCabinet the *curriculum* form now does the same (beats PPO 2.5× on
sample-efficiency at matched success), where the rigid structured form capped (Part 40) and the matched form
broke even (Part 41).

### Caveats
4 seeds; total steps = 6.5M bootstrap + release-steps-to-sustained-0.95 (fair; the bootstrap reuses an existing
skillD checkpoint but the 6.5M is counted in full vs PPO from-scratch). Success = same env metric/scorer as the
PPO baseline (n=256, 1000-step), read from `release/EVAL_boot6p5_s*.json` (never shaped reward), sustained over
the last 3 checkpoints. A complementary result: a TAMP controller (motion-planned reach + IK-push) reaches ~0.83
success analytically (no learning) — Part 43. Prior: Part 40 (cap), 41 (uncap/match), 27 (PickCube beat).
