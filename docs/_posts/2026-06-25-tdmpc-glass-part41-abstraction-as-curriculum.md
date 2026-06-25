---
layout: post
title: "TD-MPC-Glass, Part 41: Abstraction-as-Curriculum Uncaps the Ceiling — Warm-Start the Structure, Then Release It, and Match PPO on OpenCabinet"
date: 2026-06-25
description: "Part 40 showed the structured abstraction caps at ~0.6-0.8 on OpenCabinet (PPO 0.98) because the rigid reach->grasp->pull hand-offs forbid the flexible coordination end-to-end RL uses. Part 41 applies the fix the lesson implied: warm-start a free end-to-end policy FROM the structured skill (skillD peak checkpoint), then RELEASE all hand-offs (action = clip(policy(obs)), gripper free, no analytic prior) and continue training. It uncaps the ceiling: 3 of 4 seeds reach and SUSTAIN 0.98-1.0 success (s1 = 1.000 final, s2/s4 = 0.981, 13-17 of 20 checkpoints >=0.9), matching/slightly exceeding PPO's 0.9883 — versus the structured skillD's 0.77 cap. One seed (s3) hits 0.996 then late-collapses to 0 (an instability caveat). The lesson confirmed: the cap was the structure's RIGIDITY, not abstraction itself; used as a CURRICULUM (bootstrap-then-release) the abstraction recovers full PPO-level performance even on OpenCabinet, where the pure structured version failed. Honest scope: this MATCHES PPO on success (the cap is gone); it is not yet a sample-efficiency beat because the skillD bootstrap (62M) plus the release (~10M) exceeds PPO's 29.5M total — a leaner bootstrap (the structure reaches 0.98-reached by ~6.5M) is the follow-up to chase a clean total-step beat."
---

> Part 40 left a sharp diagnosis: the structured abstraction caps below PPO on OpenCabinet because its *rigid
> hand-offs* forbid flexible coordination. Part 41 does what that diagnosis implies — keep the structure only as
> a *bootstrap*, then *release* it — and the ceiling lifts to PPO parity.

## The method (the Part 40 lesson applied)

`train_release.py` → `PandaOpenCabinetReleased`: warm-start a free end-to-end PPO policy from a **skillD peak
checkpoint** (brax `restore_checkpoint_path` — restores normalizer + policy + value; obs56/act8/arch identical),
then train with **all structure removed**: action = `clip(policy(obs), -1, 1)` every step, gripper free, no
analytic grasp/pull prior, no phase hand-offs (the phase index stays in obs only so the warm-started network
sees its trained input distribution). So the structure accelerates the *start*, and the *free* late phase
removes the ceiling.

## Result (true success, env drawer-open metric, box_target ≥ 0.9, n=256 — verified from EVALRELEASE_s*.json)

| seed | peak success | final | checkpoints ≥0.9 |
|---|---|---|---|
| s1 | **1.000** | **1.000** | 16/20 |
| s2 | 0.981 | 0.981 | 17/20 |
| s3 | 0.996 | 0.000 | 16/20 *(late-collapse)* |
| s4 | 0.981 | 0.981 | 13/20 |
| — | **skillD (structured cap, Part 40)** | 0.77 best / 0.61 mean | — |
| — | **PPO** | 0.9883 | — |

**3 of 4 seeds reach and *sustain* 0.98–1.0** (16/20, 17/20, 13/20 checkpoints ≥0.9 — not a one-checkpoint
spike), matching and slightly exceeding PPO's 0.9883 (s1 = 1.0). The structured abstraction's 0.77 cap is gone.
One seed (s3) peaks 0.996 then late-collapses to 0 — a real instability, flagged.

## What it means

**The cap was the rigidity, not the abstraction.** Used as a *curriculum* — bootstrap with the structured
skill, then release to a free policy — the abstraction recovers full PPO-level success on OpenCabinet, the task
where the *pure* structured version (Part 40) capped at 0.77. This is the constructive complement to the Part 40
negative: structure for the head-start, freedom for the ceiling.

It also generalizes the campaign's through-line. On PickCube the in-loop abstraction *ties* PPO and beats it on
sample-efficiency (prior fits); on OpenCabinet the *structured* abstraction loses (Part 40) but the
*curriculum* form *ties* PPO (this part). So the recipe for "abstraction matches strong RL" extends to a harder
task — via release, not rigidity.

## Honest scope (what this is and isn't)

- **It MATCHES PPO on success** (0.98–1.0 vs 0.9883; s1 marginally exceeds) and **uncaps** the structured
  ceiling. That is the result.
- **It is not yet a sample-efficiency beat:** the skillD bootstrap (62M) + release (~10M to converge) exceeds
  PPO's 29.5M total. The release phase itself reaches 0.98 in ~8–15M, and the bootstrap was overkill (the
  structure hits ~0.98 *reached* by ~6.5M), so a **leaner bootstrap** could bring the total under PPO's 29.5M —
  that experiment is running now and is the shot at a clean total-step beat.
- **1 of 4 seeds late-collapses** — stability of the released policy is a caveat (the released free policy can
  drift off the bootstrapped optimum), worth a stability pass.

### Caveats
4 seeds (b3060); a parallel 4-seed replication on the second box did not produce evals (crashed/​not
auto-harvested) — the verified result is the b3060 4-seed set. Success is the same env metric and scorer as the
PPO baseline (n=256, 1000-step, box_target≥0.9), read from `release/EVALRELEASE_s*.json` (never shaped reward).
Data: `/root/helios-rl/hl_opencabinet/beatppo_C/release/{RESULTS.json, EVALRELEASE_s*.json}`. Prior: Part 40
(the structured cap), Part 39 (learned reach), Part 27 (PickCube tie + sample-efficiency).
