---
layout: post
title: "TD-MPC-Glass, Part 28: Toward 100% on PandaPickCube — Five Levers, One Sweet-Spot, and the Tail Finally Named"
date: 2026-06-24
description: "After tying PPO at 0.79 (Part 27), we pushed toward 100% on PandaPickCube with six experiments — and the honest result is that 0.79–0.81 is a robust ceiling for the top-down-grasp in-loop residual, but we now know exactly why. Levers: bigger residual net HURTS (512×4 → 0.65, over-capacity); a mid-size sweet-spot (64–128×4) nudges to ~0.80, best seed 0.812 = a clean tie with PPO; closed-loop retry is NULL (0.79, the tail isn't recoverable slips); hard-config curriculum is NULL even done right (0.71 — fine-tuning erodes the fragile 0.79 optimum); deploy-time best-of-8 is a modest +8pts (0.77 uniform). The two diagnostics are the payoff: (#10) the env allows ~1.0 — 0% of targets are geometrically unreachable, 100% have valid IK — so the ceiling is the policy, not the environment; and the failures are ORIENTATION, not position — only 13.6% of the hard tail admits a top-down grasp vs 66% of uniform configs, because far-reach targets need a tilted/reoriented grasp the controller never learned. The tail is named: far-reach configs requiring non-top-down grasps. That's the next lever."
---

> Part 27 left the abstraction-in-loop at **0.79, a near-tie with PPO (0.81)**, with ~20% of configs failing.
> This week we threw six experiments at that tail to chase ~0.95/100%. None breaks it — but the diagnostics
> tell us *exactly* what the tail is, which is worth more than another null.

## The six levers (real success, box_target≥0.9, n=256)

| # | lever | result | verdict |
|---|---|---|---|
| #6 | bigger residual net (512×4) | 0.65 | **hurts** — over-capacity for a residual that needs small corrections |
| mid | **mid-size sweep (64–128×4)** | mean ~0.80, **best 0.812** | **mild sweet-spot — ties PPO's 0.81** |
| #7 | closed-loop retry (re-grasp in-episode) | 0.79 | **null** — fires up to 4.9 attempts/ep, no gain |
| #8 | hard-config curriculum (as first built) | 0.10 | broken (from-scratch + 50% hard from step 0) |
| #8b | hard-config curriculum **done right** | 0.71 | **null** — and fine-tuning *erodes* the 0.79 optimum |
| #11 | deploy-time best-of-8 (no retraining) | 0.77 uniform | **modest +8pts**, recovers ~21% of hard configs |
| — | (reference) PPO / our #5b α=1.0 | 0.81 / 0.79 | the bar |

**Capacity has a sweet-spot.** 32×4 → 0.79, **64–128×4 → ~0.80–0.81** (best seed 0.812), 512×4 → 0.65. So the
residual wants a *little* more capacity than the tiny default but collapses if it's too big — and at the
sweet-spot it cleanly *ties* PPO's peak. That's the only lever that moved the number up, and only by ~0.02.

**Everything else is a null, and #8b adds a sharp sub-finding:** the 0.79 policy is a **fragile optimum** —
*any* continued fine-tuning (curriculum or uniform control, identical full-reset regime) makes both arms peak
at the carried-in checkpoint and then **decay monotonically**. So you can't fine-tune your way past 0.79;
prioritized hard-config sampling does nothing on top of that.

## The diagnostics — what the 20% tail actually is

Two cheap experiments did more than all the training levers combined.

**(#10) Is ~0.95/100% even reachable? — Yes, by construction.** Building the Panda's FK workspace (300k joint
samples) + analytical IK:
- **0% of targets are geometrically unreachable** (max reach 0.947 m; the hardest mined failure sits at
  0.919 m). **100%** of uniform targets *and* **100%** of the 1160 mined hard-failures have a valid, in-limit
  IK solution. So the environment is not the ceiling — ~0.95+ is attainable in principle.
- **The difficulty is orientation, not position.** Only **13.6%** of the hard-failure configs admit a strict
  *top-down* grasp, versus **66%** of uniform configs. Failures cluster at **far horizontal reach** (48%
  beyond 0.85 m; mean target x = 0.82 vs 0.70 box center), where the arm must **tilt and reorient** the
  gripper to grasp at all.

**(#11) Are the failures stochastic or deterministic? — Mostly deterministic.** Best-of-K (the 0.79 policy,
K independent stochastic rollouts, solved if any succeeds):

| K | uniform | hard configs |
|---|---|---|
| 1 | 0.69 | 0.12 |
| 8 | 0.77 | 0.21 |

Best-of-8 recovers ~21% of the hard tail (those are stochastically solvable) but **~79% stay unsolved across
all 8 rollouts** — deterministic failures. More sampling won't close it.

## The verdict — and the named next lever

**0.79–0.81 is a robust ceiling for *this* policy class** — a top-down-grasp analytic controller plus a
Markov-conditioned residual. It ties PPO and no capacity/curriculum/retry/sampling lever pushes meaningfully
past it. But the ceiling is **not** the environment (everything is reachable) and **not** stochastic noise
(best-of-k can't fix it). It is one specific, named policy limitation:

> **The controller only does top-down grasps; the ~20% tail is far-reach targets that require a tilted,
> reoriented grasp the policy never learned.**

That is the next experiment, and the first one motivated by a *mechanism* rather than a hope: a
**grasp-orientation-aware controller/residual** — let the grasp phase choose an approach orientation as a
function of the target's reach, so far-reach configs get a tilted grasp instead of an impossible top-down one.
Combined with deploy-time best-of-k (+8pts), that is the credible path from 0.81 toward ~0.9+.

### Caveats (kept honest)
Single-to-two seeds per lever (the 0.812 mid-size peaks are two independent seeds; treat as ~0.80 ± noise, a
tie-not-a-beat vs PPO 0.81). Reachability uses analytical IK with an orientation set, not a full grasp-physics
test (a target with a valid IK can still be hard to grasp stably — consistent with "hard-but-solvable").
best-of-k uses the policy's own stochasticity. Data: `exp/tdmpc_glass/{hl_beatppo_big, hl_midsize,
hl_retry, hl_curriculum_b, hl_tail}/`. Prior: Parts 24–27.
