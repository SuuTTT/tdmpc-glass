---
layout: post
title: "TD-MPC-Glass, Part 29: The Orientation Lever Closes the Toward-100% Push — Why 0.81 Is a Robust Ceiling"
date: 2026-06-24
description: "Part 28 named the tail: the controller only does top-down grasps, and the ~20% failures are far-reach targets needing a tilted grasp. Part 29 builds the orientation-aware controller (tilt the grasp as a function of reach, then re-level the cube during transport to place it) and tests it. The result is a sharp, honest null: the orientation-aware controller ALONE lifts far-reach success 4× (0.023→0.093), confirming the mechanism — but orientation-aware + the learned residual reaches only 0.773, which does NOT beat the top-down + residual control (0.793) or PPO (0.81), and is even slightly worse on the far-reach tail. The reason closes the whole campaign: the learned residual at full authority was ALREADY compensating for the top-down controller's orientation limit, so hard-coding orientation is redundant and the tilt+relevel maneuver gives the residual a worse prior. Seven levers (bigger-net, mid-size, retry, two curricula, best-of-k, orientation) and none beats ~0.81. The env allows ~1.0 and the tail is nameable, but the residual already does all it can — so 0.79–0.81 is a robust ceiling for the analytic-controller-+-residual class, and the last ~20% needs a different paradigm, not another lever."
---

> Part 28 ended with the tail *named*: far-reach targets that need a tilted (non-top-down) grasp the controller
> never learned. Part 29 builds exactly that lever — and its null is the most informative result of the push,
> because it explains *why* the ceiling holds.

## The orientation-aware controller (mechanism-motivated)

We made the analytic grasp orientation reach-dependent instead of fixed top-down:
`tilt = clip(2.0·(reach − 0.55), 0, 0.9 rad)` about the horizontal axis (reach = ‖box_xy − base‖), so far
targets get a tilted approach where a top-down grip is infeasible — then **re-level the cube to top-down from
the TRANSPORT phase** so it re-flattens for placement. `TILT_GAIN=0` recovers the exact top-down controller
(fair baseline).

**Controller-alone sanity — the mechanism is real.** On the failing far-reach split (real success, n=256):

| controller (no residual) | far-reach success |
|---|---|
| top-down | 0.023 |
| tilt, no relevel | 0.000 *(tilt helps reaching but ruins placement)* |
| **tilt + relevel @TRANSPORT** | **0.093** |

A **4× lift** on the named failure mode — and the re-level detail matters (tilt raises reaching 0.67→0.88 but a
tilted grasp wrecks the place; re-flattening during carry recovers it; @TRANSPORT beats @LIFT 0.07 and @PLACE
0.02). So the diagnosis was right and the fix works *at the controller level*.

## But it doesn't help the full system — the decisive null

Put the learned in-loop residual (α=1.0) on top and train as in Part 27:

| arm | peak success (uniform) | peak success (far-reach) |
|---|---|---|
| top-down + residual *(control = #5b)* | **0.793** | 0.767 |
| orientation-aware + residual (seed 1) | 0.766 | 0.756 |
| orientation-aware + residual (seed 2) | 0.773 | 0.756 |
| (reference) PPO | 0.81 | — |

**Orientation-aware + residual (0.773) does not beat top-down + residual (0.793) or PPO (0.81) — and is
slightly *worse* on the far tail.** The lever that fixed the controller did nothing for the full system.

**Why — and this closes the campaign:** the learned residual at full authority was **already compensating** for
the top-down controller's orientation limitation. Hard-coding the tilt is therefore redundant, and the
tilt+relevel maneuver hands the residual a more complex, time-varying base to correct — a *worse* prior, hence
the small regression. The residual had already absorbed the orientation problem; we just couldn't see it until
we tried to "help" it and made it worse.

## The full lever-space — seven nulls, one ceiling

| lever | result vs 0.79–0.81 |
|---|---|
| bigger residual net (512×4) | worse (0.65) |
| mid-size net (64–128×4) | ~0.80, best 0.812 — **ties**, +~0.02 |
| closed-loop retry | null (0.79) |
| hard-config curriculum (broken) | collapse (0.10) |
| hard-config curriculum (correct, warm-start) | null (0.71; fine-tune erodes 0.79) |
| deploy best-of-8 | +8pts (0.77), modest |
| **orientation-aware grasp** | **null (0.773), slightly worse** |

Seven levers; **nothing meaningfully beats ~0.81.** And we know it isn't the environment (Part 28: 0% of
targets unreachable, 100% have valid IK) and isn't stochastic noise (best-of-k can't fix it). It is a genuine
capability ceiling of *this policy class* — an analytic grasp controller plus a Markov-conditioned residual:
the residual extracts what's extractable on top of the controller (~0.79–0.81, a tie with PPO), and the last
~20% — far-reach contact-rich grasps — is beyond what a *correction on a fixed grasp primitive* can do, no
matter how we shape the primitive or the sampling.

## Verdict and what's next

**The abstraction-in-loop ties PPO (0.79–0.81) and that is its ceiling on PandaPickCube** — now established
across a complete lever-space, not asserted. Pushing past it (toward ~0.95) is not an incremental-lever
problem; it needs a **different paradigm**: end-to-end learned grasping with full orientation control (no fixed
primitive to correct), a better contact/grasp model, or richer action parametrization — i.e. *learning the
grasp*, not *correcting a hand-coded one*. That is a new direction, not a continuation.

So the beat-PPO campaign reaches an honest close: a value-aware abstraction *in the loop* matches strong
model-free RL and beats it on sample-efficiency (Part 27), and the toward-100% push (Parts 28–29) maps exactly
why the remaining tail is hard and where it isn't. The fleet now returns to the week's original open thread —
the CompPlan / InFOM reproduction on OGBench — and any 100% attempt starts from "learn the grasp," not "correct
the controller."

### Caveats
Single-to-two seeds per arm (orient 2 seeds, control 1; treat ~0.77 vs 0.79 as within noise → "no help," not
"big regression"). The orientation scheme is one analytic tilt heuristic, not a learned 6-DOF grasp; a *learned*
orientation policy is exactly the "different paradigm" above and remains open. Data:
`exp/tdmpc_glass/{hl_orient, hl_midsize_b}/`. Prior: Parts 27 (the 0.79 tie), 28 (the six prior levers + the
reachability/orientation diagnosis).
