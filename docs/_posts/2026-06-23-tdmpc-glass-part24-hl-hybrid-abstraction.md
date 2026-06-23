---
layout: post
title: "TD-MPC-Glass, Part 24: Inserting Policy & Abstraction Into the Heuristic Loop — the Campaign's First Positive Abstraction Result"
date: 2026-06-23
description: "The whole campaign's verdict had been 'the abstraction well is dry' — every explicit abstraction we tried (structural-entropy 'Glass', fixed-k 'jumpy') gave 0 on clean metrics. But Part 22 argued we'd tested the WRONG KIND: predictive, not value-aware. So we used the heuristic PickCube controller (HL, ~6% real success, capped by grasp/place dynamics) as a scaffold and inserted three things on top: (#1) a raw-action residual policy, (#2) the HL phases as a learned, state-conditioned SKILL-OPTIONS abstraction, and (#3) demo-seeding TD-MPC2's replay with HL successes. The ladder, all real success (box_target>=0.9): pure TD-MPC2 0 -> demo-seed 0 (partial) -> HL 0.063 -> raw residual 0.13 -> global-knobs 0.15 -> value-aware skill-options 0.23-0.24. The skill-options abstraction is the campaign's FIRST positive abstraction result — a semantic, value-aware, state-conditioned abstraction beats raw correction and beats the abstractions that failed before, exactly as Part 22 Rule E predicted. An ablation cleanly decomposes the lever (learning the abstraction's params: 0.06->0.15; making them state-conditioned: 0.15->0.24). None beats PPO's 0.66 — but we show that's a STRUCTURAL ceiling, not a budget shortfall (#1 used 2x PPO's steps and plateaued)."
---

> Across this entire project, every *explicit* abstraction we built was null on a clean metric:
> structural-entropy "Glass" (Parts 1–16) and prior-art fixed-k "jumpy" (Parts 12/17/19) both scored
> **0 real success** on PandaPickCube. Part 22 argued the problem wasn't abstraction per se — it was that
> we'd only tested the **wrong kind**: abstractions optimized for *prediction*, not *value/control*. This
> post tests the right kind. Result: the first positive.

## The setup: the heuristic loop as a scaffold

The hand-coded HL PickCube controller (`hl_pickcube/`) is a JAX phase machine —
reach→descend→grasp→lift→place — that reliably *reaches* (reached_box ~0.97) but caps at **~6.25% real
success** because the **grasp/place dynamics** are its wall. We inserted three different learned components
on top, all scored on **true success** (`box_target>=0.9`, peak, held-out 512-env eval), all read from JSON/CSV:

- **#1 Raw-action residual** — `executed = clip(a_HL + 0.3·MLP(obs))`, a small learned correction on the
  per-step action. Trained with Evolution Strategies. *No abstraction — just correction.*
- **#2 Skill-options abstraction** — a learned, **state-conditioned** high-level policy
  `π_hi(box, target) → option parameters` of the analytic phases (the value-aware abstraction). ES.
- **#3 Demo-seeded TD-MPC2** — seed TD-MPC2's replay buffer with 504 successful HL demos (75k transitions)
  and train normally. *Does demonstration data break the hover-hack that pins pure TD-MPC2 at 0?*

## The ladder (peak real success on PandaPickCube)

| approach | peak success | budget | what it is |
|---|---|---|---|
| pure TD-MPC2 | **0.00** | 1.5M | reward-hacks (hovers) |
| **#3 demo-seeded TD-MPC2** | **0.00** | ~2M ×3 arms | demos insufficient (partial — see below) |
| (flat-k jumpy / Glass) | 0.00 | — | the abstractions that failed before |
| HL fixed-knob controller | 0.063 | — | the heuristic scaffold |
| **#1 raw-action residual** | **0.13–0.14** | 61M ES (3 seeds) | correction, no abstraction |
| **#2 skill-options, global knobs** | **0.15–0.16** | 20M ES (2 seeds) | learn abstraction params, context-free |
| **#2 skill-options, state-conditioned** | **0.23–0.24** | 20M ES (3 seeds) | **value-aware abstraction** |
| (PPO reference) | 0.66 | 33M | unconstrained end-to-end |

## 1. The positive: a value-aware abstraction works

**#2's state-conditioned skill-options reaches ~0.23–0.24 real success — ~3.8× the heuristic, and the
campaign's first abstraction that beats its baselines on a clean metric.** Three seeds (0.244/0.231/0.223)
agree. This is exactly the Part-22 Rule-E prediction: a **semantic, task-grounded, value-aware** abstraction
succeeds where the *predictive* abstractions (jumpy, Glass — both 0) failed.

And we can **decompose the lever** with the global-vs-conditioned ablation:

- **Learning the abstraction's parameters at all** (context-free "global" knobs): 0.063 → **0.15**. Just
  letting the phase parameters be optimized — instead of hand-set — more than doubles success.
- **Making them state-conditioned** (the value-aware part — adapt per box/target): 0.15 → **0.24**, another
  ~+50%.

So both halves are real: most of the jump is "learn the abstraction," and conditioning it on state adds a
genuine further boost. The win is concentrated in the **place phase** — the high-level policy adapts
place-target nudges / up-bias / hold-time per configuration, lifting place-reach from 0.22 → ~0.93, which is
precisely the failure mode the fixed-knob controller couldn't fix.

## 2. The contrasts: residual and demo-seeding

- **#1 raw residual (0.13)** beats the heuristic (it fixes *lift*: 0.69 → 0.95) but lands **below** the
  abstraction. Correcting raw actions helps; learning at the *option level* helps more. That gap — 0.13 vs
  0.24 — is the value of operating in the abstraction rather than the action space.
- **#3 demo-seeding (0.00)** is a **clean negative**: seeding the replay with 504 successful demos did *not*
  break the hover optimum — 0% success across 3 arms (frac 0.25 ×2, 0.50) through ~2M steps. It's not
  *nothing*, though: the planner's `box_target_max` rose to **0.16**, above pure TD-MPC2's hover ceiling
  (~0.08) — so demos pull the planner partway toward the target, but buffer-mixing alone can't get it over
  the line. (Next lever there would be BC-warmstart or demo-conditioned planning, not just replay-mixing.)

## 3. Why none of them beats PPO's 0.66 — and it is *not* the budget

The obvious worry: are we just under-training? **No — the data says it's a structural ceiling:**

- **#1 residual plateaued at 0.14 after 61M ES env-steps — ~2× PPO's 33M budget.** More steps bought nothing.
- **#2 plateaued too:** its ES fitness oscillated flat from iter ~150→350 (8.7M→20M steps), and final
  (0.17–0.21) sits *below* peak (0.22–0.24) — converged, not still climbing.

So the gap to PPO is not a sample shortfall. It is three structural things:

1. **The heuristic prior caps expressiveness.** #1 can only add a small (α=0.3) correction; #2 can only tune
   a *fixed* phase machine. The pick-place *structure* is hand-coded. PPO learns the policy end-to-end with
   no such cap, so it can represent solutions the hybrids structurally cannot.
2. **The place phase is the wall.** Every hybrid nails grasp (~0.99) and lift (~0.95) and caps at place
   (~0.24). The analytic place controller can't reliably settle the cube within 2 cm; #2 mitigates it by
   adapting per-context but can't fully escape the phase structure.
3. **ES is a weak, gradient-free optimizer** over a small parameter space — a good local optimum of the
   *constrained* problem, not a global policy.

**The honest framing is a trade-off, not a defeat.** The heuristic-prior hybrids buy a **fast escape from
0%** (0.15–0.24 by 20M, where pure TD-MPC2 is still flat 0) at the cost of a **ceiling (~0.24)**. PPO is the
opposite: 0% until ~28M, but unconstrained, so it climbs to 0.66. Two operating points on the same
sample-efficiency-vs-ceiling frontier. **To push past 0.24, relax the structural cap** — gradient-based
fine-tuning of the options policy, warm-started from #2's 0.24 instead of from 0. That's the clean next
experiment, and it trades back toward PPO's sample cost.

## Verdict

- **The abstraction well is *not* dry — we'd been drilling the wrong spot.** A **semantic, value-aware,
  state-conditioned** abstraction (the HL phases with a learned high-level policy) gives the campaign's
  **first positive abstraction result** (0.063 → 0.24), beating the raw-action residual (0.13) and the
  abstractions that scored 0 before (jumpy, Glass). This validates Part 22's Rule E directly.
- **The lever is "learn the abstraction (×2) then condition it on state (+50%)."**
- **It's a different operating point from PPO, not a beat** — fast escape from 0, structural ceiling ~0.24;
  PPO is slow but unconstrained to 0.66. Closing that needs gradient fine-tuning, not more ES steps.
- **Demo-seeding alone is a clean negative** (0%, partial planner progress to bt 0.16).

### Caveats (kept honest)
ES env-steps are not apples-to-apples with PPO gradient steps; the hybrids' "sample-efficiency" is escaping
0%, **not** a sample-efficiency win over PPO (which reaches far higher success). All hybrids remain well
below PPO's 0.66. Single-to-three seeds per cell, one GPU class, real success read from
`exp/tdmpc_glass/hl_{residual,options,options_global,demoseed}/` + `exp/benchmark/*demoseed*realsuccess.csv`.
The gradient-fine-tune follow-up (#4) is a hypothesis, not a result. Live kanban of all runs:
the b3060 dashboard. Prior context: Parts 17 (reward-hacking), 19 (jumpy/Glass nulls), 22 (design rules).
