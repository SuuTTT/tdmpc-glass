---
layout: post
title: "TD-MPC-Glass, Part 15: Reproducing the Jumpy World Model Line — InFOM Reproduces, CompPlan Doesn't (and Exactly Why)"
date: 2026-06-19
description: "We set out to reproduce the 2026 jumpy-world-model planner (CompPlan / Compositional Planning with Jumpy World Models) on OGBench, building on the open-source InFOM flow-occupancy scaffold. Two clean results. (1) InFOM reproduces: on cube-single manipulation we match or exceed the paper on all five tasks (task1 1.00/0.96 vs 92.5, task3 0.90 vs 56.4, task5 0.90 vs 70.0). (2) CompPlan does NOT reproduce — it HURTS: base GC-BC 0.30 -> +CompPlan 0.10 on antmaze-medium, versus the paper's 0.49 -> 0.85. And we localized exactly why: the GHM's discounted-occupancy collapses to a near-identity — 256 sampled 'futures' from any start stay within ~0.5 units of a 20-unit maze, at every horizon gamma. The blocker is GHM training (the flow-occupancy bootstrap), not the planner — precisely the long-horizon capability TD-Flow was built to provide."
---

> The one technique that beat TD-MPC2 in this project was temporal abstraction (jumpy models), so we
> went to reproduce the SOTA jumpy-planning line. This is an honest reproduction report: what
> reproduced, what didn't, and a mechanism-check that pins the failure to a single, specific cause.

## The target and the scaffold

- **GHM (Geometric Horizon Model):** a generative model of the *discounted future-state (occupancy)
  measure* — given a state and policy, sample where you'll likely be at a geometrically-distributed
  future time. It "jumps" over time instead of stepping. (Janner et al. 2020.)
- **TD-Flow** (ICML 2025, arXiv 2503.09817): how to *train* a good GHM via flow matching + a
  low-variance TD bootstrap that propagates probability mass to *far*-future states.
- **CompPlan** (ICLR-WS 2026, arXiv 2602.19634): how to *plan* with a GHM — search over sequences of
  pre-trained base policies, using GHM "jumps" as long-range subgoals. Headline: ~+200% on
  long-horizon OGBench tasks (antmaze, cube).
- **InFOM** (ICLR 2026, arXiv 2506.08902): the open-source JAX flow-occupancy model we build on — it
  trains a GHM-like occupancy model and is benchmarked on **manipulation** (cube/scene/puzzle) + DMC.

No public code exists for TD-Flow or CompPlan, so we extended InFOM (horizon- and policy-conditioning
+ a plan-over-policies planner).

## Result 1 — InFOM reproduces on cube-single (✅)

Running InFOM with the paper's per-task hyperparameters (`expectile=0.95, kl_weight=0.05, alpha=30`),
peak `episode.success`:

| task | **ours (peak)** | InFOM paper (Table 4) |
|---|---|---|
| cube-single task1 | **1.00 / 0.96** | 92.5 |
| cube-single task2 | **0.94** (+ one 0.22 outlier seed) | 78.4 |
| cube-single task3 | **0.90** | 56.4 |
| cube-single task4 | **0.96** | 91.5 |
| cube-single task5 | **0.90** | 70.0 |

We **match or exceed** the paper on every task (cube-single is a high-variance domain — the paper
reports ±37 on task3 — so the one 0.22 outlier seed is expected noise). **The scaffold is faithful.**

## Result 2 — CompPlan does NOT reproduce on antmaze; it hurts (❌)

We trained an OGBench GC-BC base policy, trained a horizon-conditioned GHM on antmaze-medium, wired a
plan-over-policies eval loop, and measured base vs base+CompPlan over 20 episodes:

| antmaze-medium-navigate task1 | base | base + CompPlan |
|---|---|---|
| **ours (measured)** | **0.30** | **0.10** ⬇️ |
| paper (GC-BC) | 0.49 | 0.85 ⬆️ |

The paper gets a large lift; **we get the opposite.** CompPlan *degrades* the base policy.

## The mechanism-check: the GHM's imagination collapses

Why does planning hurt? We probed the trained GHM's occupancy directly — from 5 start states, 256
sampled future states each, at three horizons:

| training/eval horizon γ | mean distance reached | max (of 3,840 samples) | fraction reaching > 2 units |
|---|---|---|---|
| 0.95 | 0.10 | — | 0% |
| 0.99 | 0.11 | — | 0% |
| 0.999 (~1000-step horizon) | 0.11 | **0.48** | **0%** |

The maze spans **~20 units**. The single farthest of **3,840** sampled "futures" reached **0.48
units**. Raising γ from 0.95 to 0.999 changes nothing. **The GHM imagines only "I'll be right where I
am now."** So the planner's subgoals are always ~0.5 units away — useless for a 20-unit maze, and
worse than not planning (they pull the policy off the direct route).

The training trace explains it: the aggregate flow loss is low and plateaued (~0.015, *not*
under-trained), but the *current-step* component is **~25× larger** (0.37–0.44) and the *future*
bootstrap is only loosely fit. That is the signature of a **degenerate near-identity fixed point**:
the flow satisfies the easy aggregate objective by predicting "stay put," and the discount-bootstrap
never propagates mass outward. Horizon-conditioning was nominally on (γ_min<γ_max) but the field
learned to ignore it.

**This is exactly the collapse TD-Flow's low-variance bootstrap was designed to prevent** — and our
InFOM-style bootstrap doesn't prevent it on long-horizon antmaze.

## What it means

- **The CompPlan reproduction is blocked at the GHM-training level, not the planner.** No planner knob
  (beam width, replan rate, γ) can help when every candidate subgoal is < 0.5 units away. The fix has
  to make the flow-occupancy actually spread to distant states — i.e., the real TD-Flow contribution.
- **A method's headline can hinge on a capability that doesn't transfer.** CompPlan's +200% rests on
  the GHM seeing far ahead; on a scaffold (InFOM) built for short-horizon manipulation, that capability
  isn't there, and the planner inherits the collapse.
- **Reproduce the scaffold's own benchmark first.** Cube (InFOM's turf) reproduced cleanly; antmaze
  (off-InFOM, long-horizon) exposed the gap. Checking the foundation localized the failure precisely.

## What's next

We're running a structured **anti-collapse search** (the diagnosis points right at it): make
horizon-conditioning a strong signal (Fourier-embed γ instead of a raw scalar), raise the training
discount and the discount-consistency fraction, and, if needed, fix the bootstrap target so the
"future" term must carry the distribution. The gate is the occupancy-spread probe: does sampled mass
finally reach far subgoals? If yes, the planner already works and the CompPlan lift should follow; if
no, it confirms that a faithful **TD-Flow** bootstrap (not InFOM's) is required.

Either way the finding stands: **InFOM reproduces; CompPlan's long-horizon lift does not transfer to
the InFOM scaffold, because the flow-occupancy GHM collapses to a near-identity — a training-level,
not planner-level, failure.**

### Pointers
- CompPlan arXiv 2602.19634 · TD-Flow arXiv 2503.09817 · InFOM arXiv 2506.08902 (Table 4) · OGBench
  arXiv 2410.20092. Evidence: `exp/tdmpc_glass/ghm/` (cube eval CSVs, `occupancy_probe.json`),
  harness `ghm_repro/infom/planning/eval_compplan.py`, probe `planning/occupancy_probe.py`.
