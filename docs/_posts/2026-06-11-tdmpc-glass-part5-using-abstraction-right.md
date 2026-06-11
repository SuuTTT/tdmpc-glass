---
layout: post
title: "TD-MPC-Glass, Part 5: Using Abstraction Right — Aim It Where Sufficiency Fails"
date: 2026-06-11
description: "The plan after the redundancy criterion. The 16 nulls all attacked the state representation in-distribution — exactly where the latent is already value-sufficient. But the campaign's own data shows where sufficiency does NOT hold: the planning axes. Jumpy (a temporal abstraction) is the one real win (+90% on Orientation, CI-separated) and is unexplainedly task-dependent (null on Cabinet); MPPI's search cost never shrank no matter how good the latent got. Iteration 30: (P1) a cheap pre-test predicting WHERE temporal abstraction pays, (P2) the Hermite-spline action bottleneck, (P3) a value-equivalent macro head. Draft plan for review."
---

> Part 4 closed the state axis: explicit abstraction on the representation is redundant exactly when
> the latent is value-sufficient — and on this substrate it always was. This post is the *constructive*
> sequel. If abstraction is redundant where sufficiency holds, then using abstraction **right** means
> aiming it where sufficiency **fails**. Our own data says where that is: not the state, but the
> *planning* axes — time and action.

**Draft plan for review.** Full version with costs and protocols:
[`docs/research/abstraction-axes-plan.md`](https://github.com/SuuTTT/tdmpc-glass/blob/master/docs/research/abstraction-axes-plan.md).

## The argument in three observations

1. **The one real win of the campaign was a temporal abstraction.** The jumpy k-step world model —
   prior art, not ours — beat vanilla where sixteen state-abstraction levers could not: PandaPickCube-
   Orientation final $+90\%$, difference CI95 $[685, 1344]$, every jumpy seed above every vanilla seed
   *(source: exp/tdmpc_glass/mechcheck/anchor_jumpy_vs_vanilla.json)*. A value-sufficient latent does
   not shorten the planning horizon. That bottleneck lives outside the representation — which is
   precisely why the redundancy argument never touched it.

2. **Planning cost is a search problem, not a representation problem.** MPPI searches a $k \cdot d$-
   dimensional action sequence no matter how good the latent is. Sufficiency of the *state* never made
   the *action space* smaller. The criterion is silent about this axis — meaning it's open.

3. **The temporal win is task-dependent and we don't know why.** Jumpy: $+90\%$ on Orientation,
   $+32\%$ (not separated) on Pick, a dead tie on Cabinet (1050 vs 1053). Something mechanistic
   separates these tasks. An unexplained split in your own positive result is a research question
   with your name on it.

## Iteration 30 — three probes, all single-variable, all mechanism-check-gated

- **P1. A pre-test for WHERE temporal abstraction pays.** The mirror image of the redundancy
  criterion: from existing checkpoints, compute candidate signals (k-step-error decay slope, reward
  smoothness/contact density, plan-horizon utilization), rank-correlate against the measured jumpy
  gains ($+90/+32/0\%$), then pre-register the best signal's prediction on two unseen tasks and verify
  with a $k \in \{2,4,8\}$ sweep. Completes the paper's predictive story in both directions.
- **P2. Hermite-spline action bottleneck** (the ledger's #1 untested lever). Macro-action = spline
  knots, PD tracker executes; MPPI's search shrinks from $k \cdot d$ to $2d$ with **no learned codec**,
  so the redundancy result cannot touch it. Gate only if splines reconstruct winning replay
  trajectories with $\geq 95\%$ return preservation.
- **P3. Value-equivalent macro head.** Our VE null tested *latent* value-equivalence on the 1-step
  model; this trains the k-step head to preserve *macro-return* instead of state — the one VE variant
  the null does not cover, riding the only model where capacity actually binds.

Parked: compositional-OOD state abstraction — the synthetic gate showed value-decodability does *not*
collapse at held-out object counts ($R^2$ 0.96 → 0.92/0.94), so there is no headroom signal yet.

## Compute, honestly

Fleet right-sized today to **6 boxes ≈ \$0.57/hr ≈ \$14/day**: a 5070 Ti dev box for mechanism-checks
plus 4 A4000 workers and one cheap 3060. That sustains one full 30-run gate (~2 arms × 3 tasks × 5
seeds) every ~2 days with same-day mechanism-checks — and since the protocol's mechanism-check gates
mean at most one gate is ever worth running at a time, more GPUs would not make the science faster.
The bottleneck is decision quality, not compute.
