---
layout: post
title: "TD-MPC-Glass, Part 12: When Does Jumpy Planning Actually Help?"
date: 2026-06-16
description: "A fresh-from-scratch reproduction of the jumpy (k-step macro) world model vs vanilla TD-MPC2 across six tasks × five seeds × 500k steps. The clean result: temporal abstraction delivers genuine capability gains on contact manipulation (Pick +60%, Pick-Orient +161%, CI-separated on BOTH peak and final), provides only late-training stability on Open-Cabinet, and is neutral-to-harmful on locomotion and sparse. The peak-vs-final split caught a near-miss — a final-only read inflated the manipulation story into a clean 3/3 sweep that the peak metric does not support. And the project's recurring lesson holds: the macro-model predicts better on every task, yet that only converts to control where the dynamics are contact-structured."
---

> After the redundancy criterion closed explicit *representation* abstraction across four axes
> (state, temporal, relational, compositional), the question shifted from "does abstraction help?"
> to "where does the one technique that *did* win — the jumpy k-step world model — actually help, and
> can we say why?" This is that experiment, reproduced from scratch.

## The setup

Vanilla TD-MPC2 vs the **jumpy** variant (a k=8 macro-dynamics head + macro-MPPI planning, effective
horizon k·n_macro = 24), single-variable, on **six tasks spanning three regimes**:

- **Contact manipulation:** PandaPickCube, PandaPickCubeOrientation, PandaOpenCabinet
- **Locomotion:** CheetahRun, HopperHop
- **Sparse:** CartpoleSwingupSparse

**5 seeds × 500k env-steps each** (the budget at which the original anchor was measured), 60 runs total,
on rented GPUs. The old checkpoints were gone, so every number here is a clean re-run — and every
checkpoint and CSV streamed to HuggingFace as it was produced, so nothing was lost this time.

## The learning curves

![D2 learning curves: jumpy vs vanilla TD-MPC2 across six tasks, mean ± SEM across seeds, 500k steps]({{ '/images/d2_learning_curves.png' | relative_url }})

The picture reads cleanly by regime: on the **contact** tasks (top row) the jumpy curve (red) sits
above vanilla (blue); on **locomotion** (Cheetah, Hopper) vanilla wins — Hopper is the starkest, where
the jumpy agent never leaves the floor (flat at ~0) while vanilla hops; Cartpole-sparse is high-variance
noise for both.

## The result — and the honesty check that mattered

We score every run on two metrics the protocol requires: **peak** (best eval, a capability ceiling) and
**final** (mean of the last evals, a deployment/stability measure). They disagree, and the disagreement
is the whole story:

![D2 summary: Δpeak and Δfinal (jumpy − vanilla) per task with 95% bootstrap CIs; green = jumpy wins, red = jumpy hurts, grey = null]({{ '/images/d2_summary_bars.png' | relative_url }})

| Task | Δpeak (CI95) | Δfinal (CI95) | verdict |
|---|---|---|---|
| **Pick** | **+1017** [817, 1238] | **+1114** [829, 1369] | **genuine win** (both metrics) |
| **Pick-Orient** | **+697** [573, 794] | **+1625** [1276, 1956] | **genuine win** (both metrics) |
| Open-Cabinet | −372 [−729, 1] | +929 [542, 1345] | **stability only** (no peak gain) |
| Cheetah | −62 (null) | −87 (null) | neutral / slightly worse |
| Hopper | −121 [−182, −59] | −91 [−166, −18] | **jumpy hurts** |
| Cartpole-Sparse | +168 (null) | +110 (null) | null (high-variance) |

A **final-only** read says "manipulation 3/3, jumpy wins everywhere it touches a gripper." The **peak**
metric refuses that: on Open-Cabinet jumpy's peak is *no better* (point estimate negative). What actually
happens is that **vanilla TD-MPC2 reaches a high peak on manipulation and then collapses late in
training** (Q-overestimation), while the jumpy agent holds. So Cabinet's "win" is vanilla instability,
not jumpy capability. Only **Pick and Pick-Orient** are genuine wins — higher peak *and* higher final,
CI-separated, every seed. We caught this only because we plotted both metrics; the inflated version
nearly went in the changelog.

## The mechanism — better world model, but only contact converts

On **every task**, including the ones where jumpy loses on return, the macro-model is the more accurate
predictor: jumpy's k-step latent error is consistently below the iterated one-step error
(ratio ≈ 0.55–0.79, "WIN" at every eval). So the jumpy world model genuinely predicts the future better
*everywhere* — yet that prediction advantage only becomes a control advantage on contact manipulation,
where the dynamics are piecewise (approach → contact → lift) and a longer effective horizon pays off.
On smooth locomotion the longer horizon mostly adds variance and hurts.

This is the thread that has run through the entire project: **a better world model is not a better
controller.** Part after part, prediction-quality gains failed to convert to return. Here they convert
in exactly one place, and the place is interpretable.

## What this is, honestly

- **A real, reproduced result:** temporal abstraction (jumpy macro-planning) gives genuine capability
  gains on contact manipulation (Pick +60%, Pick-Orient +161%), and is neutral-to-harmful elsewhere.
  "When to go jumpy" has a clean answer: **contact-structured dynamics.**
- **Not a universal MBRL win.** The same technique that helps a gripper *breaks* a hopper. Anyone
  reporting jumpy/macro world models should report the regime, and should report peak *and* final —
  the final-only manipulation story is partly a vanilla-collapse artifact.

Next: whether a *cheap, pre-training* diagnostic can predict the per-task sign (the deployable version of
"when to go jumpy"), and a separate thread — error-prioritized replay — whose recurrence precondition is
being tested at proper power. Both reported when they land.
