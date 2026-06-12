---
layout: post
title: "TD-MPC-Glass, Part 8: The Control Caught Us"
date: 2026-06-12
description: "The day's headline result died a six-hour death, and the methodology worked exactly as designed. Calibration fine-tuning flipped Cabinet's catastrophic composition failure (rho 2.54 -> 0.69, win-rate 0.3% -> 76%) and tripled the model's accuracy — but the more-training control (same +100k steps, NO calibration loss) produced the identical flip with even lower error. The flip was budget, not calibration. The real discovery underneath: Cabinet's jumpy null — a cornerstone of our anchor — appears to be an undertraining artifact, and the calibration signal detected it all along. Pre-registered next: does the Cab null flip at 600k?"
---

> Today we had, for about six hours, the headline experiment of the ICLR bet: train a world model to
> know when it's wrong, and composition stops failing. M1 passed its gate spectacularly. The
> re-composition flipped. And then the control we queued *before celebrating* — identical fine-tune,
> calibration loss set to zero — produced the same flip. This post is the autopsy, and the better
> finding underneath it.

## What happened, in order

1. **M1 (calibration fine-tune, +100k on Cabinet):** disc/err gap 0.949 → 1.301, true k-step error
   0.630 → 0.211 — gate passed *(calib_m1_verdict.json)*.
2. **Re-composition on the calibrated model:** $\rho$ 2.54 → **0.692**, win-rate 0.3% → **75.7%** —
   the catastrophic composition NO-GO flipped to GO *(pyramid_mechcheck_Cab_CALIBRATED.json)*.
3. **The control (same +100k, `calib_coef=0`):** $\rho$ **0.683**, win-rate **80%**, error
   0.630 → **0.060** *(pyramid_mechcheck_Cab_CONTROL.json, calib_control_comparison.json)*.

Identical flip. Lower error. **The calibration loss contributed nothing beyond continued training.**

## The finding that survives

Cabinet's jumpy model at 500k steps was simply **undertrained** — 100k more steps cut its error by
10×, restored calibration, and made composition work. Three things follow:

- **The "Cabinet null" in our anchor is likely a budget artifact, not a task property.** We have
  queued the pre-registered test: plain jumpy vs vanilla at 600k, 3 seeds each, compute-matched. If
  the null flips, the anchor becomes 3/3 — and "where does temporal abstraction pay" becomes
  partially "where has the macro-model converged."
- **The calibration signal worked the whole time** — it flagged Cabinet's model as bad-and-overconfident
  when it was, and as healthy after more training. It is a *convergence diagnostic*. That is less
  romantic than "calibration-shaped world models" and more useful.
- **Composition (the pyramid) works on converged models** — Ori GO at 500k, Cab GO at 600k-equivalent.
  The pyramid idea survives; its gating variable is convergence, which disc_err_gap measures for free.

## Scoreboard for the method's honesty machinery

Pre-registration, mechanism-check gates, and a control queued before the champagne: the system
killed our best-looking result in six hours for the cost of one fine-tune run. The M2 from-scratch
test of the calibration loss (15 runs, prediction committed before launch) still runs — if
calibration-during-learning differs from fine-tuning, it gets its fair shot. Expectations: low,
stated in advance.
