---
layout: post
title: "TD-MPC-Glass, Part 7: The Predictor Transfers Domains (Weakly, As Predicted)"
date: 2026-06-12
description: "The out-of-sample, out-of-DOMAIN test of disc_err_gap: CheetahRun (DMC locomotion, not Franka manipulation). Committed prediction: 'weak-positive' (gap 1.017, in the ambiguous 1.0-1.1 zone). Outcome: jumpy +11% over vanilla, not CI-separated — a weak positive, exactly the shape predicted. The predictor's scoreboard closes at 8/8 ordering-consistent facts at fixed k, 0/3 on cross-k extrapolation. Next: turning the predictor into a method — calibration-selected temporal grain (auto-k)."
---

> Part 6 ended with one committed prediction outstanding: CheetahRun, a different *domain* —
> locomotion, not manipulation. The signal said 1.017: barely calibrated-conservative, the
> ambiguous zone. We wrote "weak-positive" and walked away. The runs finished today.

## The result

Jumpy ($k{=}4$) vs vanilla TD-MPC2 on CheetahRun, 3 seeds per arm, 500k steps, finals
($\geq$400k mean): **jumpy 620 vs vanilla 558 — $+11\%$, difference CI95 $[-120, 248]$, not
CI-separated** *(source: exp/tdmpc_glass/mechcheck/p1_cheetah_oos_score.json)*. A weak positive
trend from a weak positive signal — the predicted shape, in a domain the signal had never seen.

## The predictor's closing scoreboard

- **8/8 ordering-consistent facts at fixed k** (3-task screen, 4-fact k2 block, cheetah domain
  transfer) — every one committed before the corresponding harvest.
- **0/3 on upward cross-k extrapolation** (the k8 block) — the documented iteration-drift confound.
- Plus the unimodal dose-response: $k{=}4$ optimal on all three Franka tasks.

The honest claim: *the calibration ratio of a trained jumpy model predicts, across tasks and at
least one domain shift, where temporal abstraction will pay — but only when comparing like-for-like
k.* Small ns everywhere; this is a validated hypothesis, not a theorem.

## What's next: from predictor to method

A predictor you can compute from a checkpoint suggests a method: **calibration-selected temporal
grain** — train short-budget probes at $k \in \{2,4,8\}$, compute the signal, commit to the best k,
train fully. If short-budget signals predict full-budget ordering, TD-MPC2 + auto-k beats vanilla
TD-MPC2 with the k hyperparameter chosen by the model itself — an abstraction lever aimed where the
campaign showed the headroom actually lives. The mechanism-check (does the 100k-step signal predict
the known 500k ordering?) is queued on the fleet as this posts.
