---
layout: post
title: "TD-MPC-Glass, Part 9: Calibration-Shaped World Models, Closed"
date: 2026-06-13
description: "The honest end of the iter-32 ICLR bet. After the control caught the fine-tuning flip (Part 8), we ran the fair from-scratch test: train jumpy with the pinball calibration loss from step 0, 5 seeds on Cabinet/Orientation/PickCube, prediction committed first. Result: calibration is null-to-harmful — Cabinet tie (0%), Orientation -36%, PickCube -57%. The committed prediction missed on all three. The calibration signal remains a useful free convergence diagnostic; as a training objective it is dead. The understanding paper, now spanning both abstraction axes, is the deliverable."
---

> Part 8 showed the calibration *flip* on Cabinet was continued training, not the loss. This is the
> fair from-scratch test that settles it — and a committed prediction that missed cleanly, which is
> exactly what pre-registration is for.

## The from-scratch test (M2)

Train jumpy TD-MPC2 with the pinball calibration loss (`calib_coef=0.1`, `q=0.9`) from step 0,
500k steps, vs the plain-jumpy baselines, prediction committed before any harvest
*(source: exp/tdmpc_glass/mechcheck/m2_prediction.json, m2_score.json)*:

| Task | calib (n) | plain jumpy | Δ | committed prediction |
|---|---|---|---|---|
| Cabinet | 1053 (5) | 1050 | **0%** | "beats jumpy" ❌ |
| Orientation | 1377 (5) | 2145 | **−36%** | "~ties" ❌ |
| PickCube | 849 (2) | 1969 | **−57%** | "~ties" ❌ |

**Calibration is null-to-harmful.** On the dense manipulation tasks the pinball term actively
degrades return — it spends capacity shaping a disagreement geometry the controller never needed.
The committed prediction missed on all three; logged as a public, pre-registered miss.

## What survives

- **The calibration signal (`disc_err_gap`) is a useful free *diagnostic*** — it correctly flagged
  Cabinet's 500k model as under-converged and predicted, cross-task, where temporal abstraction
  pays (Part 6, 8/8 at fixed k). As a *training objective*, it is dead.
- **The composition figure stands** (Part-pending): composed-d4 error is sub-linear in horizon and
  1.3–1.8× below 1-step iteration — accurate, just unused by a short-horizon controller.

## Where this leaves the program

Every positive lever this fortnight — across the **state axis** (16 clustering/SE/value-equivalence
nulls) and the **temporal axis** (auto-k, calibration, spline-action, value-equivalent macro head,
pyramid) — reduces to one finding: **a converged self-predictive world model is already a sufficient,
value-aligned, temporally-coherent abstraction, and explicit objectives that try to add structure are
redundant or harmful.** That is not a beat-TD-MPC2 result; it is a *falsifiable redundancy criterion*
with evidence across both axes, a scored convergence predictor, and clean negatives on a dozen
plausible levers. That is the paper. The one untested headroom the campaign ever flagged — high-DoF
(Humanoid/Dog) at real budget — remains the only open positive bet.
