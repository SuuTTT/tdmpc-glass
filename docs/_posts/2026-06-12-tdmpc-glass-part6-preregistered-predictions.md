---
layout: post
title: "TD-MPC-Glass, Part 6: Pre-Registered — When Does Temporal Abstraction Pay?"
date: 2026-06-12
description: "We publish our predictions BEFORE looking at the results. One checkpoint signal survived a 9-candidate screen against the measured jumpy gains (+90/+32/0% on Ori/Pick/Cab): disc_err_gap, the ratio of the model's self-disagreement to its true k-step error. Story: temporal abstraction pays where the macro-model is accurate AND calibrated-conservative; it fails where the model is bad and overconfident. Committed predictions for the running k-sweep: k=2 underperforms k=4 on all three tasks; task ordering preserved. Also in this post: two mechanism-checks died this week (spline action bottleneck NO-GO at 36% vs 95% gate; value-equivalent macro head closed by existing data) — including a falsification-grade lesson: a mechanism-check GO licenses a test, it never predicts success."
---

> Part 5 promised three probes. Two are already dead — honestly, cheaply, and one of them in a way
> that taught us something about our own method. The third produced a falsifiable prediction, and
> this post is us nailing it to the door **before** the results come in. The finals of the k-sweep
> are sitting unread in our mirror as this goes live; the predictions below are committed to git
> (`exp/tdmpc_glass/mechcheck/p1_ksweep_prediction.json`, commit history is public). When the
> harvest happens, this post gets a scored UPDATE — either way.

## The two deaths first

**P2 — Hermite-spline action bottleneck: NO-GO.** Pre-registered gate: splines at knot-spacing 4
must preserve $\geq 95\%$ of expert return under open-loop replay. Measured: **36.4%** (zero-order
hold: 30.5%) *(source: exp/tdmpc_glass/mechcheck/spline_mechcheck_PandaPickCube.json)*. TD-MPC2's
winning Panda actions carry high-frequency content (~0.49 L2/step deviation from the spline) that a
2-parameters-per-knot bottleneck cannot express. One dev-box run instead of a 30-run gate. Caveat
(stated before the run): open-loop replay is a harsh bound — but the gate is the gate.

**P3 — value-equivalent macro head: closed by data we already had, plus a lesson.** The fresh
mechanism-check said GO on Orientation (jumpy's k-step errors systematically cost value:
value-cost ratio 0.35, $\rho = 0.57$) and NO-GO on Cabinet ($\rho = 0.23$) — a contrast that
beautifully mirrors where jumpy wins and ties. But the phasei27 archive already contained the
experiment: a value-equivalence loss on the macro head scored **583 final vs jumpy's 2145** on
Orientation (n=3 vs n=5) — catastrophic harm on exactly the task the mechanism-check green-lit.
**The falsification-grade lesson: mechanism-check NO-GOs kill reliably; GOs only license a test —
they never predict success.** That asymmetry goes into the paper.

## The survivor: one signal out of nine

With mech dumps (rollout latents + true k-step errors + ensemble-free disagreement) for all three
anchor tasks, we screened nine checkpoint-computable signals against the measured jumpy final-return
gains — Orientation $+90\%$ (CI-separated), Pick $+32\%$ (not separated), Cabinet $0\%$. Exactly one
survived the ordering screen *(source: exp/tdmpc_glass/mechcheck/p1_temporal_signals.json)*:

$$\texttt{disc\_err\_gap} = \frac{\mathrm{median}(\text{disagreement})}{\mathrm{median}(\text{true k-step error})}
\qquad \text{Ori } 1.33 \;>\; \text{Pick } 1.12 \;>\; \text{Cab } 0.95$$

where *disagreement* is the jumpy-vs-iterated-one-step prediction gap — computable from a single
checkpoint, no ground truth needed. The story: **temporal abstraction pays where the macro-model is
accurate AND knows when it's wrong (disagreement $\geq$ error, "calibrated-conservative"); it fails
where the model is bad and overconfident** (Cabinet: 3× Orientation's error at the same latent
scale, with disagreement *under*-estimating it). It is also the only scale-invariant candidate —
the absolute-scale signals are confounded across dump eras, which we flag rather than hide.

**Honesty box:** three tasks, one checkpoint per task at k=4. An ordering match at $n=3$ has chance
probability $1/6$ per signed direction. This is a *hypothesis generator*. Which is why:

## The pre-registered predictions (committed before any harvest)

We trained fresh jumpy variants at $k=2$ and $k=8$ (effective planning horizon fixed at 24 steps,
compute-matched) on all three tasks, dumped their checkpoints, computed the signal, and committed
the predictions **before reading a single final return**:

| Task | gap at $k{=}2$ | gap at $k{=}4$ | Prediction |
|---|---|---|---|
| Pick | 0.969 | 1.121 | $k{=}2$ gain **lower** than $k{=}4$ |
| Orientation | 1.028 | 1.332 | $k{=}2$ gain **lower** than $k{=}4$ |
| Cabinet | 0.672 | 0.949 | $k{=}2$ gain **lower** than $k{=}4$ |

plus: the cross-task ordering (Ori > Pick > Cab) persists at $k{=}2$. The $k{=}8$ cells and two
out-of-sample tasks (PandaPickCubeCartesian — training as this posts — and CheetahRun, a different
*domain*) follow the same protocol: signal first, prediction committed, then the gain.
*(source: exp/tdmpc_glass/mechcheck/p1_ksweep_prediction.json; git history timestamps the order.)*

**What would kill the predictor:** $k{=}2$ matching or beating $k{=}4$ on Orientation; or the task
ordering scrambling at $k{=}2$; or the out-of-sample tasks landing opposite to their signal. Any of
these and the predictor dies in public, in this post's update — and the paper keeps only the
redundancy criterion plus an honest negative on the positive direction.

*This post will be UPDATED with the scored table once the k-sweep finals are harvested.*

---

## UPDATE (13:30 UTC): The score

The harvest happened. Scoring the committed predictions against finals read for the first time
*(source: exp/tdmpc_glass/mechcheck/p1_score.json, p1_ksweep_harvest.json)*:

| Block | Prediction | Result |
|---|---|---|
| Pick $k2<k4$ | 1616 vs 1969 | ✅ |
| Ori $k2<k4$ | 1487 vs 2145 | ✅ |
| Cab $k2<k4$ | 596 vs 1050 | ✅ |
| k2 gain ordering Ori>Pick>Cab | $+358 > +261 > -457$ | ✅ |
| Pick $k8>k4$ | 1302 vs 1969 | ❌ |
| Ori $k8>k4$ | 1469 vs 2145 | ❌ |
| Cab $k8>k4$ | 694 vs 1050 | ❌ |

**4/4 on the k2 block, 0/3 on the k8 block.** The honest reading: `disc_err_gap` is a real
*cross-task* predictor at fixed $k$ (now 7/7 ordering facts across two k values) — but it is **not
k-invariant**: iterating the 1-step model $k$ times inflates disagreement mechanically with $k$, so
*upward* cross-k comparisons are confounded. The committed k8 predictions walked straight into that
confound and died in public, exactly as pre-registration is supposed to work.

Two bonus results from the same harvest: **the dose–response is unimodal — $k{=}4$ beats both
$k{=}2$ and $k{=}8$ on all three tasks** (temporal abstraction has an optimal grain, neither too
fine nor too coarse); and **the Pick anchor resolved at $n{=}8$: jumpy $+45\%$, difference CI95
$[66, 1153]$ — now CI-separated.** Jumpy's scoreboard is 2/3 tasks CI-separated (Pick, Orientation)
and one null (Cabinet). The CheetahRun cross-domain test (predicted "weak-positive", gap 1.017,
committed before any final exists) reports in Part 7 tonight.
