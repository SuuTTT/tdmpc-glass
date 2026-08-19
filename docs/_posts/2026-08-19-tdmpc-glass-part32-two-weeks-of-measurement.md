---
layout: post
title: "TD-MPC-Glass, Part 32: Two Weeks of Measurement — What Survived"
date: 2026-08-19
description: "A full accounting of ~400 runs since Part 31. Two results survive: a planner cut to 3% of its search compute is statistically indistinguishable from full MPPI on hopper (p=0.80) while recovering 91% of its value, and effective rank — the field's standard representation-health check — is uninformative on every test we can construct, including zero correlation with planning performance (rho=-0.029, p=0.923). Structural information is closed as a measure: five variants, including the proposal's own differentiable soft-assignment form, none of which track information content. The SI regulariser test came back null on two of its three stated outcomes. Also: eight retractions, two of them numbers we had already sent our supervisor, and the six-gap research agenda the failures produced."
---

> **TL;DR.** Since Part 31 we ran ~400 training runs across four control tasks, three axes where the
> true information content is known by construction, and five variants of structural information.
> **Two things survive.** (1) Most of a planner's search budget is wasted: at 3% of the compute it is
> statistically indistinguishable from full MPPI on hopper. (2) Effective rank has *zero* relationship
> to planning performance, and inflates where information is provably constant. **One thing is
> closed:** SI does not measure information content, in any variant we could build — though the
> failures were specific enough to become a six-gap agenda that six papers now answer. **Eight claims
> were retracted along the way**, including two we had already sent to our supervisor.

## The one-paragraph version

We have one finished positive result, one strong negative we think is publishable, and a closed
question. The positive: a planner running at ~3% of full MPPI's compute recovers most of its value.
The negative: effective rank — used across the literature as evidence a representation is healthy —
fails on a constructed axis, on known bits, and on downstream planning. The closed question:
structural information, in five variants including the proposal's own differentiable form, does not
measure information content. The regulariser test of the SI hypothesis came back null, with an
important caveat: we measured prediction and information retention, never planning, because a pure
JEPA has no planner.

## 1. Most of a planner's search is wasted

TD-MPC2's MPPI cut from 512 samples x 6 iterations down to 32 samples x 1 iteration — essentially
best-of-32 random guessing — against full MPPI and against no planner at all.

| task | n | full | cheap | none | planner worth | cheap recovers | cheap vs full |
|---|---:|---:|---:|---:|---:|---:|---:|
| hopper-hop | 10 | 327.3 | 311.4 | 152.5 | +174.8 (p=0.002) | **91%** | -16.0, **p=0.80** |
| walker-run | 5 | 806.4 | 767.8 | 735.1 | +71.3 (p=0.008) | 46% | -38.6 |
| acrobot-swingup | 5 | 497.2 | 452.6 | 327.4 | +169.7 (p=0.21) | 74% | -44.6, p=0.58 |
| cheetah-run* | 4 | 917.1 | 910.0 | 851.4 | +65.6 (p=0.03)* | 89% | -7.1 |

**The claim:** a planner is clearly worth having; most of its *search budget* is not. On hopper the
cheap planner is statistically indistinguishable from full MPPI at 3% of the cost.

Two caveats we would state in any paper. First, **cheetah's p=0.03 is one collapsed seed** — the
no-planner runs were `[680, 903, 911, 912]` against full's `[913, 916, 918, 921]`, so three of four
match exactly. We treat cheetah as saturated, not as evidence. Second, the recovered fraction is
**budget-dependent**: on walker at 200k steps the planner was worth +139.6 and cheap recovered 74%;
at 400k it is worth +71.3 and cheap recovers 46%. The planner's advantage *shrinks* with training,
so any headline number has to state its budget.

## 2. Effective rank is uninformative

This is the finding we would put first. To test it we needed a case where the true information
content is known, so we built one.

**The setup.** A pure JEPA — encoder, predictor, EMA target, no reward, no value, no policy — on
64x64 pixel observations of a robot arm and a cube (`visual-cube-single-v0`, OGBench). Then we add
visual clutter carrying *no information about the cube*: the distractors live in a panel appended
beside the frame, so the scene pixels are **bit-identical** at every level. Information is unchanged
by construction, and we confirmed it independently by decoding the cube's position straight from raw
pixels — that drifts by **-0.9%** across the whole axis. An honest measure must stay flat.

| measure | 0 -> 8 distractors | p (exact perm, n=5) | verdict |
|---|---:|---:|---|
| recoverable information *(ground truth)* | 0.88 -> 0.89 | 0.667 | flat ✓ |
| **effective rank** | **15.0 -> 26.8** | **0.0079** | doubles ✗ |
| SI - random-partition control | 0.29 -> 0.06 | 0.0159 | falls ✗ |
| InfoNCE *(a real MI lower bound)* | 0.89 -> 1.03 | 0.0317 | rises ✗ |
| variance | 600 -> 270 | 0.730 | flat in mean only |
| embedding std *(collapse detector)* | 0.11 -> 0.10 | 0.762 | flat ✓ |

p = 0.0079 is the *smallest attainable* value for an exact permutation test with five seeds a side,
and the groups are completely separated (level 0: 6.9-20.4; level 8: 25.8-27.3). Variance "passes"
only at its endpoints — its medians run 223 -> 9010 -> 924 -> 234, a 40x excursion driven by encoder
scale drift.

**And it does not predict planning.** Separately we trained 15 TD-MPC2 agents with deliberately
varied representation quality, scoring every agent on the *same* fixed probe set:

| measure | rho vs planning return | p |
|---|---:|---:|
| recoverable information | +0.607 | 0.020 |
| **effective rank** | **-0.029** | **0.923** |
| variance | -0.489 | 0.066 |
| SI (directed) | +0.539 | 0.043 |
| SI - null | -0.025 | 0.936 |

**The caveat, stated up front:** that sweep produced two clumps rather than a gradient — one arm at
~732 and everything else at 800-810. Dropping the broken arm, *every* correlation loses significance.
So it shows measures can spot a badly-trained representation, not that they track quality among
decent ones. We tried twice more to build a graded axis and both attempts failed.

## 3. Structural information: five variants

Each variant fixed a defect in the last.

| variant | what changed | vs known bits | outcome |
|---|---|---:|---|
| 1. k-means partition | our first proxy | +0.099 | no signal |
| 2. selib SE-optimal | proper SE optimiser | +0.285 | null scored **+0.322** — higher |
| 3. exact certificate | global optimum, O(3^n) DP | -0.070 | worse than local search |
| 4. directed + self-loops | faithful to the proposal's spec | -0.204 above null | fails, and not flat (-0.53x) |
| 5. **differentiable soft assignment** | the proposal's own form | +0.04 … +0.14 | stable, but constant |

**Variant 5 is the interesting one.** It is the only variant that robustly clears its null — 100% of
157 runs — so it behaves sensibly where our hard-clustering proxies did not. "SI doesn't work" was a
statement about our proxies, not about the proposal's formulation.

But it does not track information content at any resolution. We swept the number of structural
groups K to rule out saturation:

| K | ceiling log2(K) | SI | % of ceiling | margin vs null | rho vs bits |
|---:|---:|---:|---:|---:|---:|
| 4 | 2.00 | 1.771 | 89% | +1.72 | +0.120 |
| 8 | 3.00 | 2.499 | 83% | +2.40 | +0.042 |
| 16 | 4.00 | 2.973 | 74% | +2.80 | +0.136 |
| 32 | 5.00 | 3.301 | 66% | +2.95 | +0.081 |
| 64 | 6.00 | 3.549 | **59%** | +2.77 | +0.039 |

It converges toward ~3.8-4 bits and uses progressively *less* of its ceiling, so it estimates a
genuine finite quantity rather than merely saturating. That quantity appears to be **temporal
smoothness of the latent trajectory** — real, reliably above chance, and invariant to how much the
representation knows about the world.

## 4. The regulariser test

`L = L_pred - lambda * SI`, using the differentiable SI above.

| condition | lambda = 0 | lambda = 0.1 | lambda = 1.0 |
|---|---:|---:|---:|
| clean (readout R2) | 0.836 | 0.890 (p=0.56) | 0.890 (p=0.61) |
| heavy distractors (readout R2) | 0.883 | 0.890 (p=0.71) | 0.840 (p=0.29) |
| seed sd, clean | 0.123 | 0.039 | 0.027 |

No significant effect on retained information, and at lambda=1.0 with distractors it is nominally
worse. One-step prediction loss shows no significant movement either.

**One suggestive signal:** lambda > 0 cuts seed variance more than fourfold without moving the mean.
The regulariser may *stabilise* training rather than improve it. n=5, so we would not claim it.

**The gap in our test.** The hypothesis says SI improves "action-conditioned prediction *and
planning*". We measured prediction and information retention. We never measured **planning**, because
a pure JEPA has no planner. So this is a null on two of three outcomes, not on the hypothesis as
written.

## 5. What we got wrong

Included because it is the reason to trust anything above.

| claim | killed by |
|---|---|
| SI tracks true bits at r ~ +0.9 *(already sent to our supervisor)* | moving from 2-D toys to pixels: +0.099 |
| selib SI passes both axes | a degree-matched null scoring **higher** than the real graph |
| distractors leave information unchanged | raw-pixel decoding 0.958 -> 0.674 — our squares *occluded the cube* |
| the corruption axis measures corruption | decodability sat at 0.933/0.934/0.932/0.934 — the knob did nothing |
| cheap planner recovers 74% *(already sent)* | unmatched budgets; at 400k it is 46% |
| cheap beats full on hopper (172%) | n=5 noise; at n=10 it is 91% and indistinguishable |
| latent_dim gives a graded quality axis | between-group spread 13.0 < within-group sd 5.9 — a flat line |
| SI derives hierarchy depth | SE's encoding trees are binary by construction; depth matched a null |

The pattern is consistent: **every striking result died to a control, and the controls were cheap.**
The most instructive case is the third row — painted distractors produced the most impressive numbers
in the entire project (rank +0.97x, variance +12.5x, decodability halving) and all of it was an
artifact of the squares covering the object.

## 6. A detour: does anything predict what compression destroys?

LeCun's group published [Patch Policy](https://arxiv.org/abs/2607.18236), showing that compressing an
observation into one pooled token instead of dense ViT patch tokens costs up to **8x** in task
success. That is a published, real ground truth for "how much information does compression destroy" —
better than any axis we can build ourselves.

| task | pooling cost (truth) | SI - null | patch rank |
|---|---:|---:|---:|
| LIBERO Goal | x0.99 (none) | 1.841 | **105.3** |
| Push-T | x1.01 (none) | 1.617 | 69.0 |
| Cube | x8.24 (severe) | 1.807 | 97.9 |

Null — and the ordering is actively wrong: the task where pooling costs *nothing* has the highest
rank and the highest SI. Worth noting that with only Push-T and Cube it looked like both measures
tracked the cost; adding the third task inverted it.

## 7. The failures became an agenda

The SI failures were specific enough to name what SE theory is missing. We filed them as six gaps
([selib#10](https://github.com/SuuTTT/selib/issues/10)), and six papers now answer them. Two
independently confirmed our diagnoses:

- **Depth is unidentifiable.** They reproduced our exact pathology — "depth 6-9, branching exactly
  2.00" — on planted data, and fixed it with an MDL tree cost that makes the number of levels
  well-defined.
- **No null model.** Now a closed-form expected SE, plus the identity that calibrated SE equals
  *code-length-weighted modularity*, placing SE exactly relative to Newman-Girvan.

We tried to redo our measurements with the new calibration. It is **inconclusive**: the closed form
returns NaN on 106 of our 164 graphs (ours are dense and weighted), and those NaNs are written as
exactly 0.0, which makes a naive read give rho = +0.918 — really "which runs the estimator crashed
on". We are not quoting that number.

## Where this leaves us

The defensible claims are narrow and we would rather state them narrowly:

1. **A planner at 3% of the compute is indistinguishable from full MPPI on hopper**, and recovers
   about half its value on walker. The planner matters; its search budget mostly does not.
2. **Effective rank should not be used as evidence that a representation carries more information.**
   It inflates where information is provably constant, carries no signal about exact bits, and has
   zero correlation with planning performance.
3. **Nothing computable without labels passes all three tests.** The only quantity that is flat where
   information is fixed, tracks it where it varies, and predicts downstream performance is held-out
   decodability of the true state — which needs exactly what you do not have in practice.

One experiment is still open: 21 hopper agents trained at a fixed configuration, using their
*natural* seed spread (returns 0-576) as the quality axis instead of a knob we set — variation we do
not have to manufacture. The runs and 18 checkpoints are saved; the measurement pass has not been run.

*Everything above traces to committed data in `SuuTTT/SE-JEPA` and `SuuTTT/world-model-paper`.
Statistics are exact permutation tests throughout; where n=5 per side the smallest attainable p is
0.0079, so that value indicates complete separation rather than a marginal effect.*
