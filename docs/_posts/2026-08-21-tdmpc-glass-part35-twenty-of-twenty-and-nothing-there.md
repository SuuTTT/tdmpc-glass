---
layout: post
title: "TD-MPC-Glass, Part 35: Twenty of Twenty, and Nothing There"
date: 2026-08-21
description: "Branch 3's answer. The ten-step abstract-action graph looked far more structured than the one-step graph: the effect held on 20 of 20 maps at p < 0.0001, with the ground-truth gate passed first. It is an artifact. The raw effect tracks off-diagonal graph mass at r = +0.987, and zeroing self-loops kills it dead: +0.021, p = 0.45, 11 of 20 maps. A degree-matched null equalises the marginals, but not the estimator's sensitivity — and here sensitivity tracked the swept variable. The campaign closes with a decomposition, a refutation, and an artifact."
---

**TL;DR.** [Part 34](../../../2026/08/21/tdmpc-glass-part34-three-branches-and-a-missing-encoder.html)
left branch 3 with a well-posed question: does HWM's ten-step abstract-action transition graph carry
more structure — higher SI above a degree-matched null — than the one-step primitive-action graph
over the same latents? The answer came back looking like the best result this project has ever
produced: **coarse beats fine on 20 of 20 maps, p < 0.0001,** with the ground-truth gate passed
before we measured anything. One control killed it. The raw effect is a *sensitivity* artifact of
the estimator, correlating r = +0.987 with off-diagonal graph mass; with self-loops zeroed the
coarse graph is not more structured at all (+0.021, p = 0.45, 11/20 maps). We are reporting the
20/20 result and then reporting that it means nothing, because the control is the story.

## The measurement

Setup, per Part 34's reframing: 20 maze2d maps, 5050 frames each, run through the released HWM
checkpoint to get its (single, shared) 18×43×43 latent space — 36,980 dimensions, k-means into 48
discrete states per map. From the same latent sequence we build transition graphs at five horizons:
`skip` ∈ {1, 2, 5, 10, 20} steps per edge, where skip=1 is the primitive-action graph and skip=10
is the model's trained `l2_step_skip`. For each graph we compute SI above a degree-matched null.
Twenty maps give twenty paired replicates.

The gate ran first: a linear probe from the latents recovers agent position with **R² = 1.000 on
every one of the 20 maps.** After the SE-JEPA episode — where the "true by construction" nuisance
axis turned out to occlude the object being measured — no measurement here runs before the axis it
depends on is verified model-free. The latents genuinely encode position. Hold that thought.

## The result that looked like a slam dunk

```
 skip   offdiag        SI-null raw
    1      0.07       0.076 ± 0.014
    2      0.13       0.175 ± 0.026
    5      0.31       0.378 ± 0.071
   10      0.53       0.570 ± 0.107
   20      0.74       0.683 ± 0.133
```

SI above the null rises **nine-fold** from the one-step graph to the twenty-step graph. Paired at
the horizons that matter — coarse (skip=10, the trained abstraction) against fine (skip=1):

> raw: mean difference **+0.494, p < 0.0001, coarse wins on 20 of 20 maps.**

Every replicate. A permutation p-value with no ties to argue about. The preregistered-looking
answer to Part 34's question, in the direction that would have given hierarchical world models a
mechanism: the coarse level's transitions carry more structure, so *that* is what hierarchy buys.

This is the point in previous campaigns where the number went in the abstract.

## The control that killed it

Look at the second column of that table. Off-diagonal mass — the fraction of transitions that leave
their cluster — climbs from 0.07 to 0.74 across the same sweep. Of course it does: at skip=1,
consecutive frames almost always land in the same k-means cluster, so the graph is mostly
self-loops; at skip=20 the agent has moved. That is not structure. That is what a longer horizon
does to *any* discretised trajectory, structured or not.

And SI-above-null is not equally sensitive everywhere. A graph that is 93% self-loops has almost no
off-diagonal mass for the estimator to find community structure *in*; the same estimator on a graph
with 74% off-diagonal mass has ten times the material to work with. The null model is degree-matched
— it equalises the marginals — but matching marginals does not equalise the estimator's **power**,
and here power tracks the very variable we swept. Correlating the raw effect against off-diagonal
mass across the five horizons:

> r = **+0.987.**

The raw rise essentially *is* the mass curve. So the control: zero the self-loops in every graph,
renormalise, recompute SI above the (re-matched) null, and pair coarse against fine again on the
structure that remains:

```
 skip   SI-null nodiag
    1      1.180 ± 0.163
    2      1.250 ± 0.159
    5      1.233 ± 0.208
   10      1.201 ± 0.167
   20      1.057 ± 0.151
```

> nodiag: mean difference **+0.021, p = 0.45, coarse wins on 11 of 20 maps.**

Flat. A coin flip. Once both graphs are scored on their off-diagonal structure alone, the ten-step
abstract-action graph is exactly as structured as the one-step primitive graph. The 20/20,
p < 0.0001 effect was the estimator's sensitivity curve wearing the costume of a finding.

## Two lessons we intend to keep

**A degree-matched null is not enough when the swept variable moves the estimator's sensitivity.**
The null answers "is this graph more structured than a random graph with the same marginals?" — and
it answers it honestly at every horizon. What it cannot answer is "is the *difference between two
horizons* structure, or power?" When the thing you vary (horizon) also varies how much signal the
estimator can see (off-diagonal mass), the paired comparison inherits the confound even though
every individual measurement is clean. The r = +0.987 check costs one line and should run whenever
the compared conditions differ in graph density. Worth being honest about how it was caught: not
because the raw result looked suspicious — it looked perfect — but because self-loop handling has
burned SE measurements in this project before, so the control was on the checklist regardless of
how the headline number looked.

**A clean ground-truth gate does not protect against a sensitivity confound.** The gate did its job:
unlike SE-JEPA's occluded cube, these latents demonstrably encode position, R² = 1.000, all 20
maps. Every frame of the pipeline upstream of the estimator was sound — and the result was still an
artifact, because the failure was not in the representation, it was in how the measure's power
interacts with the design. Gates validate axes. They do not validate comparisons.

## What this means for branch 3, and the campaign

Part 34 set the stakes explicitly: if the coarse level is more structured, that is a mechanism for
why hierarchical planning helps and a target a better architecture could optimise; if the two are
equally structured, the benefit is search-space reduction, which
[Hi-LeWM](https://arxiv.org/abs/2607.12547) has already argued, and we would be adding nothing.

It is the second one. On this model, the hierarchy's coarse level carries **no additional
transition structure** — its benefit, whatever it is, is consistent with plain search-space
reduction over a shared representation. There is no mechanism here for us to claim, and we are not
claiming one.

The three-branch campaign is closed, with all three branches answered:

| branch | answer | status |
|---|---|---|
| 1 — planner economics | value = "having a planner" + "extra search"; only the first varies | decomposition; two-factor account survives 3/4 tasks, **not claimed confirmed** |
| 2 — capacity sets planning value | flat across 128× (rho = +0.064, p = 0.82) | **refuted**, as preregistered |
| 3 — coarse graph is more structured | 20/20, p < 0.0001 raw — killed by the self-loop control | **artifact** |

One decomposition, one refutation, one artifact. No positive claims. That is not the campaign
failing — the point of Yoon's critique was that this project kept producing measurements without
hypotheses sharp enough to die. All three of these were sharp enough to die, and two of them did,
cleanly and cheaply, on a single 3090 Ti.

**Repro:** `python3 repro/b3_analyze.py data/b3_graph.jsonl` in
`SuuTTT/world-model-paper`, branch `results/three-branch-2026-08-20` (data `439e9a8`).
