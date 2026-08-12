---
layout: post
title: "TD-MPC-Glass, Part 31: We Did Reproduce the Collapse — It Is Architectural, and Fixing It Doesn't Help"
date: 2026-08-12
description: "A same-day correction to Part 30. We had claimed we could not reproduce JEPA representation collapse; in fact we had never run the condition that produces it. Removing the stop-gradient collapses the latent immediately — std of normalised embeddings 0.00075 against a healthy 0.177 — which is SimSiam's documented signature. That gives us the positive control Part 30 lacked. Three findings follow: collapse is architectural rather than a regularisation failure; a uniformity loss genuinely rescues it (3/3 collapsed to 0/3); and rescuing collapse does not restore control. Along the way the standard collapse metrics disagree with information content, which is exactly the problem LeCun names as unsolved — and it reframes what structural entropy is for."
---

> **TL;DR — a correction, then a better direction.** [Part 30](../../../2026/08/12/tdmpc-glass-part30-we-could-not-reproduce-the-collapse.html)
> reported that we could not reproduce the representation collapse that JEPA world models are built
> to prevent. **That was measured without a positive control, and the control changes the reading.**
> Every arm in Part 30 kept the EMA target — which *contains* the stop-gradient — so we were asking
> "does collapse happen when the fix is already installed?" Remove the stop-gradient and the latent
> collapses immediately: **std of normalised embeddings 0.00075 against a healthy 0.177**, effective
> rank 1.21, on 3/3 seeds. That is SimSiam's signature exactly. So: **collapse is real, reproducible,
> and architectural.** A **uniformity** loss genuinely rescues it (3/3 collapsed → **0/3**). But the
> rescued representation still plans at **0.156** against a healthy agent's 0.533 — *preventing
> collapse is not sufficient for a useful representation*. And the standard collapse metrics
> **disagree with information content**: our VICReg arm reads "collapsed" by the std criterion while
> decoding the true state at **R² = 1.000**.

---

## What Part 30 got wrong

Part 30's method was to remove the anti-collapse *loss* and see whether the representation died. It
didn't, in any of four cells, so we concluded the premise did not reproduce.

The flaw is simple in hindsight. **The architecture was never ablated.** Every arm used an EMA target
— a slowly-updating copy of the encoder whose gradients are blocked. That stop-gradient is precisely
the mechanism [Chen & He (SimSiam, CVPR 2021)](https://arxiv.org/abs/2011.10566) identify as
preventing collapse. Removing the loss while keeping the architecture asks the wrong question, and
without a positive control "we saw no collapse" is indistinguishable from "our detector is blind."

This also means Part 30 and [Thread D](../../../2026/07/01/tdmpc-glass-thread-d-jepa-anticollapse-done-right.html)
were never in tension with the literature. Thread D found the **predictor + EMA asymmetry** to be
load-bearing; SimSiam says the same thing. We had the right finding and described it as a
contradiction of the field when it is a confirmation.

## The ladder

Three ways to produce the prediction target, holding everything else fixed:

| target mode | what it is | literature says |
|---|---|---|
| `ema` | slowly-updated copy, no gradient | BYOL/JEPA default — should not collapse |
| `stopgrad` | the online encoder, gradient blocked | SimSiam — should not collapse |
| `none` | the online encoder, **gradient flowing** | full symmetry — **should collapse** |

Detector is SimSiam's own: the std of the ℓ₂-normalised embedding. For a healthy 32-dimensional
latent it sits near 1/√d = **0.177**; near zero means every input mapped to the same vector.

### Rung 1 — collapse reproduces, decisively

| target | arm | std of normalised z | collapsed | eff. rank | readout R² | planning |
|---|---|---:|---|---:|---:|---:|
| **none** | none | **0.00074** | **3/3** | 1.21 | 0.827 | 0.000 |
| stopgrad | none | 0.09052 | 0/3 | 2.32 | 1.000 | 0.100 |

The no-stop-gradient arm sits at **0.4% of the healthy reference**. Collapse is not a hypothetical.
**We now have a positive control**, which is what makes Part 30's nulls interpretable at all.

### Rung 2 — a uniformity loss really does rescue it

| target | arm | std of normalised z | collapsed | eff. rank | readout R² | planning |
|---|---|---:|---|---:|---:|---:|
| none | none | 0.00074 | **3/3** | 1.21 | 0.827 | 0.000 |
| none | **uniformity** | **0.17314** | **0/3** | 9.10 | 1.000 | **0.156** |
| none | vicreg | 0.02597 | 3/3 | 3.89 | **1.000** | 0.011 |

**Uniformity does exactly what it advertises.** Given a collapsed architecture it restores the
embedding std to 0.173 — essentially the healthy reference — on every seed. The anti-collapse
literature is not selling a solution to a non-problem; the mechanism works.

**But the rescue does not buy a usable representation.** Planning success is 0.156, against 0.533 for
a plain EMA agent with no anti-collapse term at all. The latent is no longer degenerate and still
cannot be planned through.

## The finding that matters most: the metrics disagree with the information

Look at the VICReg row. By the std criterion it is **collapsed** (0.026, below threshold). Yet a
linear readout recovers the true state at **R² = 1.000** — every bit of the task-relevant
information is right there, linearly available.

A representation cannot be both "collapsed" and "perfectly informative." One of those measurements is
wrong, and it is the collapse metric.

Put that beside the result from Part 30's nuisance axis — adding 16 dimensions of **pure noise**
moved effective rank **2.43 → 22.80**, a 9× rise on zero added information — and a pattern emerges:

> **The quantities this literature optimises are not measuring information content.** Effective rank
> inflates on noise. Embedding std reports collapse on a representation that is perfectly decodable.

## Which is the problem LeCun says is unsolved

This is not our observation. It is his objection to the whole non-contrastive family:

> *"You can never maximize information because you never have appropriate measures of information
> content that is a **lower** bound. For information content, we only have **upper** bounds."*

Push an upper bound up and the true information can sit still. VICReg's variance and covariance terms
are second-order statistics — noise satisfies them for free, which is exactly what our nuisance axis
demonstrates.

**And this is where structural entropy becomes interesting again** — not as an anti-collapse loss,
which is where our supervisor's proposal aimed it and where we
[killed it](../../../2026/08/12/tdmpc-glass-part30-we-could-not-reproduce-the-collapse.html), but as
a *measure*. SI is computed on a **discretised transition graph**, so any structure it reports is
structure the representation demonstrably contains. That has the shape of a **lower** bound — the
thing LeCun says is missing.

Phase 2's null does not bind here. "Does not predict downstream return" and "does not measure
information content" are different claims; a representation can carry plenty of information that
control happens not to need.

## The proposal, restated

Three versions of this idea have now died, so here is the current one, stated plainly:

| version | claim | verdict |
|---|---|---|
| v1 | SE as a **general anti-collapse principle** | **dead** — SI does not predict downstream return (+0.000 step-controlled), and anti-collapse terms are neutral-to-harmful once the architecture is right |
| v2 | anti-collapse is **downstream-dependent** | **dead** — Thread D showed that taxonomy was nav-specific |
| **v3** | **the field cannot measure information content, and graph-structural measures are a candidate lower bound** | **open, and being tested now** |

The contribution v3 aims at is a **benchmark for information-content estimators**, on two axes where
ground truth is known by construction:

- **nuisance** — append dimensions of pure noise. Task information is *unchanged*, so an honest
  estimator stays **flat**. Effective rank already fails this (2.43 → 22.80).
- **signal noise** — corrupt the informative dimensions. Information genuinely *falls*, so an honest
  estimator must **decrease**.

Estimators under test: `Var(Z)`, effective rank, `SI`, and `SI` above its own random-partition
control, against held-out decodability of the true state as ground truth. 36 runs, currently on the
GPUs.

If SI is flat on the nuisance axis and monotone on the noise axis where variance and rank are not,
that is a measurement contribution to a named open problem — and a far better home for structural
entropy than the loss term we spent a campaign on.

## UPDATE, same day: the measurement benchmark ran, on two environments

Proposal v3 said the field cannot measure information content and graph-structural measures are a
candidate. That is now tested on three axes with ground truth known by construction, on **two
environments with opposite dynamics** — TwoRooms (translational, lattice-like transition graph) and
an Oscillator (rotational, cyclic).

**Axis 3 is the strong one:** the state is snapped to a *g*×*g* lattice, so the true information is
exactly **2·log₂(g) bits** — 4, 6, 8, 12 — and an estimator can be scored on whether it tracks
*actual bits* rather than merely moving the right way.

| estimator | r with true bits — TwoRooms | r with true bits — Oscillator |
|---|---:|---:|
| **`Var(Z)`** | **−0.983** | **−0.932** |
| **effective rank** | **−0.944** | **−0.959** |
| SI | +0.900 | +0.859 |
| SI − random-partition | +0.858 | +0.860 |
| InfoNCE | +0.831 | +0.821 |

> **Variance and effective rank are anti-correlated with information content.** Not weak proxies —
> they move the *wrong way*, on both environments, against exact ground truth.

On the nuisance axis (where information is unchanged by construction) effective rank inflates
**10.8×** and **8.8×** respectively, on zero added information. That is LeCun's *"we only have upper
bounds"* objection, measured.

### What this does and does not buy structural entropy

SI tracks bits on both environments (+0.900 / +0.859), beating InfoNCE (+0.831 / +0.821) with no
learned critic, no negatives and no batch-size ceiling. That is real.

But **no SI variant is honest on every criterion, and the two variants fail on opposite ones:**

| | nuisance-invariance | bits-tracking |
|---|---|---|
| raw SI | **fails** (0.53 / 0.70) | **robust** — +0.747…+0.994 across 5 quantisation settings |
| SI − random-partition | 1.15 on TwoRooms but **0.69 on the Oscillator** | **fragile** — collapses to ≈0 at M=50 |
| **InfoNCE** | **0.98 / 0.97** | +0.82 |

The variant that looked like a measure on one environment is fooled on the next, and the variant
that tracks bits robustly is fooled by nuisance. **InfoNCE is the better-behaved estimator overall.**
Anyone claiming SI *is* the missing measure — us included — is over-reaching.

### The honest position

1. **Robust negative:** what this literature optimises is anti-correlated with information content.
   Two environments, three axes, exact ground truth on one. Needs no method to work.
2. **Supporting positive:** a graph-structural quantity tracks information content somewhat better
   than a learned-critic bound, without a critic.
3. **Limit:** SI is not a drop-in measure, and we have the counterexample ourselves.

## Limits

- **n = 3 per cell**, and the ladder's EMA rows were still running when this was written. The
  collapse signal is enormous (0.00074 vs 0.177) but the control numbers are not yet at n=5.
- **The ladder uses a cosine loss**, to make the collapse signature the one SimSiam reports rather
  than a scale artefact of an MSE. That changed the healthy baseline's planning from 1.000 (Part 30,
  MSE) to 0.533, so **planning numbers are not comparable across the two posts** — within the ladder
  they are, across posts they are not. The ladder is a good instrument for the collapse axis and a
  poor one for control.
- Our environment and JEPA remain small and state-based. Nothing here speaks to pixel-scale training.

---

*Part 30 stays up unedited as the record of what it measured; this post is the correction. Code and
per-seed values in `SuuTTT/SE-JEPA`.*
