---
layout: post
title: "TD-MPC-Glass, Part 34: Three Branches, and a Missing Encoder"
date: 2026-08-21
description: "All three directions are running. Branch 1 splits the planner's value into two quantities and finds that only one of them varies: extra search depth buys little everywhere and never grows, while the value of having a planner at all swings from -2 to +159 on one task. Branch 2 is launched with a preregistered prediction over a 128x sweep of representational capacity. Branch 3 hit its gate immediately, and the answer overturns our own plan: the released hierarchical world model has no second encoder. Its level-2 backbone is the identity, and the authors' own output directory is named l2_wo_encoder."
---

**TL;DR.** Branch 1: the planner's advantage decomposes into *search depth* and *having a planner
at all*, and only the second one varies — which is why our Part 33 prediction failed. Branch 2 is
running on a 128x capacity sweep with the prediction written down first. Branch 3's gate came back
in an hour and the answer is that **the hierarchical world model we planned to measure per-scale
has only one scale of representation.** Its level-2 encoder is the identity function. That kills
the measurement we proposed and points at the one we should have proposed.

## Branch 1: the planner's value is two quantities, not one

[Part 33](../../../2026/08/20/tdmpc-glass-part33-from-findings-to-claims.html) claimed that within
a task, the planner's value decays with training budget. Our own eval curves refuted it the same
day: hopper's advantage *grows eighty-fold*. The correction proposed a two-factor account. Here is
the test of it, and it costs nothing — the numbers were already on disk.

The planner advantage we had been quoting is `full − none`: full MPPI (512 samples, 6 iterations)
against the policy prior with no search. That single number silently adds together two different
things. Splitting them:

| what varies | 50k | 100k | 200k | 300k | 400k |
|---|---|---|---|---|---|
| **hopper — extra search** (full − cheap) | 4 | 7 | 50 | 24 | 16 |
| **hopper — having a planner** (cheap − none) | −2 | 11 | 49 | 125 | **159** |
| **walker — extra search** | 89 | 58 | 45 | 39 | 39 |
| **walker — having a planner** | 139 | 102 | 70 | 33 | **33** |
| **acrobot — extra search** | 27 | 101 | 33 | 30 | 45 |
| **acrobot — having a planner** | 163 | 101 | 176 | 169 | 125 |
| **cheetah — extra search** | 10 | −51 | −100 | −6 | 7 |
| **cheetah — having a planner** | 40 | 55 | 80 | 50 | 59 |

"Cheap" is 32 samples and 1 iteration — about 3% of full MPPI's compute.

Two things read straight off this table.

**Extra search depth buys little on every task, and it never grows.** The largest column in any
"extra search" row is walker's 89 at 50k, and it decays to 39. On cheetah it is negative for three
consecutive checkpoints — the extra search actively hurts. This is the earlier 3%-planner result
seen over training rather than at one point, and it is the more robust of the two halves.

**All the interesting variation lives in whether there is a planner at all.** Hopper's growth
(−2 → 159) and walker's decay (139 → 33) are both entirely in this row. So the Part 33 prediction
was not merely wrong about the direction; it was attached to the wrong quantity.

The two-factor account survives this test on three tasks: hopper's policy is stuck near the floor
(14 → 152 out of ~1000), so as the model improves the planner has more to exploit and its value
grows; walker's policy climbs (523 → 735), absorbs what search was providing, and the value falls.
Cheetah breaks the ordering — but its whole range is 40 to 80, close enough to noise that we are
not going to lean on it. **We are not calling the two-factor account confirmed.** With n=4 tasks
and one of them uninformative, it is a hypothesis that has survived one test, and that is all.

## Branch 2: the prediction, written down first

Yoon's objection to the measurement work was that it had no hypothesis — that "a good metric should
ignore irrelevant clutter" is a sanity property, not a claim. The version with a claim in it is:

> **Representational capacity is what search consumes.** If a planner's value comes from rolling
> out a learned model, then how much the representation can carry should set how much search can
> buy — and planning value should rise with capacity and saturate where the information saturates.

We have the experiment for this already trained: the h3c sweep is hopper-hop with `latent_dim` in
{4, 8, 16, 64, 512} across 3 seeds — a **128x range in representational capacity with everything
else held fixed.** For each of the 15 checkpoints we toggle the planner at deployment time, so the
weights are identical across the two arms and only search changes. That isolates what the
representation supports from anything search did during training.

Preregistered, before the sweep finishes:

- **If the hypothesis holds:** planning value is monotone increasing in `latent_dim`, with most of
  the rise between 4 and 64 and little between 64 and 512.
- **If planning value is flat across 128x:** capacity is not the binding constraint, the hypothesis
  is dead, and we report it dead. This is the outcome our track record makes likely — every
  representation measure we have tested has come back uncorrelated with planning.

A single-episode smoke test at `latent_dim=4` gives prior 164, MPC 340, planning value 175. That is
one episode and means nothing yet; it is here to show the harness loads a 4-dimensional latent
without silently falling back to the default.

## Branch 3: the gate answered, and it moves the goalposts

The proposal promised a deliberately small first step on HWM: pull the repo, confirm the latents
are extractable per scale, report back before committing. That step is done, and the answer is no —
for a reason that matters more than the answer.

The released checkpoint's state dict contains 28 tensors under `level1` and 30 under `level2`.
Every one of level 2's is a *predictor*. There is not a single `level2.backbone` parameter. Their
config says why in one line:

```yaml
level2:
  backbone:
    arch: identity_encoder     # level 2 has no encoder
  predictor:
    z_dim: 8                   # a learned 8-dimensional abstract action
step_skip: 10                  # level 2 predicts ten environment steps at a time
```

and the output directory the authors chose for this configuration is `l2_wo_encoder` — level two,
without encoder. Their naming, not ours.

Corroborating shapes: level 1's backbone ends in a 1x1 convolution to 16 channels, joined by a
2-channel proprioceptive component, giving an 18-channel spatial latent. Level 1's predictor maps
20 → 18 (latent plus a 2-channel action). Level 2's predictor maps 26 → 18 (the same 18-channel
latent plus an 8-channel encoded abstract action) and its final layer-norm has shape (18, 43, 43).
**Both levels live in the identical latent space, at identical spatial resolution.**

So the hierarchy in this hierarchical world model is not a hierarchy of representations. It is a
hierarchy of *time and action*: the same state space, predicted in jumps of ten steps, driven by a
learned 8-dimensional macro-action.

This kills the measurement we proposed. "Measure the information content at each scale" has nothing
to measure at scale two — level 2's representation is level 1's representation, so any per-scale
information measure returns the same number twice by construction. Had we not run the gate, we
would have spent a week producing that tautology and reporting it as a finding.

It also says what to measure instead. What actually differs between the levels is the *transition
structure*: one-step versus ten-step, primitive action versus learned macro-action. Transition
structure is what structural information is a measure of. So the question becomes well-posed and
narrow:

> Does the ten-step abstract-action transition graph carry more structure — higher SI above a
> degree-matched null — than the one-step primitive-action graph over the same latents?

If the coarse level is more structured, that is a mechanism for why hierarchical planning helps and
a target a better architecture could optimise. If the two are equally structured, the benefit is
search-space reduction, which [Hi-LeWM](https://arxiv.org/abs/2607.12547) has already argued, and
we would be adding nothing.

We are being explicit that this is a *reframing after a negative gate*, not a plan we had all
along. The dataset download is running so we can run their model forward and build both graphs.

## Status

| branch | state | cost |
|---|---|---|
| 1 — planner economics | decomposition done, from data already on disk | $0 |
| 2 — capacity vs planning value | 15 checkpoints sweeping, ~2h | ~$0.40 |
| 3 — hierarchical WM | gate answered, dataset downloading | ~$0.20 |

One 3090 Ti at $0.205/h is carrying all three. No new box was rented.
