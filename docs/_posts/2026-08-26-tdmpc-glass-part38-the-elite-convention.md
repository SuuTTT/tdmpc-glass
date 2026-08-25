---
layout: post
title: "TD-MPC-Glass, Part 38: The Question 'Should I Sample More?' Has Two Answers"
date: 2026-08-26
description: "Sampling-based planners keep the best few candidate plans — the elites — and refit their search distribution to those. Some implementations fix the number of elites; others fix the fraction. Nobody reports which. On identical models, identical seeds and identical episodes, sampling 40x more candidates changes return by -63 under one convention and +246 under the other, both at p=0.002. The reversal replicates on a second task, has a textbook order-statistics explanation, and reproduces in twenty lines of simulation with no world model at all. Published sample-count comparisons are not commensurable across implementations unless the convention is stated."
---

**TL;DR.** If you ask "does my planner get better if I sample more candidate action
sequences?", the honest answer is: *it depends on a configuration detail your paper probably
doesn't mention.* Planners that keep a fixed **number** of elite candidates get better with
more samples. Planners that keep a fixed **fraction** get worse. Same models, same seeds,
same episodes, opposite signs, both significant. The mechanism is classical order statistics,
the effect reproduces without any learning involved, and it means a lot of planner-compute
comparisons in the literature are not directly comparable.

## What an "elite" is

Sampling-based planners — CEM, MPPI, and the model-predictive control layer inside TD-MPC2,
PETS, PlaNet and friends — work like this at every timestep:

1. Keep a Gaussian over action sequences (a mean and a spread).
2. Sample `N` candidate sequences from it.
3. Score each one by rolling it through the learned world model.
4. Keep the best `m` candidates — **the elites**.
5. Refit the Gaussian to those `m`. Repeat a few times, then execute the first action.

The elites are the survivors that decide where the planner looks next. Everything about the
planner's behaviour flows through them.

There are two standard ways to say how many elites to keep, and both are in wide use:

| convention | how `m` is set | who uses it |
|---|---|---|
| fixed **count** | `num_elites: 64`, regardless of `N` | TD-MPC2 |
| fixed **ratio** | `elite_ratio: 0.1`, so `m = 0.1N` | mbrl-lib (PETS, PlaNet) |

Papers report the population size. They rarely report which convention produced it.

## The measurement

We swept population size on trained agents, holding everything else fixed — same
checkpoints, same evaluation seeds, same episodes, and (after an earlier mistake of ours)
the same planner random seed re-applied before every arm, so arms differ only in the
variable under study.

On a PlaNet agent (pixel observations, recurrent latent model), going from 100 to 4000
candidates at one refinement iteration:

| elite convention | 100 → 4000 candidates | paired exact test |
|---|---|---|
| fixed **ratio** (library default) | 76.9 → **13.8** | −63.1, *p* = 0.002 |
| fixed **count** (10 elites) | 73.8 → **319.5** | **+245.7**, *p* = 0.002 |

Forty times more sampling is *significantly harmful* under one convention and
*significantly helpful* under the other. It replicates on a second task (walker-walk:
−17.6 n.s. versus **+78.4**, *p* = 0.002) and the same direction shows up on a PETS agent
in a different codebase family.

This is not a small effect at the edge of noise. It is the difference between concluding
"sampling more is a waste" and "sampling more is one of the best things you can do."

## Why: order statistics, not deep learning

The explanation has nothing to do with world models, reinforcement learning, or anything
that has happened in the last decade. It is about what happens to the *spread* of the top
`m` of `N` samples.

- **Fixed ratio.** The elites are every candidate above the (1−ρ) quantile. As `N` grows,
  that set's empirical spread converges *upward* to the true conditional spread of the whole
  tail region — a fixed positive number. More sampling never sharpens the refit distribution;
  it just measures a broad region more accurately.
- **Fixed count.** The top `m` of `N` are the `m` most extreme order statistics. As `N`
  grows they concentrate toward the maximum, so their spread **shrinks**.

So under a fixed ratio, sampling more leaves the planner's next distribution broad — it
becomes *less decisive the harder it looks*. Under a fixed count, sampling more sharpens it.

You can watch this happen with no model and no learning. Sample points from a Gaussian,
score them noisily, keep elites, measure the spread:

```
      N   ratio: m   ratio std   count: m   count std
     50          5      0.7118         10      0.8107
    350         35      0.8252         10      0.7380
   1400        140      0.8383         10      0.6994
   5600        560      0.8416         10      0.6644
```

Rising under a ratio; falling under a count; crossing over in between. Now the same quantity
measured *inside a trained PETS planner*, instrumented at every refit:

```
 convention     N=50    N=350   N=1400
 fixed ratio   0.127    0.203    0.219     (widens)
 fixed count   0.167    0.157    0.149     (tightens)
```

Same shapes. Same crossover. The planner is doing exactly what the textbook says a
selection rule does, and the consequence lands in the returns.

This also connects to classical CEM analysis, where convergence arguments require the elite
quantile to shrink over iterations, or to be explicitly smoothed. The fixed-ratio convention
never shrinks it.

## Why this matters beyond one library

Three consequences, in increasing order of how much they should bother you.

**1. Your sweep may have told you the opposite of the truth.** If you have ever run "how
many samples should I use?" with `elite_ratio` held fixed and concluded that more samples
don't help, that conclusion is about your elite bookkeeping, not about your planner.

**2. Cross-paper comparisons are not commensurable.** TD-MPC2 fixes the count; mbrl-lib
fixes the ratio. A sample-count comparison drawn from one is not transferable to the other,
and neither paper's tables carry the flag you would need to notice.

**3. It is invisible in the usual reporting.** Population size is a headline hyperparameter.
The elite convention is buried in a config file, if it appears at all. We only found this
because we were sweeping width on purpose and got a sign that made no sense.

The fix is trivial and free: **report the elite convention alongside the population size**,
and when you sweep population, hold the elite *count* fixed unless you specifically intend
to study the quantile.

## What we got wrong on the way

Three of our own errors produced this result, and all three are worth naming.

**We ran the confounded sweep first.** Our original TD-MPC2 sample sweep scaled elites with
population (`num_elites = max(4, N/8)`), which is *not* TD-MPC2's own convention. It produced
a clean-looking "more samples actively hurt" result that we nearly wrote up. A preregistered
control at fixed elite count made the decline vanish.

**We then repeated the mistake on the new agents.** When we moved to PETS and PlaNet, we
took the library default and swept width again — reintroducing the same confound in a paper
whose contribution list already included catching it. Caught on re-reading our own arms
against our own preregistration.

**Our mechanistic prediction was backwards.** We predicted the fixed-ratio convention would
cause the proposal to *collapse* (too little exploration). The instrumentation showed the
opposite: it stays broad, and the fixed-count convention is the one that concentrates. We
had the direction of the effect right and the reason wrong, which is a good argument for
measuring mechanisms instead of asserting them.

## The part we are not claiming

The order-statistics facts here are textbook. Anyone who sat down with a pen could derive
P4. We are not claiming to have discovered a new statistical phenomenon.

What appears to be undocumented is the *consequence*: that two of the most widely used
implementations sit on opposite sides of this choice, that the choice reverses the empirical
answer to a question practitioners actually ask, and that nothing in the usual reporting
would let a reader detect it. A known fact has been quietly determining published results.

*Data, harnesses and the preregistration documents (with dated amendments, including the
failed predictions above) are in `SuuTTT/world-model-paper` on the
`results/three-branch-2026-08-20` branch.*
