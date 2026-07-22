---
layout: post
title: "TD-MPC-Glass, Part 21: The Audit — Retracting Our Own Headline, and What an Ablation Actually Ablates"
date: 2026-07-22
description: "A milestone review six days before the AAAI-27 deadline. We audited our TD-MPC2 reimplementation line-by-line against the official release and audited the experimental design against its own claims. Two things broke: the hopper-hop 'world models can hurt' headline was one collapsed seed, and our 'stripped' arm was never model-free at all. One thing held up under the strongest test we could aim at it. This post is the audit, the retraction, and the plan for the last six days."
---

> A milestone review, written between Part 20 and the AAAI-2027 deadline (July 28).
> Part 20 ended with a finished 7-page paper and a confident headline. This post is
> what happened when we pointed the same skepticism we use on nulls at our own
> positives — a full-fidelity audit of the TD-MPC2 reimplementation against the
> official release, and a design audit of the ablation itself. The headline did not
> survive. The core finding did, and is stronger for it.

## The one-paragraph version

We asked two questions we had been deferring: **is our TD-MPC2 actually TD-MPC2**,
and **does our ablation ablate what we say it ablates?** The answers were *mostly,
with three disclosable deviations* and *no*. The "no" is the important one: setting
`consistency_coef=0` does not remove the world model — the latent dynamics is still
trained by the reward and value heads, and the planner still plans through it. Our
"stripped" agent is a **purely value-equivalent model**, not a model-free agent.
Separately, the hopper-hop result we led with — *stripping the model improves return
by +150* — turned out to be **one reproducibly-collapsing seed** in a median over
n=3. At n=5 it evaporates, and the official TD-MPC2 numbers confirm it. We have
retracted it from the paper, in the paper, with the arithmetic shown.

## Part 1 — Is our TD-MPC2 actually TD-MPC2?

Every within-implementation ablation rests on an unstated premise: that the "full"
arm is a faithful instance of the method it claims to be. If our full agent were
already a degraded TD-MPC2 along the very axis we ablate, then "removing the model
barely hurts" would be an artifact, not a finding. So we checked the axis first.

### The loss coefficients look catastrophically wrong, and aren't

Official TD-MPC2 uses `consistency_coef=20, reward_coef=0.1, value_coef=0.1`.
Ours uses `2.0 / 2.0 / 1.0`. Read naively, our consistency term is weighted 100×
lower relative to reward — which would mean our full arm was half-stripped before
we touched it, and the whole paper collapses.

It isn't, because the *reductions* differ:

- official consistency is `F.mse_loss(z, next_z)` — a **mean** over the 512-dim latent
- ours is `mean_batch(sum over 512 dims)` — a **sum**
- official value loss divides by `num_q`; ours sums over the ensemble

Unit-matching all three and normalizing to the value term:

| | consistency : reward : value |
|---|---|
| official | **1.96** : 5.0 : 1 |
| ours | **2.00** : 2.0 : 1 |

**The quantity we set to zero is weighted within 2% of official.** (Adam is
scale-invariant per-parameter, so only these ratios matter — the absolute loss
magnitude is irrelevant.) The full arm is not a straw man. This was the single most
important thing to establish and it is the reason the rest of the paper survives.

### Three deviations that do matter, and are now in the paper

**Our planner is CEM, not MPPI.** The elite update takes an unweighted mean and
standard deviation over the top-64 of 512 sampled trajectories. Official TD-MPC2
uses a softmax-weighted update at temperature 0.5 — that is the "MPPI" in the name.
Both our arms are planned identically, so the full-vs-strip contrast is unaffected,
but every *magnitude* in the planning channel is implementation-specific. Disclosed.

**The policy's `log_std` head is never trained.** Our pi loss is
`-E[min_i Q_i(z, tanh(mu(z)))]` — the mean action, no entropy term, no
reparameterized sample. Official is `-(entropy_coef * scaled_entropy + qs).mean()`,
which puts gradient on the std. Consequence: our policy's exploration std sits at
initialization forever, exploration comes from an additive noise schedule
(σ=0.3 for the first 25k steps), and MPPI seeds 24 of its 512 trajectories by
sampling with that untrained std. This is closest to a bug, and it lands squarely
on the exploration channel. Disclosed, because fixing it means re-running
everything and the deadline is six days out.

**Smaller, cancelling in-contrast:** 2 Q-heads not 5, value support ±20 not ±10, a
`max(·, 0)` clamp on the TD target, and half the update-to-data ratio (0.5 vs 1.0).
That last one is almost certainly the source of the parity gaps we already
reported — cheetah −5%, walker −17%, acrobot −23% — since it means half the
gradient work at equal step count.

## Part 2 — The retraction

Part 20's headline: on hopper-hop, stripping the world model *improves* return by
+150 (strip 436 vs full 286). It was the "actively harmful" end of the spectrum and
it was in the title-adjacent sentence of the abstract.

Here is the per-seed data, which we should have looked at first:

```
full   n=3:  {286, 425, 6}        <- median 286, mean 239
strip  n=3:  {437, 225, 436}      <- median 436
```

The 6 is a collapsed run. A median over three numbers where one is a collapse is
not a robust statistic; it is a coin flip that happened to land on "world models
hurt". We re-ran that seed from scratch: **it collapsed again (0.0)** — so it is a
reproducible seed-specific training instability, not noise. Two fresh seeds trained
normally (262 and 265 at 1.0M steps). Strip has never collapsed.

Then the external check. The official TD-MPC2 release publishes per-seed CSVs:

```
official hopper-hop @1.5M:  {335, 364, 331}   mean 343, ZERO collapses
```

Our non-collapsing full runs match that. Our strip runs also match it. There is no
effect. What we actually measured was **a 1-in-5 training-instability rate in the
full agent** — a real observation, worth a sentence in a stability discussion, and
nowhere near an abstract.

The official baselines make the same point independently: hopper-hop is
collapse-prone *across methods* — TD-MPC v1 collapses on 2/3 seeds, SAC on 1/3.
This is exactly Voelcker et al.'s ["Can we hop in general?"](https://arxiv.org/abs/2404.09991)
finding, that hopper rankings do not transfer across variants. We had the citation
in our bibliography and did not apply it to ourselves.

**The general hazard, stated so we do not repeat it:** with n=3 and a median
statistic, one collapsed seed is enough to invert the sign of an ablation. We have
adopted a **collapse screen** as a stated inclusion criterion for any "removing the
model helps" claim — a run is excluded if its return or its latent effective rank
degenerates — and we apply it uniformly. It is what retracts hopper-hop, and it is
what excludes the JEPA arm below.

## Part 3 — What "stripping the world model" actually does

This one is a design error, not a data error, and it is more interesting than the
retraction.

In TD-MPC2 the latent dynamics `d(z,a)` receives gradient from three places:

1. the **self-predictive** (latent consistency) term, pulling `d(z_t,a_t)` toward a
   target encoding of the next observation
2. the **reward head**, evaluated on latents rolled out *through* `d`
3. the **value head**, likewise

Setting `consistency_coef=0` removes (1) and leaves (2) and (3) fully intact. The
dynamics model is still trained. The planner still plans through it. **Nothing in
our experiment is model-free.**

So the honest description of the strip arm is: a **purely value-equivalent** latent
model, in exactly Grimm et al.'s sense. And the honest question the paper answers is
not "does a world model help?" but:

> **Does the self-predictive, observation-grounded signal earn its keep on top of
> value equivalence?**

That is a narrower claim and a better one. It is also *explanatory* in a way the old
framing wasn't. Look at the results again with it in mind:

| task | full π | strip π | reading |
|---|---|---|---|
| pendulum-swingup | 819 | 189 | self-prediction essential |
| cheetah-run | 746 | 544 | helps |
| walker-run | 700 | 647 | helps a little |
| finger-spin | 983 | 958 | redundant |
| hopper-hop | 286 | 436 | **null** (retracted) |
| acrobot-swingup (Π) | 387 | **144** | planner needs it |

Strip stays competitive wherever the reward and value gradients alone are enough to
shape a usable latent — which is most dense-reward locomotion. It falls apart where
they aren't: pendulum, and acrobot's *planner*, where planning three steps through a
latent that was never grounded in observations produces garbage. The old framing had
to call these unrelated; the new framing predicts them.

And with hopper's −150 gone, the spectrum is **monotone non-negative**: the value of
self-prediction decays smoothly to zero as value becomes compressible, and never
crosses below it. Less punchy. Much more defensible.

## Part 4 — Two more things the design audit found

**The four cells attribute, they do not identify.** Both arms collect their own data
with their own planner. So `J_full(π) − J_strip(π)` mixes a representation effect
with the different state distribution each arm visited. The channel labels come from
the *probes*, not from the arithmetic. The paper said the cells "isolate all three
channels at once"; it now says attribution, with the buffer-swap control (train strip
on full's replay) named as the experiment that would actually identify.

**The exploration channel is our weakest leg.** Ball-in-cup is 1/3 vs 0/3 successes —
Fisher exact p = 1.0. Cartpole-sparse is zero in all four cells, which is a
non-result, not evidence. This matters because the retraction promotes ball-in-cup
toward the headline. It is now running to **n=8**.

**JEPA is out.** The joint-embedding arm produced a striking number — stripped 1.00
success vs full 0.53 on PointMaze, a third-family "stripping helps". All four runs
report `latent_collapsed: true`, effective rank ~1e-5. The diagnostic went further:
the VICReg anti-collapse term *is* applied and *is* saturated at its maximum-collapse
value from the first evaluation onward, because simnorm (V=4) goes one-hot
immediately and kills the gradient path back to the encoder. So the number measures a
saturated softmax, not a world model. Under our own collapse screen it is excluded.
Trading one artifact for another six days before a deadline is exactly the failure
mode this post is about.

## Where the GPUs are pointed now

| box | job | why |
|---|---|---|
| 3060 · box1 | acrobot n=3 | done: full 419/414, strip 127/208 — the planning channel's anchor |
| 5060 | hopper n=5 | the retraction's evidence; s3/s4 finishing |
| 3060 · mwm | **DreamerV3 hopper**, 4 cells | cross-family check *on the retraction* |
| 4070 | **ball-in-cup s2–7** | the exploration channel, from n=3 to n=8 |

The Dreamer hopper cell is the one to watch. We are claiming hopper is a null in
TD-MPC2; an unmodified official DreamerV3 van-vs-strip cell on the same task either
corroborates that or contradicts it. Either outcome is publishable and we have
committed to the answer in advance.

## The plan for the last six days

1. **Land ball-in-cup n=8.** If the exploration cliff holds at 8 seeds it becomes a
   real headline; if it doesn't, we say so and the paper leads on representation
   alone.
2. **Land Dreamer hopper.** Cross-family test of the retraction.
3. **Buffer-swap control** if there is time — it is the difference between
   attribution and identification, and reviewers will ask.
4. **Do not fix the planner or the policy std.** Both are real deviations; both are
   now disclosed quantitatively; fixing either means re-running the whole grid.
   Disclosure beats a rushed re-run that lands the night before.

## What this week was actually about

Part 20 said "every number is read from disk and the nulls are reported as loudly as
the positives." That was true and it was not enough, because we were applying the
skepticism asymmetrically — hard on results we didn't like, soft on the one that made
the best headline. The +150 sat in the abstract for a week. It took someone asking
*"is that claim correct? check the official data"* to look at the three per-seed
numbers behind a median.

The audit cost about a day and cost us our headline. It also caught the JEPA artifact
before it went in, produced a stated inclusion criterion, and turned a vague claim
about world models into a sharp one about self-prediction versus value equivalence.
That is a good trade six days out. It would have been a catastrophic one to skip.

---

*Numbers in this post are read from the run CSVs on the boxes and from the official
TD-MPC2 release's published per-seed results. The retraction, the fidelity
deviations, and the JEPA exclusion are all written into the paper itself rather than
silently dropped.*
