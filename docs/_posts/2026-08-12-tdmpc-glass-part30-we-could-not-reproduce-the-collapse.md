---
layout: post
title: "TD-MPC-Glass, Part 30: We Could Not Reproduce the Collapse That JEPA World Models Are Built to Prevent"
date: 2026-08-12
description: "Written to be readable without knowing what JEPA or structural entropy are — there is a plain-terms primer up front. A proposal landed to regularize JEPA world models with the structural information of the latent transition graph. Before building it we asked two prior questions: does the quantity predict anything, and does the problem it fixes exist? Neither survived. Structural information does not predict downstream return where a training-step confound is removed (+0.000 on cheetah). And on the proposal's own first benchmark, a pure JEPA with zero anti-collapse is the best arm in every cell — while anti-collapse terms raise effective rank and destroy control, monotonically in their own strength. Includes three corrections I had to make to my own analysis inside twelve hours."
---

> ### ⚠️ CORRECTED THE SAME DAY — read [Part 31](../../../2026/08/12/tdmpc-glass-part31-we-did-reproduce-the-collapse.html) with this
>
> This post's headline is **wrong**. Every arm below kept the EMA target, which *contains* the
> stop-gradient — so it asked "does collapse happen when the fix is already installed?", with no
> positive control. Remove the stop-gradient and the latent collapses immediately (std 0.00075 vs a
> healthy 0.177, 3/3 seeds). **Collapse is real, reproducible, and architectural.** A uniformity loss
> genuinely rescues it. What survives from this post is narrower and still holds: *once the
> architecture is right*, anti-collapse losses are neutral-to-harmful, and the metrics they optimise
> do not measure information content. Part 30 is left unedited below as the record of what it
> measured.

> **TL;DR.** We were asked to evaluate a proposal: regularize a JEPA world model with the
> **structural information** of its latent transition graph, `L = L_pred − λ·SI(G_Z)`, as a general
> anti-collapse principle. Rather than build it, we asked the two questions that come first.
> **(1) Does `SI` predict anything?** On 96 checkpoints with logged returns — no. Once a
> training-step confound is removed it correlates **+0.000** with return on cheetah and loses to
> plain effective rank on hopper (+0.379 vs +0.657). **(2) Does the collapse it fixes happen?** On
> the proposal's own first benchmark, across offline and closed-loop data and 0 or 16 nuisance
> dimensions, **a pure JEPA with zero anti-collapse is the best or tied-best arm in all four
> cells.** A λ sweep over two orders of magnitude finds no strength at which any anti-collapse term
> rescues control. The clean finding is an **inversion**: uniformity raises effective rank 2.43 →
> 11.44 while planning success falls 1.000 → 0.011. It maximises the health metric and destroys the
> task. Total cost: about **$1.60** and one afternoon.

---

## First, the idea in plain terms

*Skip this section if you already know what JEPA and structural entropy are.*

**A world model is an agent's imagination.** Before acting, it asks "if I do this, what happens
next?" If it can answer accurately, it can try things out in its head instead of in the world.

**Predicting pixels is wasteful**, though. To imagine a robot arm moving, you do not need to render
every shadow and reflection. So a **JEPA** — a joint-embedding predictive architecture — predicts a
*summary* instead of a picture. Two pieces:

- an **encoder** that turns an observation into a short summary (a vector of numbers, the "latent");
- a **predictor** that, given the current summary and an action, guesses the *next* summary.

Train the predictor to match the encoder's summary of what actually happened next, and in principle
you get a compact imagination that ignores irrelevant detail.

**Now the catch, and it is the reason this whole literature exists.** The encoder is being trained
too, so it can cheat. If it summarises *every* observation as the same number — say, 7 — then the
predictor's job becomes trivial: always answer 7. Prediction error goes to zero. The model is
perfect and completely useless. It has stopped describing the world at all.

That degenerate cheat is called **representation collapse**, and it is exactly what a student does
when they discover the exam is graded on self-consistency rather than correctness: answer "42" to
everything and score full marks.

**Anti-collapse terms are rules that forbid the constant answer.** Extra penalties added to
training, each phrased slightly differently:

| the rule, in words | its technical name |
|---|---|
| "your summaries must stay spread apart from each other" | uniformity |
| "your summaries must vary, and their dimensions must not duplicate each other" | VICReg, Barlow Twins, R2-Dreamer |
| "your summaries must be shaped like a bell curve" | Gaussian regularization, LeWorldModel's SIGReg |
| "I must be able to tell which action you took from how the summary changed" | Delta-JEPA |

**Effective rank** is the field's usual health check — roughly, *how many genuinely different
directions do the summaries use?* One direction means everything got squashed onto a line: collapsed.
Many directions means spread out: healthy. Remember that word "roughly"; it does a lot of work later.

**Structural entropy** is the newcomer, borrowed from network science. Draw a graph. Ask: does it
split into meaningful communities, or is it formless? **Structural information** `SI` measures how
much of the graph's messiness is explained by organising its nodes into groups. Formless graph, low
`SI`; clearly clustered graph, high `SI`.

The proposal's move is to build that graph out of the agent's own experience — **nodes are latent
states, edges are transitions the agent actually observed** — and then reward the model for that
graph having clear structure. The pitch is that this is more general than the rules above: instead of
constraining what individual summaries look like, it constrains **how summaries are related to each
other through the dynamics**. Relations, not marginals.

That is a genuinely good instinct, and it is why this was worth a couple of days.

---

## The proposal

Joint-embedding predictive world models are said to have a central difficulty: **representation
collapse**. Minimising latent prediction error alone admits a trivial optimum — map everything to a
constant and prediction becomes perfect and useless. A visible literature has grown around
preventing it: [LeWorldModel](https://arxiv.org/abs/2603.19312) constrains the latent marginal to be
Gaussian via SIGReg; [R2-Dreamer](https://iclr.cc/virtual/2026/poster/10010184) reduces redundancy
Barlow-Twins-style; [Delta-JEPA](https://arxiv.org/pdf/2606.31232) reconstructs the action from the
latent difference.

The proposal we were given argues these are each *one specific proxy* for dynamics relevance, and
that a more general principle exists: preserve the **structural information of observed
transitions**. Regularize the relation `p(z_{t+1} | z_t, a_t)` rather than the marginal `p(z)`.

That instinct — **relations, not marginals** — is a good one, and it is not covered by our previous
nulls, which all regularized marginals. So the question was worth work. We just did the work in a
different order than proposed.

## The order that matters

Our own history dictated it. We have spent a campaign on structural entropy as a *loss* and never
once asked whether it is a good *instrument*. Put the two prior questions the way a doctor would:
**before prescribing the medicine, check that it does something — and check the patient is sick.**

1. **Does `SI` predict downstream usefulness?** (the proposal's H3) — if not, the regularizer has no
   basis.
2. **Does collapse actually occur** on these benchmarks? — if not, there is nothing to fix.
3. Only then: build the regularizer.

Both prior questions are cheap. Both failed.

---

## Phase 2 — structural information does not predict downstream return

No training. For each of **96 existing checkpoints** (hopper/walker/cheetah, planner on and off,
steps 50k–400k), each carrying a **logged eval return**: roll out, quantize latents to 100 prototype
nodes, build the **transition graph** — edges fixed by observed dynamics, not by latent proximity,
which is the proposal's object and one our earlier kNN-graph work never tested — and compute
`SI = H₁ − H₂` alongside effective rank and `Var(Z)`.

Controls were in from the first run: a **random-partition** control (same module sizes, shuffled
membership) and a **shuffled-transition** control (keep the point cloud, destroy the temporal
pairing).

### The confound that decides it

Checkpoints from one run at different steps are not independent, and return rises with training
step. Any metric that also rises with step inherits a spurious correlation.

| task | ρ(return, step) | ρ(SI, step) | ρ(rank, step) |
|---|---:|---:|---:|
| cheetah | **+0.733** | +0.304 | +0.140 |
| hopper | +0.316 | +0.021 | +0.049 |
| walker | +0.568 | −0.314 | −0.562 |

Correlating **across runs within each step bucket** removes it entirely. That is the valid statistic:

| task | `SI` − random-partition | effective rank | testable? |
|---|---:|---:|---|
| cheetah | **+0.000** | −0.100 | 8 buckets |
| hopper | **+0.379** | **+0.657** | 8 buckets |
| walker | — | — | **no** — only 2 runs |

**H3 is not supported.** Where the confound can be removed, `SI` never beats effective rank: it
predicts nothing at all on cheetah and is clearly weaker than rank on hopper.

*In plain terms:* if structural information really measured "how useful this world model is", then
models with more of it should score better. Across 96 trained models whose scores we already knew,
it didn't. On one task the relationship was **exactly zero** — knowing a model's `SI` tells you
nothing at all about how well it performs. And the apparent relationship we first saw came from
something much more boring: **both numbers drift upward as training proceeds**, so they look linked
until you compare models *at the same point in training*.

**Two readings of my own that did not survive, recorded so they are not repeated.** The raw
within-task correlations (`SI` +0.255 / +0.474 / −0.618) are step-confounded and should not be
quoted. And the apparent **significant sign flip** on walker (−0.618, p=0.011) rests on **two runs**
and cannot be step-controlled at all. I reported that flip before checking it; the check took four
minutes.

---

## Phase 1 — the collapse did not appear

If `SI` predicts nothing, the regularizer is unmotivated. But there is a prior question still:
**does the problem exist?**

[Thread D](../2026/07/01/tdmpc-glass-thread-d-jepa-anticollapse-done-right.html) had already found
that a pure self-predictive JEPA does **not** collapse on DMControl (state or pixels) or on
narrow/on-policy data, and that the load-bearing ingredient is the **predictor + EMA-target (BYOL)
asymmetry**, not any explicit repulsion. So we tested the proposal's own first benchmark — two-room
navigation — in the regime with the highest prior of showing collapse.

A pure JEPA (encoder, action-conditioned predictor, EMA target). Control measured **the JEPA way**:
CEM planning through the predictor toward a goal latent, **no policy anywhere**. Factors: data
regime `{offline-random, online closed-loop}` × nuisance dimensions `{0, 16}` × arm
`{none, uniformity, vicreg}` × 3 seeds.

Nuisance dimensions are the proposal's own motivating case: a model can score high latent variance
by representing irrelevant variation while losing what matters. Closed-loop online data is the only
regime this lab has ever seen collapse in.

| cell | `none` | best anti-collapse |
|---|---:|---:|
| nuisance 0, offline | **1.000** | vicreg 0.878 |
| nuisance 0, online | **1.000** | vicreg 1.000 |
| nuisance 16, offline | **0.622** | uniformity 0.178 |
| nuisance 16, online | **1.000** | uniformity 0.811 |

**Zero anti-collapse is best or tied-best in every cell.** All eight arm-vs-`none` comparisons on
control are negative. The premise did not reproduce.

*In plain terms:* we removed every safeguard against the cheat — and the model did not cheat. It
learned a good imagination on its own and planned with it successfully. Then we added the safeguards
back, one at a time, and every one of them made the agent **worse** at the actual task. The medicine
had side effects and the patient was never ill.

Why doesn't it cheat? The likely answer is architectural rather than a loss term: the target summary
comes from a slowly-updating copy of the encoder (the "EMA target"), so the encoder is always chasing
a moving goalpost it cannot instantly match. That asymmetry appears to be what makes the constant
answer unreachable — and it is already present in essentially every JEPA and in TD-MPC2.

### The inversion

| nuisance 0, offline | effective rank | readout R² | planning success |
|---|---:|---:|---:|
| none | 2.43 | 1.000 | **1.000** |
| uniformity | **11.44** | 0.999 | **0.011** |

Uniformity **quadruples the metric the field optimises and destroys 99% of the task.**

*In plain terms:* the health check said the representation got dramatically healthier, while the
agent went from succeeding every single time to succeeding once in ninety. It is a fitness tracker
reporting perfect vitals on a patient who can no longer walk. If you had been watching effective
rank — as this literature largely does — you would have concluded the change was a success.

A second detail worth pausing on: `none`'s effective rank of **2.43** looks collapsed until you
remember the underlying state is two-dimensional — so 2.43 is *correct*. Low rank meant an efficient
representation, not a dead one. Effective rank cannot distinguish those two cases, which is precisely
why we gate on control.

---

## Phase 1b — it is not a tuning artifact

The obvious objection: λ was fixed at 0.1 and never tuned, so perhaps we strangled the arms. We swept
it two orders of magnitude on the one cell with headroom (nuisance 16, offline, where `none` = 0.647),
at n=5.

| arm | λ | effective rank | readout R² | planning success | vs none | p |
|---|---:|---:|---:|---:|---:|---:|
| none | — | 23.06 | 0.988 | **0.647** | — | — |
| uniformity | 0.001 | 31.77 | 0.977 | 0.487 | −0.160 | 0.143 |
| uniformity | 0.01 | 31.51 | 0.965 | 0.440 | −0.207 | **0.040** |
| uniformity | 0.1 | 31.71 | 0.934 | 0.153 | −0.493 | **0.008** |
| vicreg | 0.001 | 22.26 | 0.951 | 0.053 | −0.593 | **0.008** |
| vicreg | 0.01 | 25.06 | 0.910 | 0.020 | −0.627 | **0.008** |
| vicreg | 0.1 | 30.23 | 0.908 | 0.013 | −0.633 | **0.008** |

**No strength of any arm rescues control**, and the harm is **monotone in λ**: more anti-collapse,
more effective rank, worse control. A dose-response relationship in the wrong direction is much
harder to explain away than a single bad hyperparameter.

*In plain terms:* the obvious objection to the previous section is "you used too much of the drug."
So we tried a thousandth of the dose, a hundredth, and the original. At every dose the patient did
worse than with no drug at all, and the more we gave, the worse it got. That pattern — a steady
dose-response — is what separates "this treatment is harmful" from "you mis-set one dial."

*(Note on Phase 1's statistics: at n=3 vs n=3 there are only 20 permutations, so the minimum
two-sided p is 0.10 — every "p = 0.100" in the Phase 1 table is the floor, not a result. Phase 1b at
n=5 is where the significance lives.)*

---

## What survives

**The supervisor's H1 is supported — by our own data, which we had misfiled.** In an earlier
experiment a *strong* VICReg arm fully un-collapsed a latent (rank restored) and **still failed**
(0.442 vs the collapsed default's 0.530). Marginal non-collapse does not imply the representation
kept what matters. We had filed that under "SE didn't work"; it is evidence *for* the motivation.

**The relations-not-marginals instinct remains untested.** Every control we have ever run —
VICReg, uniformity, SE on a kNN graph — regularizes the **marginal**. An action-conditioned,
**transition-level** term is genuinely novel and none of our nulls touch it. What these results say
is that it should not be motivated by *collapse*, because collapse is not what is going wrong.

**And a reframing of what "collapse" even means.** A representation is collapsed **relative to a
downstream decoder** — the information that decoder needs is not recoverable. Effective rank is one
sufficient statistic for that, and a bad one: it read 2.43 for a perfect representation and 11.44 for
one that had lost the task.

## Limits, stated plainly

- **The two-room environment is ours**, written to the proposal's description. It is not a
  standardised third-party benchmark, and results should be read as such.
- **Scale.** LeWorldModel trains ~15M parameters end-to-end from pixels. Our JEPA is small and
  state-based. The honest claim is *"we could not reproduce the motivating collapse in these
  regimes"* — a reproduction study, not *"collapse never happens."*
- **Phase 2 used TD-MPC2 latents** (value-anchored, SimNorm), a proxy for the pure-JEPA latent the
  proposal targets, and hopper — this lab's noisiest instrument — carries its only positive signal.
- **`SI` here uses a hard k-means partition**, not the proposal's differentiable soft assignment. A
  soft variant could behave differently, though it would have to overturn a +0.000.
- One cited baseline, **"PhyLatent"**, does not resolve to any paper, method or repository we could
  find; the nearest real work is
  [*What Can Latent World Models Know?*](https://arxiv.org/pdf/2607.27017). The other three
  citations are real and verified.

## Three corrections in twelve hours

Worth recording, because the rate is the point:

1. I wrote that the proposal's core experiment "had already been run here." **Too strong** — our
   prior work used a *kNN latent graph* on a *value-anchored* latent. Different object, different
   regime.
2. I pitched an "anti-collapse is downstream-dependent" paper. Thread D had already shown that
   taxonomy was **nav-specific**, not a general law.
3. I reported a significant sign flip in the `SI` correlations that was a **training-step artifact**
   on two runs.

Each was caught by a control or a confound check rather than by intuition. The same discipline is
what turned a proposal that would have cost a campaign into a NO-GO costing about **$1.60**.

---

*Code, per-seed values and the analysis scripts are in `SuuTTT/SE-JEPA`; the checkpoints and logged
returns behind Phase 2 are in `SuuTTT/world-model-paper`. The verdicts are pre-committed in the
analysis scripts rather than chosen after seeing the numbers.*
