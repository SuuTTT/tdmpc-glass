---
layout: post
title: "TD-MPC-Glass, Part 33: From Findings to Claims"
date: 2026-08-20
description: "Our supervisor's critique of Part 32 was that we had produced findings where we needed claims: the planner work was 'yet another which-tasks-need-a-planner result', and the measurement work was testing that a metric should be invariant to noise, which is a sanity property rather than a hypothesis. Both criticisms are right. This post reframes them. The planner result becomes a rule with a mechanism: the planner's value is set by how far the learned policy is from its ceiling, which predicts a perfect rank ordering across our four tasks and explains why the advantage halves between 200k and 400k steps. The measurement work gets an honest verdict: as posed it has no hypothesis, and we say what would give it one. And hierarchical planning becomes the focus, aimed at HWM."
---

> **TL;DR.** Yoon's critique of [Part 32](../../../2026/08/19/tdmpc-glass-part32-two-weeks-of-measurement.html)
> was that we had findings, not claims. He is right on both counts. **The planner result now has a
> rule and a mechanism**: planning is a scaffold for a not-yet-good policy, its value is set by the
> policy's distance from ceiling, and that predicts a perfect rank ordering across our four tasks
> (rho = 1.000, n=4) and explains the 49% decay between 200k and 400k steps. **The measurement work,
> as posed, has no hypothesis** — "a good metric should ignore irrelevant clutter" is a sanity check,
> and discovering that SI fails it is not a result worth a paper. **Hierarchical planning is the
> focus**, and it shares the planner work's through-line: both are about what *search* buys.

## The critique

Three points, paraphrased:

1. **The planner work is another "which tasks need a planner" result.** More tasks buy more empirical
   suggestions, never a principle. We need a rule.
2. **The measurement work has no hypothesis.** We are testing that a measure should be invariant to
   visual clutter and reporting that SI is not. That is a property a metric ought to have, not a
   claim about the world.
3. **Hierarchical world models are where to focus.**

All three land. Below is what we do about them.

## 1. The planner result, restated as a rule

The finding was: a planner at ~3% of MPPI's compute recovers 91% of its value on hopper and 46% on
walker. That is a catalog entry. Here is the rule underneath it.

**Claim: planning is a scaffold for a policy that is not yet good, and its value is set by how far the
policy is from its ceiling.**

Two pieces of evidence, one within a task and one across tasks.

**Within a task.** Walker at two budgets, everything else fixed:

| arm | 200k | 400k | gain |
|---|---:|---:|---:|
| no planner | 652.0 | 735.1 | **+83.1** |
| cheap planner | 754.9 | 767.8 | +12.9 |
| full MPPI | 791.5 | 806.4 | +14.9 |

The planner's advantage falls from **+139.5 to +71.3 — a 49% decay**. And the mechanism is visible in
the columns: the gap closes because the **policy caught up (+83.1)**, not because planning got worse
(+14.9). Planning was standing in for a policy that could not yet act well on its own.

**Across tasks.** If that is the mechanism, then the planner should be worth most exactly where the
policy alone is worst. It is:

| task | policy alone | headroom to 1000 | planner gain | relative gain |
|---|---:|---:|---:|---:|
| hopper-hop | 152.5 | 847.5 | +174.8 | **1.146** |
| acrobot-swingup | 327.4 | 672.6 | +169.8 | 0.519 |
| walker-run | 735.1 | 264.9 | +71.3 | 0.097 |
| cheetah-run | 851.4 | 148.6 | +65.7 | 0.077 |

`rho(policy-alone return, relative planner gain) = **-1.000**` — a perfect rank ordering across all
four tasks. With n=4 that is p = 0.083 two-sided, so it is a **hypothesis with a perfect ordering, not
an established law**. But it is the right shape: it *predicts*, rather than cataloguing.

**What it predicts, and how to kill it:**

- Within any task, the planner's value should decay monotonically with training budget, and approach
  zero at convergence.
- Across tasks, relative planner value should track policy headroom — so "which tasks need a planner"
  is not a task property at all, but a *training-state* property.
- **Falsifier:** a task where the policy is near its ceiling and the planner still helps a lot. Or a
  task where the policy is far from ceiling and planning does not help.

This also retires our own earlier framing. We had reported cheetah as "saturated, so the planner does
nothing" as though saturation were a quirk of that environment. Under this rule, saturation *is* the
explanation — cheetah's policy is at 851 of ~1000, so there is nothing for a scaffold to hold up.

## 2. The measurement work: an honest verdict

The critique here is sharper and we accept it. Our test was: hold information constant, add visual
clutter, see which measures stay flat. **That is a sanity property.** A metric that fails it is
broken; a metric that passes it has demonstrated nothing except not being broken. Reporting that five
SI variants fail a sanity check is not a contribution.

What *would* make it a hypothesis? Looking at what we actually measured, the candidate measures sort
by **what they read**:

| class | measures | result |
|---|---|---|
| the marginal `p(z)` | variance, effective rank, embedding std | fail — inflate on information-free clutter |
| a learned bound on `I(Z;S)` | InfoNCE | fails — inflates too |
| the transitions `p(z'\|z)` | SI, five variants | fails differently — tracks temporal smoothness |
| task-anchored | held-out readout | works, and needs labels |

That table suggests a real claim: **task-relevant information is not a property of a representation
alone — it is defined relative to a decoder, so no label-free measure of it can exist.** Every
candidate we tested is measuring some surrogate — spread, smoothness — that correlates with usefulness
in some regimes and not others. The falsifier is clean: exhibit a label-free quantity that predicts
downstream performance across regimes.

We are **not** claiming that. It is an impossibility-flavoured hypothesis and we have four failed
measures, which is evidence but not proof. What we will do is stop running the sanity check and stop
reporting its failures as results.

The one thing from this line that survives on its own is narrower and we will state it narrowly:
**effective rank has zero correlation with planning performance** (rho = -0.029, p = 0.923, 15 agents)
while being used across the literature as evidence a representation is healthy. That is a warning
about a specific practice, not a theory.

## 3. Hierarchical planning is the focus

And it shares a through-line with §1, which is why the two belong in one program rather than two.

**The target is [HWM](https://arxiv.org/abs/2604.03208)** — *Hierarchical Planning with Latent World
Models* (Zhang, Terver, Zholus, Chitnis, Sutaria, Assran, Balestriero, Bar, Bardes, **LeCun**, Ballas).
World models at multiple temporal scales in one shared latent space, trained purely by next-latent
prediction; the long-horizon model's predictions become subgoals for the short-horizon one. On a real
Franka arm: **70% versus 0%** for single-level planning, from a single goal image, with 3x less
planning compute. Code and checkpoints are public.

**Two papers independently name the same bottleneck.** HWM: *"to keep long-horizon search tractable,
HWM learns an action encoder that compresses primitive action chunks into latent macro-actions."*
[Hi-LeWM](https://arxiv.org/abs/2607.12547): unconstrained high-level search "can select latent
macro-actions that appear favorable under the learned model but produce poor control targets", and
constraining that space is worth +11.3 and +14.7 percentage points. Their diagnosis is explicit — the
frozen low-level controller executes well-aligned subgoals fine, so **subgoal generation is the
bottleneck**, not control.

**The through-line.** Section 1 says a planner is a scaffold whose value depends on how much the
policy still needs it. Hi-LeWM says hierarchy's value depends on whether the high-level search space
is constrained to things the low level can execute. Neither is a claim about *prediction quality* —
both are claims about **what search is buying and when**. That is the program:

> **Search in a world model is not a free improvement. Its value is set by what the rest of the system
> cannot yet do, and by whether the space being searched is one the controller can act in.**

That framing makes §1 and §3 the same paper, and it demotes §2 to a methods note.

**The plan, cheapest first.**

1. Load HWM's released checkpoints, extract latents per temporal scale, and ask which measure predicts
   planning success across their conditions. No training. *This is also the fair version of the test
   §2 kept failing to construct — HWM's 0% → 70% is a real quality gradient we did not build and
   therefore cannot have broken.*
2. Use SI as a **diagnostic** on the learned macro-action space: does it separate working
   configurations from failing ones? *Falsifier: if it cannot, SI adds nothing here and we stop.*
3. Only if both survive: an SI-derived constraint on high-level search — the only step that would be a
   method contribution.

**What we are not proposing**, stated so it is not re-proposed: not replacing HWM's learned action
encoder with a graph partition (that is the representation-swap pattern SE's own case studies record
as its main failure mode), and not using SE to choose the number of levels (already gated and killed —
SE is provably indifferent between a multiway tree and its binary refinement, so it always returns a
dendrogram; we measured branching exactly 2.00 and depth indistinguishable from a null).

## What changed in one line

We had three directions and a pile of measurements. We now have **one program with a claim** — search
is a scaffold, and its value is set by what the rest of the system cannot yet do — a **rule that
predicts** rather than catalogues, and a **target system** where that rule can be tested on a real
robot with released checkpoints.

*Numbers above trace to `SuuTTT/world-model-paper` and `SuuTTT/SE-JEPA`. The planner-headroom ordering
is n=4 and is offered as a hypothesis with its falsifier attached, not as a result.*
