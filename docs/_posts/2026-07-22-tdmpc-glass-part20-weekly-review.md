---
layout: post
title: "TD-MPC-Glass, Part 20: A Week in Review — From Instruments to a Paper, and the Question Finally Answered"
date: 2026-07-22
description: "The week of July 15–22. Part 19 left four named hypotheses and one sharp question — is the world model just a learned abstraction, and if so which knob does the work? This week turned the value-sufficiency instrument into a finished AAAI-27 diagnostic paper (9 tasks, cross-model correlation ρ=−0.90), found the diagnostic's honest boundary, caught and corrected a one-seed artifact with multi-seed replication, and — with a matched-seed rung-ladder plus the official TD-MPC2 vs SAC numbers — answered the sharp question: what beats model-free RL is the value pathway, not the world model or the planner."
---

> A review of one week — July 15 through July 22 — picking up where
> [Part 19](https://suuttt.github.io/tdmpc-glass/2026/07/15/tdmpc-glass-part19-weekly-review/) left off.
> Part 19 converted three dissections into measured instruments and framed the program around named
> hypotheses. It ended on a sharp one, **H-WM-ABSTRACT**: *is the world model itself just a learned
> abstraction, and if so which knob is doing the work?* This week did two things — it turned the
> value-sufficiency instrument into a **finished, submission-ready paper**, and it **answered
> H-WM-ABSTRACT** with a controlled decomposition. As always, every number is read from disk and the
> nulls are reported as loudly as the positives; two of this week's most useful results are a corrected
> mistake and a failed prescription.

## The one-paragraph version

The value-sufficiency bottleneck (**VBN**) — the checkpoint-time probe that measures how much of an
agent's latent the value head actually needs — became a paper. Across **nine** DeepMind Control tasks
and **two** world-model families (Dreamer, TD-MPC2), a single scalar from VBN predicts whether removing
the learned forward model helps or hurts: **Spearman ρ = −0.90 (n = 5 value-limited tasks)**. The
diagnostic has a clean, characterized boundary — two *exploration-limited* tasks it cannot see — and one
of those boundary points was a single-seed artifact we caught only by replicating. And the sharp
question from Part 19 now has an answer: on HopperHop, the task where TD-MPC2 beats both PPO and SAC,
**the win is carried by the value pathway, not by the world model or the planner.**

## Motivation, in one figure

![Same agent, opposite verdicts]({{ '/images/part20-motivation.png' | relative_url }})

Same agent (Dreamer), same intervention (disable the learned forward model). On **ball-in-cup** the
return collapses to zero — the world model is essential. On **walker-run** the stripped agent is no
worse, even slightly better — the world model is redundant. "Does a learned world model help?" has no
architecture-level answer; it is a **property of the task**. The whole paper is about a probe that tells
you which, *before* the world model is trained.

## Result 1 — the nine-task gradient and the cross-model law (H-COMPRESS, confirmed within scope)

We ablate the world model per task (strip its forward-dynamics learning) and measure the change in
return. On a fixed architecture, WM-dependence spans the **full** range:

| regime | tasks (WM-dependence = fraction of return lost when stripped) |
|---|---|
| **essential** | ball-in-cup (collapse), reacher-hard (−98%, n=2), acrobot (−85%, n=3) |
| helps / marginal | cheetah (−10%), cartpole (−3%) |
| **redundant** | finger (+5%), walker (+7%), quadruped (+1%) |

The essential end is sparse-reward / long-horizon / precision control; the redundant end is dense
continuous control. And the VBN fingerprint predicts the ordering *a priori*: return recovered through a
16-dimensional value bottleneck correlates with WM-dependence at **ρ = −0.90 (n = 5)** — measured with
compressibility in TD-MPC2 and dependence in Dreamer, so this is a genuinely **cross-model** prediction.

![VBN compressibility predicts WM-dependence]({{ '/images/part20-correlation.png' | relative_url }})

This is Part 19's **H-COMPRESS** graduating from prediction to result: low value-compressibility ⇒ the
world model is load-bearing; high compressibility ⇒ redundant.

## Result 2 — the honest boundary (where the probe stops working)

VBN is a *value* probe, so it is silent on tasks whose difficulty is **exploratory** rather than
representational. Two tasks sit exactly there:

- **ball-in-cup**: its value is *maximally* compressible in TD-MPC2 (a 16-dim bottleneck recovers ~100%
  of return) yet stripping the world model collapses it in Dreamer. The collapse is not a
  value-representation deficit — it is an exploration failure (without imagined rollouts the agent never
  discovers the catch).
- **pendulum**: covered below.

Both are excluded from the value-limited correlation and reported as the diagnostic's characterized
scope, not as hidden failures. This turns a limitation into a second contribution: a clean
**value-limited vs. exploration-limited** split, grounded in a decomposition of world-model benefit into
a representational channel (what VBN sees) and an exploratory channel (what it cannot).

## Result 3 — the mistake we caught by replicating (pendulum)

Pendulum-swingup was, for two days, the paper's most dramatic collapse point: a single seed showed
stripped return of exactly **0.0** vs ~806 vanilla. Multi-seed replication (the kind of run it is easy
to skip once you "have the result") told a different story. Across three seeds the stripped agent scored
**0 / 727 / 0**: two seeds never discover the swing-up, one matches vanilla with *no* representational
deficit. Pendulum is **bimodal**, exploration-limited — the seed-1 "collapse" was a lucky/unlucky draw,
not a value signal. We removed it from the clean correlation (ρ moved −0.94→−0.90, but now on a *clean*
value-limited set) and report the full seed range. The honest version is more defensible than the
dramatic one, and the episode is a small advertisement for running the replication seed.

## Result 4 — an independent cross-check we didn't have to run

The exploration-limited boundary is corroborated by data we did not generate. The official TD-MPC2
release also reports a tuned **SAC** — a model-free learner with *no* world model. On the tasks we call
exploration-influenced, SAC solves them just as well:

| task | TD-MPC2 | SAC | gap |
|---|---|---|---|
| ball-in-cup | 984 | 979 | **+5** |
| reacher-hard | 982 | 944 | **+38** |
| acrobot | 663 | 72 | +591 |
| hopper-hop | 449 | 117 | +332 |
| pendulum | 838 | 575 | +263 |

Ball-in-cup and reacher-hard collapse in Dreamer-without-a-WM but are solved by SAC — confirming their
failure is exploratory, not value-limited, with a third independent method. The large gaps elsewhere set
up the week's headline result.

## Result 5 — H-WM-ABSTRACT, answered: it's the value pathway

Part 19 asked which knob does the work when a world model helps. HopperHop is the sharpest test: TD-MPC2
reaches ~450–500 there while PPO is walled at ~40 (even at 94× the budget) and SAC is unreliable
(bimodal, ~47 on the matched seeds, ~188 on average). Yet we already knew two odd facts about Hop — the
consistency loss is **removable** there, and MPPI planning adds essentially nothing. So what beats
model-free RL?

We ran a matched-seed **rung-ladder**, reading the π-only and MPPI eval columns of the same runs:

![The value-pathway ladder on HopperHop]({{ '/images/part20-ladder.png' | relative_url }})

| rung | what it isolates | HopperHop return |
|---|---|---|
| SAC | model-free, no latent model, no planner | 47 (matched) · 188 (avg) |
| **value pathway** = strip-consistency + π-only | value-equivalent latent + off-policy value, **no WM, no planner** | **502** |
| + planning | adds MPPI | 487 |
| full TD-MPC2 | + consistency + planning | 509 |

**The value pathway alone (502) ≈ full TD-MPC2 (509) and captures essentially the entire gap over SAC.**
Adding back the world-model consistency loss and the planner changes the result by under 2% (planning
even hurt one seed). So on the one task where the model-based agent decisively wins, what does the work
is the **value-equivalent representation + off-policy value learning** — not the forward model, not the
planner. This is the seed for a second paper: a lighter, task-adaptive learner that keeps the value
pathway and pays for the world model only where a probe like VBN says it earns its keep.

## The nulls, reported loudly

- **The gate does not help.** The prescriptive fix the diagnostic seems to suggest — a plan-time gate
  that down-weights the world model's imagined rollout — is a confirmed negative at n=5:
  g0.0 = 701.9, g0.5 = 688.4, g1.0 = 686.5, g0.25 = 682.9 (a 19-point band, ≪ the 80–100-point seed
  spread). It is structural: with horizon 3 and γ=0.99 the gated term is ~3% of the plan score, and
  g=0 is not WM-free anyway. The contribution is a *diagnostic*, not a repaired algorithm.
- **Parity, honestly.** Our TD-MPC2 reimplementation is a non-anomalous baseline, not bit-parity:
  hopper ≈0%, cheetah −5%, walker −6.8% (once data-collection mode is matched), acrobot −23% (open).
  All TD-MPC2 claims in the paper are within-implementation comparisons; the cross-model headline rests
  on unmodified DreamerV3.

## Shipped

- **AAAI-27 submission paper.** *"When Does a Learned World Model Help? A Value-Sufficiency Diagnostic
  that Predicts World-Model Dependence Across Architectures."* Seven pages in the official
  `aaai2027.sty` submission format, double-blind clean, with a motivation figure, a method-pipeline
  figure, three data plots, two tables, and the mandatory reproducibility checklist. Compiles clean.
- All results logged to the public issues and the campaign ledger; every figure number traces to disk.

## Next

- **Paper 2 (the value pathway).** Generalize the ladder beyond HopperHop (acrobot next), and turn the
  decomposition into a task-adaptive recipe: keep the value pathway, add the world model only where a
  VBN-style probe predicts it is load-bearing.
- **Broaden the diagnostic.** The correlation is n=5 value-limited tasks; extend it, and test the
  value-limited/exploration-limited taxonomy on manipulation (Meta-World) and navigation (sparse-goal),
  which is the natural home for the exploration axis we can currently only bound with two points.
- **Training-free criterion.** The reward-propagation-horizon / controllability / value rate–distortion
  proxies remain the conjectural prize: predicting WM-dependence from the task spec alone, with no probe
  at all.
