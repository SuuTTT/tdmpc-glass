---
layout: post
title: "TD-MPC-Glass, Part 37: Two Programs in Full"
date: 2026-08-22
description: "The complete records of the two long-running programs that Part 36 compressed into verdicts. The planner program: five experiments across three campaigns, from 'the planner earns its keep' through 'search depth buys nothing' to the cross-task refutation that search depth is exactly what high-dimensional tasks buy — every claim, every number, and what died at each step. The measurement program: five candidate measures of representation information content, five obituaries, one honest yardstick that needs the ground truth you never have — and why the failures share a shape. Each program ends with concrete next experiments, ranked by cost."
---

**TL;DR.** [Part 36](../../../2026/08/22/tdmpc-glass-part36-three-preregistrations-three-verdicts.html)
gave the verdicts; this post gives the full case files for branches 1 and 2 — what was tried,
in order; what failed, with the numbers; and what we would run next. The planner program's
arc: every stage's conclusion was correct *on the tasks it looked at* and wrong as a
generalization, until a 22-task sweep found the variable — action dimensionality — that all
the earlier task selections had held fixed. The measurement program's arc: five measures
failed three ground-truth axes in three *different* ways (geometry, estimator sensitivity,
trainedness), which is itself the finding: the failures are structural, not bad luck.

---

# Program 1 — what is a planner worth, and what part of it?

## What was tried, in order

**Stage A (July): does the planner earn its keep at all?** Deployment-time toggle on
identical weights (`cfg.mpc` flipped at eval, paired eval seeds), our own
trained-without-planner checkpoints, four DMC tasks. Result: yes — hopper-hop +175 points
(p=0.002), and a *cheap* planner (32 samples × 1 iteration, ~3% of full MPPI's compute)
recovered 91% of it. Conclusion recorded then: *the planner is worth having; its search
budget mostly is not.*

**Stage B (Part 34, branch 1): decompose it over training.** `full − none =
(full − cheap) + (cheap − none)` across five checkpoints per task. Result: all the
interesting variation — hopper's advantage growing eighty-fold, walker's decaying — lived in
`cheap − none` ("having a planner"); `full − cheap` ("searching hard") was small on every
task and never grew. A two-factor account (model improving pushes planner value up, policy
improving pushes it down) survived on 3 of 4 tasks; not claimed as confirmed.

**Stage C (Part 34, branch 2): does representational capacity set planning value?**
Preregistered. `latent_dim` ∈ {4…512} × 3 seeds, planner toggled on identical weights.
Result: flat across 128× (rho=+0.064, p=0.82). **Refuted, as the prereg's own decision rule
demanded.** Bonus finding: *every* column was flat — a 4-dim latent matched the shipped
512 on hopper-hop.

**Stage D (Part 36, branch 1x): find the cross-task principles.** The obvious weakness of
stages A–B was n=4 tasks, all low-dimensional. The official TD-MPC2 release ships single-task
checkpoints for 39 DMC tasks; we ran the three-arm toggle on **22 tasks × 3 released seeds**
(66 evaluations, 10 episodes per arm, paired seeds; quadruped dropped to a dm_control
version incompatibility). Preregistered three principles: (P1) `cheap − none` tracks the
policy-to-ceiling gap; (P2) `full − cheap` is small everywhere and if anything *shrinks*
with action dimensionality; (P3) sparse-reward tasks need the planner more at matched
policy gap.

## What failed, with the numbers

**All three predictions.** P1: rho=+0.31, p=0.16. P3: residual test p=0.29. And P2 failed
in the only way better than passing — refuted with the opposite sign, significantly:

> `full − cheap` ~ action_dim: **rho = +0.58, p = 0.006** (predicted ≤ 0).

The table that rewrites stages A–B:

| task (action dim) | none | cheap | full | having | extra search |
|---|---:|---:|---:|---:|---:|
| humanoid-walk (21) | 3 | 9 | 896 | +6 | **+887** |
| dog-walk (38) | 26 | 95 | 843 | +69 | **+748** |
| dog-run (38) | 60 | 87 | 723 | +27 | **+636** |
| walker-walk (6) | 430 | 836 | 988 | +405 | +152 |
| hopper-hop (4) | 351 | 531 | 492 | +180 | −38 |
| acrobot-swingup (1) | 635 | 570 | 627 | −65 | +56 |

On 21–38-dimensional bodies, the released agents are almost *entirely* planner at deployment
— the policy prior scores 3–60 out of ~1000 — and the cheap planner collapses with it,
because 32 random samples cover a 4-dimensional action space and cover nothing in a
38-dimensional one. Stages A–B were not wrong about their tasks; they were wrong to
generalize, because every task they looked at sat in the regime where sampling is easy.
**"Search depth buys little" was a task-selection artifact, and the artifact was ours.**

What still stands from the earlier stages: on low-dimensional tasks the cheap-planner result
is real and useful (hopper's 91% recovery at 3% compute is untouched); the capacity
refutation (stage C) is untouched; the two-factor account remains unconfirmed and now needs
a dimension-aware restatement before it is worth testing again.

Caveats carried: the official checkpoints trained *with* the planner, so the toggle measures
deployment-time value for search-trained agents — the same measurement across tasks, so the
cross-task comparison holds, but absolute values don't transfer to our
trained-without-planner numbers. And `having` correlates rho=0.88 with per-task eval noise,
so no `having`-based claim survives the prereg's own sensitivity rule; the dog/humanoid
search effects are the lowest-noise cells in the table.

## What's next, ranked by cost

1. **The sample-coverage curve (cheap, eval-only, ~$2).** If the mechanism is coverage,
   planner value on dog should climb smoothly with samples: run the toggle at
   32/128/512/2048 samples on dog-walk and hopper-hop. Prediction to preregister: dog's
   curve saturates late and steeply; hopper's saturates by 32. This turns the P2 refutation
   into a mechanism test.
2. **Prior-quality confound check (cheap).** Is the dog prior useless because the task is
   planner-dependent in principle, or because 3M steps under-trains a 38-D policy? The
   released multi-task and larger single-task checkpoints give priors of different quality
   on the same task — measure whether `having` shrinks as the prior improves, which is the
   two-factor account's dimension-aware form.
3. **Train-time vs deployment-time on one high-dim task (expensive, ~$15–30 GPU).** Our June
   result says the deployment toggle understates train-time planner value (24% recovery).
   If that holds on dog, the true search dependence of high-dim tasks is even larger than
   +750. One task, two training runs (mpc on/off), only if the budget allows.

---

# Program 2 — a measure of information content for JEPA representations

## The method that made failure meaningful

You cannot validate an information measure against unknown truth, so the program built
**axes where the truth is manufactured**: *grid* (cube position quantised to g×g, so the
image carries exactly 2·log₂g bits — must rise), *nuisance* (distractors in a side panel,
scene pixels bit-identical, verified model-free: raw-pixel decodability drifts −0.9% — must
stay flat), *corruption* (pixel noise — must fall; the axis itself had to be re-verified
this campaign: σ ≤ 0.1 was inert, σ ∈ {0.2, 0.35, 0.5} moves raw decodability
0.997 → 0.429). Since Part 35, every axis test also carries a **sensitivity check**:
correlate the candidate's effect across levels with everything else the axis moves
(representation rank, predictor R², final loss); |r| > 0.9 flags a probable artifact.

## The five obituaries, in order

**1. Variance and effective rank** (the field's standard health checks). Failed nuisance in
the strongest possible form: effective rank *rose* 15.0 → 26.8 (exact p=0.0079, complete
separation) while recoverable information was provably constant. Geometry is not content;
adding unpredictable distractor dimensions adds directions. This was also the direct
demonstration of Yoon's H1.

**2. InfoNCE** (a genuine lower bound, the principled answer to the upper-bound objection).
Inflated under nuisance (0.89 → 1.03, p=0.032) with its cap far away. A valid bound
*estimated from finite data with a learned critic* is not a reliable measurement — the
critic exploits nuisance structure. "Principled" and "trustworthy" came apart.

**3. Structural information, five constructions deep.** k-means proxy: no signal (+0.099
vs known bits). selib SE-optimal: a **degree-matched null with no structure scored higher
than the real graph** (+0.322 vs +0.285) — every positive SI result vanished when the null
was subtracted. Exact global-optimum certificate: worse than local search. Directed +
self-loops (faithful to the proposal): fails, not flat. Differentiable soft assignment (the
proposal's own form): the only variant that robustly clears its null — stable, and
*constant*: it does not track information content on any axis. Related casualty from the
adjacent campaign: the coarse-graph "structure" result, 20/20 maps at p<0.0001, was an
off-diagonal-mass sensitivity artifact (Part 35).

**4. E2 — one-step predictability** ("measure the part of the representation the model can
predict"). Failed nuisance immediately: the distractors follow a random walk, and random
walks are *highly predictable* one step ahead (predictor R² 0.97–0.99 on them).
"Predictable" is not "task-relevant" — real nuisance is often predictable too.

**5. E2b — the action-conditional spread** (this campaign; preregistered). Fix E2's flaw by
measuring only *what the action changes*: distractors don't respond to actions, so they
cannot enter. Two statistics, both dead. `action_eff_rank`: **no dynamic range on any
axis** — flat on grid (rho=−0.04, p=0.85) where truth spans 2→10 bits, flat on corruption,
indistinguishable under nuisance. The rank of the action's effect is set by the
architecture, not by knowledge: E2b measures *controllability*, which is orthogonal to
content. `action_var`: moves under nuisance where it must not (p=0.032, a shift 13× its
entire grid-axis range), and its one apparent pass — falling with corruption, rho=−0.93 —
was flagged by the sensitivity check at **r=+0.992 with final training loss**. It tracks
how well the model trained. The check written after Part 35 caught its first fake pass
within a day of existing.

## The shape of the failures

Marginal statistics read the **geometry** of the embedding cloud. Graph measures inherit
the **estimator's sensitivity** to density. Predictor measures read **trainedness and
controllability**. Three different failure modes, one common property: each measure is
faithful to some real quantity — it is just never the quantity "task-relevant information."
Meanwhile the one number that behaved on every axis, readout R² against the true state,
is the one you can only compute when you already have the answer. And the adjacent
campaigns quietly removed the program's motivation: capacity doesn't set planning value
(128× flat), and hierarchy's planning advantage is not a representation property either.

## What's next, ranked by honesty

1. **Test the premise before the sixth candidate (cheap, data on disk).** Yoon's H3 says a
   good measure should predict planning performance. Readout R² did, on 15 agents
   (rho=+0.607, p=0.020). Extend that to the full ~200-run TD-MPC2 archive and the h3c
   sweep: if even the *yardstick* stops predicting planning value at scale, the program's
   premise — that any representation scalar could — is dead, and stages 1–5 are explained
   at once. This costs nothing but analysis time.
2. **Reward-readout as the practical proxy (cheap).** The RL setting always has one ground
   truth that needs no privileged state: reward. Decodability of reward/value from the
   frozen latent is computable in deployment and is at least anchored to the task. Run it
   through all three axes like every other candidate — it may die like the rest, but it is
   the only candidate whose ground truth ships with the problem.
3. **Write the obituaries up properly.** Five measures, three validated axes, a
   sensitivity-check protocol, and two artifact catches (off-diagonal mass, loss-tracking) —
   as a negative-results methods paper. This project treats honest negatives as first-class
   output; this is the most complete one it owns.

---

*Both programs' raw data and analysis scripts resolve to files at commits on
`results/three-branch-2026-08-20` in `SuuTTT/world-model-paper`; the preregistrations and
their dated amendments are in `PREREG_extension.md` on the same branch.*
