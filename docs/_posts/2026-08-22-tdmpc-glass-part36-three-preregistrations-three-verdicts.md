---
layout: post
title: "TD-MPC-Glass, Part 36: Three Preregistrations, Three Verdicts"
date: 2026-08-22
description: "The extension campaign closes. Branch 1x: all three predicted planner principles failed, and the failure is the finding — extra search RISES with action dimensionality (rho=+0.58, p=0.006), worth +600 to +900 points on dog and humanoid where the cheap planner collapses; our own 'search buys little' claim was a low-dimensional artifact. Branch 2x: the fifth candidate measure of information content is dead on all three axes, its one apparent pass exposed by the sensitivity check Part 35 taught us. Branch 3x: the hierarchical planner's advantage on hard mazes survives a matched compute budget (19/20 vs 13/20, p=0.031) — which rejects our own Part 35 reading and points at the optimization horizon, not search-space reduction. One box, ~$29, and every claim was written down before its number existed."
---

**TL;DR.** All three branches of the extension campaign are answered, each against a
preregistration committed before any result existed
([PREREG_extension.md](https://github.com/SuuTTT/world-model-paper/blob/results/three-branch-2026-08-20/PREREG_extension.md),
`2cb4751`, three dated amendments). Branch 1x: our predicted planner principles all failed, and
one prediction was **refuted in the direction that matters** — search depth is what
high-dimensional tasks buy, the opposite of what four low-dimensional tasks had taught us.
Branch 2x: the action-conditional predictor measure joins four predecessors in the graveyard of
information-content measures, killed cleanly by the axes. Branch 3x: hierarchical planning's
advantage **persists at matched compute**, rejecting the search-space-reduction story we
ourselves proposed in Part 35. Two of the three verdicts correct claims *we* made. That is the
preregistration working.

## Branch 1x — the planner principles we predicted don't exist; a better one does

[Part 34](../../../2026/08/21/tdmpc-glass-part34-three-branches-and-a-missing-encoder.html)'s
branch 1 decomposed planner value into *having a planner* (cheap − none) and *searching hard*
(full − cheap) on four tasks, and concluded that extra search buys little everywhere. The
extension ran the same three-arm deployment toggle on the **official released TD-MPC2
checkpoints** across **22 DMControl tasks × 3 seeds** (66 evaluations, paired eval seeds,
10 episodes per arm; quadruped dropped to a dm_control version incompatibility).

Preregistered: (P1) *having a planner* tracks the policy-to-ceiling gap; (P2) *searching hard*
is small everywhere and, if anything, shrinks with action dimensionality; (P3) sparse tasks
need the planner more at matched policy gap.

**All three failed.** P1: rho = +0.31, p = 0.16. P3: p = 0.29. And P2 was not merely
non-significant — it was **refuted with the opposite sign**:

> extra search ~ action_dim: rho = **+0.58, p = 0.006** (we predicted ≤ 0)

| task | none | cheap | full | having | extra search |
|---|---:|---:|---:|---:|---:|
| dog-walk (38-D) | 26 | 95 | 843 | +69 | **+748** |
| dog-run (38-D) | 60 | 87 | 723 | +27 | **+636** |
| humanoid-walk (21-D) | 3 | 9 | 896 | +6 | **+887** |
| hopper-hop (4-D) | 351 | 531 | 492 | +180 | −38 |
| walker-walk (6-D) | 430 | 836 | 988 | +405 | +152 |

On high-dimensional bodies the released agents are almost *entirely* planner at deployment —
the policy prior alone scores 3–60 out of 1000 — and the cheap 32-sample planner collapses with
it. Everything our four-task study attributed to "having a planner at all" was a property of
low-dimensional action spaces, where 32 random samples cover the space. At 21–38 dimensions
they cover nothing, and search depth is the whole game. **The June "search buys little" claim
was a task-selection artifact, and we are the ones on record saying so.**

Honesty items: these checkpoints trained *with* the planner (the toggle understates train-time
value; identical measurement across tasks keeps the comparison valid), and `having` correlates
rho = 0.88 with per-task eval noise — but the dog/humanoid search effects are the *lowest*-noise
cells in the table. Per the decision rule, no predicted principle is claimed; the
dimensionality relationship is reported as a refutation of P2 plus a labelled exploratory
finding. Data: `data/b1x.jsonl`, `repro/b1x_{eval,analyze}.py`.

## Branch 2x — the fifth measure obituary

The candidate was E2b: the effective rank of *what the action changes* in the predictor's
output — anchored to dynamics like SI, immune (by construction) to the predictable-nuisance
failure that killed plain predictability. Preregistered across all three validated axes, with
the corruption axis first verified model-free (raw-pixel decodability 0.997 → 0.429 across
σ = 0 → 0.5 — unlike the inert σ ≤ 0.1 attempt).

| axis | truth demands | `action_eff_rank` | `action_var` |
|---|---|---|---|
| grid (2→10 bits) | rise | flat (rho=−0.04, p=0.85) | flat (p=0.78) |
| nuisance (constant) | flat | n.s. but > its own grid range | **moves, p=0.032**, 13× its grid range |
| corruption | fall | flat (p=0.62) | rho=−0.93, p<0.0001 * |

\* — and that lone pass is exactly what Part 35 taught us to distrust: across the σ levels it
correlates **r = +0.992 with final training loss**. It tracks how well the model trained, not
information. The mandatory sensitivity check, added to the prereg after the off-diagonal-mass
artifact, caught its first fake pass within a day of being written down.

`action_eff_rank` deserves its own line in the obituary: it has **no dynamic range on any
axis**. The rank of the action's effect is set by the architecture, not by what the
representation knows — E2b measures *controllability*, which is orthogonal to information
content. Five candidates in (variance/rank, InfoNCE, five SI constructions, E2, E2b), the only
quantity that has ever behaved on every axis is the readout R² that requires the ground truth
you never have. Data: `data/b2x_{grid,sigma}.jsonl`, `repro/b2x_analyze.py`.

## Branch 3x — we were wrong about why hierarchy helps, and the experiment said so

Part 35 ended branch 3 with: no extra transition structure at the coarse level, so the benefit
is presumably search-space reduction — nothing to claim. The extension tested that
presumption directly, on HWM_PLDM's own planning stack, with a comparison cleaner than we had
any right to expect: the released flat (PLDM) and hierarchical (HWM) checkpoints have
**bitwise-identical level-1 weights**, so the arms differ only in the planner. Same start/target
files across arms; paired exact tests.

| level | H (hier) | F (flat) | gap | paired p (n=40) |
|---|---:|---:|---:|---:|
| easy | 1.000 | 0.975 | +0.025 | 1.0 |
| medium | 0.825 | 0.700 | +0.125 | 0.30 |
| hard | 0.925 | 0.500 | **+0.425** | **0.0002** |

Prediction 1 — the advantage grows with goal distance — confirmed. Then the decisive arm: flat
planning given the hierarchical arm's **entire sample budget** (5000 at hard, which
over-provisions flat compute per replan, since its rollouts are 250 steps long). Preregistered:
if the gap closes, hierarchy is search-space reduction and Part 35's reading stands; if it
persists, our reading was wrong and we say so with equal prominence.

> hard, same 20 trials: **H 19/20 vs F+ 13/20** (exact p = 0.031).
> And F+ over F: 13/20 vs 11/20, p = 0.73 — **twenty times the samples bought nothing.**
> Steps to goal: H **143**, F 478, F+ 487.

**It persists. Prediction 2 rejected; Part 35's mechanism reading is revised.** The two results
are consistent once the right distinction is drawn: the coarse level's transition *graph*
carries no extra structure (Part 35 stands), but the coarse *planner* solves a different
optimization problem — MPPI over ~47 macro-actions has a landscape that 5000 samples navigate
easily, while MPPI over 250 primitive actions has one that 5000 samples cannot. The
world-model *representation* doesn't explain hierarchy's value; the **optimization horizon**
does. That is a claim about planning, not about representation learning — which is precisely
the distinction every structure measurement we built was blind to.

Caveats, stated plainly: n=20 in the decisive cell (6/0 discordant pairs); one environment;
three dated protocol amendments (probe-backed latent bounds after the 130 GB train-image set
proved unobtainable — sanity gate passed; F+ trimmed to the hard level and then to one env
batch on runtime grounds, both committed before any F+ number existed).

## What the campaign leaves us believing

Across seven preregistered questions in two campaigns, every attempt to locate planner value in
the *representation* — capacity (refuted), information measures (five obituaries), transition
structure (artifact) — has failed. The two things that demonstrably move planner value are
properties of the **search problem**: action dimensionality (branch 1x) and optimization
horizon (branch 3x). As a working hypothesis, labelled as interpretation and not as a tested
claim: **the world model's contribution is realized through the planner's optimization
landscape, and improving that landscape — not the latent's information content — is where the
leverage is.** That is testable, and it is sharp enough to die.

## Ledger

| branch | verdict | cost |
|---|---|---|
| 1x — planner principles | P1/P3 fail; **P2 refuted**: search depth ∝ action dim | ~$2 |
| 2x — info measure E2b | **dead on all three axes**; fake pass caught by the sensitivity check | <$1 |
| 3x — hierarchy mechanism | gap survives matched compute; **our Part 35 reading revised** | ~$6 |

One RTX 3090 Ti carried everything; total box spend across both campaigns ≈ $29. The box's own
budget guard stopped it twice mid-campaign at its original $20 cap — both stops were the guard
doing exactly what it was built for, and both were survived by pull-verify automation. The box
is destroyed; every number in this post resolves to a file at a commit on
`results/three-branch-2026-08-20`.

**Repro:** `repro/b1x_analyze.py data/b1x.jsonl` · `repro/b2x_analyze.py data/b2x_grid.jsonl
data/b2x_sigma.jsonl h2.jsonl` · `repro/b3x_analyze.py data/b3x/` in `SuuTTT/world-model-paper`.
