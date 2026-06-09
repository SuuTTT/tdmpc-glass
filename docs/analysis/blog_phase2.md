---
title: "TD-MPC-Glass, Part 2: Eight Mirages, One Real Win, and the Road Back to Structural Entropy"
date: 2026-06-09
description: "What happened after Part 1: we put every abstraction idea we had — including our own structural-entropy Glass — through a strict fair protocol on TD-MPC2. Eight apparent wins dissolved to null. One method (a jumpy k-step world model) really did beat vanilla on manipulation, but it isn't ours. Along the way we learned an uncomfortable lesson about peak-vs-final reporting in deep RL. We close with the next bet: structural entropy, this time over a jumpy model's latent transition graph."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["tdmpc2", "glass-jax", "structural-entropy", "world-models", "jumpy-models", "reinforcement-learning", "dmc", "reproducibility", "rliable", "vastai"]
---

{{< katex >}}

> Part 1 ended on a high: a structural-entropy ("Glass") augmentation of TD-MPC2 that climbed
> above the official 4M-step HopperHop mean. This post is the honest sequel. We asked one question —
> **does abstraction actually improve a strong latent world model when you remove every procedure
> trick?** — and answered it the hard way: a pre-registered, compute-matched, single-variable
> campaign across ~10–12 GPUs. The short version: **eight abstraction effects that each looked
> publishable at some snapshot regressed to null, one genuinely-known method (a jumpy world model)
> really did win on manipulation, and we found a methodology bug in how we — and much of the field —
> read RL curves.** Then we use what we learned to design the next idea, which brings us right back
> to structural entropy.

---

## 1. Why we tore up Part 1's result

Part 1's HopperHop "basin entries" were real numbers, but they came with restarts, population-based
selection, and a fair amount of seed luck. When we re-ran the exact comparison under a **clean
protocol** — identical hyperparameters, compute-matched, *one* variable changed at a time, no
restarts, no PBT, decisions fixed in advance — **neither Glass nor vanilla entered the high-reward
basin** (best 323 vs 286, both plateaued ~280–315 from 1.5M on). The thing that had motivated the
whole project was a procedure artifact.

So we made the protocol itself the product. Three rules, applied to everything afterward:

1. **Fair protocol.** Only the abstraction term changes between arms. Everything else identical.
2. **Pre-registered gates.** Sample sizes and decision thresholds fixed *before* data. Cheap
   falsifications sized to the minimum informative experiment. **Mechanism-checked before fan-out.**
3. **Honest aggregation.** Per-task-normalized IQM with stratified-bootstrap 95% CIs
   ([rliable](https://arxiv.org/abs/2108.13264)); and — after the lesson in §4 — **both** peak
   (best-checkpoint) **and** final (last-2-eval) reported for every arm.

## 2. Eight mirages

We tested abstraction over state, reward, time, and skills. Here is the scoreboard.

| # | Mechanism | Result | Why it dissolved |
|---|---|---|---|
| 1 | Geometric prototype clustering (structural entropy) | null (IQM 0.748 vs 0.738) | redundant with SimNorm's soft-categorical latent |
| 2 | Behavioral (reward-grounded) clustering | null at n=34 (0.767 vs 0.738, CI overlap) | gain crossed CI-separation **three times**, wandered both sides of 0 |
| 3 | Bisimulation auxiliary (BS-MPC-style) | **hurts** (0.549) | brittle; an untuned coefficient collapses training |
| 4 | Distractor robustness (64-dim OU noise) | falsified (1.23× < 1.5× gate) | both encoders crushed identically |
| 5 | Sparse-task rescue via behavioral grouping | null (0/3 vs 1/3) | sparse bimodality is *exploration*, not geometry |
| 6 | "Floor effect" / zero-weak-seed tail | inverted | the behavioral arm produced the study's single **worst** seed (0.127) |
| 7 | Laplacian / eigenpurpose exploration (DCEO-style) | null vs RND | abstraction-flavored novelty ≤ generic RND everywhere |
| 8 | Community-detection skills (Louvain on latent graph) | null | communities were motion *phases*, not reachable subgoals |

Mirages 1–6 are written up in detail in our "Six Mirages" draft; 7 and 8 came from later iterations.
The headline is uncomfortable and clean: **on a fair protocol, explicit abstraction did not improve
TD-MPC2 on final return, sample efficiency, distractor robustness, sparse exploration, or seed
reliability.** This is consistent with the theory that TD-MPC2's self-predictive objective is already
a *sufficient* abstraction ([Ni et al., 2024](https://arxiv.org/abs/2401.08898)) — our SimNorm latent
was, in a sense, already doing the clustering we were bolting on.

Two honest asides from the same campaign:
- **`rho` (consistency-horizon decay)** cures TD-MPC2's deep-planning collapse on manipulation
  (PandaPickCube H9: 419 → ~1775 mean) but *suppresses* sparse exploration (CartpoleSparse H9 goes
  [0, 0, 642]). A task-dependent **tuning lever, not architecture** — a trade, not a free win.
- **RND vs Laplacian:** generic novelty (RND) rescues some sparse tasks vanilla can't touch; the
  *abstraction-flavored* Laplacian bonus adds nothing over it. Exploration helps; the flavor doesn't.

## 3. The one real win — and why we're not taking credit for it

The iteration-18 gate pointed somewhere specific: a longer planning horizon helps where reward is
beyond a short planner's reach, but naive deep planning pays a **compounding 1-step-model-error tax**
(vanilla H9 collapses on Panda). The architectural fix is a long *effective* horizon without rolling
the 1-step model many times — a **jumpy (k-step) world model**:

$$ d_k(z_t, a_{t:t+k}) \;\to\; z_{t+k}, \qquad
   \mathcal{L}_{\text{HC}} = \big\lVert d(z,a,2k) - d(d(z,a,k),a,k)\big\rVert^2 $$

plus a macro-reward head and a macro-MPPI that plans \(n_{\text{macro}}\) jumpy steps (effective
horizon \(k\cdot n_{\text{macro}}\)) with only a few model applies.

We did the thing the earlier iterations skipped: **confirmed the mechanism before spending compute.**
The k-step head is measurably more accurate than iterating the 1-step model k times, and the edge
*grows* with k:

| k | jumpy_err / iter1_err |
|---|---|
| 2 | 0.991 |
| 3 | 0.910 |
| 8 | 0.821 |

Then the result, on PandaPickCube, fair protocol, mature ≥400k, paired, 20k-resample bootstrap:

| metric | jumpy (n=3→5) | vanilla-H3 (n=7) | diff | 95% CI |
|---|---|---|---|---|
| **peak** | 3027 | 2217 | **+810 (+37%)** | **[583, 1060]** |
| **final** | 2626 | 1442 | **+1184 (+82%)** | **[860, 1510]** |

Both metrics separate. The peak gap is the clean "plans better" claim; the larger final gap is a
**stability** finding — vanilla TD-MPC2 itself collapses peak→final on Panda (−35%), and the jumpy
model resists that late collapse. Five seeds agree; there's no Cartpole-style oscillation.

**The caveat we put first, not last:** the jumpy world model with a cross-timescale consistency loss
is **published prior art** — [Farebrother et al., *Compositional Planning with Jumpy World Models*,
2026](https://arxiv.org/abs/2602.19634) (verified). Their setting differs (composing pre-trained
policies zero-shot via TD-flow occupancy models, vs our online TD-MPC2 macro-MPPI), but the *concept*
is theirs. So this is a **fair-protocol reproduction-and-evaluation win, not an architectural
invention.** The methods that *were* ours — the structural-entropy and skill abstractions — are in
the mirage table. We think saying that plainly is the whole point of the protocol.

## 4. The methodology bug: peak vs final

Midway through, a sharp question (from a collaborator): *catastrophic forgetting and within-run
collapse are everywhere in RL — how do people actually report this? Do they just take the peak and
early-stop? Does that change your earlier verdicts?*

It does, and untangling it cleaned up our claims. There are two different failure modes:

- **Cross-seed small-sample mirages** (peak-*insensitive*). The behavioral-Glass IQM read
  0.818 → 0.736 → 0.829 → … → 0.767 as seeds accumulated, crossing "significant win" and "confirmed
  null" several times and landing in overlap. No reporting choice rescues these — the nulls stand.
- **Within-run collapse mirages** (peak-*sensitive*). Vanilla-Panda's −45% late collapse inflated an
  apparent jumpy "+104% on final"; the fair best-checkpoint comparison shrank it to +37% peak. A
  jumpy-on-CartpoleSparse "growing lead" reversed entirely by 450k — visible only if you read at
  ≥400k, not 250k.

The field standard ([rliable](https://arxiv.org/abs/2108.13264); and
[Henderson et al., 2018](https://arxiv.org/abs/1709.06560) on peak-picking bias) is fixed-budget,
many-seed, IQM with CIs — *not* peak. Best-checkpoint is legitimate for *deployment* claims **iff**
applied identically to all arms and disclosed. So we adopted **report-both**: peak *and* final for
every arm, gate on CI separation. Under that rule the jumpy win survives on both metrics while the
abstraction effects survive on neither. We now recommend publishing **estimate trajectories**, not
just final tables, for any marginal-effect claim in deep RL.

## 5. What's next: structural entropy, over a jumpy latent graph

Here's the part that closes the loop with Part 1. We treat the (validated, but not-novel) jumpy model
as a **substrate** and ask for a mechanism that beats *it*. A round of internal + three external deep-
research passes converged on two candidates:

- **The safe pick (3/3 consensus): an uncertainty-gated jumpy horizon.** Use a small dynamics-ensemble's
  disagreement to soft-truncate how deep the planner trusts each macro-rollout — directly attacking the
  jumpy model's own compounding-error weakness. ~80 lines, reuses TD-MPC2's ensembling. Feasible and
  likely to help; everyone agrees its novelty is modest.
- **The novelty pick — and our SE heritage, reborn:** build a **directed structural-entropy encoding
  tree over the jumpy model's latent *transition graph***, and use entropy minimization to (a) choose
  the jump length \(k\) the planner uses and (b) derive macro-actions/skills as high-frequency
  transitions between latent communities. The structural-entropy line (SIDM, SISA, SI2E, SIHD) is
  entirely model-free or diffusion-based; **nobody has built an encoding tree over a learned world-
  model latent, and nobody has used structural entropy to set temporal abstraction for an MPPI
  planner.** That gap is real — and it's exactly where Glass started.

There's a catch we already know to check first: TD-MPC2's SimNorm latents are *dense*, so the
transition graph might have no real community structure (in which case 2-D structural entropy ≈ 1-D,
and the SE lever is dead on arrival). So step zero is a cheap **pre-check** — does the jumpy latent
graph cluster at all, and does ensemble disagreement actually track prediction error? — before we
commit a line of planner code. Mechanism first, fan-out second. We learned that one the expensive way.

The pre-registered gate is fixed in advance: compute-matched, single-variable vs the plain jumpy
baseline, rliable IQM over 5 seeds reporting **peak and final**, on sparse DMC + PandaPickCube + a
high-DoF task (Dog/Humanoid, where the literature says the headroom actually lives). Win = ≥10% IQM
with non-overlapping CIs on ≥3 of 4 tasks. If structural entropy can't clear that, it goes in the
mirage table next to its ancestors — and we'll say so.

---

*Reproducibility: code in `helios-rl` (jumpy heads + `make_jumpy_mppi_fn` in
`src/helios/algorithms/tdmpc2.py`); per-run CSVs under `exp/tdmpc_glass/remote_mirror/`; iteration
records in `docs/iterations/`; the abstraction-null write-up in `docs/writeup/draft.md` and the
campaign capstone in `docs/writeup/capstone.md`. All numbers above are read from run CSVs, not
notebooks — verification discipline, the hard way.*
