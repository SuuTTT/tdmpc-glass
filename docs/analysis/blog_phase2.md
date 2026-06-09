---
title: "TD-MPC-Glass, Part 2: Eight Mirages, One Real (Borrowed) Win, and a Mechanism-Check That Saved a Campaign"
date: 2026-06-09
description: "After Part 1 we put every abstraction idea we had — including our own structural-entropy Glass — through a strict fair protocol on TD-MPC2. Eight apparent wins dissolved to null; one method (a jumpy k-step world model) really did beat vanilla on manipulation, but it isn't ours. We learned a hard lesson on peak-vs-final reporting, and our next novel bet — structural entropy over the jumpy latent graph — was killed by a cheap mechanism-check before it cost a multi-week campaign. A field report on how hard a strong world model is to beat."
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
> read RL curves.** Then we use what we learned to design the next idea — which brings us right back
> to structural entropy, where a cheap mechanism-check killed it before it cost us a campaign.

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

| metric | jumpy (n=5) | vanilla-H3 (n=7) | diff | 95% CI |
|---|---|---|---|---|
| **peak** | 3183 | 2217 | **+966 (+44%)** | **[714, 1248]** |
| **final** | 2708 | 1442 | **+1266 (+88%)** | **[877, 1642]** |

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
  apparent jumpy "+104% on final"; the fair best-checkpoint comparison shrank it to +44% peak. A
  jumpy-on-CartpoleSparse "growing lead" reversed entirely by 450k — visible only if you read at
  ≥400k, not 250k.

The field standard ([rliable](https://arxiv.org/abs/2108.13264); and
[Henderson et al., 2018](https://arxiv.org/abs/1709.06560) on peak-picking bias) is fixed-budget,
many-seed, IQM with CIs — *not* peak. Best-checkpoint is legitimate for *deployment* claims **iff**
applied identically to all arms and disclosed. So we adopted **report-both**: peak *and* final for
every arm, gate on CI separation. Under that rule the jumpy win survives on both metrics while the
abstraction effects survive on neither. We now recommend publishing **estimate trajectories**, not
just final tables, for any marginal-effect claim in deep RL.

## 5. The next novel bet — and how a cheap mechanism-check killed it in an afternoon

Here's the part that closes the loop with Part 1. We treated the (validated, but not-novel) jumpy model
as a **substrate** and asked for a mechanism that beats *it*. Internal + three external deep-research
passes converged on a structural-entropy lever, right back where Glass started: build a directed
**structural-entropy encoding tree over the jumpy model's latent transition graph**, and use entropy
minimization to pick the jump length \(k\) — long jumps inside a coherent "motion-phase" community,
short jumps at community boundaries (contacts, turning points), where a long jump should be least
accurate. The SE line (SIDM, SISA, SI2E, SIHD) is all model-free or diffusion-based; nobody had used
structural entropy to set temporal abstraction for an MPPI planner over a learned world model. A real gap.

We did the thing the early iterations didn't: **two cheap pre-checks before a single multi-seed run.**

**Pre-check 1 — does the latent even cluster?** Yes, strongly. On real jumpy CheetahRun latents the
2-D vs 1-D structural-entropy gap is **53%** on the k-step transition graph (47% via kNN) — *provided*
you sparsify the graph first (the raw SimNorm transition graph is a dense "blob" at ~0%). So the
abstraction exists. Green light to the real test.

**Pre-check 2 (the kill-test) — does that structure track where the model is actually wrong?** This is
the load-bearing question, and the answer was **no**. The community-boundary score does not correlate
with the jumpy model's k-step prediction error (Spearman \(+0.09\) on Panda, \(-0.18\) on Cartpole).
Digging in, the reason is decisive: **the k-step error is small and nearly uniform** — there are no
"hard regions." We even tried to salvage it as an uncertainty-gated horizon: an ensemble-free
disagreement signal (jumpy-prediction vs iterated-one-step) turned out to be a *great* error proxy
(Spearman \(+0.72\)) — but under MPPI-scale action perturbations the error barely moves (inflation
\(1.06\times\)). The jumpy model is **uniformly accurate**, in-distribution and out. Which is exactly
*why* fixed-\(k\) jumpy already works — and why adaptive jump-length has nothing to adapt to.

So the structural-entropy lever, and the whole adaptive-\(k\) family with it, goes in the mirage table
next to its ancestors. The difference from Part 1: this time the negative cost an **afternoon of latent
dumps**, not a multi-week seed campaign. Mechanism-check first, fan-out second — that's the discipline
the whole project is really about. (Silver lining: that jumpy-vs-iterated-one-step disagreement is a
validated, ensemble-free uncertainty signal — useful elsewhere, just not for gating an already-uniform
horizon.)

## 6. The ledger: what didn't work, what's promising, what to probe next

Stepping back over the whole campaign, here is the honest accounting.

**What did NOT work (nulls, in order of how thoroughly):**
1. Geometric / behavioral latent clustering (structural-entropy Glass) — redundant with SimNorm.
2. Bisimulation auxiliary — actively hurts; brittle (failed twice for us).
3. Distractor-robustness from reward-grounded abstraction — falsified.
4. Sparse-task rescue from latent grouping — it's an exploration problem, not geometry.
5. Laplacian / eigenpurpose exploration — a generic RND bonus beats it.
6. Community-detection *skills* — communities are motion phases, not reachable subgoals.
7. `rho` consistency-horizon schedule — a task-dependent tuning knob, not architecture.
8. **SE-k adaptive jump-length** and **uncertainty-gated horizon (F)** — the jumpy model is uniformly
   accurate, so there's nothing to gate; killed by mechanism-check before any campaign.

**What DID work:**
- The **jumpy (k-step) world model** beats vanilla TD-MPC2 on PandaPickCube manipulation, n=5, peak
  +44% / final +88%, CI-separated. (Honest caveat: a *known* method, fairly evaluated — not our invention.)
- The **methodology**: peak-AND-final reporting + pre-registered, CI-separated gates + mechanism-checks
  before fan-out caught every mirage, several of which looked publishable at an interim snapshot.

**What's PROMISING (untested, not killed by the uniform-error finding):**
- **Hermite-spline action bottleneck** — parametrize macro-actions as smooth cubic splines (target
  joint pos/vel) + a PD tracker; shrinks the MPPI search space (\(k\cdot d \to 2d\)) with no learned
  codec (so no online representation drift). Doesn't depend on error-variance, so this verdict doesn't
  touch it.
- **Value-equivalent macro head** — train \(d_k\) to be *return*-equivalent over k steps (predict the
  same macro-\(Q\)) rather than state-faithful, so the abstraction keeps only what matters for control.
- **The disagreement signal we found** (jumpy vs iterated-one-step, Spearman \(0.72\) vs true error)
  — reusable for exploration bonuses or safe/abstained planning, just not horizon-gating.
- **High-DoF, done right** — our Humanoid probe floored at 500k (it needs millions of steps); a proper
  high-DoF run is the place the literature says abstraction headroom actually lives.

**Next probes, in priority order:**
1. **Hermite-spline action bottleneck** — mechanism-check first (does spline-restricting actions keep
   the achievable-return envelope?), then the pre-registered beat-jumpy gate. Highest novelty-per-risk.
2. **Value-equivalent macro head** — single-variable loss change on the existing jumpy head; clean test
   of "abstract what matters for control, not the full state."
3. **Reuse the disagreement signal** for an exploration/abstention probe (cheap, orthogonal).
4. **Humanoid/Dog at a real budget** (millions of steps) to settle whether jumpy's win extends to
   high-DoF — only worth it with the compute to do it honestly.

The thread through all of it: a strong latent world model is a *high bar*, and most "abstraction"
ideas are redundant with what it already learns. The wins, when they come, will be small, specific, and
only believable behind a pre-registered, peak-and-final, mechanism-checked gate.

---

*Reproducibility: code in `helios-rl` (jumpy heads + `make_jumpy_mppi_fn` in
`src/helios/algorithms/tdmpc2.py`; SE pre-check + mechanism-check in `scripts/se_precheck.py`); per-run
CSVs under `exp/tdmpc_glass/`; iteration records in `docs/iterations/` (iter-22 = jumpy, iter-23 = the
SE-k null); campaign capstone in `docs/writeup/capstone.md`. All numbers are read from run CSVs, not
notebooks — verification discipline, the hard way.*
