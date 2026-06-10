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

## 0. First, the bottom line: TD-MPC-Glass does **not** beat TD-MPC2 (recap of the last post)

The [previous post](/projects/2026-05-27-tdmpc-glass-iterations-8-9/) ended on a hopeful note — "Glass
off-at-1M beats our internal TD-MPC2 mean on HopperHop, 5-seed confirmations continuing." Here is the
honest closure of that thread: **under a fair, single-variable, adequately-powered protocol, Glass does
not beat TD-MPC2.** Per-task-normalized IQM over five DMC tasks, with 95% bootstrap CIs:

![TD-MPC-Glass does not beat TD-MPC2: IQM ± 95% CI, all three arms overlap](/images/glass_vs_tdmpc2_null.png)

- vanilla TD-MPC2: **0.950** [0.922, 0.968] (n=37)
- geometric Glass: **0.940** [0.851, 0.992] (n=16)
- behavioral Glass: **0.970** [0.952, 0.984] (n=34)

All three intervals **overlap** — no arm is statistically separated from vanilla. The behavioral arm's
+0.02 nominal edge sits inside the noise (and, as Part 1's §5 showed, its per-snapshot estimate wandered
to *both* sides of baseline as seeds accumulated). The hopeful HopperHop "win" was **basin-lottery**: under
the clean protocol neither Glass nor vanilla entered the high-reward basin (best 323 vs 286), so the earlier
edge came from restarts/seed-luck, not the representation.

**Why it didn't work — in one line:** TD-MPC2's **SimNorm latent is already a soft-clustering** (it softmaxes
V latent groups → V soft codebooks), and the self-predictive loss already makes the dynamics coherent — so
Glass's structural-entropy clustering mostly **re-derives structure the model already has**. We later
*measured* this directly (the latent shows a 53% structural-entropy community gap with **no** added
objective). The non-redundant part of Glass (its transition-graph entropy) captures real **motion-phase**
structure that turns out to be irrelevant to control. Full retrospective:
`docs/analysis/why-glass-failed-simnorm-redundancy.md`. Tell-tale sign we should have read sooner: the best
Glass recipe was to *turn Glass off* at 1M — a method whose optimal schedule is "use it then remove it" is
quietly reporting **asymptotic value ≤ 0**.

The rest of this post is what we did *after* accepting that: the fair-protocol campaign, the one real
(borrowed) win, the methodology, and four further novel bets that also came back null.

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

| metric | jumpy (n=7) | vanilla-H3 (n=6) | diff (paired bootstrap) | 95% CI |
|---|---|---|---|---|
| **peak** | 3226 | 2233 | **+992 (+44%)** | **[790, 1230]** |
| **final** | 2494 | 1388 | **+1106 (+80%)** | **[599, 1593]** |

Both metrics separate (updated to the **full n=7** as all seeds matured; the win held). The peak gap is
the clean "plans better" claim; the larger final gap is a **stability** finding — vanilla TD-MPC2 itself
collapses peak→final on Panda (−35%), and the jumpy model resists that late collapse.

![Jumpy world model vs vanilla TD-MPC2 on PandaPickCube, mean ± 95% bootstrap CI over seeds](/images/jumpy_panda_ci.png)

*A reporting nuance worth being precise about: the shaded bands above are each arm's **marginal** 95% CI,
and they lightly **touch** at the final step. The number that is CI-separated is the **paired difference**
(jumpy−vanilla per the matched bootstrap: +992 peak, +1106 final, both >0 with margin) — the paired test
is the correct, more powerful one for a head-to-head. We show the marginal bands anyway, because hiding
the overlap would be exactly the kind of thing this whole project is against.*

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

## 7. The full post-jumpy sweep: four novel levers, four (near-)nulls — fast

Treating the jumpy model as a *substrate*, we tried to beat **it** with a genuinely new abstraction.
Every lever got a cheap mechanism-check before any multi-seed spend. The scoreboard:

| iter | lever | mechanism-check | verdict |
|---|---|---|---|
| 23 | **SE-k**: structural-entropy adaptive jump-length | boundary score vs k-step error: Spearman \(+0.09\)/\(-0.18\) | **null** — boundaries don't mark where the model errs |
| 23 | **F**: uncertainty-gated horizon | disagreement tracks error (Spearman \(0.72\)) **but** error is uniform under MPPI perturbation (inflation \(1.06\times\)) | **null** — nothing to gate; jumpy is uniformly accurate |
| 24 | **SI2E / VCSE + `wmsi2e`** (SE-exploration over the world-model latent) | full sparse-task gate, 75 runs | **null/negative** — no rescue; at coef 1.0 the bonus *hurts* (collapses even on the easy task); the WM-latent novelty adds nothing |
| 25 | **Hermite-spline action bottleneck** | spline fit to action windows: needs K=4 for R²≈0.8 (only 2× compression) | **lean-negative, low-EV** — the open-loop proxy is invalid for a closed-loop planner; premise (big search cut) fails |
| 26 | **value-equivalent macro head** (train \(d_k\) to preserve *value*, not state) | trains clean; gate (distractor tasks) **running** | TBD |

The unifying empirical fact: **the jumpy model is already uniformly accurate over the states a good
policy visits**, and **TD-MPC2's MPPI is already policy-prior-warm-started** — so adaptive-k, smarter
search distributions, and exploration bonuses each have little to grab onto. Five distinct "abstraction
beats it" ideas, and the model keeps not needing them.

## 8. SimNorm's hidden structure — what structural entropy actually says

Our oldest idea (Glass) clustered TD-MPC2's latent with a structural-entropy objective and it was a null
(§2, mirage #1). Iter-23's pre-check asked *why*, directly: **does the SimNorm latent even have
exploitable community structure?** Answer, measured on a trained jumpy model's latent transition graph:

- **Raw transition graph: ~0% structural-entropy gap — a dense "blob"** (40–76% edge density). SimNorm's
  bounded simplex latents are *diffuse*; naive clustering sees one cluster.
- **After top-fraction sparsification / kNN, with an SE-optimal partition: 53% gap on the k-step
  transition graph, 47% on the kNN geometry graph.** So the structure is *real and strong* — once you
  build the graph right.

The twist is the punchline of the whole project: **the structure is real but not useful for control.**
- Iter-19's communities were *motion phases*, not reachable subgoals → useless as skills.
- Iter-23's boundaries didn't coincide with model-error regions → useless for adaptive-k.
- Iter-24's SE-coverage didn't beat generic novelty → useless for exploration.

Why? SimNorm is *already* a soft-categorical code (V softmax groups). It hands the model a built-in
abstraction, and the self-predictive objective already extracts what's control-relevant
([Ni et al., 2024](https://arxiv.org/abs/2401.08898): TD-MPC2's loss is a *sufficient* abstraction). So
re-clustering it, or steering exploration/jump-length by its communities, is **redundant**: you're
re-deriving structure the policy/value heads already exploit.

**Future research this opens (the honest, interesting part):**
1. **SE-*shaped* SimNorm, not SE-*read* SimNorm.** We only ever *read* structure out of a fixed SimNorm
   latent. An encoder loss that *minimizes* the 2-D structural entropy of the latent transition graph
   would *create* crisp dynamics-communities. The caveat: this is geoglass-adjacent (a null for dense
   return), so it must target *planning amenability* (cleaner macro-abstraction), not raw return, and be
   tested where temporal abstraction pays (high-DoF, phase-rich locomotion).
2. **Structured SimNorm.** SimNorm's V groups are arbitrary fixed slices; let an SE objective *learn*
   which latent dims group together from transition structure — a latent whose categorical groups *are*
   dynamics regimes.
3. **SE as an analysis/interpretability tool, not a controller.** The 53%-gap communities are a clean
   lens on *what a world model represents* (motion phases, contact regimes). That may be more publishable
   as **understanding** than as a performance lever — "what structural entropy reveals about learned
   world-model latents."
4. **The redundancy question itself.** *Quantify* how much of SimNorm's structure the value/policy heads
   already use (e.g. mutual information between SE-communities and Q-relevant directions). A crisp
   "abstraction is redundant with a sufficient world model — here's the measurement" is a real result.

## 9. Our delta vs *Compositional Planning with Jumpy World Models* (2026)

To be exact about credit and contribution:

| | Farebrother et al. 2026 | This work |
|---|---|---|
| multi-step ("jumpy") model | ✅ (the originators) | reused |
| cross-timescale consistency | ✅ (Temporal-Difference Flows) | reused (horizon-consistency form) |
| **use** | zero-shot **composition of pre-trained policies** (occupancy models) | **online TD-MPC2**, jumpy head **planned by macro-MPPI** |
| **our additions** | — | fair single-variable head-to-head vs vanilla (peak+final, CI); the mechanism-check-before-fanout methodology; a catalog of what does **not** beat a strong WM |

So the **jumpy mechanism is theirs**; our contribution is its **online model-predictive-control
instantiation + a rigorous evaluation/negative-results methodology**, not a new model class. We say this
plainly because the alternative — quietly implying the jumpy win is ours — is exactly the failure mode
this project exists to avoid.

## 10. Is this publishable?

Honestly:

- **As a "novel method beats SOTA" paper — no.** The one win is a known method; every genuinely novel
  lever we tried is a null. Main-track novelty isn't there.
- **As a negative-results / reproducibility / methodology paper — yes, plausibly**, at a workshop (e.g.
  ICLR/NeurIPS "I Can't Believe It's Not Better", reproducibility, or negative-results tracks) or as a
  tech report. The contribution is: (a) a **multi-mechanism fair-protocol confirmation** that explicit
  abstraction is redundant with a strong self-predictive world model (~13 mechanisms, one consistent
  story); (b) the **methodology** — peak-AND-final reporting, the small-sample "mirage" anatomy, and
  **mechanism-check-before-fanout** that killed four levers in afternoons instead of multi-week
  campaigns; (c) the **SimNorm structural-entropy analysis** (real structure, not useful for control).
- **The single most transferable piece** is the methodology + the SimNorm-already-sufficient finding —
  useful to anyone tempted to bolt abstraction onto TD-MPC2/Dreamer.

Bottom line: a credible, honest **negative-results-with-methodology** paper, not a SOTA claim. Whether
that's worth writing up depends on appetite for the negative-results genre — but the data and the
discipline behind it are solid.

---

*Reproducibility: code in `helios-rl` (jumpy heads + `make_jumpy_mppi_fn` in
`src/helios/algorithms/tdmpc2.py`; SE pre-check + mechanism-check in `scripts/se_precheck.py`); per-run
CSVs under `exp/tdmpc_glass/`; iteration records in `docs/iterations/` (iter-22 jumpy, iter-23 adaptive-k null, iter-24 SE-exploration
null, iter-25 spline, iter-26 value-equiv); SE pre-check `scripts/se_precheck.py`; CI figure from per-seed CSVs; campaign capstone in `docs/writeup/capstone.md`. All numbers are read from run CSVs, not
notebooks — verification discipline, the hard way.*
