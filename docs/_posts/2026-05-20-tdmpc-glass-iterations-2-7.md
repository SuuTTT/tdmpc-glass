---
layout: post
title: "TD-MPC-Glass Iterations 2–7: Basin Lottery, Glass Internals, and the K_UPDATE Hypothesis"
date: 2026-05-20
description: "Seven iterations of TD-MPC-Glass on HopperHop: what the μ/S/c vectors actually do, what the video rollouts revealed about cluster→gait alignment, why 25 phases failed to beat vanilla TD-MPC2, and why K_UPDATE=64 may be the root cause we ignored for two weeks."
---

> This post is the sequel to [Phase 1b](https://suuttt.github.io/projects/2026-05-13-tdmpc-glass-phase1b/).
> It covers seven days of iteration — ~25 experimental phases, 8 GPUs, two
> goals that still aren't both solved — and ends with the current best hypothesis:
> **we've been training at 4× too low a gradient-update rate the entire time.**
> Live dashboard: [bus-brussels-fate-performed.trycloudflare.com](https://bus-brussels-fate-performed.trycloudflare.com/)

---

## 1. Two goals

We track every run against two success criteria:

| ID | Name | Criterion | Status |
|----|------|-----------|--------|
| **G1** | Consistency | *All 5* seeds reach best MPPI ≥ 500 | Never achieved. Best: 2/5 |
| **G2** | Ceiling | *At least one* seed reaches MPPI ≥ 600 | Phase-t (knee-penalty shaping) hit 612 — but benchmark-unfair |

G1 is the hard goal. It requires that the agent reliably escapes the gait basin
lottery on *every* seed — not just the lucky ones. G2 only needs one seed to
push the upper limit; it confirms the physical ceiling exists. The benchmark-fair
objective is G1 without reward shaping, behaviour cloning, or environment edits.

---

## 2. TD-MPC-Glass internals: what μ, S, and c actually do

We explain the design top-down: start from *what we want*, then show how each
component is introduced to make it achievable.

### 2.1 The goal: a transition graph over behavioural states

The core idea behind Glass is simple: **if we could draw a graph whose nodes
are behavioural states and whose edges represent how often the agent moves
between them, we could measure and minimise the structural entropy of that
graph.** Minimising structural entropy pushes the graph toward a clear modular
structure — a few tightly-connected clusters with sparse edges between them.
For hopping, the ideal graph would be a clean 4-node cycle:
push-off → takeoff → flight → landing → push-off.

The challenge: RL observations are raw 15-dim sensor readings. There are no
predefined "behavioural states" — we have to learn them. Everything that follows
is the machinery for doing that.

### 2.2 Prototypes μ — what an "anchor" means in deep learning

\(\mu \in \mathbb{R}^{N \times d}\), \(N=16\), \(d=512\). Each row \(\mu_n\) is
a learnable vector in the latent space — a fixed reference point we call a
**prototype** or **anchor**.

The word *anchor* has a consistent meaning across several DL contexts:

- **Contrastive learning (SimCLR, MoCo)**: the anchor is the query sample;
  other embeddings are pulled toward it (positives) or pushed away (negatives).
- **Prototype networks (ProtoNets, SwAV, DINO)**: anchors are cluster centroids
  — fixed or slowly-moving reference points in embedding space that the rest of
  the data is organised around. A new datapoint is described by its distance to
  each anchor rather than by its raw coordinates.
- **Object detection (YOLO, Faster-RCNN)**: anchors are predefined bounding-box
  shapes; detection is formulated as residuals from these anchors.

In all three cases the intuition is the same: **replace an unconstrained
continuous space with "how close am I to each reference point?"** — a much
lower-dimensional and more structured description.

In Glass, the anchors \(\mu_1 \ldots \mu_{16}\) are the behavioural reference
points. After training, one anchor might point in the latent direction of "leg
fully extended mid-hop," another toward "knee on ground recovering from a fall."
Every current observation is then described as a soft mixture over these 16
anchors via the assignment vector \(c_t\) — and *that* compact description is
what we use to build the transition graph.

**Why N=16 anchors instead of K=8 clusters directly?** Two reasons:

1. *Graph size.* Building a transition graph directly over data points requires
   a B×B matrix per batch (B=256) — \(O(B^2)\) per step. With N=16 anchors
   the graph is always 16×16 — constant cost, 256× smaller.

2. *Parameter stability.* The mapping from anchors to clusters (matrix S,
   described below) is a learnable parameter, not a function of the current
   batch. The cluster partition is therefore consistent across training steps
   and identifiable post-hoc. This is why you can watch a rollout video and
   see a stable `K=` label per frame.

### 2.3 The latent z and SimNorm(V=8)

The TD-MPC2 encoder maps every observation \(o_t\) to a 512-dim latent \(z_t\)
using **SimNorm(V=8)** as the final activation.

Standard normalisation choices and why they don't quite work here:
- *BatchNorm / LayerNorm*: normalise to zero mean and unit variance — addresses
  covariate shift but the latent can still collapse to a low-rank subspace
  (well-documented in RL: [Nikishin et al., 2022](https://arxiv.org/abs/2205.07802)).
- *L2 normalisation*: projects everything to the unit sphere — used in
  contrastive learning but loses magnitude information and can saturate gradients
  through long dynamics rollouts.

SimNorm's approach: **split the 512 dimensions into V=8 groups of 64, then apply
softmax within each group.** Each group becomes a proper probability distribution
over 64 "codes." The result is that \(z\) looks like 8 concatenated soft
histograms:

\[
z = \bigl[\,\text{softmax}(z^{(1)})\,,\; \text{softmax}(z^{(2)})\,,\; \ldots\,,\; \text{softmax}(z^{(8)})\,\bigr]
\]

**Why this helps:**
1. *No collapse.* Softmax is bounded in \((0,1)\) and sums to 1 per group — the
   representation can never degenerate to a zero vector or collapse to a single
   active dimension, which kills RL training.
2. *Well-defined cosine similarity.* Because every group is on the probability
   simplex, the inner product \(z \cdot \mu^\top\) is a sum of 8 group-wise dot
   products with a consistent scale. Cosine similarity to the anchor prototypes
   is well-behaved from the first step.
3. *Implicit group structure.* Each of the 8 groups can specialise
   independently — one group might encode body orientation, another joint
   velocities, another contact state — without any explicit supervision.

### 2.4 Soft assignment c — mapping latents to anchors

Given the current latent \(z_t\) and the N=16 anchor prototypes \(\mu\), the
soft assignment vector \(c_t \in \Delta^{N-1}\) is:

\[
c_{t,n} = \frac{\exp\!\bigl(\hat{z}_t \cdot \hat{\mu}_n / \tau_p\bigr)}{\sum_{n'} \exp\!\bigl(\hat{z}_t \cdot \hat{\mu}_{n'} / \tau_p\bigr)}
\]

where \(\hat{z}, \hat{\mu}\) denote unit-normalised vectors (cosine similarity),
and \(\tau_p = 0.2\) is the temperature. A low temperature makes \(c_t\) sharply
peaked: one anchor dominates, and the assignment behaves like a soft nearest-
neighbour lookup. The entry \(c_{t,n}\) answers: *how much does the current
observation resemble anchor \(\mu_n\)?*

### 2.5 Building the transition graph A

Now we have a compact description of each state: instead of a raw 15-dim
observation we have a 16-dim soft assignment vector. We accumulate these over
a replay batch of B=256 consecutive transition pairs \((z_t, z_{t+1})\):

\[
P_{\text{counts}} = \sum_{t=1}^{B} c_t \otimes c_{t+1} \in \mathbb{R}^{N \times N}
\]

The outer product \(c_t \otimes c_{t+1}\) distributes the probability mass of
the current state across all anchor pairs. Row-normalise and symmetrise:

\[
P = \text{row\_norm}(P_{\text{counts}} + \epsilon), \qquad
A = \tfrac{1}{2}(P + P^\top)
\]

**A is the prototype transition graph** we set out to build in §2.1. Entry
\(A_{mn}\) records how often the agent moves from a state near anchor \(\mu_m\)
to one near anchor \(\mu_n\). For a good hopper this graph should look like a
sparse 4-cycle; for a kneeling policy it looks like a 2-node oscillator.

### 2.6 Cluster assignment S — coarsening anchors into behavioural phases

N=16 anchors is more granularity than needed for a 4-phase gait. We coarsen them
into K=8 clusters via a learnable matrix:

\[
S = \text{softmax}(\text{assign\_logits},\; \text{axis}=1) \in \mathbb{R}^{N \times K}
\]

\(S_{nk}\) is the probability that anchor \(n\) belongs to cluster \(k\). In
well-trained Glass, each row is near one-hot. This is a two-level hierarchy:
anchors capture fine-grained variation (is the knee slightly more bent here?),
clusters capture coarse behavioural phases (is this "stance" or "flight"?).

### 2.7 Structural entropy loss

With the transition graph A and the cluster partition S in hand, Glass minimises
the **2D structural entropy**:

\[
H^2(A;\,S) = -\sum_{k=1}^{K} p_{\text{cut},k} \log p_{\text{vol},k}
           + H^1(A) + \sum_{k=1}^K p_{\text{vol},k} \log p_{\text{vol},k}
\]

where \(p_{\text{vol},k} = V_k / \text{vol}(A)\) is the volume fraction of
cluster \(k\) and \(p_{\text{cut},k} = g_k / \text{vol}(A)\) is the fraction
of edges that cross out of cluster \(k\). Minimising this pushes toward **small
boundary cuts** (transitions stay within clusters) and **balanced volumes** (no
dead clusters) — exactly the pressure to align clusters with gait phases.

The full Glass loss:

\[
\mathcal{L} = \mathcal{L}_{\text{TD-MPC2}}
            + \lambda_{\text{SE}} \cdot H^2(A,S)
            + \lambda_{\text{bal}} \cdot (\mathcal{L}_{\text{cluster-bal}} + \mathcal{L}_{\text{proto-bal}})
            + \lambda_{\text{temp}} \cdot \mathcal{L}_{\text{temporal}}
\]

Defaults: \(\lambda_{\text{SE}} = 10^{-4}\), \(\lambda_{\text{bal}} = 10^{-3}\), \(\lambda_{\text{temp}} = 10^{-4}\).

### 2.8 The K= label in rollout videos

The overlay number on each rendered frame is:

```
argmax(S[argmax(z · μᵀ)])
```

1. Compute cosine similarities to each anchor: \(\text{sim}_n = \hat{z} \cdot \hat{\mu}_n\).
2. `n_star = argmax(sim)` — the nearest anchor.
3. `cluster_id = argmax(S[n_star])` — which cluster that anchor belongs to.

In plain English: *which behavioural cluster does the current observation most
resemble, according to the encoder and Glass partition?*

---

## 3. What the rollout videos revealed

We have video rollouts for Phase-f seeds 1, 3, 4 and the best-ever checkpoint
Phase-t seed 2 (MPPI 612). K= is overlaid per frame.

### Phase-t seed 2 — G2 winner (MPPI 612) — the most important video

This is the highest-scoring checkpoint we've ever produced. The K= behaviour is
strikingly simple:

| Episode phase | K= value |
|---------------|----------|
| Initial drop (hopper spawns and falls to ground) | K=A (one cluster) |
| **Entire steady-state hop — all jumps, all frames** | **K=B (one cluster, UNCHANGED)** |

Two clusters in total. K does **not** switch between push-off and takeoff, does
not switch mid-flight, does not flicker. The entire hopping motion — from the
first jump to the last — is described by a single stable cluster assignment.

This is the strongest video finding of the project. The best policy we have
ever produced uses the **simplest possible cluster structure**: one state for
"I am falling / recovering", one state for "I am hopping". No sub-phase
discrimination, no 4-state gait machine. Just a binary: falling vs. hopping.

### Phase-f seed 1 — secondary winner (MPPI 571)

The K= label alternates **K=3 → K=4 → K=3 → K=4** during the hop cycle, with
K=6 and K=7 appearing only during fall recovery. Within the steady hop,
K switches once per jump — K=3 during push-off, K=4 during takeoff.

| Cluster | Gait sub-phase |
|---------|---------------|
| K=3 | Hip leaves ground / push-off |
| K=4 | Leg extending, mid-takeoff |
| K=6 | Air-recovery (post-fall only) |
| K=7 | Ground-contact recovery (post-fall only) |

This is more granular than the 612 winner — 2 clusters within the hop, not 1.
And the score is lower (571 vs 612). **More cluster detail during hopping does
not improve performance.**

### Phase-f seed 3 — stuck (MPPI 262)

K cycles **2 → 4 → 5 → 1 → 4 → 2** with 5 distinct clusters active. The
behaviour is a knee-walk with nose dragging the ground. Glass found a rich
partition of the kneeling gait — but a rich partition of the wrong gait is useless.

### Phase-f seed 4 — stuck (MPPI 266)

Only 2 clusters active (K=0 ↔ K=5), oscillating once per knee-walk cycle. This
looks superficially like the 612 winner (2 clusters, stable oscillation) — but
the gait is completely different: "knee on ground" vs "knee-thrust off ground",
not "falling" vs "hopping". **The cluster count alone does not predict performance;
what the clusters represent does.**

### The revised key insight: stability, not count

Comparing all four rollouts side by side:

| Checkpoint | MPPI | Active K | K during hop | Verdict |
|------------|------|----------|--------------|---------|
| Phase-t s2 | **612** | 2 | **One stable K throughout** | G2 winner |
| Phase-f s1 | 571 | 4–5 | K=3/K=4 alternating per sub-phase | Winner |
| Phase-f s4 | 266 | 2 | K=0/K=5 oscillating (knee-walk) | Stuck |
| Phase-f s3 | 262 | 5 | Rapid switching (knee-walk) | Stuck |

The ranking by MPPI is **inversely correlated with cluster complexity during
the active gait phase.** The 612 winner uses 1 cluster for hopping; the 571
winner uses 2; the stuck seeds use 2–5 for kneeling.

**The "K=4 basin is needed for hopping" hypothesis is falsified.** The
best-ever checkpoint uses 2 clusters total. The K=3 vs K=4 basin distinction
we tracked across iterations was measuring cluster *count* — but the real
predictor is cluster *stability within the active gait phase*.

**What the stuck seeds lack is not cluster count — it is that their cluster
assignment oscillates within a single gait phase.** Seed 3 switches K five times
per knee-walk cycle; seed 4 switches once per cycle but for the wrong gait.
Neither case is "stable on hopping", because neither is hopping.

**The core failure mode:** Glass's SE loss minimises boundary cuts in the
*aggregate* transition graph, which encourages Markovian cluster structure on
average. But it does not enforce that **the same cluster fires throughout a
repeated gait phase** across time. The 612 winner learned this temporal stability
by accident (or through the knee-penalty shaping gradient). The stuck seeds
never did.

---

## 4. Iteration history — all 25 phases

### Iteration 1–2 (Phases 1b, 1c, d–h)

| Phase | Change | Best | Verdict |
|-------|--------|------|---------|
| 1b | TD-MPC-Glass default | 562 s5 | Baseline: 2/5 > 500. Seed lottery. |
| 1c | Act-noise anneal 0.30→0.10 | 412 | FALSIFIED: hurts winners |
| d_v1 | H=5 + noise=0.40 | 114 | FALSIFIED: Warp 901 crash |
| d_v2 | H=5 only | 199 | FALSIFIED: no help |
| e | Q-reset | 228 then collapse | FALSIFIED: impl bug, zeroed all opt state |
| **f** | latent_smooth=1e-3 | **571 s1** | First >500 ever; 1/5 survived |
| g | consistency_coef=1.0 | 482 s2 | Partial lift; caps below 500 |
| h | smooth+ccoef combined | 490 | No additive benefit |

**Key finding (iter 2):** latent action smoothing 1e-3 flipped seed-1's gait
from knee-walk to foot-hop, producing the first 571 winner. But 4/5 seeds didn't
make the same flip.

### Iteration 3 (Phases i–n)

| Phase | Change | Best | Verdict |
|-------|--------|------|---------|
| i | smooth=1e-4 (too weak) | 312 | FALSIFIED: below threshold |
| **j** | curriculum smoothing (off<250k) | **518 s2** | 1/5 > 500; curriculum helps |
| k | smooth + λ_temporal=0.05 | 292 | FALSIFIED: over-regularised |
| l_v1 | tdmpc2 + smooth, Glass OFF | 289 | FALSIFIED: Glass IS needed |
| m | basin perturbation (param noise) | 286 | Doesn't help |
| n | basin perturbation v2 | 56 | FALSIFIED |
| **o** | Glass OFF after 2M (hybrid) | **577 s3** | 1/3 > 500; Glass helps early |

**Key finding (iter 3):** K=3 basin is a structural cap (~310 average) — 4-phase
gait needs K=4. Curriculum smoothing (Phase-j) rescues some seeds by letting the
basin lock *before* smoothing engages.

### Iteration 4 (Phases p, t, and hierarchy)

| Phase | Change | Best | Verdict |
|-------|--------|------|---------|
| **p** | EXPL_UNTIL=500k (wider random phase) | **538 s4** | 1/3 > 500; slow-burn winner |
| **t** | Knee penalty reward shaping | **612 s2** | G2 hit! 2/3 > 500. Benchmark-unfair |
| y | Hierarchical Glass K_super=4 | 462 | 0/3 > 500; close but no cigar |

Phase-t confirmed the physical ceiling (612) but used reward shaping. Reward
shaping is excluded from the benchmark-fair objective.

### Iteration 5 (Paths 7, 9, 10 and intrinsic rewards)

| Phase | Change | Best | Verdict |
|-------|--------|------|---------|
| v | Cluster soft-dist as pi/q obs (Path 7) | 232 | FALSIFIED: representation drift |
| **x** | NS=2048 MPPI planner | **523 s3** | ~40% hit rate; planner scale helps |
| y | Hierarchical Glass K_super=4 (Path 10) | 462 | 0/3 > 500 |
| P | Cluster entropy intrinsic reward | 91 then 2.4 | FALSIFIED: non-stationary |
| Pa | Same with linear decay 0.1→0 | 25 | FALSIFIED: 3.6× worse |

**Key finding (iter 5):** Planner scale (NS=2048) raises good-seed ceiling but
doesn't rescue stuck seeds. Cluster entropy intrinsic reward is fundamentally
incompatible with the reward scale (max bonus ≈ 210, target ≈ 600).

### Iteration 6 — reward shaping audit (concluded)

We ran a systematic audit of benchmark-*unfair* reward shaping to understand the
ceiling, then studied whether stacking helps.

| Phase | Config | Seeds | G1 hit | Best |
|-------|--------|-------|--------|------|
| phasez | Vanilla tdmpc2 baseline (NS=512) | 5 | 1/5 | ~500 |
| phaseq_knee | Knee penalty + NS=2048 | 12 | 4/12 (33%) | 557 |
| phaser1_soft | Soft stand bonus 0.1→0@3M | 5 | 1/5 (20%) | 553 |
| phaser2_gait | Gait fall penalty + action smooth | 4 | 1/4 (25%) | 510 |
| **phaserstack** | **ALL shaping stacked** | 3 | **0/3 (0%)** | **5–16 COLLAPSED** |
| phaserstack_nosmooth | Stack − action_smooth | 2 | 0/2 | 10.2 |
| phaserstack_nosoft | Stack − soft_stand_bonus | 2 | 0/2 | 254 |

**Key finding (iter 6):** The stack collapse is **combinatorial**, not attributable
to any single component. Dropping either ablated component still mostly collapsed.
Individual shaping raises hit rate to 20–33% but no individual component gets 5/5.
Also: **Glass never clearly beat vanilla TD-MPC2** — phasez already produced one
seed above 500 with no Glass at all.

### Iteration 7 — K_UPDATE audit (in progress)

The strongest unresolved benchmark-fair lever: **gradient update rate.**

Current codebase: `K_UPDATE=64` → 64 gradient updates per 256-env-step collection
batch → ~0.25 updates per env step. Official TD-MPC2: ~1.0 updates per env step.
We've been training at **4× lower update rate for 7 iterations.**

Phase-aa sweep: vanilla tdmpc2, NS=2048, K_UPDATE ∈ {64, 128, 256}, seeds 1–3:

| K_UPDATE | Seed 1 | Seed 2 | Signal |
|----------|--------|--------|--------|
| 64 | 234 @ 3.25M | running | Weak |
| **128** | **538.6 G1** | 285 | **Strong** |
| **256** | **531.0 G1** | 324 | **Strong** |

k64 seed-1 is stuck at 234 — the same basin-lottery failure mode we've seen for
7 iterations. k128 and k256 both escaped it in seed-1. **This is the first time
any benchmark-fair change has rescued a seed that k64 gets stuck on.**

---

## 5. Why Glass didn't help (yet) and what to try next

### 5.1 What Glass does accomplish

Glass does find meaningful partitions. The video analysis confirms:
- Winner seeds: 4–5 active clusters, stable K= during hop phases, partition maps
  cleanly to push-off / takeoff / air / recovery.
- Stuck seeds: 4–5 active clusters too, but the partition describes the *kneeling*
  gait, not the hopping gait.

Glass is doing structurally correct representation learning. The problem is that
**a good representation of the wrong gait is useless.**

### 5.2 The stability problem — and why cluster count is a red herring

The video analysis (§3) forces a revision of our working hypothesis.

**Old hypothesis**: K=4 basin → good (4 gait phases → 4 clusters → hopping).
K=3 basin → structural cap (~310 average).

**New evidence**: the 612 winner (Phase-t s2) uses **2 clusters total** — one
for falling, one for the entire hop. It doesn't distinguish push-off from
takeoff, flight from landing. The 571 winner (Phase-f s1) uses 2 clusters
*within* the hop (K=3 push-off, K=4 takeoff). The 612 winner is more abstract
and scores higher.

**Revised diagnosis**: what distinguishes winners from stuck seeds is not the
number of active clusters but **whether the cluster assignment is stable
throughout the active gait phase**. The 612 winner has K frozen on a single
value for every frame of every hop. The stuck seeds have K oscillating within
a single kneeling cycle.

This reframes the problem entirely. We don't need Glass to learn a 4-state gait
machine. We need it to learn a **stable binary abstraction**: "hopping" and
"not hopping". Any number of clusters works, as long as the cluster that fires
during the hop phase stays fixed across all frames of all hops.

Glass's structural entropy loss minimises boundary cuts in the *aggregate*
transition graph — it encourages Markovian cluster structure on average over
a replay batch. But it does not enforce that **the same cluster fires
consistently throughout a single repeated gait phase across time**. A latent
can visit prototype A during one push-off and prototype B during the next
push-off, and the SE loss will not penalise this as long as both A and B belong
to the same cluster in S. The 612 winner achieved stability without this
being explicitly trained — the stuck seeds did not.

### 5.3 Proposed Glass integration directions

**Direction 1: Temporal consistency on cluster assignments (not latents)**

Currently `λ_temporal` penalises \(\|z_t - z_{t+1}\|^2\) — smoothness in the
latent space. Instead, penalise entropy of the assignment distribution over a
short window:

\[
\mathcal{L}_{\text{assign-consistency}} = \mathbb{E}_t\!\left[H\!\left(\frac{1}{W}\sum_{\tau=t}^{t+W} c_\tau\right) - \frac{1}{W}\sum_{\tau=t}^{t+W} H(c_\tau)\right]
\]

This is the mutual information between time index and prototype assignment within
a window of length W. Minimising it encourages the same prototype to be active
throughout a gait phase, without constraining the *values* of the latent.

**Direction 2: Phase-locked prototypes via temporal contrastive loss**

Positive pairs: two frames within the same gait phase (e.g., consecutive frames
during steady-state hop). Negative pairs: frames from different gait phases. Use
NT-Xent on the soft assignment vector \(c_t\) rather than on z:

\[
\mathcal{L}_{\text{phase-contrastive}} = -\log \frac{\exp(c_t \cdot c_{t'}^+/\tau)}{\exp(c_t \cdot c_{t'}^+/\tau) + \sum_{j} \exp(c_t \cdot c_j^-/\tau)}
\]

This directly trains the prototype assignments to be consistent within a phase
and discriminative across phases — exactly the property the winning seed exhibits
but the stuck seeds don't.

**Direction 3: Coarser cluster count — but matched to stability, not gait sub-phases**

The 612 winner used 2 stable clusters (fall / hop). Reducing to N=4, K=2 would
force Glass toward exactly this binary abstraction. This is a *different*
motivation from matching "4 gait sub-phases" — the video evidence shows the
best result doesn't distinguish sub-phases at all. Reducing K=8 → K=2 or K=3
removes the capacity for fine-grained sub-phase tracking, which has not helped
and may be actively harmful by giving the SE loss too many degrees of freedom
to fill with arbitrary, low-stability partitions.

**Direction 4: Prototype-conditioned planning in MPPI**

Currently MPPI is cluster-unaware — it samples action sequences and evaluates
them with the same Q ensemble regardless of which cluster z is in. A lightweight
extension: learn cluster-specific Q offsets \(\Delta Q_k\) that capture
cluster-conditional value. Effectively this is a mixture-of-experts Q with K
components, using Glass to route.

**Direction 5: Fix K_UPDATE first, then re-evaluate Glass**

The most urgent question: does Glass add anything over vanilla TD-MPC2 *once the
training ratio is correct?* Phase-ac (K_UPDATE winner + 5-seed Glass vs. vanilla)
will answer this directly. If Glass is neutral at K_UPDATE=128, it means the
representation benefit was masked by under-training — and the architecture
changes above become much more interesting on a properly-trained base.

---

## 6. Infrastructure this week

- **8-GPU fleet**: Local 4070 Ti + ssh6 (4060 + 3080) + ssh17637 GPU0/1 (3060 Lap) + ssh1 (2080 Ti) + ssh3 (3070 + 3060 Ti)
- **Web dashboard** (port 5055): live learning curves, 95% CI bands per phase, multi-phase overlay with stable per-phase colors, canonical phase merging (e.g., phasex_local + phasex_4060 → phasex), comprehensive phase descriptions for all 40 phases
- **Auto-queue**: per-box queue files; idle boxes self-assign next experiment

---

## 7. What's running now

All 8 GPUs on Phase-aa K_UPDATE sweep. ETA: 2–3 days for seeds 2–3 to mature.
Next: pick K_UPDATE winner → Phase-ab (5-seed vanilla) → Phase-ac (Glass comparison).

If K_UPDATE fixes stuck-seed rate, the answer is: we were under-training for
7 iterations. The basin lottery may not be a lottery at all — just an artefact
of insufficient gradient updates in the first 500k steps.

---

## 8. Complete phase ledger

| Phase | Feature | Best MPPI | Hit rate |
|-------|---------|-----------|----------|
| 1b | TD-MPC-Glass default | 562 | 2/5 |
| 1c | Act-noise anneal | 412 | FALSIFIED |
| d_v2 | H=5 only | 199 | FALSIFIED |
| e | Q-reset (buggy) | 228 | FALSIFIED |
| f | smooth=1e-3 | **571** | 1/5 |
| g | consistency_coef=1.0 | 482 | 0/5 |
| h | smooth+ccoef | 490 | 0/2 |
| i | smooth=1e-4 | 312 | 0/1 |
| j | curriculum smooth | **518** | 1/5 |
| k | smooth+λ_temp=0.05 | 292 | 0/1 |
| l | tdmpc2+smooth (Glass OFF) | 289 | 0/1 |
| m | basin perturbation | 286 | 0/5 |
| n | basin perturb v2 | 56 | FALSIFIED |
| o | Glass OFF after 2M | **577** | 1/3 |
| p | EXPL_UNTIL=500k | **538** | 1/3 |
| t | Knee penalty (unfair) | **612** | 2/3 |
| v | Cluster obs (Path 7) | 232 | FALSIFIED |
| x | NS=2048 | **523** | ~40% |
| y | Hierarchical K_super=4 | 462 | 0/3 |
| P/Pa | Cluster entropy intrinsic | 91/25 | FALSIFIED |
| z | Vanilla tdmpc2 baseline | ~500 | 1/5 |
| phaseq_knee | Knee+NS=2048 | 557 | 4/12 |
| phaser1_soft | Soft stand bonus | 553 | 1/5 |
| phaser2_gait | Gait fall penalty | 510 | 1/4 |
| phaserstack | All shaping stacked | 16 | **COLLAPSED** |
| phaseaa k128 | K_UPDATE=128 (early) | **538** | promising |
| phaseaa k256 | K_UPDATE=256 (early) | **531** | promising |

---

## 9. Seed statistics: is seed 1/2 always best and seed 4 always worst?

A common intuition after watching many runs: "seed 1 and 2 always win, seed 4
always gets stuck." We ran the numbers across all 21 HopperHop phases with at
least two seeds present.

### 9.1 Aggregate per-seed statistics

| Seed | N phases | Mean MPPI | Median | Max | Min | Times best | Times worst |
|------|----------|-----------|--------|-----|-----|-----------|------------|
| 3 | 12 | **386** | **432** | 578 | 0.1 | 6 | 1 |
| 1 | 13 | 345 | 372 | 572 | 25 | 5 | 3 |
| 2 | 14 | 333 | 319 | **612** | 7 | **6** | **7** |
| 4 | 9 | 321 | 267 | 538 | 227 | 1 | 4 |
| 5 | 9 | 290 | 286 | 562 | 27 | 2 | 4 |

**Seed 3 is the most consistently good** (median 432, 5 times above 500, worst
only once). Seed 1 is a reliable mid-upper performer. The intuition about seeds
1/2 being best is half-right — but **seed 2 is simultaneously the most frequent
best (6 times) AND the most frequent worst (7 times).** It has by far the highest
variance: Phase-t seed 2 = 612 (G2 winner); Phase-Pa seed 2 = 7; Phase-x seed 2 = 6.
Seed 4 is genuinely below-median but seed 5 is comparably weak.

### 9.2 Per-phase breakdown

| Phase | s1 | s2 | s3 | s4 | s5 | Best seed |
|-------|----|----|----|----|----|----|
| baseline (pre) | 284 | 341 | **526** | 356 | 264 | s3 |
| phase1b | 526 | 526 | 294 | 227 | **562** | s5 |
| phasef (smooth) | **572** | 284 | 262 | 266 | 255 | s1 |
| phasej (curriculum) | 452 | **518** | 322 | 266 | 354 | s2 |
| phaseo (Glass-off-late) | — | — | **578** | 254 | 33 | s3 |
| phasep (EXPL=500k) | — | — | 197 | **538** | 27 | s4 |
| phaset (knee penalty) | 375 | **612** | 534 | — | — | s2 |
| phasez (vanilla baseline) | — | 268 | **535** | 448 | 467 | s3 |

No seed dominates across all phases. The same seed that wins in one phase
finishes last in another.

### 9.3 Why different seeds produce different outcomes

The JAX random seed controls three things simultaneously:

1. **Initial network weights** — all five networks (encoder, dynamics, reward, Q,
   policy) draw from different random initializations. Different loss surfaces at
   step 0 mean the early gradient steps point in different directions.

2. **Initial exploration trajectory** — the first 25k–500k env steps use random
   or near-random actions drawn from the seed's PRNG chain. Different seeds yield
   different state-action pairs in the replay buffer, different early experience
   coverage, and different first gradient targets.

3. **MPPI action noise sequences** — all planning rollouts sample action
   perturbations from the same seed. This affects which action sequences get
   evaluated during early policy improvement.

These three interact to determine **which gait basin the policy enters** in the
first ~200k env steps. Once a basin is locked, it is almost never escaped — we
confirmed this across 25 phases and every architectural intervention we tried.

### 9.4 The basin lottery in concrete terms

Earlier analysis described two basin types (K=3 vs K=4) with K=4 being necessary
for hopping. The Phase-t seed 2 video falsifies this: the 612 winner uses only
2 stable clusters and outperforms all K=4-basin seeds.

The revised picture has two basin types defined by **cluster stability**, not count:

**Stable-hop basin** — the cluster assignment locks onto a single K value for
the entire hopping phase, regardless of how many clusters are nominally active.
The encoder has found a representation where "hopping" is one coherent region of
latent space. Phase-t s2 (612) and Phase-f s1 (571) are both in this basin,
though s1 uses 2 sub-clusters within the hop.

**Unstable basin** — the cluster assignment oscillates within a single gait
phase, either because the gait is wrong (knee-walk, seeds 3 and 4 in Phase-f)
or because the representation hasn't found a stable "hopping" region. Seeds in
this basin are stuck regardless of cluster count: Phase-f s3 had 5 active
clusters and scored 262; Phase-f s4 had 2 active clusters and scored 266.

What determines which basin a seed enters: the same three factors as before
(initial weights, exploration trajectory, MPPI noise) — but the quantity being
determined is **whether the early policy finds a foot-hop gait at all**, not
whether it allocates 3 or 4 clusters. If it finds foot-hopping early, the
representation stabilises around it; if it finds kneeling, any number of
clusters will describe the kneeling precisely and stably.

### 9.5 Why seed 2 is bimodal

Seed 2's initialisation happens to be highly sensitive to the training signal.
When Glass or shaping provides a clear gradient (Phase-t s2=612, Phase-j s2=518,
Phase-1b s2=526), seed 2 commits strongly to a productive K=4 configuration.
When the signal is weak or noisy (Phase-Pa s2=7, Phase-x s2=6), it enters a
degenerate basin and collapses. High initialisation sensitivity = high outcome
variance.

### 9.6 Why seed 4 is reliably below median

Seed 4 has a strong prior toward the K=3 basin. It escaped only once: Phase-p
seed 4 = 538, where the 500k random exploration phase gave enough state coverage
to stumble into a K=4 configuration before basin lock. In every phase with
shorter exploration (EXPL_UNTIL=25k), seed 4 goes K=3 and caps around 200–270.
This also explains why Phase-p (EXPL_UNTIL=500k) produced the one seed-4 winner:
longer exploration delayed basin lock long enough for the K=4 basin to be sampled.

### 9.7 Implication for the K_UPDATE audit

If more gradient updates per env step shift the early loss landscape sufficiently
to nudge seeds away from K=3 configurations, we would expect k128/k256 to improve
seed 4 specifically — it is the clearest test of whether basin entry is
determined by gradient quality or by initialisation alone. The Phase-aa results
so far show k64 seed 1 stuck at 234 while k128 seed 1 reached 538 — exactly the
basin-escape pattern. Whether seed 4 also escapes under k128 will be decisive.

---

## 10. Brainstorm: how to actually fix Glass

The diagnosis from §3 and §5 is clear: Glass finds a good partition of whatever
gait the policy learned, but it does not push the policy toward a better gait,
and it does not enforce that the same behavioural phase maps to the same cluster
across time. Here are concrete directions, roughly ordered by expected cost and
risk.

### 10.1 Fix the training ratio first (K_UPDATE=128, currently running)

Before any architectural change, verify that the stuck-seed rate drops at
K_UPDATE=128. If it does, every previous Glass experiment was under-trained and
the comparison is invalid. Re-run Phase-ac (Glass vs. vanilla at the same
K_UPDATE) before spending GPU time on any of the ideas below.

### 10.2 Temporal assignment consistency loss ← highest priority given §3 findings

**Problem**: the SE loss minimises cut edges in the *aggregate* graph, but a
single anchor can be visited at both knee-walk and hop phases without penalty —
only the average transition pattern matters. The 612 winner (§3) achieved
temporal stability without this being trained — it was emergent. The stuck seeds
did not achieve it. This loss would make stability a first-class training objective.

**Fix**: add a loss that penalises the entropy of the assignment distribution
within a short sliding window \(W\) of consecutive frames:

\[
\mathcal{L}_{\text{window}} = \mathbb{E}_t\!\Bigl[
  H\!\Bigl(\tfrac{1}{W}\textstyle\sum_{\tau=t}^{t+W} c_\tau\Bigr)
  - \tfrac{1}{W}\textstyle\sum_{\tau=t}^{t+W} H(c_\tau)
\Bigr]
\]

This is the mutual information between time index and anchor assignment inside
the window. Minimising it forces *the same anchor to dominate throughout a gait
phase*, not just on average. W=5–10 env steps should cover one hop phase (~0.2 s
at 50 Hz). Cost: one extra scan over the rollout buffer, no new parameters.

### 10.3 Phase-contrastive loss on assignments (not on latents)

**Problem**: contrastive losses applied directly to \(z\) (e.g. BYOL, SimCLR)
have been tried in RL and do not reliably help. Applying them to the soft
assignment vector \(c_t\) instead is cheaper and more semantically targeted.

**Fix**: positive pairs = two frames within the same gait phase (consecutive
frames during steady-state motion); negatives = frames from different phases
or different episodes. NT-Xent on \(c_t\):

\[
\mathcal{L}_{\text{phase-con}} = -\log \frac{\exp(c_t \cdot c_{t'}^+ / \tau)}
{\exp(c_t \cdot c_{t'}^+ / \tau) + \sum_j \exp(c_t \cdot c_j^- / \tau)}
\]

This directly trains the anchor assignments to be phase-consistent without
constraining the latent geometry. The tricky part is defining positive pairs
without ground-truth gait labels — one proxy: two frames are "same phase" if
their contact state (foot on ground vs. not) matches, which is available from
the MuJoCo state.

### 10.4 Reduce N and K to match the known gait structure

**Problem**: N=16 anchors and K=8 clusters give the SE loss 16×8 degrees of
freedom to fill. Only 4 of the 8 clusters are active in winning seeds; the other
4 are wasted capacity that can absorb arbitrary patterns.

**Fix**: set N=4, K=4 to directly match the 4-phase hop cycle. This forces the
two-level hierarchy to compress to gait phases. If the encoder can only describe
a latent as a mixture of 4 anchors, and the anchors must be grouped into 4
clusters with small SE, the partition *has* to align with the dominant
behavioural structure. Risk: losing fine-grained within-phase discrimination.
Mitigation: keep N=8, K=4 as a compromise (two anchors per phase, K=4 clusters).

### 10.5 Prototype-conditioned Q (mixture-of-experts critic)

**Problem**: MPPI plans using a single Q ensemble regardless of which gait
phase the agent is in. A policy in the flight phase and one in the stance phase
get identical value estimates for the same action — even though the physics are
completely different.

**Fix**: learn K=4 Q-offset vectors \(\Delta Q_k \in \mathbb{R}^{101}\) (one
per cluster, in two-hot space). The effective Q is:

\[
Q_{\text{eff}}(z, a) = Q_{\text{base}}(z, a) + \sum_k \bar{S}_k(z)\,\Delta Q_k
\]

where \(\bar{S}_k(z) = \sum_n c_{t,n} S_{nk}\) is the soft cluster membership.
This is a mixture-of-experts critic with Glass as the router, adding only
\(K \times 101 \approx 400\) parameters — negligible. The planner can then
optimise actions conditional on which gait phase it expects to be in.

### 10.6 Early basin detection and restart

**Problem**: basin lock happens in the first 200k steps. If we could detect at
200k that a seed entered K=3, we could restart it with a different random seed
for the exploration phase only.

**Implementation**: monitor the number of active prototypes (Glass diagnostic:
`active`) at step 200k. If `active < 4` (K=3 basin), kill and restart with
`seed += 100`. This is not a change to the algorithm — it is a training
curriculum that selects for K=4 initial conditions. Benchmark-fair because
evaluation is still on the original reward.
Cost: at most 2–3 restarts per seed (< 600k extra env steps total).

### 10.7 Anchor initialisation from exploration trajectories

**Problem**: anchors initialise randomly (SimNorm-shaped noise). The first 100k
gradient updates spend time moving anchors from random positions to meaningful
latent directions, delaying basin lock.

**Fix**: after the initial EXPL_UNTIL=500k random phase, run one pass over the
replay buffer, cluster the latents with k-means (k=16), and use the cluster
centroids as the initial \(\mu\). Glass then starts from semantically grounded
anchors rather than noise. The SE loss can immediately act on a meaningful graph
instead of spending 200k steps bootstrapping anchor positions.

### 10.8 Multi-task pre-training to force generic gait representations

**Problem**: Glass trained on HopperHop alone can represent a knee-walk gait
perfectly well — there's no pressure to learn a hopping gait from the structure
alone.

**Fix**: jointly train on HopperHop + HopperStand. The shared Glass partition
must now describe both tasks' state spaces. A knee-walk cluster that works for
HopperHop must also be consistent with HopperStand's standing states — which
are physically orthogonal. This cross-task pressure may force the partition toward
anatomically grounded clusters (joint angles, contact states) rather than
task-specific gaits. Risk: the shared partition might just grow larger rather
than become more meaningful. Mitigation: use task-specific heads with shared
encoder + Glass partition.

### Summary

| Idea | Params added | GPU cost | Risk | Priority |
|------|-------------|----------|------|----------|
| 10.1 K_UPDATE=128 | 0 | running now | low | **now** |
| 10.2 Window consistency | 0 | +5% | low | high |
| 10.6 Basin detection + restart | 0 | +10% | low | high |
| 10.7 Anchor init from k-means | 0 | +1% | low | medium |
| 10.4 N=8/K=4 reduction | −50% params | neutral | medium | medium |
| 10.3 Phase-contrastive | 0 | +10% | medium | medium |
| 10.5 Mixture-of-experts Q | +400 params | +5% | low | medium |
| 10.8 Multi-task pre-training | 0 | 2× data | high | low |
