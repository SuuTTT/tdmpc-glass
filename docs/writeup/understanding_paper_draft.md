# A Strong Self-Predictive World Model Is Already a Sufficient, Value-Aligned Abstraction: A Fifteen-Lever Negative Result and a Mechanism-Check-Before-Fan-Out Methodology

> **VENUE / FRAMING TBD — author to finalize.** Two natural homes: (a) **ICLR
> understanding/blog-track** (the result is mechanistic, the contribution is "why a
> family of ideas fails," the methodology is the takeaway); (b) **TMLR** (no novelty
> bar, rewards rigorous negative results and reproducibility — arguably the better
> fit given that the one positive is prior art). The abstract and intro below are
> written to serve either; pick the frame and trim the methodology section's
> emphasis accordingly.

*Draft v0.1 — 2026-06-11. Consolidates the iter-8→iter-29 TD-MPC-Glass campaign.
Every numerical claim is read from run CSVs / probe JSON and cited to its source
document; this project's hard rule is no fabricated numbers. Items still in flight
or not yet measured are marked **TBD**. Compute: ~10–12 vast.ai GPUs
(A4000/2080Ti/TitanV/5070Ti), mujoco_playground (MJX) + Warp Franka manipulation.*

---

## Abstract

We ask whether an explicit *abstraction* objective — over state, reward, time, or
skills — can improve TD-MPC2, a strong self-predictive latent world model, under a
strict fair protocol. Across **fifteen distinct abstraction levers** the honest
answer is **no**: every genuinely novel lever returned to null, and the only method
that beat vanilla TD-MPC2 (a jumpy *k*-step world model on manipulation) is published
prior art that we reproduce and evaluate fairly, not our invention. We then explain
*why*. A trained TD-MPC2 latent is already (i) a soft-clustering — SimNorm partitions
the latent into groups and softmaxes each, so the encoder output is *V* parallel soft
codebooks by construction; (ii) temporally coherent — the self-predictive consistency
loss shapes transitions to be smooth and predictable; and (iii) value-aligned — on a
trained PandaPickCube checkpoint, value is *linearly* decodable from the latent at
\(R^2 = 0.9994\), criticality is near-uniform (CV \(= 0.36\)), and the value-relevant
subspace is low-dimensional (\(\approx 7\) of \(\approx 7\) effective latent dims),
while \(\approx 98\%\) of the latent's variance is value-irrelevant slack the value
head already ignores [iter28]. This is precisely the *sufficient self-predictive
abstraction* of Ni et al. (2024), measured on our own checkpoints rather than asserted
— which reframes the fifteen nulls as **one finding fifteen times**: explicit
abstraction is *redundant* with what a strong self-predictive world model already
learns, and where it is not redundant it is *misaligned* (the structural-entropy
communities of the latent transition graph capture real *motion-phase* structure — a
**53%** two-dimensional-vs-one-dimensional structural-entropy gap [glass-analysis] —
that is genuinely beyond SimNorm but irrelevant to control). Our central
contribution is methodological: a **mechanism-check-before-fan-out** discipline —
single-variable, compute-matched, pre-registered peak-AND-final CI gates, paired
difference bootstrap, read-from-JSON anti-fabrication, and above all a cheap
kill-test of each lever's load-bearing assumption on a frozen checkpoint *before*
any multi-seed campaign. This discipline called every null correctly, often in an
afternoon where the original effort spent months. We ship it as an
honest-RL-benchmarking recipe alongside the negative result.

---

## 1. Introduction

The motivating question is concrete: **can an explicit abstraction objective beat a
strong self-predictive world model at the architecture/algorithm level?** TD-MPC2
[Hansen et al., 2024] reaches strong continuous-control performance with a
representation trained only by reward, value, and latent self-consistency losses. A
large literature suggests that *adding* abstraction on top — bisimulation grouping,
discrete codebooks, hierarchical/structural objectives, temporal abstraction, skills
— should help. We set out to demonstrate exactly that, with a structural-entropy
latent regularizer ("Glass") and a parade of successors.

The honest answer, after a campaign spanning iterations 8 through 29, is **no — with
one prior-art exception**. Fifteen levers; fifteen nulls among the ideas that were
ours. The one method that beat vanilla TD-MPC2 — a jumpy (*k*-step) world model on
Franka manipulation — is published prior art [Farebrother et al., 2026], which we
reproduce and evaluate under our fair protocol. That is not a tragedy: it is a
question. *Why does a strong latent world model keep being un-improvable by
abstraction?* The contribution of this paper is to answer that question
mechanistically rather than gesture at it, and to extract the methodology that let
us answer it cheaply.

**Contributions.**
1. A **fifteen-lever negative result**: explicit abstraction (over state, reward,
   time, skills, exploration, dynamics architecture, and value-equivalence) does not
   improve TD-MPC2 across final return, sample efficiency, distractor robustness,
   sparse exploration, or seed reliability, under a single-variable compute-matched
   protocol with peak-AND-final CI gates.
2. A **measured mechanism** for the null: cheap probes on frozen checkpoints show the
   trained latent is already value-sufficient (\(R^2 = 0.9994\)) and criticality is
   near-uniform — the headroom these levers assume does not exist. This is Ni et
   al.'s (2024) sufficiency theorem rendered as a number.
3. The **mechanism-check-before-fan-out methodology**, the most transferable output:
   interrogate each lever's load-bearing assumption on a frozen checkpoint before
   spending a multi-seed campaign.
4. An **interpretability result**: the SimNorm latent carries a real 53%
   structural-entropy community structure — but it is *motion-phase* structure,
   useful for *describing* the latent and useless for *improving* control — together
   with a validated differentiable structural-entropy implementation (`se_jax`,
   cross-checked vs `selib` to \(\sim 10^{-12}\)) and an audit clearing the campaign's
   nulls of an SE-implementation confound.
5. A **fair-protocol generalization result** for the one real (prior-art) win: the
   jumpy world model sustains final return where vanilla degrades across the Franka
   manipulation suite.

---

## 2. Background

**TD-MPC2 and the SimNorm self-predictive latent.** TD-MPC2 [Hansen et al., 2024]
trains an encoder → latent, latent dynamics, reward, twin-Q, and policy heads, and
plans with MPPI. The latent is passed through **SimNorm**: partition the latent into
*V* groups and softmax each group, so the encoder output is, by construction, *V*
parallel soft-categorical assignments — i.e. *a soft clustering* [glass-analysis §2].
The **self-predictive consistency loss** (with stop-gradient targets) shapes the
latent so transitions are smooth and predictable.

**Sufficient self-predictive abstraction.** Ni et al. (2024) show, in theory, that a
self-predictive objective learns a *sufficient* abstraction — one retaining all
information needed to predict its own future and value. This sets a high bar: any
explicit abstraction added on top must capture something the sufficient abstraction
is *missing*.

**Structural entropy (SE).** Li & Pan (2016) define the *k*-dimensional structural
entropy of a graph: the minimum number of bits to encode the position of a
degree-stationary random walk under a (hierarchical) partition. The
two-dimensional-vs-one-dimensional *gap* measures how much community structure the
partition captures. SE has been proposed as a graph-hierarchy objective; "Glass"
applied it to a prototype transition graph over the TD-MPC2 latent.

**Value-equivalence.** Grimm et al. (value-equivalence) argue a model need only be
accurate where it affects value/decisions, not state-faithful everywhere. The
value-equivalent macro-head lever trains the *k*-step dynamics to be
return-equivalent rather than state-faithful.

**Jumpy temporal abstraction.** A jumpy (*k*-step) world model predicts *k* steps in
a single call, enabling a long effective planning horizon without the compounding
1-step error that sinks deep naive planning. Jumpy world models with cross-timescale
consistency are published prior art [Farebrother et al., 2026].

**Reliable evaluation.** rliable [Agarwal et al., 2021] prescribes fixed-budget,
many-seed, IQM with stratified-bootstrap CIs rather than peak-picking — the
statistical backbone of our protocol.

---

## 3. Methodology: the fair protocol (a core contribution)

The methodology is not scaffolding around the result; in a campaign that is fifteen
nulls and one prior-art win, **the discipline is the deliverable.** It has six
components.

**(1) Single-variable, compute-matched changes.** The only difference between arms is
the abstraction term; all hyperparameters, network sizes, planner budgets, env steps,
and eval schedules are identical. No restarts, no population-based training, no
per-task tuning. This rule overturned the project's own origin story: the iter-11/12
"basin entries" on HopperHop that motivated everything occurred only under procedure
interventions (restarts/PBT) or seed luck — under the clean protocol *neither* Glass
nor vanilla entered the high-reward basin (best 323 vs 286) [glass-analysis §4.4;
draft §3]. The "win" was procedure, not representation.

**(2) Pre-registered peak-AND-final CI gates.** Within-run instability is real in deep
RL (deadly triad, plasticity loss, policy churn). It creates two mirage modes:
cross-seed small-sample mirages (an effect at n≤9 that dissolves as n grows) and
within-run collapse mirages (an effect visible only at one checkpoint). We therefore
report **both** peak/best-checkpoint AND final/last-2 for every arm, and a win
*requires CI separation on both*. Several abstraction "wins" were publishable at an
interim snapshot and survived neither metric: a behavioral-Glass IQM that crossed
"significant win" and "confirmed null" readings six times and wandered to both sides
of baseline past n=30 [draft §5]; a jumpy-Cartpole "growing lead" that reversed by
450k [capstone §4].

**(3) Paired-difference bootstrap.** For head-to-head claims we compute the
paired arm-vs-baseline difference with a stratified bootstrap (20k resamples) and
gate on CI separation. The headline jumpy result (§5d) is a paired jumpy-vs-MPPI
bootstrap.

**(4) Mechanism-check before fan-out.** The single most transferable rule: *before
fanning a lever into a multi-seed campaign, run the cheapest possible test of the
mechanism it depends on.* Not "does it work?" (that needs the campaign) but "does the
thing it *assumes* even exist?" Every abstraction lever has a load-bearing assumption
interrogable on a frozen checkpoint in an afternoon (§5a–b). The contrast with the
original Glass effort is the whole argument: iters 8–9 spent months tuning a
"turn-Glass-off-at-1M" schedule when an iter-23-style mechanism-check ("does Glass's
structure track anything control needs?") would have returned *no* in an afternoon
[glass-analysis §6].

**(5) Win definition.** A win is **≥10% improvement with non-overlapping CI on ≥3/4
tasks**, against the *right* baseline (the strongest config — jumpy, not vanilla —
once jumpy is the incumbent), reading at ≥400k steps and distrusting any single
snapshot [ledger gate-discipline].

**(6) Read-from-JSON anti-fabrication.** Every number is read from run CSVs or probe
JSON, never from a notebook or memory. This project has a documented history of
fabricated numbers; the discipline is enforced mechanically (probes write JSON,
tables are regenerated from disk) and every claim in this paper cites the document it
was read from.

---

## 4. Results

### 4a. The fifteen-lever null table

Each lever was killed at the cheapest sufficient level — many by a mechanism-check
before any fan-out, others by a fully-powered multi-seed run. The table below is the
campaign ledger [ledger §❌]; **iter** is the iteration where it died.

| # | lever | iter | how it died |
|---|---|---|---|
| 1 | Geometric prototype clustering (structural-entropy Glass) | 14 | redundant with SimNorm's soft-categorical latent (IQM 0.748 vs 0.738, CI overlap) |
| 2 | Behavioral / reward-grounded clustering | 14 | null at n=34; gain crossed CI-separation 3× then settled in overlap (0.767 vs 0.738) |
| 3 | Bisimulation auxiliary (BS-MPC style) | 14 | actively hurts (0.549); brittle to coef; failed twice |
| 4 | Distractor robustness from abstraction | 14 | falsified (1.23× < 1.5× gate); both encoders crushed equally |
| 5 | Sparse-task rescue via grouping | 14 | it's an exploration problem, not latent geometry (0/3 vs 1/3) |
| 6 | "Floor effect" / weak-seed tail | 14 | inverted — behavioral arm produced the study's worst seed (0.127) |
| 7 | Laplacian / eigenpurpose exploration | 21 | generic RND ≥ it everywhere (unsupervised-abstraction fail #2) |
| 8 | Community-detection *skills* | 19 | communities = motion phases, not reachable subgoals |
| 9 | `rho` consistency-horizon schedule | 20 | task-dependent tuning knob, not architecture (helps Panda, hurts sparse) |
| 10 | SE-k adaptive jump-length (structural entropy → k) | 23 | SE pre-check PASSED (53% gap) but mechanism-check FAILED: boundary score does not track k-step error (Spearman +0.09 Panda / −0.18 Cart) |
| 11 | Uncertainty-gated horizon ("F") | 23 | signal valid (disc↔err Spearman 0.72) but no headroom: k-step error uniform in-dist AND under MPPI-perturbed actions (inflation 1.06×) |
| 12 | SI2E / VCSE SE-exploration | 24 | null: no rescue of sparse Cart/Acro; doesn't beat RND (all 0/n); at coef 1.0 mildly HURTS |
| 13 | wmsi2e — SE-exploration over the WORLD-MODEL latent (the novel bet) | 24 | null: ties si2e at 0/n; WM-latent + critic-value conditioning adds nothing over random-encoder SI2E or RND |
| 14 | Value-equivalence loss (return-equivalent, not state-equivalent) | 26–28 | null: mechanism-check says latent already value-sufficient (linear V-decode R²=0.9994); coef sweep {0.05,0.1,0.2,0.5} never beats jumpy baseline, monotone harm |
| 15 | Alt dynamics backbone (resmlp gated-residual / attn over SimNorm groups) | 27–28 | null/mirage: resmlp beats MLP on weak vanilla (+40%/+26%) but HURTS strong jumpy; best config stays jum/mlp |

**Root cause for #10–11 (and the whole adaptive-k family):** the trained jumpy model
is *uniformly accurate* over the states/actions a near-optimal policy visits — which
is exactly why fixed-k jumpy already works, and why adaptive jump-length has nothing
to adapt to [ledger].

**Predictive validity of the mechanism-checks.** The cheap kill-tests *called the
campaign verdicts in advance*. Where we spent the multi-seed compute anyway (#10–11,
#14), the fully-powered result matched the mechanism-check's prediction. The
predictive-validity table:

| lever | the assumption it needs | cheap kill-test | result | matched fan-out verdict? |
|---|---|---|---|---|
| value-equivalent macro head (#14) | value is hard to recover from the latent | linear V-decode \(R^2\) | **0.9994** → no headroom [iter28] | yes — coef sweep monotone harm |
| value-critical horizon | criticality varies across states | criticality CV | **0.36** → near-uniform [iter28] | yes — not fanned out (killed by check) |
| SE adaptive jump-length (#10) | boundaries mark where the model errs | boundary-vs-error Spearman | **+0.09 / −0.18** → uncorrelated [ledger] | yes — null |
| uncertainty-gated horizon (#11) | error varies under planning perturbations | error inflation under MPPI noise | **1.06×** → uniform [ledger] | yes — nothing to gate |

**Audit: the nulls are not an SE-implementation artifact.** A natural worry is that
the SE-flavored nulls (#1, #8, #10) reflect a buggy SE computation rather than a real
redundancy. We audited this directly [se_jax header; git 1358411]. The *old* iter-8/9
Glass computed a **directed** SE variant (out-degree / out-cut-only) on an asymmetric
transition graph, which is a well-defined quantity but *not* canonical undirected
Li–Pan SE and disagrees with `selib`. The **production** iter-14 Glass symmetrized
correctly (`tdmpc_glass.py:313`), so the fifteen nulls were **not** SE-bug artifacts;
the iter-29 re-audit confirms 13/15 are truly dead (mechanism-proven or
already-Panda-tested-null), with only two weak Panda re-test candidates (~20–25%
prior: behavioral Glass floor-raising, community-skills spatial subgoals) outstanding
**TBD** [git 1358411]. Independently, we contribute `se_jax`, a JAX-differentiable
1D/2D SE cross-checked against `selib` to \(\sim 10^{-12}\) on karate, les_misérables,
and planted-SBM graphs [se_jax header].

### 4b. The value-sufficiency mechanism-check

The deepest lever — *value-organized* abstraction — is also where the mechanism is
cleanest. We ran two cheap kill-tests on a trained PandaPickCube checkpoint
(`scripts/value_probe.py`, standalone, no hot-path edits; n_ep=12, ~12k states; all
from JSON) [iter28].

**Probe 1 — value-equivalence headroom.** A *linear* decode of value from the frozen
latent gives \(R^2 = 0.9994\): value is **already** trivially decodable. The
supporting numbers: `effective_dim_latent` \(\approx 6.96\),
`effective_dim_value_subspace` \(\approx 7.08\), `value_irrelevant_variance_frac`
\(\approx 0.978\). The value-relevant subspace (~7 dims) and the latent's effective
dimensionality (~7 dims) *match*: value is not buried, it is cleanly carried in a
low-dimensional subspace a linear map reads off perfectly, and ~98% of latent variance
is value-irrelevant slack along directions the value head already ignores. There is
nothing for a value-equivalence objective to *add*.

When run anyway, value-equivalence at coef 0.5 *hurt*: PandaPickCube `ve` 1616/916 vs
`vebase` 2692/2243 (Δ −1076 peak / −1327 final); CheetahRun −129/−83 [iter28]. The
full coef sweep is monotone degradation — coef 0.05: 2752/1939; 0.1: 2638/2118; 0.2:
2084/1930; 0.5: 1616/916 vs vebase 2692/2243 — no coef beats the baseline on peak AND
final [iter28]. A latent probe on the trained `ve` checkpoint confirms the loss barely
moved the representation: `value_irrelevant_frac` 0.978 (self-pred) → 0.953 (ve@0.1) →
0.977 (ve@0.2), with value still linearly decodable at \(R^2 \approx 1.0\) [iter28].
The latent was already value-sufficient; the VE term just traded off against the
consistency the planner needs. Triple confirmation (mechanism + returns + latent).

**Probe 2 — value-criticality variation.** `crit_cv` \(= 0.36\) (< 0.5 bar),
`flat_state_frac` \(= 0.029\): criticality is near-uniform across the states a good
policy visits. There is essentially nothing for an adaptive horizon to gate on — the
same "nothing to adapt to" that killed error-gated adaptive-k (#10–11) [iter28].

This is Ni et al.'s sufficient-abstraction theorem rendered as a number on our own
checkpoints; it is the spine of the analysis.

### 4c. The architecture mirage (helps-weak-baseline-only)

The campaign's most persistent *apparent* win was an alternative dynamics backbone
(`resmlp`, a deeper gated-residual MLP). For three harvests it looked real:
**van/resmlp 2699/1561 vs van/mlp 1925/1238** on PandaPickCube @≥450k (n=2 each,
+40% peak / +26% final) [iter28]. But tested against the *strongest* config it
collapses [iter28]:

| config | peak | final |
|---|---|---|
| van/mlp | 1925 | 1238 |
| van/resmlp | 2699 | 1561 |
| **jum/mlp** | **2645** | **2319** ← BEST |
| jum/resmlp | 1796 | 1381 |

resmlp *helps the weak vanilla baseline* (+40%/+26%) but *hurts the strong jumpy
model* (jum/resmlp 1796/1381 ≪ jum/mlp 2645/2319, a ~−850 final gap far beyond n=2
noise). The attention backbone behaves the same: van/attn 1627/900 < mlp, jum/attn
2081/1184 < jum/mlp [iter28]. Neither backbone improves the best config → not a real
architecture win. This is the canonical mirage the protocol exists to catch: **test a
lever against the strongest config, not the weak baseline.** The arch lever is closed
(#15).

### 4d. The one real win = jumpy (prior art), and its generalization

The lone durable positive is the jumpy (*k*-step) world model on manipulation. Its
mechanism was pre-confirmed: the *k*-step head is more accurate than iterating the
1-step model, and the advantage *grows* with k (jumpy_err/iter1_err: 0.991 at k=2 →
0.910 at k=3 → 0.821 at k=8) [capstone §3]. On PandaPickCube under the fair protocol
(paired jumpy-vs-MPPI, 20k-resample bootstrap), it separates on both metrics:
**peak +966/+44%, CI [714, 1248]; final +1266/+88%, CI [877, 1642]** [ledger §✅;
capstone §3].

**The honest caveat (load-bearing).** This is published prior art [Farebrother et al.,
2026, *Compositional Planning with Jumpy World Models*, arXiv:2602.19634]. Their
setting differs (TD-Flow occupancy models composing pre-trained policies zero-shot vs
our online TD-MPC2 macro-MPPI), but the concept is not ours. This is a fair-protocol
*reproduction-and-evaluation* win, not the architectural innovation the campaign set
out to find.

**Generalization across the Franka suite (the empirical anchor).** On FINAL return,
the jumpy WM *sustains* return where vanilla degrades, robust in *direction* across all
three discriminating Franka tasks; current read-from-disk values @≥450k, MPPI
[ledger iter-27/28 update; iter28]:

| task | jumpy final | vanilla final | Δ | n (jum / van) |
|---|---|---|---|---|
| PandaPickCube | 2319 | 1154 | **+101%** | 2 / 4 |
| PandaPickCubeOrientation | 2323 | 1329 | **+75%** | 2 / 3 |
| PandaOpenCabinet | 1261 | 596 | **+112%** | 2 / 1 |

Peak is mixed (jum +23% / +5% / −30%) [ledger]. The direction is robust across all
three tasks; the *magnitude* settles downward as vanilla's seed count grows (Ori
137%→75% from n=2→3), so the firmest claim is *directional* sustained-final-return,
with magnitudes still moving. **Honest caveat:** jumpy lags in seed count (n=2 vs van
n=3–4); n is firming to 5 [ledger]. This is the paper's empirical anchor: a prior-art
method, evaluated under a fair protocol, generalizes across the manipulation suite.

---

## 5. Interpretability: what world-model latents represent

It would be tidy to conclude "there is no structure in the latent, so abstraction has
nothing to grab." That is wrong, and the most interesting part of the project is *why*
it is wrong.

The structure is unambiguously there. Building the latent transition graph correctly —
sparsifying the dense SimNorm "blob" first — and measuring its structural entropy
yields a **53% two-dimensional-vs-one-dimensional structural-entropy gap** on the
k-step transition graph (47% via kNN geometry); the raw dense graph is a ~0%-gap blob
(40–76% edge density), and the communities pop out crisply only after sparsification
[glass-analysis §3]. So the latent carries rich, real community structure.

The punchline is the next sentence: **the structure is real but not useful for
control.** Across three independent probes [glass-analysis §3; ledger #8, #10]:

- the communities are **motion phases** (swing/stance/contact arcs of a limit cycle),
  not reachable subgoals → useless as skills;
- their boundaries **do not coincide** with model-error regions → useless for
  adaptive-k (Spearman +0.09 / −0.18);
- SE-driven coverage **did not beat** generic novelty → useless for exploration.

Glass's pressure decomposes into a *geometric clustering* part that is **redundant**
with SimNorm (re-deriving a soft-categorical simplex code the encoder already has) and
a *transition-graph SE* part that is **not** redundant but **control-irrelevant** (it
shapes the prototype transition graph toward motion-phase coherence the model did not
need) [glass-analysis §3, §5]. A tell-tale: the best iter-8/9 recipe was to *turn
Glass off* at 1M steps — a method whose optimal schedule is "use it then remove it" is
signalling asymptotic value ≤ 0, the signature of a redundant-or-harmful auxiliary
read as a clever schedule [glass-analysis §4]. The durable salvage is the
*measurement*: SimNorm latents carry rich motion-phase structure (53% gap). That is an
**interpretability** result about what world-model latents represent — more
publishable as understanding than as a performance lever [glass-analysis §7].

(The SE measurements here rest on the validated `se_jax` implementation and the
directed-vs-undirected audit of §4a: the production Glass symmetrized correctly, so
the 53% gap is canonical undirected Li–Pan SE, not the directed artifact of the
iter-8/9 code.)

---

## 6. Limitations and honest scope

This is a negative result with a measured mechanism, not a closed book.

- **Single agent family (TD-MPC2).** "Sufficient self-predictive abstraction" is a
  property of TD-MPC2's *self-predictive* objective. DreamerV3 learns a
  *reconstruction*-based recurrent latent and trains its actor purely in imagination —
  a reconstruction objective is *not* value-sufficient by construction, so genuine
  abstraction headroom may exist there. The DreamerV3 generality probe is **in
  progress** but currently **blocked**: both DreamerV3 and the DreamerV4 transformer-WM
  ran at ~1% GPU util / >60 min stuck in warmup with no checkpoint on the 5070Ti — a
  dispatch-bound, non-vectorized per-step collection loop (a dreamer-family issue, not
  transformer-specific). It needs a vectorized (`lax.scan`) collection loop before the
  cross-family value-sufficiency probe can run. **TBD** [iter28].
- **\(R^2\) is necessary, not sufficient, for "no headroom."** Value being linearly
  decodable on the *visited* distribution does not prove decodability off-distribution
  (under distractors, transfer). The sharp open question the data leaves is whether an
  explicit value-equivalence objective helps where the *implicit* one is
  capacity-limited (distractors, transfer), even though it is redundant on-distribution
  [blog §7].
- **Low-DoF regime; high-DoF untested/floored.** Every probe is on PandaPickCube and
  DMC-scale tasks (~7 effective latent dims). The theory permits abstraction headroom
  exactly where there is *more value-irrelevant capacity to discard* — high-DoF control
  (Humanoid, Dog, dexterous manipulation). Our one high-DoF attempt (Humanoid) floored
  at 500k steps (needs millions); we could not test it honestly [blog §7; ledger]. The
  right move is the same one the paper argues for: run a high-DoF `value_probe` *first*.
- **Directions not covered.** Two action/value-abstraction levers independent of the
  null mechanisms remain untested and are *not* claimed null: a **Hermite-spline action
  bottleneck** (macro-action = target (q,v), cubic-Hermite interp + PD tracker; shrinks
  MPPI search dim, no learned codec → no online representation-shift) and a refined
  value-equivalent macro head under distractors [ledger §🔭]. Pixel/natural-video
  distractor settings (where the behavioral-abstraction literature locates its gains)
  were not tested (no Madrona renderer) [draft §6].
- **Seed counts.** The jumpy generalization anchor is n=2–3 (firming to 5); magnitudes
  are still moving even as direction is robust (§4d).

---

## 7. Future work

The honest read is that this closes a *family* — re-organizing a monolithic,
already-sufficient vector latent — not the whole question. It points to two regimes
where "sufficient" may stop holding.

- **Graph / structured world models + differentiable SE.** The redundancy that killed
  Glass was specific: SE on a SimNorm latent is redundant because the softmax is
  *already* a soft clustering. A graph-structured world model (nodes as
  entities/objects, edges as relations, GNN/transformer dynamics) has **no built-in
  clustering**, so structural entropy is non-redundant there *by construction*. Our
  validated `se_jax` (differentiable in both adjacency and soft community assignment)
  is purpose-built to *shape* such a graph as a loss. The concrete first step is a
  **mechanism-check** (`scripts/se_attention_graph.py`): does a *trained*
  transformer-WM's attention graph carry community (SE-gap) structure analogous to the
  53% SimNorm gap? If yes, the graph-WM + SE north star is alive; if it is a
  structureless blob (gap ≈ a degree-preserving random graph), the direction is dead.
  **SE-attention-graph mechanism-check result: TBD — RUNNING/GATED.** The script exists
  and is read-only over the model, but it is blocked on the same dreamer-family
  collection-loop perf fix (no trained transformer-WM to run it on yet) [iter28;
  se_attention_graph header]. *(Placeholder for the per-layer SE-gap, trained-vs-
  untrained-vs-shuffled comparison, and GO/NO-GO verdict.)*
- **High-DoF, done right.** A high-DoF `value_probe` *before* any big-budget run: if
  value is *not* linearly decodable at \(R^2 \approx 1.0\) on Humanoid/Dog, the headroom
  is real and worth the budget; if it is, the verdict extends and the budget is saved.
- **Generality across world models.** Completing the DreamerV3 value-sufficiency probe
  once the collection loop is vectorized (§6).
- **Reusable byproduct.** The ensemble-free disagreement signal (jumpy prediction vs
  iterated one-step) tracks true k-step error at Spearman **+0.72** — useless for
  horizon-gating (error is uniform) but a validated uncertainty signal for exploration
  bonuses or safe/abstained planning [ledger §✅].

---

## 8. Reproducibility

- **Public repo.** TD-MPC-Glass (helios-rl), with the fair-protocol queue/worker
  harness (file-backed central queue, worker registry, control-plane daemons).
- **Mechanism-check scripts.** `scripts/value_probe.py` (standalone value-sufficiency /
  criticality probe; writes JSON), `scripts/se_attention_graph.py` (graph-WM SE
  mechanism-check, dump/analyze modes), `src/helios/se_jax.py` (validated
  differentiable 1D/2D SE, cross-checked vs `selib` to \(\sim 10^{-12}\);
  `tests/test_se_jax.py`).
- **Fair-protocol harness.** `scripts/run_benchmark.py` (the only training driver) with
  `--glass_*`, `--jumpy_k/--jumpy_plan/--jumpy_n_macro`, value-equivalence and arch
  flags; rliable IQM + stratified bootstrap analysis.
- **Honest-RL-bench tutorial.** A how-to for the peak-AND-final + mechanism-check-
  before-fan-out discipline (the §3 protocol) as a reusable recipe. **TBD — to be
  written for camera-ready.**
- **Data and records.** Per-run CSVs under `exp/tdmpc_glass/`; iteration records in
  `docs/iterations/` (iter-26 value-equiv, iter-28 value-organized mechanism-check,
  iter-29 audit); campaign verdicts in `docs/iterations/RESEARCH_LEDGER.md`; the SimNorm
  SE retrospective in `docs/analysis/why-glass-failed-simnorm-redundancy.md`. **All
  numbers are read from CSVs / probe JSON, not notebooks** — verification discipline,
  the hard way.

---

### References

TD-MPC2 (Hansen et al.) 2310.16828 · Ni et al. (sufficient self-predictive
abstraction) 2401.08898 · Grimm et al. (value-equivalence — *full citation TBD*) ·
Farebrother et al. (Compositional Planning with Jumpy World Models, prior art)
2602.19634 · rliable (Agarwal et al.) 2108.13264 · Henderson et al. (Deep RL That
Matters) 1709.06560 · Li & Pan (structural information) IEEE TIT 2016 · DBC 2006.10742
· BS-MPC 2410.04553 · DC-MPC 2503.00653 · behavioral-metric study 2506.00563 · TAP
2208.10291.

*Source documents cited inline by short tag: [ledger] = docs/iterations/RESEARCH_LEDGER.md;
[iter28] = docs/iterations/iteration_28_plan.md; [glass-analysis] =
docs/analysis/why-glass-failed-simnorm-redundancy.md; [se_jax header] =
src/helios/se_jax.py; [git 1358411] = iter-29 audit commit; [blog] =
docs/blog/blog_phase3_understanding_draft.md; [draft] = docs/writeup/draft.md;
[capstone] = docs/writeup/capstone.md.*
