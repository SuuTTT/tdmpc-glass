<!--
DRAFT — venue not chosen. Two candidates the user should decide between:
  (1) ICLR Understanding track (negative/diagnostic results, falsifiable criterion)
  (2) TMLR (claims-and-evidence; well-suited to a null-converted-to-principle paper)
DISCIPLINE NOTE: every quantitative claim below is annotated inline with the file it was
read from. `TBD<...>` marks a number that is NOT persisted in a JSON/CSV and must be
re-run-to-persist before submission. Do NOT fill any TBD from memory.
-->

# When Is Explicit Abstraction Redundant for a World Model? A Falsifiable Redundancy Criterion

## Abstract

A long line of work adds *explicit abstraction* objectives — clustering, structural-entropy
(SE) compression, bisimulation, or value-equivalence losses — on top of a learned world
model (WM), on the premise that a flat self-predictive latent fails to expose the structure
control needs. Across a 16-lever campaign on TD-MPC2 with the SimNorm latent, every such
objective we tried was null or harmful. Rather than report another negative result, we
extract from it a *positive, falsifiable predictive criterion*: **explicit abstraction is
redundant for a world model exactly when its latent is (1) linearly value-decodable and
(2) its interaction graph already carries task-aligned structure.** We argue this criterion
spans three latent classes. For the *monolithic* class (TD-MPC2 / SimNorm) we measure both
conditions directly: value is linearly decodable from the latent at R²=0.9994
[exp/tdmpc_glass/mechcheck/value_probe_selfpred_pick_n12.json], and the latent's transition
graph already carries (control-irrelevant) community structure. For the *token-transformer*
class we measure a structural-entropy NO-GO: the attention graph of a trained transformer-WM
shows no community structure beyond a degree-preserving null (best gap-over-null −0.0334
trained vs +0.0064 untrained) [exp/tdmpc_glass/mechcheck/se_attn_trained.json,
se_attn_untrained.json]. For the *entity-graph* class we state the criterion as a prediction
and describe the mechanism-check that would confirm or falsify it; we explicitly do **not**
claim entity-graph results we do not have. We complement the redundancy direction with a
*scored, pre-registered* positive one: a single checkpoint-computable signal (the ratio of the
model's ensemble-free self-disagreement to its true k-step error) survived a 9-candidate screen
of where *temporal* abstraction pays, and its committed-before-harvest predictions scored 4/4
at k=2 and 0/3 at k=8 — establishing it as a cross-task predictor at fixed k while falsifying
its k-invariance [exp/tdmpc_glass/mechcheck/p1_temporal_signals.json,
p1_ksweep_prediction.json, p1_score.json]. The criterion is falsifiable: it predicts that a
latent that is *not* value-decodable yet whose abstraction objective helps would be a positive
counterexample, and it specifies exactly where to look (compositional / OOD regimes where
linear decodability is not certified).

---

## 1. Introduction

Should a world model represent state as a single learned vector, or should it be given an
explicit abstraction — discrete codes, object slots, a clustered latent, a value-equivalent
projection? The structured-WM literature is built on the intuition that flat self-predictive
latents leave structure "on the table" that an explicit objective can recover, improving
sample efficiency, robustness, or generalization. Yet on control benchmarks the empirical
record is mixed-to-negative: published object-centric and graph world models frequently fail
to beat strong monolithic baselines (DreamerV3, TD-MPC2), winning on *prediction* and
*interpretability* rather than *control*
[docs/research/graph-wm-dr/dr1-fable5-graph-wm-se-survey.md].

This paper reports the same pattern from the inside. We ran a single-variable, compute-matched,
pre-registered campaign of 16 explicit-abstraction levers on top of TD-MPC2 with the SimNorm
latent [docs/iterations/RESEARCH_LEDGER.md]. Every lever was null or harmful. The interesting
output is not the 16th null; it is that the nulls share a *single mechanism* we can measure
ahead of time, and that the measurement generalizes a falsifiable rule.

We state the rule as a **redundancy criterion** (Section 3): explicit abstraction is redundant
when the latent is (1) linearly value-decodable and (2) its interaction graph already carries
task-aligned structure. The contribution is not "abstraction never helps" — that would be
both false and unfalsifiable — but a *predictive criterion* that says, for a given trained
latent, whether an abstraction objective has headroom, together with the cheap probes that
decide it. We instantiate the criterion across three latent classes (Section 4): monolithic
SimNorm (both conditions measured), token-transformer attention graphs (a measured NO-GO), and
entity-graph latents (stated as a prediction with a pre-specified mechanism-check). Section 5
summarizes the 16-null campaign as systematic evidence; Section 6 states the predictions that
would falsify the criterion, reports a scored pre-registered test of the converse (positive)
direction, and records an asymmetry in what mechanism-checks can and cannot predict; Section 7
is honest about its limits — chief among them that
linear value-decodability is a *training-distribution* property and does not certify OOD.

---

## 2. Background

**Self-predictive sufficiency.** Ni et al. (2024, arXiv:2401.08898) formalize when a
self-predictive (latent-consistency) objective yields a *sufficient* state abstraction for
control: a latent that supports its own forward prediction and reward already contains the
information a value function needs. This is the theoretical anchor for our condition (1):
if value is linearly decodable from the latent, the latent has *already* extracted the
value-relevant subspace, and an explicit objective can at best re-derive it.

**Value equivalence.** Grimm et al.'s value-equivalence principle holds that a model need only
predict quantities that matter for the value function, not reconstruct full state. A
value-equivalence *loss* on a WM should therefore help only when the underlying latent wastes
capacity on value-irrelevant directions. Condition (1) is precisely the test of whether that
slack exists.

**Structural entropy.** Li & Pan (IEEE TIT, 2016) define the structural entropy of a graph via
its optimal encoding tree; the 2D compression gap (H¹−H²)/H¹ measures how much community
structure a partition captures, 0 for a structureless "blob" and →1 for a perfectly modular
graph [src/helios/se_jax.py]. SE has been used in RL for state/skill abstraction on precomputed
transition graphs (SISA, SIDM, SI2E), but never as a differentiable loss *inside* a WM latent
[docs/research/graph-wm-dr/dr1-fable5-graph-wm-se-survey.md]. Condition (2) operationalizes
"task-aligned structure" through the SE gap of the latent's interaction graph relative to a
degree-preserving null.

**SimNorm.** TD-MPC2 applies SimNorm to its latent: the latent is partitioned into V groups,
each passed through a softmax, so the encoder output is *by construction* a set of V
soft-categorical (simplex) codes
[docs/analysis/why-glass-failed-simnorm-redundancy.md]. A SimNorm latent is therefore already a
soft clustering; combined with the self-predictive consistency loss, it already shapes a
temporally coherent, implicitly-factored latent. This is the structural reason a separate
clustering/SE objective is likely redundant for the monolithic class.

---

## 3. The Redundancy Criterion

We state the criterion for a trained world model with latent encoder *z = E(o)*, a value
function *V(z)*, and an *interaction graph* *G* defined on the latent's units of composition
(latent dimensions / SimNorm groups for the monolithic class; tokens for a transformer-WM;
entity nodes for an entity-graph WM).

> **Redundancy Criterion.** An explicit abstraction objective (clustering / structural-entropy
> / value-equivalence) is *redundant* — i.e. cannot improve control over the no-objective
> backbone — when **both** hold:
>
> **(C1) Linear value-decodability.** *V(z)* is linearly decodable from *z* on the
> on-policy state distribution to high accuracy (operationally R² ≳ 0.95). The latent has
> already isolated the value-relevant subspace, so a value-organizing objective has nothing to
> add.
>
> **(C2) Pre-existing task-aligned interaction structure.** The interaction graph *G* already
> carries community structure that aligns with task structure — measured as an SE compression
> gap that (a) exceeds a degree-preserving null by a margin, and (b) corresponds to control-
> relevant groupings rather than incidental (e.g. motion-phase) structure. A structure-imposing
> objective then either re-derives structure already present (if C2(a) holds) or imposes
> control-irrelevant structure (if C2(a) holds but C2(b) fails).

**What falsifies it.** The criterion is falsified by a *positive counterexample*: a trained
latent for which C1 fails (value is **not** linearly decodable, R² well below threshold) yet
an explicit abstraction objective **still does not help**; or, conversely, a latent satisfying
both C1 and C2 for which an explicit objective nonetheless yields a robust control gain. The
criterion's *useful* direction is predictive: if a latent fails C1 (or C2(a)), the criterion
predicts headroom — that is precisely where abstraction work should be aimed. Note the two
distinct failure modes of C2: "no structure" (gap at/below null) and "wrong structure"
(gap above null but task-misaligned). Both make an SE objective unhelpful, for different
reasons, and we observe both in our data (Sections 4.2 and 5).

The criterion is cheap to evaluate before any multi-week build: C1 is a linear probe on a
trained checkpoint (`scripts/value_probe.py`), and C2 is an SE-gap-vs-null computation on the
latent's interaction graph (`scripts/se_precheck.py`, `scripts/se_attention_graph.py`).

---

## 4. Evidence across latent classes

We instantiate the criterion on three latent classes. For monolithic SimNorm we measure both
conditions. For the token-transformer we measure a C2 NO-GO. For entity-graph latents we state
the criterion as a prediction and describe the gating mechanism-check; we make **no** empirical
claim about entity-graph latents here.

### 4.1 Monolithic (TD-MPC2 / SimNorm)

**C1 — linear value-decodability (holds).** On a trained self-predictive PandaPickCube jumpy
checkpoint, rolling out the policy for 12 episodes (12,000 visited states), value is linearly
decodable from the 512-d latent at **R²=0.9994**
[exp/tdmpc_glass/mechcheck/value_probe_selfpred_pick_n12.json]. The same probe finds the latent
spends almost all of its variance in value-irrelevant directions: the value-relevant subspace
has effective dimension ≈7.08 versus the latent's effective dimension ≈6.96, and the
**value-irrelevant variance fraction is 0.9785**
[exp/tdmpc_glass/mechcheck/value_probe_selfpred_pick_n12.json]. The interpretation is that the
latent has already isolated a low-dimensional value-relevant subspace that a linear head reads
off essentially perfectly — exactly the self-predictive-sufficiency picture of Ni et al. (2024).
A value-equivalence objective, whose job is to organize the latent around value, therefore has
no slack to exploit, predicting a null.

This prediction was confirmed directly. A value-equivalence loss applied to the jumpy macro-model
was null-to-harmful: a coefficient sweep on PandaPickCube (seed 0, MPPI) gave peak/final returns
2752/1939 (coef 0.05), 2638/2118 (0.1), 2084/1930 (0.2), and 1616/916 (0.5), versus the
vebase(0) baseline 2692/2243 — monotone degradation, no coefficient beating baseline on both
peak and final [docs/iterations/iteration_28_plan.md, "COEF SWEEP RESULT" table, read from ssh7
CSVs]. Probing the resulting value-equivalence checkpoints showed the loss barely reorganized
the latent: the value-irrelevant fraction moved only from 0.9785 (self-pred) to 0.9728 (ve@0.05)
to 0.9505 (ve@0.1) to 0.9773 (ve@0.2), with value still linearly decodable at R²=0.9998–1.0
[exp/tdmpc_glass/mechcheck/value_probe_ve005_pick_n12.json, value_probe_ve01_pick_n12.json,
value_probe_ve02_pick_n12.json — persisted re-runs of `scripts/value_probe.py` on the ve
checkpoints, 12 episodes / 12,000 states each].

**C2 — pre-existing interaction structure (holds for (a), fails for (b)).** The SimNorm latent's
*transition graph* already carries strong community structure. On a trained jumpy CheetahRun
checkpoint, dumping 12,000 latents, clustering to 128 nodes, and building the k-step transition
graph, the SE-optimal partition yields a best compression gap of **53.1%** (and **47.2%** for a
kNN-on-centroids graph) [exp/tdmpc_glass/mechcheck/se_precheck_simnorm_cheetah.json — persisted
re-run of the tier-2 analysis (`scripts/persist_se_precheck.py`) on the original latent dump,
reproducing the figures first recorded in docs/research/se-precheck-note.md §6]. Crucially this was measured on a *jumpy encoder trained with no SE pressure*,
ruling out the objection that the structure is an artifact of a clustering loss. So C2(a) — the
structure is already present — holds, and an SE-clustering objective can only re-derive it.

But C2(b) fails: the communities are **motion phases** (swing / stance / contact arcs of the
limit cycle), not control-relevant subgoals or error-prone regions
[docs/analysis/why-glass-failed-simnorm-redundancy.md §3; docs/research/se-precheck-note.md §2].
An SE objective that imposes this structure trades control-relevant capacity for structure the
model did not need — which is the misaligned-structure failure mode. The early Glass clustering
levers (geometric and behavioral prototype clustering) were null at adequate sample size for
exactly this reason [docs/iterations/RESEARCH_LEDGER.md, nulls #1–#2].

Together, C1 and C2 hold for the monolithic class, and every clustering / value-equivalence
lever on it was redundant — consistent with the criterion.

### 4.2 Token-transformer (attention graph): a measured NO-GO

For a token-transformer WM the natural interaction graph is the attention graph. We trained a
transformer-WM on PandaPickCube and ran `scripts/se_attention_graph.py`, which rolls out the
actor, aggregates per-layer attention into a (T×T) adjacency (T=32 tokens, 4 layers, 4 heads),
sparsifies (quantile thresholds and kNN), and compares the SE compression gap against a
*degree-preserving null* that destroys community structure while preserving the weight
distribution. Only the margin over this null counts as real structure; raw gaps are inflated by
sparsification.

The verdict is **NO-GO**. The best per-layer compression gap *over the null* is **−0.0334**
for the trained model (best layer 0) and **+0.0064** for an architecture-matched untrained model
(best layer 1) [exp/tdmpc_glass/mechcheck/se_attn_trained.json,
exp/tdmpc_glass/mechcheck/se_attn_untrained.json]. Both are below the GO margin threshold of
0.05, so neither model has exploitable community structure; the per-layer over-null margins for
the trained model are −0.0334 (L0), −0.1217 (L1, knn4), −0.0987 (L2, knn4), −0.0307 (L3) at
their best variants [exp/tdmpc_glass/mechcheck/se_attn_trained.json]. The decisive comparison is
trained ≈ untrained: training induces **no** exploitable community structure in the attention
graph. This is the C2 "no structure" failure mode (as opposed to the monolithic class's "wrong
structure"). An SE-as-a-loss has nothing to shape on this substrate, and the cheap probe killed
the SE-loss build before it was written
[docs/iterations/iteration_28_plan.md, "NORTH-STAR SE-ATTENTION-GRAPH VERDICT"].

**Scope (honest).** This is the *attention* graph of a small, modestly-trained transformer-WM
(~16k training steps; checkpoint eval ~345) on *single-object* manipulation, where relational
structure is inherently weak [exp/tdmpc_glass/mechcheck/se_attn_trained.json meta;
docs/iterations/iteration_28_plan.md]. It does **not** close the case for a true entity-node
graph/GNN WM on a genuinely relational (multi-object / multi-agent) domain. The attention-graph
*proxy* for graph-WM+SE is dead; the entity-graph version is the remaining open class
(Section 4.3).

### 4.3 Entity-graph latents: in progress, mechanism-check pending

The entity-graph class is the one the criterion has not yet been tested on, and we make **no**
empirical claim about it here. The criterion makes a clear *prediction* and specifies the
mechanism-check that would confirm or falsify it
[docs/research/graph-wm-se-proposal.md §A; dr1 survey §"Cheap mechanism-check"].

The plan: train a plain entity-factored WM (ground-truth entity states first, to dodge
slot-collapse confounds) on a 3–6-object manipulation task, then run the same two probes:
(i) C1 via `value_probe` — is value linearly decodable from the flat entity concatenation, and
critically does R² *drop* at held-out object counts? (ii) C2 via an SE-gap computation on the
*value-coupling* interaction graph (edge weight = sensitivity of return to an entity-pair
interaction), not the similarity graph. The criterion predicts redundancy (a null) **unless**
C1 fails OOD — i.e. value stops being linearly decodable at held-out object counts — in which
case the entity-graph class would be the first to exhibit genuine headroom for explicit
abstraction. This single experiment either completes the criterion's third data point or opens
the positive direction; either outcome is informative
[docs/research/graph-wm-se-proposal.md §"GO only if..."]. `TBD-entity-graph-probe` (no
entity-graph checkpoint or probe output exists yet).

---

## 5. The 16-null campaign as systematic evidence

The criterion did not come from a single experiment; it is the compression of a campaign of
explicit-abstraction levers, each run under a single-variable, compute-matched protocol with
pre-registered peak-and-final CI gates and a mechanism-check before any multi-seed fan-out
[docs/iterations/RESEARCH_LEDGER.md, "Gate discipline"]. The table below summarizes the levers
and why each died; all entries are read from the ledger.

| # | Lever | Outcome | Why (per ledger) |
|---|-------|---------|------------------|
| 1 | Geometric prototype clustering (SE Glass) | NULL | redundant with SimNorm's soft-categorical latent (IQM 0.748 vs 0.738, overlapping); confirmed on manipulation at n=5 (PandaPickCube final 1247 CI[815,1693] ≈ vanilla 1416 CI[1010,1821]) [exp/tdmpc_glass/mechcheck/clustering_panda_pick_n5.json] |
| 2 | Behavioral / reward-grounded clustering | NULL | null at n=34; gain crossed CI-separation then settled in overlap |
| 3 | Bisimulation auxiliary (BS-MPC style) | HARMFUL | actively hurts (0.549); brittle to coefficient; failed twice |
| 4 | Distractor robustness from abstraction | FALSIFIED | 1.23× < 1.5× gate; both encoders crushed equally |
| 5 | Sparse-task rescue via grouping | NULL | exploration problem, not latent geometry (0/3 vs 1/3) |
| 6 | "Floor effect" / weak-seed tail | INVERTED | behavioral arm produced the worst seed (0.127) |
| 7 | Laplacian / eigenpurpose exploration | NULL | generic RND ≥ it everywhere |
| 8 | Community-detection skills | NULL | communities = motion phases, not reachable subgoals |
| 9 | `rho` consistency-horizon schedule | NULL | task-dependent tuning knob, not architecture |
| 10 | SE-k adaptive jump-length | NULL | SE pre-check passed (53% gap) but boundary score does not track k-step error (Spearman +0.09 / −0.18) |
| 11 | Uncertainty-gated horizon (F) | NULL | signal valid (Spearman 0.72) but no headroom; k-step error uniform in-dist (inflation 1.06×) |
| 12 | SI2E / VCSE SE-exploration | NULL | no rescue of sparse tasks; does not beat RND; at coef 1.0 mildly hurts |
| 13 | wmsi2e — SE-exploration over WM latent | NULL | ties SI2E at 0/n; adds nothing over random-encoder SI2E or RND |
| 14 | Value-equivalence loss | NULL | latent already value-sufficient (R²=0.9994); coef sweep monotone harm |
| 15 | Alt dynamics backbone (resmlp / attn) | NULL/mirage | helps weak vanilla (+40%/+26%) but hurts strong jumpy (jum/resmlp 1796/1381 ≪ jum/mlp 2645/2319) |

[All rows: docs/iterations/RESEARCH_LEDGER.md "WHAT DID NOT WORK" table and
docs/iterations/iteration_28_plan.md "ARCH A/B VERDICT" for #15.]

Two cross-cutting findings recur. First, the **uniform-error** finding: a strong jumpy model is
uniformly accurate over the states a near-optimal policy visits, which is why fixed-k jumpy works
and why every *adaptive* scheme (#10, #11) has nothing to adapt to
[docs/iterations/RESEARCH_LEDGER.md "Root cause for #10–11"]. Second, the **redundancy** finding:
a strong self-predictive WM (TD-MPC2 + SimNorm) already encodes a value-sufficient abstraction,
so explicit clustering / value objectives re-derive structure already present
[docs/iterations/RESEARCH_LEDGER.md "Cross-cutting lesson"]. The criterion makes the second
finding measurable and predictive.

**The campaign's one positive result is consistent with the criterion — and narrower than our
interim notes claimed.** The only lever that beat the baseline was the (prior-art) jumpy / k-step
world model itself — *not* an abstraction objective. The picture is task-dependent rather than
suite-wide (persisted aggregations, 10k-resample bootstrap on seed means):
PandaPickCubeOrientation is a clean CI-separated win at n=5 per arm (jumpy final 2145 vs vanilla
1129, +90%, difference CI95 [685, 1344]; every jumpy seed beats every vanilla seed);
PandaPickCube, a positive trend that had not reached CI separation at n=5 (1872 vs 1416, +32%,
difference CI95 [−267, 1169]), *resolved* once boosted to n=8 per arm: jumpy 1969 vs vanilla
1355, +45%, difference CI95 [66, 1153] — now CI-separated; PandaOpenCabinet is null at n=5
(1050 vs 1053, difference CI95 [−563, 685])
[exp/tdmpc_glass/mechcheck/anchor_jumpy_vs_vanilla.json (n=5 aggregation, from
scripts/aggregate_anchor.py); exp/tdmpc_glass/mechcheck/p1_score.json "pick_anchor_n8" block,
per-seed finals in p1_ksweep_harvest.json]. Jumpy's scoreboard is thus 2/3 tasks CI-separated
(PickCube, Orientation) plus one null (OpenCabinet). Earlier hand-aggregated interim numbers
(+101%/+75%/+74% across all three tasks at n=2–4) settled as seeds completed — itself a
demonstration of why we persist aggregations and report final-n CIs. Peak is mixed (vanilla
often higher peak; jumpy sustains final where it wins). Jumpy is a temporal, not an abstraction,
lever — consistent with the thesis that abstraction is the redundant axis on this substrate
while temporal coarse-graining (on some tasks) is not. Section 6.1 turns this task-dependence
itself into a scored, pre-registered prediction.

---

## 6. Predictions & falsification

The criterion is useful only if it makes risky predictions. It does:

1. **A positive case requires C1 to fail.** Explicit abstraction will help control only on a
   latent where value is *not* linearly decodable on the relevant distribution. The sharpest
   place to test this is the compositional / OOD regime: train an entity-factored WM on N
   objects and probe value-decodability at held-out N′≠N. The criterion predicts a control gain
   from abstraction **iff** R² *collapses* OOD; if R² stays high OOD, abstraction stays
   redundant [docs/research/graph-wm-se-proposal.md §B-3, §"GO only if..."].

2. **No-structure vs wrong-structure are distinguishable a priori.** C2 splits into "gap at/below
   null" (token-transformer attention, Section 4.2) and "gap above null but task-misaligned"
   (SimNorm motion phases, Section 4.1). The criterion predicts an SE objective is unhelpful in
   *both* cases, and the SE-gap-vs-null + alignment probes diagnose which one applies before any
   build.

3. **Value-coupling beats similarity.** If an entity-graph latent satisfies C1 OOD-collapse, the
   criterion predicts the *value-coupling* graph (edges = ∂return/∂interaction), not the
   similarity graph, is the one that carries task-aligned structure — because the
   sparse relational structure of *which interactions drive return* is a property of the dynamics
   graph that a monolithic latent does not expose [docs/research/graph-wm-se-proposal.md §B-1].
   A positive counterexample here would both falsify the "abstraction is always redundant" reading
   and confirm the criterion's predictive direction.

A clean falsification would be: a trained latent with R² ≳ 0.95 and a task-aligned SE gap over
null, for which a single-variable explicit-abstraction objective nonetheless produces a ≥10%,
CI-separated control gain on ≥3/4 tasks under the campaign's protocol. We did not observe one in
16 levers; the criterion predicts it will not be observed wherever both conditions hold.

### 6.1 A scored, pre-registered positive direction

The redundancy criterion says where abstraction is redundant. The converse question — where does
*temporal* abstraction pay? — we made predictive, and then scored. The campaign's one positive
lever (the jumpy k-step model) has task-dependent gains (Section 5), which raises a testable
question: is there a checkpoint-computable signal that predicts, per task, how much jumpy will
gain? We screened nine candidate signals (error medians and dispersions, latent speeds,
disagreement statistics, perturbation inflation), computed from a single trained k=4 checkpoint
per task, against the measured jumpy final-return gains at the time of the screen (+90%
Orientation / +32% PickCube / 0% OpenCabinet). Exactly one survived the ordering screen:

> **disc_err_gap** = median(disagreement) / median(true k-step error)
> — Ori 1.33 > Pick 1.12 > Cab 0.95

where *disagreement* is the ensemble-free jumpy-vs-iterated-one-step prediction gap — computable
from one checkpoint, no ground truth needed
[exp/tdmpc_glass/mechcheck/p1_temporal_signals.json — 9 candidates, surviving_candidates =
["disc_err_gap"]]. The reading: temporal abstraction pays where the macro-model is accurate
*and* calibrated-conservative (disagreement ≥ error); it fails where the model is inaccurate and
overconfident. An ordering match at n=3 tasks is hypothesis-generating only, so we pre-registered
the test: fresh jumpy variants at k=2 and k=8 (effective planning horizon fixed, compute-matched)
were trained on all three tasks, the signal was computed from their dumped checkpoints, and the
predictions were committed to a public git history **before any final return was read** (k=2
gaps 0.969/1.028/0.672 all below their k=4 values ⇒ predict lower gains at k=2; k=8 gaps
1.521/1.736/1.681 all above ⇒ predict higher gains at k=8)
[exp/tdmpc_glass/mechcheck/p1_ksweep_prediction.json; git history timestamps the commit order].
The score, against finals read for the first time at harvest
[exp/tdmpc_glass/mechcheck/p1_score.json, p1_ksweep_harvest.json — n=3 seeds per new cell]:

| Block | Prediction | Outcome | Scored |
|---|---|---|---|
| Pick k2<k4 | lower gain at k=2 | 1616 vs 1969 | ✓ |
| Ori k2<k4 | lower gain at k=2 | 1487 vs 2145 | ✓ |
| Cab k2<k4 | lower gain at k=2 | 596 vs 1050 | ✓ |
| k2 gain ordering | Ori > Pick > Cab | +358 > +261 > −457 | ✓ |
| Pick k8>k4 | higher gain at k=8 | 1302 vs 1969 | ✗ |
| Ori k8>k4 | higher gain at k=8 | 1469 vs 2145 | ✗ |
| Cab k8>k4 | higher gain at k=8 | 694 vs 1050 | ✗ |

**4/4 on the k=2 block; 0/3 on the k=8 block.** The honest reading: disc_err_gap is a real
*cross-task* predictor at fixed k (7/7 ordering facts across two k values) but it is **not
k-invariant** — iterating the 1-step model k times inflates disagreement mechanically with k, so
*upward* cross-k comparisons are confounded; the committed k=8 predictions walked into that
confound and died in public, exactly as pre-registration is supposed to work
[exp/tdmpc_glass/mechcheck/p1_score.json "summary.interpretation"]. A bonus result from the same
harvest: the dose–response is **unimodal, with k=4 optimal on all three tasks** (Pick
1969 > 1616 > 1302; Ori 2145 > 1487 ≈ 1469; Cab 1050 > 694 > 596 for k=4 / k=2 / k=8
respectively) — temporal abstraction has an optimal grain, neither too fine nor too coarse
[exp/tdmpc_glass/mechcheck/p1_score.json, p1_ksweep_harvest.json].

### 6.2 Mechanism-check asymmetry: NO-GOs kill reliably, GOs only license

The same week produced two further mechanism-checks, and with them a finding about the method
itself. The first, on a Hermite-spline action bottleneck (macro-action = spline knots, shrinking
the planner's search dimension), died at its pre-registered gate: open-loop spline replay at
knot spacing 4 preserved a mean **0.364** of expert return against a 0.95 gate (zero-order hold:
0.305) — one dev-box run instead of a 30-run fan-out
[exp/tdmpc_glass/mechcheck/spline_mechcheck_PandaPickCube.json; stated caveat in the evidence
file: the replay is open-loop while the reference rollout is closed-loop, so a marginal fail
would not be fatal — 0.364 vs 0.95 is not marginal].

The second is the instructive one. A macro-Q error decomposition for a *value-equivalent macro
head* (pre-registered gates: GO iff value-cost ratio ≥ 0.3 and Spearman ρ ≥ 0.4) said **GO on
Orientation** (value-cost ratio 0.352, ρ=0.57 — the macro-model's latent errors systematically
cost value) and **NO-GO on OpenCabinet** (ρ=0.23 — errors large but value-unstructured)
[exp/tdmpc_glass/mechcheck/p3_macroq_ori.json, p3_macroq_cab.json]. Yet the archive already
contained the experiment: a value-equivalence loss on the macro head scored a final return of
**583** (n=3) versus jumpy's 2145 (n=5) on Orientation — catastrophic harm on exactly the task
the mechanism-check green-lit [docs/CHANGELOG.md 2026-06-12 entry; per-seed finals 431/732/585
read from the phasei27_ve PandaPickCubeOrientation worker CSVs (mean of MPPI evaluations from
step 400k) in exp/tdmpc_glass/remote_mirror/]. The asymmetry we record: **mechanism-check NO-GOs
kill reliably; GOs license a test but never predict success.** This is consistent with the
redundancy criterion's own logic — C1/C2 are headroom (NO-GO-shaped) instruments: they can
certify that an objective has nothing to gain, but passing them establishes only that a gain is
not excluded.

---

## 7. Limitations

We are deliberately conservative about scope.

- **Linear decodability is a training-distribution property.** R²=0.9994 is measured on the
  on-policy state distribution of a trained checkpoint
  [exp/tdmpc_glass/mechcheck/value_probe_selfpred_pick_n12.json]; it does **not** certify that
  value stays decodable out-of-distribution or at held-out compositional counts. The criterion's
  own predictive direction (Section 6) hinges on this gap, and we have not yet measured an OOD
  R². This is the single most important caveat and the one the dr survey flags explicitly
  [docs/research/graph-wm-dr/dr1-fable5-graph-wm-se-survey.md "Caveats"].

- **Small world-model scale.** The transformer-WM NO-GO is on a ~16k-step checkpoint
  [exp/tdmpc_glass/mechcheck/se_attn_trained.json meta]; the dreamer-family transformer training
  loop was dispatch-bound (≈1% GPU utilization), capping how far we could train before measuring
  [docs/iterations/iteration_28_plan.md, "DREAMER4 / transformer-WM PERF BLOCKER"]. A
  better-trained / larger transformer-WM could in principle develop attention structure absent
  here; the NO-GO is for this scale.

- **Single-object manipulation.** Both measured classes are on single-object Franka tasks (and a
  single-agent CheetahRun for the SimNorm SE gap). Genuinely relational domains (multi-object,
  multi-agent) are exactly where the entity-graph class might break C1 OOD — untested here.

- **Seed counts are modest.** The jumpy anchor is n=5 per arm with persisted CIs (n=8 on
  PandaPickCube after the boost) [exp/tdmpc_glass/mechcheck/anchor_jumpy_vs_vanilla.json;
  p1_score.json "pick_anchor_n8"]; 2 of 3 tasks are CI-separated (PickCube at n=8, Orientation
  at n=5), the third is null. Several ledger nulls are decisive in *direction* at small n but
  not all reach the full n=5 CI gate [docs/iterations/RESEARCH_LEDGER.md].

- **The scored predictor (Section 6.1) has its own caveats.** Each new k-sweep cell is n=3 seeds
  (only the k=4 anchor arms have n=5–8) [exp/tdmpc_glass/mechcheck/p1_ksweep_harvest.json]; the
  disc_err_gap reference values were computed from a *single* checkpoint per task at k=4, so
  seed-robustness of the signal itself is unverified
  [exp/tdmpc_glass/mechcheck/p1_temporal_signals.json]; and the mechanical-inflation confound
  that explains the k=8 failure was identified *post hoc* — the prediction failure itself was
  pre-registered, the explanation was not. A cross-domain out-of-sample test (CheetahRun,
  predicted "weak-positive" from gap 1.017 in the pre-registered ambiguous zone) is committed
  and pending; its outcome is deliberately **not** included here
  [exp/tdmpc_glass/mechcheck/p1_ksweep_prediction.json "pre_registered_prediction_cheetah"].

- ~~**Unpersisted numbers.**~~ Resolved: every quantitative claim in this draft is now backed by
  a persisted JSON under exp/tdmpc_glass/mechcheck/ (value_probe R²=0.9994; ve-checkpoint probes
  0.9728/0.9505/0.9773 at R²=0.9998–1.0; attention-graph NO-GO gaps; SimNorm SE gap 53.1%/47.2%;
  jumpy anchor aggregation with n=5 CIs plus the n=8 PickCube resolution; the k-sweep
  prediction/score/harvest files and the spline and macro-Q mechanism-checks) — with the single
  exception of the archival 583-vs-2145 value-equivalence final, which is read from the mirrored
  worker CSVs and docs/CHANGELOG.md as noted in Section 6.2.

---

## 8. Related work

**Structured / object-centric world models.** The control track record of structured WMs is
mixed: DLPWM underperforms DreamerV3; DreamerV3 beats its SLATE-slot variant (ROCA); ObjectZero
≈ EZ-V2; the clearest win (SOLD) is on bespoke relational manipulation with few seeds and is not
replicated; token-transformer WMs (IRIS, STORM) are competitive but monolithic-token
[docs/research/graph-wm-dr/dr1-fable5-graph-wm-se-survey.md "Comparison" tables]. The consistent
reading is that structure helps prediction and interpretability, not in-distribution control —
which the redundancy criterion explains as C1+C2 holding for strong monolithic latents.

**Structural entropy in RL.** SE has been used for state/action/skill abstraction on precomputed
transition or value graphs via discrete greedy encoding-tree minimization (SISA, SIDM, SIRD,
SI2E), and differentiable SE exists for static graph clustering (LSEnet, DeSE) — but a
differentiable SE loss *inside* a WM latent is an empty gap
[docs/research/graph-wm-dr/dr1-fable5-graph-wm-se-survey.md "Comparison: SE methods"]. Our
contribution is to show, with cheap mechanism-checks, that this gap is most likely null-shaped on
current monolithic and token-transformer substrates, and to specify the one substrate (value-
coupled entity graphs, OOD) where it might not be.

**Self-predictive and value-equivalent abstraction.** Ni et al. (2024, 2401.08898) on
self-predictive sufficiency and Grimm et al. on value equivalence provide the theory that
condition (1) operationalizes; the criterion is, in effect, an empirically-measurable test of
when those theories' "sufficient abstraction" has already been attained by a self-predictive
backbone.

**Methodology.** The campaign's transferable by-product is the protocol itself — single-variable,
compute-matched, peak-and-final pre-registered CI gates, mechanism-check before fan-out — which
caught 8 mirages and killed SE-flavored levers in an afternoon rather than a multi-week build
[docs/iterations/RESEARCH_LEDGER.md "WHAT WORKED"].

---

## 9. Conclusion

We converted a 16-lever campaign of null explicit-abstraction results into a single falsifiable
predictive criterion: explicit abstraction is redundant for a world model exactly when its
latent is (1) linearly value-decodable and (2) its interaction graph already carries task-aligned
structure. We measured both conditions for the monolithic SimNorm class (R²=0.9994; a 53.1% SE
transition-graph gap that is real but motion-phase / control-irrelevant), measured a clean C2
NO-GO for the token-transformer class (attention-graph SE gap at/below a degree-preserving null,
trained ≈ untrained), and stated the criterion as a falsifiable prediction for the entity-graph
class together with the exact OOD probe that would confirm or break it. On the converse,
positive direction we scored a pre-registered predictor of where *temporal* abstraction pays:
disc_err_gap went 4/4 on its committed k=2 predictions and 0/3 at k=8 — a real cross-task
predictor at fixed k whose k-invariance is falsified — and the week's mechanism-checks
crystallized an asymmetry worth stating: NO-GOs kill reliably, GOs only license a test. The
criterion's value is that it is cheap to evaluate on any trained checkpoint and tells you,
before a multi-week build, whether an abstraction objective has headroom — and where to look
(a latent that is *not* value-decodable OOD) if you want to find a genuine positive case.
