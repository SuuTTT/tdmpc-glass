# Six Mirages: A Pre-Registered Multi-Axis Null for Latent Abstraction in TD-MPC2
## (and an Anatomy of How Small-Sample Effects Dissolve)

*Draft v0.2 — 2026-06-07. All numbers verified from run CSVs (mirror), data as of this date;
final tables to be regenerated at submission. Compute: ~12 vast.ai GPUs (A4000/2080Ti/TitanV/
1660S), mujoco_playground (MJX), pinned code SHA 4d3b935.*

---

## Abstract

Does latent-space abstraction improve a state-of-the-art latent world model? We add three
abstraction mechanisms to TD-MPC2 under a strict fair protocol — identical hyperparameters,
compute-matched, single-variable changes, pre-registered decision gates — on five
mujoco_playground DMC tasks: (i) **geometric prototype clustering** with a structural-entropy
objective ("geometric Glass"), (ii) the same clustering **grounded in reward prediction**
("behavioral Glass"), and (iii) a **pairwise bisimulation auxiliary** (BS-MPC-style). Three
findings. **First**, geometric clustering is statistically indistinguishable from vanilla
TD-MPC2 (IQM 0.748 [0.695, 0.815] vs 0.738 [0.715, 0.771]) — consistent with TD-MPC2's
SimNorm already providing soft-categorical latent structure. **Second**, the bisimulation
auxiliary *hurts* (0.549 [0.527, 0.599]; an untuned coefficient collapses training outright),
replicating recent large-scale negative results for behavioral metrics. **Third**, behavioral
Glass is also, finally, indistinguishable from vanilla (0.767 [0.757, 0.779] vs 0.738
[0.715, 0.771] at n=34 vs 37 final; CIs still overlap (0.757 < 0.771) despite a +0.029 point
gap — and the gap wandered to *both* sides of zero across maturity snapshots, never stably
separating) —
but only after its apparent advantages dissolved one by one with growing samples: an IQM
gain that crossed in and out of CI separation **three times** (peaking at +0.051), a
"+30%" sample-efficiency edge that inverted, and a zero-weak-seed "floor effect" that held
through 16 runs and then *inverted* — behavioral Glass ultimately produced the worst seed
in the study, a late-training collapse to 0.127 (weak-seed rates 2/27 vs 3/33,
Fisher p≈1.0). Together with distractor and
sparse-exploration falsifications, **six distinct effects — each separately publishable at
some interim snapshot — regressed to null**. We report the complete estimate trajectories
as an anatomy of small-sample mirages in deep RL, alongside the durable findings: under
16–64 dimensions of temporally-correlated nuisance input both encoders degrade identically
(dose–response curve included), and on three literally 0-vs-solved sparse tasks behavioral
grounding rescues nothing. Our behavioral-arm IQM oscillated 0.818 → 0.736 → 0.829 → 0.749
→ 0.785 → 0.726 → 0.754 → 0.767 as seeds accumulated to n=34, crossing "significant win" and
"confirmed null" readings and wandering to both sides of baseline — CIs never stably
separating — a case study in why pre-registered gates and adequate n are non-negotiable
for marginal-effect claims in deep RL.

---

## 1. Introduction

Latent world models such as TD-MPC2 [Hansen et al., 2024] achieve strong continuous-control
performance with a representation trained only by reward, value, and latent self-consistency
losses. A natural hypothesis is that an explicit *abstraction* over this latent space —
grouping behaviorally equivalent states, discretizing, or imposing hierarchical structure —
should further improve learning. The literature offers encouragement (bisimulation metrics
[Zhang et al., 2021], discrete codebooks [DC-MPC, 2025]) but also strong cautions: theory
identifies TD-MPC2's objective as already satisfying the conditions of a sufficient
self-predictive abstraction [Ni et al., 2024], and a recent large-scale study finds
behavioral-metric losses add little beyond self-prediction [2506.00563].

We test the hypothesis directly, with the discipline that marginal-effect claims demand:

1. **Fair protocol.** The only change between arms is the abstraction term; all
   hyperparameters, network sizes, planner budgets, env steps, and eval schedules are
   identical. No restarts, no population-based training, no per-task tuning.
2. **Pre-registered gates.** Sample-size bars and decision thresholds fixed before
   data arrival; falsification probes sized to the minimum informative experiment.
3. **Aggregate statistics.** Per-task-normalized IQM with stratified-bootstrap 95% CIs
   [Agarwal et al., 2021]; sustained (last-2-eval) values, never best-of-run peaks.

## 2. Methods

**Base agent.** TD-MPC2 (JAX/MJX reimplementation, verified to reproduce published-scale
returns on CheetahRun/WalkerRun/FingerSpin/WalkerWalk): encoder → SimNorm latent; latent
dynamics, reward, twin-Q, policy heads; MPPI planning; latent-consistency loss with
stop-gradient targets.

**Arm G (geometric Glass).** N=32 prototypes (SimNorm-normalized), soft cosine assignment,
K=8 clusters via learned assignment logits; a prototype-transition graph from consecutive
latents; differentiable two-level structural-entropy loss + one-sided balance hinges +
temporal-consistency terms. λ_SE = 5e-3 (legacy-tuned).

**Arm B (behavioral Glass).** Arm G plus a learned per-prototype reward vector r_p; loss
λ_b · (c(z)ᵀ r_p − r)², λ_b = 0.5, gradients to encoder, prototypes, and r_p. Prototypes are
thereby pushed to group *reward-equivalent* states — a soft, hierarchical analogue of
bisimulation grouping, supplying the one signal the consistency loss lacks (cross-state
behavioral equivalence rather than along-trajectory predictability).

**Arm BS (bisimulation reference).** BS-MPC-style permuted-pair π*-bisimulation encoder
loss: (‖z_i−z_j‖₁ − |r_i−r_j| − γ‖sg(z'_i)−sg(z'_j)‖₂)², coefficient 0.01 after 0.1 was
observed to collapse training (Section 4.3).

**Tasks.** mujoco_playground (MJX) DMC: CheetahRun, WalkerRun, FingerSpin (primary);
AcrobotSwingup, WalkerWalk (breadth). 1M env steps, eval every 50k (MPPI eval), metric =
mean of final two evals, normalized /1000.

## 3. Main result: a complete null

**Aggregate (mature runs ≥950k steps; stratified bootstrap over tasks):**

| arm | n | IQM | 95% CI |
|---|---|---|---|
| behavioral Glass | 34 | 0.767 | [0.757, 0.779] |
| geometric Glass | 16 | 0.748 | [0.695, 0.815] |
| vanilla TD-MPC2 | 37 | 0.738 | [0.715, 0.771] |
| bisimulation aux | 6 | 0.549 | [0.527, 0.599] |

*(FINAL — fixed cutoff 2026-06-08: all live runs matured; a dead tail of 12 early-crashed
runs was excluded (never reached 950k). behavioral Glass [0.757,0.779] vs vanilla
[0.715,0.771] still **overlap** (0.757 < 0.771) → no separation. This is the regenerate-once
table; the per-snapshot values below are kept verbatim as the small-sample case study.)*

**No stable difference on any measure.** The behavioral-vs-vanilla IQM gap crossed in and
out of CI separation and wandered to *both sides* of baseline as n grew (+0.051 at n=17/23,
separated; +0.046 at n=19/32, separated; +0.015 at n=20/37, overlapping; +0.008 at n=22/37;
−0.012 at n=28/37 — *below* baseline; +0.016 at n=31/37; **+0.029 at n=34/37 final, still
overlapping** [0.757,0.779] vs [0.715,0.771]). Six snapshots, six different stories, never a
stable separation. The lower-tail
("floor") effect — zero weak seeds through n=16, which we provisionally framed as the
mechanism — did not merely dissolve but inverted: behavioral Glass's weak seeds arrived at
n=18 (0.619 final) and n≈24 (0.127, a late-training collapse from 0.643 at 800k — the
worst seed of any arm in the study). Final weak-seed rates 2/24 vs 3/33 (Fisher p≈1.0).
The three non-bisimulation arms are equivalent in mean, IQM, floor, and weak-seed rate;
only the bisimulation arm differs (worse).

**The lower tail, final (3 primary tasks):**

| arm | n | mean | std | min | seeds < 0.65 |
|---|---|---|---|---|---|
| behavioral Glass | 30 | 0.777 | 0.169 | **0.127** | 2 / 30 |
| geometric Glass | 12 | 0.781 | 0.166 | 0.505 | 2 / 12 |
| vanilla | 33 | 0.764 | 0.137 | 0.419 | 3 / 33 |

The arms' *best* seeds are equivalent (all reach ≈0.99 on FingerSpin, ≈0.75–0.77 on
CheetahRun), their weak-seed rates are equivalent (8–17%, all pairwise Fisher p≈1.0), and
the study's single worst outcome belongs to the behavioral arm. At the n=16 interim
snapshot this table read "0/16 weak, min 0.700" and supported a floor-raising mechanism
story we had drafted in full; the final column is the strongest single illustration of
this paper's methodological point.

*Interpretation.* The original observation that motivated the project (high across-seed
variance in basin-entry on HopperHop) was itself, in retrospect, a small-sample reading of
seed luck. Nothing in the final data suggests reward-grounded grouping changes the seed
outcome distribution in either direction. A dedicated 5M-step HopperHop replication pair
under the fair protocol closes the loop: **neither arm entered the high-reward basin**
(behavioral Glass best 323, vanilla best 286, both plateaued ≈280–315 from 1.5M onward) —
the iteration-11/12 "basin entries" that motivated the project occurred only under
procedure interventions (restarts, PBT) or luck, never under the clean protocol.

## 4. Pre-registered falsifications

### 4.1 Geometric clustering is redundant (the SimNorm explanation)
Across every sample size, geometric Glass tracked vanilla (final: 0.748 vs 0.738,
overlapping). TD-MPC2's SimNorm already projects latents onto softmax simplices — a soft
categorical structure — so latent-similarity prototype clustering duplicates existing
machinery. Eleven prior internal iterations of geometric-Glass variants on HopperHop
(documented in the project log) produced no robust gain; this experiment explains why.

### 4.2 No distractor robustness (falsified at minimal cost)
Hypothesis: reward-grounded abstraction should help the encoder ignore behaviorally
irrelevant input. Probe: append 64 temporally-correlated OU nuisance dims to observations;
pre-registered metric (per-arm mean of last-2-eval averages) and gates (≥2× = signal,
<1.5× = falsified), two rounds of 2 arms × 2 seeds × 500k:

| round | vanilla | behavioral Glass |
|---|---|---|
| CheetahRun+64d | 153 (245, 61) | 233 (216, 250) |
| WalkerRun+64d | 90 (111, 69) | **67** (62, 72) |
| **combined** | 121 | 150 → ratio **1.23× < 1.5 → falsified** |

The round-1 edge reversed in round 2. Both encoders are equally crushed. Total cost: 8
half-length runs — the falsification protocol working as intended.

**Dose–response (CheetahRun, last-2 mean):**

| nuisance dims | vanilla | behavioral Glass |
|---|---|---|
| 0 | ≈551 | ≈539 |
| 16 | 508 | ≈430 |
| 32 | 179 | ≈175 |
| 64 | 153 | 233 / 67 (two rounds) |

The encoder tolerates ≈1× its native observation dimensionality (17) of correlated nuisance
and collapses between 16 and 32 dims — identically for both arms.

### 4.3 Bisimulation auxiliaries hurt
At coefficient 0.1 the pairwise bisimulation loss dominates (its L1 pairwise scale, ~O(10),
dwarfs the O(1) reward+next-latent target) and collapses training (returns 5–10× below
vanilla on all tasks). Retuned to 0.01 it still finishes last (0.549). This replicates, on a
modern strong baseline, the finding that behavioral-metric losses add nothing beyond
self-prediction and are brittle [2506.00563; cf. BS-MPC's TD-MPC-v1-only gains].

### 4.4 Sparse-task seed bimodality is not an abstraction problem

The probe that closed our temporal-abstraction direction (Section 6 limitations) revealed
that three sparse mujoco_playground tasks exhibit literal 0-vs-solved seed bimodality under
flat TD-MPC2 — the most extreme form of the weak-seed phenomenon. If behavioral grounding
raised the floor by improving latent geometry generally, it should rescue these seeds.
It does not (solve counts, 1M budget; † = run still short of 1M at draft time):

| task | vanilla solves | behavioral Glass solves |
|---|---|---|
| CartpoleSwingupSparse | 1/3 (841, 0, 14†) | **0/3** (3, 1†, 2) |
| BallInCup | 1/3 (975, 0, 0†) | 2/3 (975, 0†, 958) |
| AcrobotSwingupSparse | 2/3 partial (~215 unstable) | 0/3† (early) |

No cell pair differs meaningfully at n=3, and the direction flips across tasks. The
parsimonious reading: sparse-task bimodality is an *exploration* failure — whether the
random action sequence ever touches reward — and latent-geometry regularization neither
causes nor cures it. (At the time this probe ran, the floor effect of Section 3 was still
alive; the probe was designed to test whether it extended to exploration-driven weak seeds.
It did not — and the dense-reward floor effect itself subsequently dissolved.) The probe
remains informative as the fifth falsification: even where seed bimodality is most extreme,
behavioral abstraction does not move it.

## 5. The small-sample hazard (a case study)

The behavioral-Glass aggregate IQM, recomputed at every maturity snapshot:

```
n=3: 0.818   n=4: 0.736   n=5: 0.829   n=6: 0.801   n=7: 0.793
n=8: 0.749   n=9: 0.743   n=17: 0.785  n=20: 0.753  n=22: 0.746
n=25: 0.747  n=28: 0.726  n=31: 0.754  n=34: 0.767  (vanilla reference: 0.738)
```

Several of these snapshots support a "significant win" reading (CI-separated at n=17 and
n=19); others support a "confirmed null"; the estimate dipped *below* the baseline it
twice "significantly beat" at n=28, then drifted back above it (n=31, n=34) — wandering to
both sides of the reference even past n=30, with CIs overlapping throughout. A
sample-efficiency side-metric (reward@300k) likewise swung
from +30% to −22% before settling at parity, and a zero-weak-seed tail effect held for 16
runs before the arm produced the study's single worst seed. Every intermediate certainty
was wrong. With heavy-tailed seed outcomes, the IQM of n≤9 runs is dominated by which
seeds happen to have finished. We took many interim looks (documented in the project
log); the seed and task schedule, however, was fixed in advance rather than adaptively
extended, and the final-n estimate with stratified-bootstrap CI is reported as the result.
We recommend (i) pre-registered n-bars *per claim type* — variance-reduction claims need
roughly 3× the n of mean-shift claims; (ii) reporting estimate trajectories, not just final
tables, for marginal effects.

## 6. Limitations

- Proprioceptive observations only; the literature locates behavioral-abstraction gains in
  pixel + natural-video distractor settings, which we did not test (no Madrona renderer).
  Our synthetic-distractor falsification lowers but does not eliminate that prior.
- Single base agent (TD-MPC2) and five dense + three sparse tasks from one suite; the null
  is about this regime, not all of RL.
- The bisimulation arm used one published formulation at two coefficients; gentler schedules
  may exist (though the cited large-scale study suggests not).
- behavioral-Glass adds ~2× wall-clock per step at equal env steps (structural-entropy +
  prototype machinery); the comparison is env-step-matched, not wall-clock-matched. A
  wall-clock-matched comparison would favor vanilla.

## 7. Conclusion

On a fair, pre-registered protocol, explicit latent abstraction does not improve TD-MPC2
on any axis we measured: final returns, sample efficiency, distractor robustness,
sparse-task exploration, or seed reliability. Geometric clustering is redundant with
SimNorm; reward-grounded clustering is a null at adequate n; bisimulation auxiliaries are
brittle and harmful. This is consistent with the theoretical position that TD-MPC2's
self-predictive objective already constitutes a sufficient abstraction [Ni et al., 2024],
and we offer it as a strong, multi-mechanism empirical confirmation on a modern baseline.
The second contribution is the anatomy: six effects — two procedure-confounded "wins," a
thrice-CI-separated IQM gain, a sample-efficiency edge, a distractor-robustness edge, and
a zero-weak-seed floor — each of which a reasonable researcher could have published from
some interim snapshot, and each of which regressed to null (or inverted) as n grew under a
fixed protocol. With n<10 seeds per arm, both "significant" wins and "confirmed" nulls are
frequently artifacts; we recommend estimate-trajectory reporting as a standard supplement
to final tables.

---

### Reproducibility appendix (to complete)
- Code: helios-rl @ 4d3b935 (+ iter-14 flags: `--glass_lambda_behav`, `--bisim_coef`,
  `--distractor_dims`); queue/configs in `scripts/queues/archive_done_failed.jsonl`.
- Per-run CSVs: `exp/tdmpc_glass/remote_mirror/**/phasei14v2*/`, `phase2a*`, `phase2ad*`.
- Analysis: `control/eval_rliable.py` + inline scripts (IQM, stratified bootstrap, 20k
  resamples, seed 0).
- References: TD-MPC2 2310.16828 · Ni et al. 2401.08898 · DBC 2006.10742 · BS-MPC
  2410.04553 · DC-MPC 2503.00653 · behavioral-metric study 2506.00563 · rliable 2108.13264.
