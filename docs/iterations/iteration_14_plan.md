# Iteration 14 Plan — Behavior-Aware Abstraction for TD-MPC2 (fair protocol)

Created 2026-06-04 · synthesizes 3 deep-research reports (`docs/research/dr-{claude,gemini,gpt}.md`)
+ 4 internal lit agents. Goal restated: **beat vanilla TD-MPC2 at the architecture/algorithm
level via an abstraction idea, under a fair compute-matched protocol with NO procedure tricks**
(no restart, no PBT, no per-seed tuning).

## 1. What the research agrees on (high confidence)

1. **TD-MPC2 already implements self-predictive representation learning** (latent-consistency
   loss = SPR/DeepMDP-class; Ni et al. 2401.08898 group them together). → **Family D
   (self-predictive/contrastive) is REDUNDANT. Do not pursue.** Unanimous (all 3 reports).
2. **SimNorm already gives sparse, quasi-discrete, collapse-safe latents.** → plain
   latent-similarity clustering / sparsification (the *original* Glass) largely **duplicates
   SimNorm** and won't add a new signal. This is the core reason Glass hasn't beaten vanilla.
3. **Family A (MDP homomorphisms / symmetry)** has no demonstrated clean win on TD-MPC2's turf;
   needs known symmetries proprio control lacks. → **Drop.**
4. **Fair protocol is non-negotiable**: `rliable` IQM + 95% stratified-bootstrap CIs, **≥5
   seeds**, **sample-efficiency curves** (score vs steps at fixed budget), single-variable,
   compute-matched. **Drop HopperHop best-of-peak** (basin-entry luck, not representation;
   the DMC-Hopper critique 2410.08870 confirms it's luck-dominated). Unanimous.
5. **The one signal TD-MPC2 genuinely lacks**: its consistency loss only constrains
   *consecutive* latents (z_t→z_{t+1}); it never forces **arbitrary** behaviorally-equivalent
   states across the buffer to be close, nor task-irrelevant variation out. That gap is the
   only place an abstraction can add value.

## 2. Where the research splits (the real decision)

| | **Family B — behavioral / bisimulation** | **Family C — temporal / hierarchy** |
|---|---|---|
| Champions | dr-claude, dr-gpt, 3/4 internal agents | dr-gemini |
| Direct TD-MPC prior art | **BS-MPC (2410.04553)** — same architecture + a bisimulation encoder loss; beats TD-MPC on Dog/Humanoid + distractors, ~15–20% *faster* | Director (2206.04114) on Dreamer; **no** direct TD-MPC2 result |
| Win location | high-dim locomotion stability + **distractor robustness** + transfer | **long-horizon sparse** (Ant Maze, Crafter) where flat TD-MPC2 = 0% |
| Win shape | incremental (vs TD-MPC2 specifically: "comparable/slightly better") | **step-function** (0%→high) — most reviewer-undeniable |
| Risk | thin margin vs TD-MPC2; bisim collapse/instability; per-task c₄ | abstract-model exploitation (2406.00483 = flat ties/wins); non-stationarity; hierarchy hurts dense tasks; big build |
| Compute | feasible — BS-MPC uses **permuted-pair O(B)**, not O(B²) | manager+worker+goal-autoencoder = large addition |
| Fit to *our* "Glass" | **direct** — Glass becomes behavior-grounded clustering | would replace Glass with a Director-style hierarchy |

## 3. The core reframe (the contribution)

**Re-ground Glass from latent-similarity clustering → behavior-aware hierarchical state
abstraction.** Concretely: prototypes/clusters should group latents that are **behaviorally
equivalent** (similar immediate reward + similar next-latent distribution + similar value),
*not* latents that are merely close/visually similar (SimNorm already does that).

- **Novelty vs BS-MPC**: BS-MPC is a *pairwise* bisimulation loss. Glass is a **hierarchical,
  prototype-based soft bisimulation** — structural entropy supplies a multi-scale,
  minimal-description partition over a *behavior* graph (edges = reward+next-latent similarity).
  The hypothesis worth a paper: **hierarchical behavioral abstraction > pairwise bisimulation**
  (more stable, global, multi-scale) — but we must *show it beats BS-MPC*, not just vanilla.
- This is the only framing where the existing Glass machinery (prototypes, super-clusters,
  structural-entropy loss) survives the "TD-MPC2 already does that" objection.

## 4. Decision

- **Primary bet: behavior-aware Glass (Family B reframe).** It keeps the Glass thread,
  compute-feasible, has a direct must-beat baseline (BS-MPC), and a clean test bed.
- **Must-beat baselines (both)**: vanilla TD-MPC2 **and** a BS-MPC-style pairwise bisimulation
  auxiliary. Beating vanilla alone is not enough — we must beat pairwise bisimulation to claim
  the *hierarchical* abstraction earns its place.
- **High-ceiling fallback (Family C)**: if behavior-aware Glass only reaches parity with
  vanilla/BS-MPC on the locomotion+distractor axes, pivot to a Director-style temporal
  hierarchy on a long-horizon sparse task (Ant Maze) where the win is step-function — but
  accept the heavy negative evidence (2406.00483) and the larger build.

## 5. Fair protocol (fixed before running)

- **Single variable**: official-equivalent TD-MPC2; the ONLY change is the abstraction loss.
  Match param count, grad steps, replay ratio, planner budget, env steps, **report compute
  overhead explicitly**. (BS-MPC's "only the c₄ term differs" is the gold standard.)
- **Metric**: `rliable` IQM + 95% CI, performance profiles, probability-of-improvement; ≥5
  seeds (target 8–10 for separated CIs on high-variance DMC); sample-efficiency curves +
  AUC, not best-of-peak. Pre-register the G-threshold/metric.
- **Mechanism probes** (turn "number went up" into "mechanism works"): (a) latent distance ∝
  reward+next-latent (behavioral) distance; (b) prototypes group behaviorally-equivalent
  states; (c) under distractors, probe accuracy high on task factors, low on nuisance.

## 6. Benchmarks

- **Stage 1 (feasible on current proprio infra)** — DMC proprioceptive **hard-locomotion**
  sample efficiency: Dog Run, Dog Trot, Humanoid Walk, Humanoid Run, Acrobot Swingup,
  Quadruped Run + 3–4 easier (Walker Run, Cheetah Run, Finger Spin, Reacher Hard) for breadth.
  This is exactly where BS-MPC/DC-MPC show their TD-MPC(2) edge (stability on Dog/Humanoid).
- **Stage 2 (needs pixel-obs + video-bg infra)** — Distracting Control Suite / DMC-VB: the
  cleanest *isolation* of behavioral abstraction (nuisance held variable, capacity/exploration
  fixed). Bigger lift; do only if Stage 1 clears the gate.
- **NOT HopperHop** for claims (keep only as a smoke test).

## 7. Staged experiments + go/no-go gates

- **Stage 0 — reproduce + reference (foundation).** (a) Confirm our JAX/Flax TD-MPC2 reproduces
  published vanilla TD-MPC2 on 4–6 DMC proprio tasks incl. Dog Run / Humanoid Walk (3 seeds);
  (b) implement the **BS-MPC permuted-pair bisimulation auxiliary** as the reference arm.
  Gate: baseline within noise of published curves; bisim arm trains stably.
- **Stage 1 — behavior-aware Glass vs {vanilla, BS-MPC}** on the hard-locomotion suite, ≥5
  seeds, IQM@500k & 1M + curves. **GATE**: Glass IQM beats vanilla with separated 95% CIs on
  sample-efficiency OR final, on the locomotion subset, AND is ≥ BS-MPC.
- **Stage 2 — distractor robustness** (if Stage 1 passes): pixel + video-bg; show Glass > both.
- **Stage 3 — transfer/multi-task** (optional): multi-task TD-MPC2 ± Glass; few-shot to
  held-out tasks (abstraction should help most where representation is shared).
- **KILL criterion (pre-registered)**: if, under compute-matching + ≥5 seeds, behavior-aware
  Glass does NOT beat vanilla on ANY axis with separated CIs — and does not beat BS-MPC —
  the hierarchical-behavioral abstraction does not earn its place. Report the negative result
  honestly; do NOT rescue it with a procedure trick. Then consider the Family-C fallback.

## 8. Implementation roadmap (our JAX/Flax codebase)

1. **DMC harness**: verify Dog/Humanoid/Quadruped/Acrobot proprio tasks available via
   mujoco_playground/dm_control; add task configs + per-task eval. (Pixel/distractor infra is a
   separate later milestone for Stage 2.)
2. **rliable aggregation**: add IQM/CI/perf-profile reporting over ≥5 seeds to the analysis
   tooling (read-from-JSON discipline — see [[nbeatsx-dc-verification]]).
3. **BS-MPC reference loss**: permuted-pair π*-bisimulation encoder loss term (the must-beat).
4. **Behavior-aware Glass (core change)**: replace latent-similarity prototype assignment with
   **behavioral** assignment (reward + next-latent + value), structural-entropy over the
   behavior graph. Keep stop-gradient targets; watch for embedding collapse (norm constraint).
5. **Compute parity harness**: param-count + grad-step + wall-clock matching; overhead report.

## 9. What runs NOW (keeps fleet busy, on-thesis)

- Keep the current **clean HopperHop vanilla-vs-Glass batch** finishing — it's the last useful
  HopperHop datapoint (confirms vanilla≈Glass, justifying the pivot) and keeps GPUs busy while
  the DMC harness lands. After it, the fleet redirects to **Stage 0 DMC reproduction**.
- First engineering task: Stage 0 step 1 (DMC task availability) — gates everything else.

## RESULTS (2026-06-06, pre-registered n≥6 verdict — recorded honestly)

**Setup run:** 4 arms (vanilla TD-MPC2 / bsmpc bisim-ref coef 0.01 / geoglass λ_behav=0 /
behavglass λ_behav=0.5) × {CheetahRun, WalkerRun, FingerSpin} (mujoco_playground/MJX), 1M
steps, eval@50k, compute-matched, no procedure tricks. Metric: per-task-normalized (/1000)
final MPPI, IQM + stratified-bootstrap 95% CI over mature (≥950k) runs.

| arm | n | FINAL IQM | 95% CI | @300k IQM |
|---|---|---|---|---|
| **behavglass** | 6 | **0.801** | [0.780, 0.828] | 0.344 (n=8) |
| vanilla | 15 | 0.756 | [0.730, 0.789] | 0.439 |
| geoglass | 9 | 0.757 | [0.690, 0.817] | 0.465 |
| bsmpc | 6 | 0.549 | [0.527, 0.599] | 0.289 |

**VERDICT (per the pre-registered separated-CI criterion): NEAR-MISS, not a win.**
behavglass leads vanilla by +0.045 IQM (≈6% relative) and led the per-task mean on every
task throughout, but its CI lower bound (0.780) overlaps vanilla's upper (0.789) by 0.009
at the n≥6 bar. The strong claim ("behavior-aware abstraction beats vanilla TD-MPC2") is
therefore **not established**; the kill-criterion fires for the headline claim. What
survives, with full CI separation: **behavglass ≫ bisimulation reference** (0.780 vs 0.599).

**Confirmed findings (stable across all n):**
1. **Geometric Glass ≡ vanilla** — geoglass converged to 0.757 vs vanilla 0.756. Latent-
   similarity prototype clustering is fully redundant with SimNorm, exactly as the
   deep-research synthesis predicted. This retroactively explains iterations 1–11.
2. **Bisimulation auxiliary hurts** on clean proprioceptive DMC (0.549, separated below all;
   coef 0.1 collapsed outright; 0.01 still loses) — consistent with the 2025 large-scale
   behavioral-metric study.
3. **TD-MPC2 is a near-optimal baseline** on these axes (Ni et al. 2401.08898's prediction
   borne out): neither geometric nor behavioral latent abstraction, nor bisimulation,
   produced a separated improvement.

**The honesty case-study (why the pre-registered bar mattered):** every apparent edge
regressed as n grew — behavglass FINAL IQM trajectory 0.818(n=3) → 0.736(n=4) → 0.749 →
0.829(n=5) → 0.801(n=6); its @300k "sample-efficiency edge" went +30% (0.572) → 0.462 →
0.344 (BELOW vanilla) as seeds accumulated; geoglass's early @300k edge evaporated the same
way. Calling any of the early snapshots would have been wrong. Small-n IQM with bimodal
outcomes is acutely composition-sensitive.

**HIGHER-N REVISION (2026-06-06 late, n=17 vs 23, 5 tasks — clearly marked as superseding
the n=8 "null is final" note below, which was itself premature):** with seeds 6–7 and the
breadth tasks (AcrobotSwingup, WalkerWalk) matured, **behavglass = 0.785 [0.771, 0.799] vs
vanilla = 0.734 [0.701, 0.762] — CIs SEPARATED at the largest sample.** The mechanism is
**seed-variance reduction (floor-raising), not peak-raising**: on the 3 main tasks behavglass
min=0.700 with 0/13 seeds <0.65, vs vanilla min=0.419 with 3/19 <0.65 (geoglass 2/12).
Behavioral grounding prevents weak-seed outcomes — the project's original "seed robustness"
question, answered positively on a fair benchmark. Full volatility trajectory (0.818→0.749→
0.829→0.801→0.793→0.749→0.785) is retained below as the small-n case-study; the n≈17–23
estimate with tight stratified-bootstrap CI is the credible one. Caveat: many interim looks
were taken (documented); the seed/task schedule was predetermined, not adaptively selected.

**FINAL UPDATE (2026-06-06, n=8 — the null is FINAL):** the convergence completed. At n=8,
behavglass = **0.749 [0.724, 0.766]**, geoglass = 0.752 [0.691, 0.808], vanilla = 0.756
[0.730, 0.789] — all three are statistically and numerically identical; behavglass's point
estimate is now *below* vanilla's. Full behavglass IQM trajectory: 0.818(n=3) → 0.736 →
0.749 → 0.829(n=5) → 0.801(n=6) → 0.793(n=7) → **0.749(n=8)** — a textbook
regression-to-the-mean curve terminating exactly at the baseline. **Conclusion: on clean
proprioceptive mujoco_playground DMC under a fair compute-matched protocol, NO latent
abstraction tested (geometric Glass, behavior-aware Glass, pairwise bisimulation) changes
final performance vs vanilla TD-MPC2; bisimulation actively hurts (0.549).** The earlier
near-miss paragraph above is retained as part of the honesty record.

**Disposition / open doors:** (a) ~~more seeds may revisit at n≥10~~ RESOLVED — converged
into the null at n=8; remaining breadth seeds (AcrobotSwingup/WalkerWalk) extend the null
to 5 tasks for the writeup;
(b) Stage-2 distractor-pixel setting remains the literature's best-supported home for
behavioral abstraction (untested here — needs pixel infra); (c) Family-C temporal
abstraction remains the designed fallback; (d) accepting the null on this benchmark family
is a legitimate, well-evidenced outcome.

## STAGE-2a (2026-06-06, user-directed): distractor robustness, minimal-falsification design

User directives: (1) test the distractor setting (the literature's actual home for behavioral
abstraction) — Stage 2a uses SYNTHETIC distractors first since madrona_mjx (pixels) is a heavy
uninstalled build; (2) NEW STANDING DEV-LOOP RULE: run the MINIMUM experiment that can safely
falsify a stage before scaling.

**Mechanism**: `--distractor_dims N` appends N temporally-correlated OU nuisance dims
(scale=1, rho=0.95, JIT/vmap-safe via state.info) to the observation. Hypothesis: vanilla's
encoder wastes capacity on predictable nuisance; behavioral grounding filters it.

**Kill-probe (4 runs, ~6 GPU-h total)**: vanilla vs behavglass × CheetahRun+64dims × 2 seeds
× 500k steps. Reference clean curves @500k are known (vanilla ≈0.60, behavglass ≈0.55-0.68).
GATE: if both arms degrade equally → hypothesis falsified cheaply, NO Madrona build, Stage-2
closes. If behavglass degrades clearly less → scale up (more seeds/tasks), then justify
Madrona pixel infra (Stage 2b).

**Round-1 result (2026-06-06, CheetahRun+64dims, pre-registered metric = per-arm mean of
last-2-eval averages):** distractors crush BOTH arms (clean ~550 @500k → distracted
60–340). vanilla = 153 (seeds 245, 61); behavglass = **233** (216, 250) → lands in the
pre-registered AMBIGUOUS band [230, 306) by 3 points. Honest read: behavglass's seeds equal
vanilla's good seed; the arm difference is entirely vanilla's one dead seed — n=2 ambiguity
as designed. **Per pre-registration: exactly one round-2 pair queued (WalkerRun+64dims,
same gates, 4 runs).** Total Stage-2a cost so far: 8 short runs.

**STAGE-2a FINAL VERDICT (2026-06-06): FALSIFIED.** Round-2 (WalkerRun+64dims): vanilla=90
(111, 69), behavglass=**67** (62, 72) — behavglass came out WORSE. Combined over rounds:
vanilla 121, behavglass 150 → ratio **1.23× < 1.5 gate → falsified**. The r1 "edge" did not
replicate; it reversed. Conclusion: **behavior-aware Glass does NOT confer robustness to
synthetic (OU) observation distractors — both encoders are equally crushed** (clean ~550 →
distracted ~60–250). Stage-2a closed at a total cost of 8 × 500k runs (~12 GPU-h) — the
minimal-falsification protocol working as intended (iteration-14's null cost ~60 × 1M runs).
A dose-response study (0/16/32/64 dims × both arms) is completing as writeup material;
early read: 16 dims is a moderate dose (vanilla still reaches ~430@300k), 64 is crushing.
**Next direction (user decision pending): Family-C temporal abstraction / write up the
nulls / pause / Madrona pixels anyway (note: with the synthetic-distractor null, the prior
for pixel-distractor gains is now lower).**

## FAMILY-C TESTBED PROBE — VERDICT (2026-06-06): no demonstration ground; build skipped

Probe: vanilla TD-MPC2 on the three sparse mujoco_playground tasks, 2 seeds × 1M.
Strict gate (flat ~0 everywhere) **FAILED**: AcrobotSwingupSparse reaches ~215 unstably
(tails 40–177) on both seeds; CartpoleSwingupSparse and BallInCup are seed-bimodal
(818/0 and 975/0). Flat TD-MPC2 does not *structurally* fail on any available task →
**SPlaTES-style temporal abstraction has nothing to demonstrate in this env library; the
jumpy-K build is skipped** (weeks saved by a 6-run probe). **Discovered instead:** the
sparse tasks exhibit literal 0-vs-solved seed bimodality — the sharpest available testbed
for the floor-raising finding — and were redirected into the SPARSE-FLOOR program
(behavglass × 3 sparse tasks × 3 seeds vs flat n=3).

## DECISION (2026-06-04, user): **Family B — behavior-aware Glass** is the primary bet.
C (temporal hierarchy) is the documented fallback if B only reaches parity vs {vanilla, BS-MPC}.
Next action: Stage 0 step 1 — verify DMC hard-locomotion proprio tasks (Dog/Humanoid) exist in
our worker infra (via mujoco_playground; do NOT add dm_control — it drifts mujoco off 3.8.0).
