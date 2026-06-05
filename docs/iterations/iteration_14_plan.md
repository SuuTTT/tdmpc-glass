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

## DECISION (2026-06-04, user): **Family B — behavior-aware Glass** is the primary bet.
C (temporal hierarchy) is the documented fallback if B only reaches parity vs {vanilla, BS-MPC}.
Next action: Stage 0 step 1 — verify DMC hard-locomotion proprio tasks (Dog/Humanoid) exist in
our worker infra (via mujoco_playground; do NOT add dm_control — it drifts mujoco off 3.8.0).
