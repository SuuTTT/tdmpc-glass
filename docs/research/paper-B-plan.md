# Paper B — Control-Aware Compositional Abstraction for World Models

*2026-06-15. The positive companion to Paper A (the redundancy criterion). A synthesizes every null
into "when is abstraction redundant"; B asks the constructive question — "build the abstraction that
*isn't* redundant" — and is disciplined by every lesson we paid for.*

## The thesis (positive, falsifiable)
A world-model abstraction improves **control** iff it is **(1) control-coupled** (built from
sensitivity-of-return, not visual/latent *similarity*) **and (2) compositionally targeted**
(object-factored + permutation-equivariant, so it transfers to held-out object configurations).
On the in-distribution, value-sufficient regime it is redundant (Paper A). The single headroom is the
**control-coupled × compositional** intersection — everywhere else, a converged monolithic latent
already suffices.

## Why this is the right bet (every clause is a paid-for lesson)
- **"Control-coupled, not similarity":** our value-coupling probe failed *because* similarity graphs
  don't recover task structure (iter-34); SimNorm's SE structure was real but motion-phase, not value
  (Paper A). Abstraction must be wired to return.
- **"Compositional / OOD":** the *only* place monolithic value-decodability collapses (synthetic OOD)
  and the *only* place graph latents showed a real edge (iter-34: graph OOD value-R² holds 0.57 vs
  monolithic 0.21). SOLD's own Table 1 agrees — TD-MPC2 ≈ SOLD on non-relational; graph wins only on
  the relational/Distinct variants.
- **"Control, not representation":** the recurring killer — representation advantages don't convert
  (iter-34 control-benefit null; iter-36 inconclusive). B's headline metric must be **return at
  held-out object count**, never a probe R².
- **"Object-factored + permutation-equivariant":** the structural reason a graph WM *can* extend to
  unseen counts where a fixed monolithic vector cannot.

## Contributions
1. **Principle:** control-coupled × compositional as the necessary-and-sufficient headroom condition
   for WM abstraction (the positive of A's criterion).
2. **Method:** an object-factored WM with a *control-coupling* abstraction objective
   (edge/feature weight ∝ |∂return/∂interaction|, the value-coupling idea done at the dynamics level)
   trained for compositional transfer — vs a param-matched monolithic WM.
3. **Benchmark + protocol:** compositional-OOD control (train N objects, test N′) with our cheap
   pre-registered **diagnostics as go/no-go gates** — value-decodability-OOD-collapse and the
   disc_err_gap calibration signal — so the method is only deployed where the criterion predicts it
   helps.
4. **Honest negative controls (the methodology contribution):** similarity-abstraction, in-distribution,
   and representation-only baselines that *don't* work — plus mechanism-check-before-fanout,
   pre-registration, fair-control (we caught the calibration "flip" + the pooling artifact), and
   reproduce-before-cite (we reproduced SOLD). This discipline is itself a contribution.

## What we already have (de-risks B)
- The criterion + cross-axis null campaign (A).
- The graph OOD-representation advantage (iter-34) — the seed result.
- SOLD reproduced (98–100%) → legitimate baseline + Table-1 corroboration.
- The diagnostics (disc_err_gap, value-decode R², SE-gap-vs-null) — the gates.
- A GPU-vectorized object-factored WM + monolithic baseline (entity_wm/monolithic_wm) + a
  Dreamer-style learner (coods_train.py) — the machinery, already running on Blackwell.

## The one missing result (what makes B a paper, not a plan)
**Demonstrate the method beats monolithic on compositional-OOD CONTROL on an adequate benchmark.**
iter-36 was *inconclusive* — our synthetic env's control-signal ceiling (~10%, within noise) +
BPTT-NaN meant no controller separated from random, so WM quality had no leverage. B needs a benchmark
with: **(a) variable object count, (b) high control-SNR (a competent policy clearly beats random),
(c) GPU-fast sim** (to avoid SOLD's mujoco-py single-core wall).

## The benchmark decision (this is the unlock)
**ManiSkill (GPU-vectorized sim), not SOLD's mujoco-py suite.** The SOLD wall was mujoco-py
single-core (~200 steps/hr); ManiSkill runs thousands of envs on-GPU (we measured ~5k SPS PPO). Pick a
*learnable* multi-object ManiSkill task with controllable object count (e.g. stacking/pushing with a
configurable number of cubes; PickCube-class so a policy actually learns — StackCube floored PPO, avoid
it as the *only* task). Train object-factored WM + monolithic at N objects, eval return at held-out N′.
This sidesteps both walls: GPU sim (no throughput wall) + real control signal (unlike our synthetic).

## Phased plan (mechanism-check gated, our discipline)
- **B0 (gate, ~days):** stand up a *learnable* variable-count ManiSkill task; confirm a vanilla
  monolithic WM-agent (TD-MPC2/DreamerV3-class) **beats random** there (the validity check iter-36
  lacked) and that value-decodability **collapses** at held-out N (the headroom signal). If no collapse
  or nothing beats random → STOP, fold into A as "no compositional headroom found."
- **B1:** object-factored WM vs monolithic, compositional-OOD return, ≥3 seeds, pre-registered CI gate.
- **B2:** add the control-coupling abstraction objective; ablate vs object-factored-only and vs
  similarity-coupled (the negative control). Show the control-coupling × compositional intersection is
  where the win lives.
- **B3:** the diagnostics predict the per-task outcome (disc_err_gap / OOD-R²-collapse → win), closing
  the loop with A.

## Honest odds + relationship to A
- **Paper A is done and self-contained** — ship it regardless.
- **Paper B prior: ~30–40%** of a clean positive (higher than past bets because the benchmark choice
  fixes the two walls that sank iter-35/36, and the regime is the one our own evidence flags). If B0's
  gate fails, B's effort *strengthens* A (another scoped negative) — so B is low-regret.
- Cost: ManiSkill GPU sim is cheap; the bottleneck is engineering (object-factored WM + the abstraction
  objective + compositional eval), ~weeks on one good GPU box (the 5070 Ti class), not the 3-week-per-run
  SOLD wall.

## Immediate next step
B0 on a fresh GPU box: ManiSkill learnable multi-object task + monolithic WM-agent validity check
(beats random?) + value-decode-OOD-collapse probe. One run decides whether B is live.
