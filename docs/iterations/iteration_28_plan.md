# Iteration 28 — value-organized abstraction: mechanism-checks + coef sweep

*2026-06-10. Tests the "value-organized abstraction" thesis (organize the abstraction around value/
decisions, not state-prediction) — the one axis the campaign hadn't fairly tested. Two cheap kill-tests
on a trained PandaPickCube checkpoint via `scripts/value_probe.py` (standalone; no hot-path edits).*

## Mechanism-check RESULT (n_ep=12, 12k states, self-predictive PandaPickCube ckpt; from JSON)
**Probe 1 — value-equivalence headroom (gates the `ve` lever):**
- `linear_V_decode_r2 = 0.9994` → value is ALREADY perfectly decodable from the latent.
- `effective_dim_latent ≈ 6.96`, `effective_dim_value_subspace ≈ 7.08`, `value_irrelevant_variance_frac ≈ 0.978`.
- Read: the self-predictive latent is already value-sufficient. Nothing for value-equivalence to *add*.

**Probe 2 — value-criticality variation (gates the value-critical adaptive-horizon lever):**
- `crit_cv = 0.36` (< 0.5 bar), `flat_state_frac = 0.029` → criticality near-uniform.
- Read: little for an adaptive horizon to gate on — the same "nothing to adapt to" that killed error-gated adaptive-k (ledger #10–11).

**Cross-check vs returns (iter-26 ve vs vebase, MPPI):** value-equivalence @coef 0.5 HURTS —
PandaPickCube `ve` 1616/916 vs `vebase` 2692/2243 (Δ −1076 peak / −1327 final); CheetahRun −129/−83.

**VERDICT:** both value-organized levers fail their mechanism-check. Same root cause as all 13 prior
nulls — a strong self-predictive world model (TD-MPC2 + SimNorm) already encodes a value-sufficient
abstraction. The probes now *quantify* it (V decodable at R²≈1.0), which is the spine of the analysis paper.

## Live: cheap coef sweep (the last-gasp on the win-path, user-chosen)
On the **dedicated 5070 Ti** (ssh7, reserved out of the daemon pool): `jumpy_ve_coef ∈ {0.05, 0.1, 0.2}`
× PandaPickCube × seed 0, single-variable vs vebase(0) / ve(0.5). Driver `/root/helios-rl/sweep_driver.sh`
(nohup'd). Gate: if any coef ≥ vebase at seed 0 → expand to seeds 1,2; else value-equivalence is closed.
Honest prior: low (mechanism-check predicts redundant/neutral-at-best).

## Operational notes
- 5070 Ti reserved as MAIN DEV PLATFORM (daemon now polls 9 boxes); replaces the contended ssh4 for probes/dev.
- PandaRobotiqPushCube DROPPED from ti27 (Warp Robotiq sim OOMs across fleet GPUs even at MEM_FRACTION=0.5);
  suite now Pick / Cart(floored ~10) / Ori / Cabinet → effective discriminators = Pick, Ori, Cabinet.
- `ti27_jum_Cart20` finished on ssh7 (final MPPI ~10 = Cartesian floored); marked done manually.

## If the sweep is null → pivot
The analysis/understanding paper ("a strong self-predictive WM is already value-sufficient — why 13
explicit-abstraction levers were redundant") is the ICLR-viable output; the only remaining headroom the
theory permits is high-DoF (more value-irrelevant capacity), gated by a high-DoF value_probe before any big-budget run.

## COEF SWEEP RESULT (2026-06-10, read from ssh7 CSVs, PandaPickCube seed0 MPPI)
| coef | peak | final | vs vebase(2692/2243) |
|---|---|---|---|
| 0.05 | 2752 | 1939 | peak ~tie, final −304 |
| 0.1  | 2638 | 2118 | peak −54, final −125 |
| 0.2  | 2084 | 1930 | both worse |
| 0.5  | 1616 | 916  | much worse |
**VERDICT: value-equivalence lever CLOSED = NULL.** Monotone degradation with coef; no coef beats the
jumpy baseline on peak AND final (best case 0.05 ties peak, loses final). Confirms the iter-28 mechanism-check
(latent already value-sufficient, linear V-decode R²=0.9994 → VE redundant; trades off vs consistency the
planner needs). n=1/coef (seed0) — direction unambiguous, not worth more compute. Both value-organized
levers (value-equivalence + value-criticality) now closed; the win-path is exhausted → understanding paper
is the output, graph-WM+SE the sequel.

## VE LATENT PROBE (ve01 ckpt, 2026-06-10): value_irrelevant_frac 0.978(selfpred)→0.953(ve), linR2=1.0
The value-equivalence loss barely reorganized the latent (95% still value-irrelevant variance) and value
stays perfectly linearly decodable (R²=1.0) — i.e. VE didn't even meaningfully change the representation,
let alone help. Mechanistic confirmation of the null: the latent was already value-sufficient; the VE term
just traded off against the consistency the planner needs (hence the return regression).

## DREAMER4 / transformer-WM PERF BLOCKER (2026-06-10) + resmlp arch lead
- DreamerV4 transformer-WM: PE bug fixed, imports+inits clean, but the 60k dev run on the 5070ti ran
  ~65min at **1% GPU util** (no ckpt) → the per-step Python collection/eval loop is dispatch-bound
  (GPU-starved), as the build agent warned. NEEDS a vectorized/lax.scan collection loop before it's a
  usable WM. The SE-over-attention-graph north-star result is GATED on this perf fix (deferred to a
  hands-on session; scripts/se_attention_graph.py not yet written — no trained transformer-WM to run it on).
- ARCH A/B (the campaign's ONE positive signal): **van/resmlp 2699/1561 vs van/mlp 1925/1238** on
  PandaPickCube @≥450k, n=2 each → +40% peak / +26% final. Confirm seeds (Pick3,4 + Ori/Cab) queued
  (ti27a2_*). If it holds at n≥4 + generalizes, it's the paper's constructive contribution (a deeper
  gated-residual dynamics backbone beats the MLP) alongside the negative abstraction thesis. van/attn
  hurts (1627/900); jum/attn helps vs noisy baseline.
- Generality: value_probe running on a CheetahRun (DMC locomotion) jumpy ckpt — does value-sufficiency
  hold beyond Panda manipulation? (result next harvest).

## JUMPY GENERALIZATION firming (2026-06-10, Panda suite @≥450k, MPPI)
jumpy vs vanilla FINAL return: Pick 2458 vs 1238 (+99%, n=2), Ori 2170 vs 982 (+121%, n=1),
Cab 1261 vs 596 (+112%, n=1). Jumpy SUSTAINS final return where vanilla degrades, across all 3
Franka tasks; PEAK is mixed (jum wins Pick peak, ~ties Ori, loses Cab peak). The jumpy-WM win
(prior art, Farebrother 2026) GENERALIZES across the manipulation suite on final return. n building to 5.
Note: jumpy final (2458) > resmlp final (1561) on Pick → resmlp beats MLP but not jumpy; the open
question = does resmlp help ON TOP of jumpy (jum/resmlp vs jum/mlp, ti27a2_jum_resmlp_* queued).

## ARCH A/B VERDICT (2026-06-10, PandaPickCube @≥450k, n=2 each) — resmlp NULL (mirage)
| config | peak | final |
|---|---|---|
| van/mlp    | 1925 | 1238 |
| van/resmlp | 2699 | 1561 |
| jum/mlp    | 2645 | 2319 |  ← BEST
| jum/resmlp | 1796 | 1381 |
**resmlp helps the WEAK vanilla baseline (+40%/+26%) but HURTS the strong jumpy model
(jum/resmlp 1796/1381 ≪ jum/mlp 2645/2319). attn likewise: van/attn 1627/900 < mlp, jum/attn
2081/1184 < jum/mlp.** Neither backbone improves the best config → NOT a real architecture win;
best config stays jum/mlp. Classic "helps weak baseline, not the strong one" mirage. n=2 caveat but
the jum/resmlp gap (~−850 final) is far beyond noise. The arch lever is closed; the campaign's only
win remains the (prior-art) jumpy WM. Lesson reaffirmed: always test a lever against the STRONGEST
config, not the weak baseline (this is why the resmlp lead looked real for 3 harvests).

## VE coef-latent trend complete (2026-06-10): value_irrelevant_frac vs coef
self-pred 0.978 | ve@0.1 0.953 | ve@0.2 0.977 — VE barely moves the value-irrelevant fraction at ANY
coef AND value stays decodable (R²≈1.0). The loss doesn't reorganize the latent toward value-sufficiency
because it's ALREADY value-sufficient. Triple confirmation of the null (mechanism + returns + latent).
