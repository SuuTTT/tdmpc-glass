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
