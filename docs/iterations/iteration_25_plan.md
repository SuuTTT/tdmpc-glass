# Iteration 25 — Hermite-spline action bottleneck (probe #1, non-SE)

*2026-06-10. SE-flavored levers are exhausted (adaptive-k iter-23 null; SE-exploration iter-24 null/harmful).
Pivot to ACTION-space abstraction, untouched by those nulls. Gemini DR's lever: parametrize the planner's
action sequence as a smooth cubic-Hermite spline instead of k independent action vectors.*

## The claim (single-variable, falsifiable, vs the jumpy/MPPI baseline)
MPPI samples a length-H action sequence as H×act_dim independent Gaussians — a high-dim, temporally-jagged
search. **Replace it with a cubic-Hermite spline over a few control points** (e.g. start+end action +
tangents): the planner searches ~2·act_dim params instead of H·act_dim, and every sampled rollout is a
smooth, physically-plausible action trajectory. Expected: better sample-efficiency / higher return at a
matched MPPI sample budget, because the search space is lower-dim AND in-distribution (smooth).

## Why novel (precedent honesty)
- Alvarez-Padilla et al.: cubic-Hermite-spline MPPI, but in RAW joint/action space with an analytic MuJoCo
  sim (no learned model). TAP/PLAS: learned latent action codecs (online repr-shift; we'd avoid that).
- NEW: a spline action bottleneck INSIDE a learned-world-model planner (TD-MPC2/jumpy MPPI), **no learned
  codec** (so no online representation drift — its edge over TAP/PLAS), single-variable vs MPPI. Not killed
  by the uniform-error finding (this is about search-space shape, not model error).

## Mechanism-check FIRST (the kill-test, cheap, before any multi-seed build)
The whole bet rests on: **are good action trajectories smooth enough that a few-control-point Hermite spline
expresses them without losing return?** If the optimal policy is bang-bang/jagged, the spline bottleneck
caps achievable return -> dead on arrival.
- Load a trained checkpoint (Panda jumpy = the task where we have a real win; + a DMC locomotion ckpt).
- Run eval rollouts recording the executed action sequence + return.
- Fit cubic-Hermite splines with K in {2,3,4,6} control points; measure (a) reconstruction R^2 of the
  action trajectory, (b) RETURN when the smoothed actions are replayed open-loop from the same init.
- KILL CONDITION: if even K=6 control points loses >~25% return on replay (splines can't express the
  policy) -> the bottleneck is too restrictive, abandon. PASS: K<=4 preserves >=85% return + high R^2
  -> build the spline-MPPI and run the beat gate.

## Pre-registered beat gate (only if mechanism-check passes)
spline-MPPI vs vanilla-MPPI (same n_samples, same model), single-variable, rliable IQM, 5 seeds, peak+final,
CI, on PandaPickCube + 2 DMC locomotion (CheetahRun/WalkerRun) + 1 sparse. WIN = spline-MPPI >= +10% IQM
(sample-eff to threshold OR asymptote) with non-overlapping CI on >=2/4 tasks. If tie -> honest null
(smoothness neither helps nor hurts); if worse -> bottleneck too tight.

## Honest prior
~30%. Spline-MPPI works in analytic-sim locomotion (Alvarez-Padilla), but (a) TD-MPC2's MPPI is already
warm-started by the policy prior (less to gain from a better search distribution), and (b) contact-rich
manipulation may need non-smooth actions. The mechanism-check decides cheaply before fanout.

## Status
- [ ] mechanism-check: HERMITE_CHECK mode in run_benchmark (env-gated, reuses resume path) — record actions,
      fit splines, replay, report R^2 + return-preservation per K. py_compile + offline spline-fit self-test.
- [ ] run on a (now-idle) box vs a Panda + a DMC checkpoint -> verdict.
- [ ] if PASS: spline-MPPI in make_mppi_fn behind --spline_ctrl flag; beat gate. If FAIL: record null, ledger.
- HOLD multi-seed fanout for explicit user go (mechanism-check first).

## MECHANISM-CHECK RESULT (2026-06-10): LEAN-NEGATIVE / low-EV; open-loop proxy INVALID for closed-loop
Panda + CheetahRun, pi-rollout actions, K control pts per H=8 window:
  CONTROL (replay ORIG actions open-loop) = 100% of free on BOTH -> envs deterministic but
  ACTION-PRECISION-CHAOTIC (~4% action error collapses an OPEN-LOOP rollout).
  Hwin_R2:  Panda K2 0.13 / K3 0.65 / K4 0.80 / K6 0.90 ; Cheetah K2 0.27 / K3 0.79 / K4 0.90 / K6 0.96
  spline open-loop replay: 2-14% of control even at K6.
INTERPRETATION:
 (1) The spline open-loop-replay COLLAPSE is NOT a valid kill — it's precision-chaos (control proves it);
     spline-MPPI is CLOSED-LOOP (re-plans each step) so it would NOT suffer open-loop compounding. The
     cheap open-loop proxy cannot decide a closed-loop lever -> no valid cheap kill/pass.
 (2) Valid signal = Hwin_R2. The lever's PREMISE (big search cut, K~2 << H*d) FAILS: at K=2 spline fits
     action windows poorly (0.13-0.27); need K=4 for decent fidelity (0.80-0.90) = only 2x compression
     over H=8. TD-MPC2 MPPI is already policy-prior warm-started -> small upside from a better search dist.
VERDICT: lean-negative, low expected value. A definitive test REQUIRES building closed-loop spline-MPPI
(--spline_ctrl in make_mppi_fn) + the beat gate; the cheap proxy is invalid. Given (a) only 2x search
reduction viable, (b) warm-started MPPI baseline, expected upside is low. RECOMMEND: pivot to probe #2
(value-equivalent macro head — a clean single-variable LOSS change with a valid cheap mechanism-check),
unless user wants the spline-MPPI build. HOLD for user decision.
