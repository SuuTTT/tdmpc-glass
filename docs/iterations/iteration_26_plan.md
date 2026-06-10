# Iteration 26 — Value-equivalent macro head (probe #2)

*2026-06-10. The jumpy k-step head trained to preserve VALUE (V=min-Q(z,pi(z))), not just reconstruct the
latent. Single-variable: --jumpy_ve_coef adds L_ve=(V(jpred)-sg V(z_true))^2 to the jumpy loss (Q/pi
stop-gradded -> trains only jdyn). Smoke PASSED (trains clean, no NaN, jumpy mechanism intact).*

## Claim / why it could win where others didn't
Value-Equivalence Principle (Grimm 2020/MuZero): a model only needs to predict return-relevant structure.
State-faithful jumpy wastes capacity reconstructing task-IRRELEVANT latent dims (e.g. distractors). A
value-equivalent head keeps only what affects macro-Q -> should be more robust under distractors / limited
capacity. NOT killed by prior nulls (adaptive-k=uniform-error; exploration=coverage; spline=search-shape).

## Pre-registered gate (single-variable: jumpy_ve_coef 0 vs 0.5; else identical jumpy k4)
Arms: jumpy_base (ve=0) vs jumpy_ve (ve=0.5). Tasks (the discriminator = distractors, where VE should help):
  CheetahRun+32distractors, WalkerRun+32distractors (VE should beat state-faithful), PandaPickCube clean
  (NO-REGRESSION check — VE must not hurt the existing win). 4 seeds, 500k, peak+final, CI.
WIN = jumpy_ve > jumpy_base by >=10% (peak AND final), CI-sep, on >=1 distractor task AND no regression on Panda.
If tie on distractors -> value-equivalence adds nothing here (honest null). If hurts Panda -> VE over-restricts.

## Status
- [x] code + smoke (trains clean). [ ] gate (queued, idle fleet). HOLD nothing — user said build+run both.
