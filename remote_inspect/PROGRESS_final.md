# Experiment #5 — live option-controller + learned residual (Markov in (s,z)) — RESULTS

GOAL: beat PPO by keeping the abstraction IN the loop. Dual criterion:
  (A) reach >= 0.82 real success, AND (B) cross 0.66 in < 33M env-steps.

## Verdict: NO-GO on the dual criterion (but the structural cap WAS broken)

| variant | alpha | PEAK real success | mean_max_bt | cross 0.66 | vs anchors |
|---|---|---|---|---|---|
| **fixed alpha=0.5** | 0.5 const | **0.484 @ 36M** (rising) | 0.84 | never | **2x #2 cap, 4x #1** |
| anneal5k (0->1 @10M) | 0->1 | 0.074 (collapses to 0.008) | 0.37 | never | FAILS |
| anneal10k s1 (0->1 @20M) | 0->1 | ~0.098 (declines as alpha->0.8) | - | never | FAILS |
| anneal10k s2 | 0->1 | (confirmatory, same pattern) | - | never | FAILS |

Anchors (real success, same protocol/step-counter): scratch PPO 0.809 peak, crosses
0.66 at 32.77M (the BAR); #2 options 0.24; #1 residual 0.11; HL controller 0.0625.

## Findings
1. **Markov-in-(s,z) residual DOES break the abstraction ceiling.** Constant
   alpha=0.5 reaches ~0.48 real success — double the #2 option-abstraction cap
   (0.24), ~4x the #1 raw residual (0.11). Keeping the controller live + feeding
   phase z into the obs works to exceed the structural cap.
2. **But it does NOT match PPO's 0.82, and never crosses 0.66** -> dual criterion
   not met. The alpha=0.5 authority cap limits final-placement precision:
   mean_max_box_target ~0.84 (boxes get very close) but success(>=0.9) stalls ~0.48.
3. **The alpha-ANNEALING mechanism (the planned design) BACKFIRES.** As alpha->1,
   real success COLLAPSES (anneal5k 0.074->0.008; anneal10k peaks ~0.10 at
   alpha~0.5 then declines). Annealing = non-stationary executed-action
   distribution + removes the controller scaffold before the residual is a
   competent standalone policy.
4. **Implication / next lever:** controller-in-the-loop helps as a PERSISTENT
   scaffold (constant moderate alpha), NOT a vanishing prior. Try a HIGHER fixed
   alpha (0.7-1.0) or a LEARNED per-state alpha gate, not a global time-anneal.

## Verification
- Smoke: alpha=0 + zero residual -> 0.0938 real success (== controller baseline,
  NOT 0 -> phase z threaded & controller live; conditioning correct).
- All numbers from eval_residual_curve.py real rollouts (n=256, 1000-step,
  box_target>=0.9), training-matched alpha. JSON-validated. PEAK+FINAL reported.

## Artifacts
- Code: helios-rl/hl_pickcube/residual_env.py; baselines_ppo_sac/{run_ppo_residual,
  residual_patch,eval_residual_curve}.py
- Curves: hl_subgoal/{curve_fixed0p5,interim_anneal5k,curve_s1_anneal10k,
  curve_s2_anneal10k,curve_s1_anneal5k}.json ; RESULTS.json ; this PROGRESS.md
