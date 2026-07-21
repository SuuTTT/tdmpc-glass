# Exp#6 — Is the 0.79 abstraction-in-loop result CAPACITY/BUDGET-limited?

**Box:** ssh3 (vast 41721730), GPU0+GPU1 only. **Date:** 2026-06-24.

## Question
#5b/#5c reached **0.79 +/- 0.01** real success on PandaPickCube with an in-loop
residual `a = clip(a_option(s,z) + alpha*pi_res(s,z))`, alpha=1.0, phase z in obs
(Markov). The residual policy/value nets were SMALL (default manipulation config:
policy (32,32,32,32), value (256,256,256,256,256)). Near-tie with PPO 0.81 peak,
but no seed cleared 0.82. **Does a BIGGER residual net + LONGER training + dense
keep-best push peak above 0.81 (beat PPO) / above 0.82?**

## Bars (from FILES, baselines_ppo_sac/)
- PPO peak (20M-class reference): **0.81** (the standing bar).
- PPO @ very-long 100M (PPO_100_RESULT.json): peak **0.832** @108M, final 0.7188.
- #5b/#5c abstraction-in-loop alpha=1.0: **0.79 +/- 0.01** (the result under test).
- Analytic controller alone (alpha=0): ~0.05-0.09 success (residual off).

## Design (single variable = network capacity + budget)
- KEEP: alpha=1.0 fixed (RES_ALPHA_FIXED=1.0), same ResidualPickCube env (env-var
  knobs only, residual_env.py NOT edited), same stock reward/shaping.
- CHANGE: policy **(512,512,512,512)** [vs 32x4], value **(512,512,512,512,512)**
  [vs 256x5]; budget **70M** [vs #5b 40-50M]; eval+ckpt every ~2M (num_evals=35) so
  eval_residual_curve.py per-ckpt scan = implicit KEEP-BEST/peak tracking.
- 2 arms: big-net seed1 (GPU0), big-net seed2 (GPU1).
- Net sizes set via train_jax_ppo CLI flags --policy_hidden_layer_sizes /
  --value_hidden_layer_sizes (already plumbed; verified applied in ckpt
  ppo_network_config.json: policy [512,512,512,512], value [512,512,512,512,512]).

## SMOKE (done, ~3M steps each)
- big-net alpha=1.0: trains end-to-end ("Done training"), GPU 98-100%, ~1.2GB. OK.
- big-net alpha=0.0 sanity: PEAK success 0.0781 / reached 0.8281 (residual OFF =
  analytic controller baseline, far below the alpha=1.0 result). Sanity PASS.

## FULL runs (in flight)
- big_a1p0_s1  GPU0  pid 672737  log logs/run_big_a1p0_s1.log
- big_a1p0_s2  GPU1  pid 672738  log logs/run_big_a1p0_s2.log
- launched 2026-06-24 ~07:21 UTC, 70M steps, num_evals=35.

## EVAL (when training done)
`bash hl_beatppo_big/eval_big.sh`  (alpha=1.0, n=256, 1000-step, per-ckpt =>
peak/final/steps_to_cross_0.66) -> curve_big_a1p0_s1.json / curve_big_a1p0_s2.json.
Then make RESULTS.json with verdict vs 0.81 / 0.82.

## FULL RESULTS (from FILES: curve_big_a1p0_s{1,2}.json, n=256, alpha=1.0, 1000-step)
- Training overshot 70M target to ~111M (brax num_resets_per_eval inflation, same as
  PPO 20M->32.8M). 34 checkpoints/seed evaluated = full keep-best scan.
- Net VERIFIED in ckpt config: policy [512,512,512,512], value [512,512,512,512,512], obs 77.
- **s1: PEAK 0.6758 @104.9M ; FINAL 0.6445 @111.4M ; crosses 0.66 @104.9M**
- **s2: PEAK 0.6328 @65.5M ; FINAL 0.5508 @111.4M ; never crosses 0.66**
- best peak 0.676 ; mean peak 0.654.
- Shaped TRAIN reward climbed high (~900) but did NOT translate to real success >0.79
  (grasp/lift shaping bonus != box_target>=0.9).

## VERDICT (FINAL)
**The 0.79 result is NOT capacity/budget-limited. Bigger net + longer/keep-best
UNDERPERFORMS the small-net #5b 0.79.**
- best peak 0.676 = -0.114 vs #5b 0.79 ; -0.134 vs PPO 0.81 ; -0.156 vs PPO-100M 0.832.
- Does NOT beat PPO; does NOT clear 0.82; does NOT even match #5b.
- Direction: more residual capacity HURTS. Likely mechanism: the residual rides on an
  already-strong analytic controller and needs only small corrections; a 512-wide policy
  has too much capacity -> noisier/over-aggressive correction + optimization instability
  (seed spread 0.633-0.676, s2 declines to 0.55 final). The small default policy (32x4)
  was the better regularizer. So 0.79 is an INTRINSIC ceiling of in-loop-residual on
  PandaPickCube under this protocol, not a small-net artifact -- a valid negative finding.
- CAVEAT: single big-net size x2 seeds; not a capacity sweep. Confident on "bigger does
  not help / hurts at 512-wide"; a mid-size (e.g. 128x4) was not tested.
