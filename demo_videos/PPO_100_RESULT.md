# Can official PPO reach ~100% on PandaPickCube? (Task 2)

Box b3060 GPU1. Official manipulation_params PandaPickCube config (2048 envs, lr 1e-3,
entropy 2e-2, policy 32x4, value 256x5, unroll 10, discount 0.97), num_timesteps target=80M
(brax env-step counter overshoots to ~113M via num_resets_per_eval=10), num_evals=24.
Real success eval = max over 1000-step rollout of metrics[box_target]>=0.9 (gated by grasp),
n=256 per checkpoint. Numbers READ FROM PPO_100_RESULT.json (eval_ckpts.py).

## VERDICT: PPO does NOT reach 100%. It plateaus at ~80-83%.
- PEAK  real success = 83.2% @ brax-step 108.13M (reached 100%, mean max box_target 0.948)
- FINAL real success = 71.9% @ 113.05M (last ckpt dips; peak earlier)
- Plateau band ~80-83% sustained from ~24.6M all the way to 108M (no climb toward 100%).

## Curve (brax-step : success / reached / mean_max_box_target)
   4.9M 0.000 0.820 0.062
   9.8M 0.000 0.992 0.120
  14.7M 0.000 0.996 0.200
  19.7M 0.480 0.984 0.851   <- threshold first crossed (place emerging)
  24.6M 0.789 1.000 0.932
  29.5M 0.809 1.000 0.938
  34.4M 0.816 1.000 0.941
  39.3M 0.816 1.000 0.948
  49.2M 0.816 0.996 0.942
  68.8M 0.812 0.992 0.941
  88.5M 0.820 1.000 0.949
 103.2M 0.816 1.000 0.953
 108.1M 0.832 1.000 0.948   <- PEAK 83.2%
 113.0M 0.719 0.992 0.902   <- FINAL (late dip)

## Comparison to the earlier 20M-target PPO run
That run peaked 66.0% @ 32.77M and crossed threshold only at ~29M. THIS 80M run with the SAME
config crossed at ~19.7M and reached ~80% by 24.6M -> faster + higher ceiling (seed/variance +
longer budget). Net: PPO ceiling on stock-reward PandaPickCube real-success is ~80-83%, not 100%.
The remaining ~17% are episodes where the arm grasps (reached~100%) but does not get box_target
fully to >=0.9 within the rollout (placement precision), i.e. a residual hard-instance tail, not a
budget problem (flat from 24M to 108M).
