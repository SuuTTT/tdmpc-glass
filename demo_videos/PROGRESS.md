# Official PPO/SAC baselines on PandaPickCube (box_target>=0.9 real-success metric)

Box: b3060 (4x RTX 3060, 12GB each). venv /root/helios-rl/.venv (jax 0.10.2, brax ppo+sac OK).
Metric: success = max over a 1000-step rollout of state.metrics["box_target"] >= 0.9
  (metrics["box_target"] = raw box_target * reached_box, i.e. gated by grasp).
  reached = max reached_box > 0.5. SAME protocol as eval_curve.py / TD-MPC2 scoring.
Reward = STOCK: PICKCUBE_W_* env vars UNSET; lift_bonus/grasp_bonus default 0.0.

## Plan
- PPO GPU0: official config (20M ts, 2048 envs, unroll10, lr1e-3, ent2e-2, policy(32,32,32,32),
  value(256x5)). num_evals high + checkpoints for learning curve.
- SAC GPU1: brax SAC (NO official manipulation SAC config -> LABELED unofficial).
- After training: eval periodic checkpoints with peak-box_target>=0.9 protocol.

## Status log

[update] PPO (official) DONE training. Crashed only at post-train video render
(EGL/OpenGL, harmless) -- all 20 checkpoints saved (to 32.77M brax-counted steps;
num_resets_per_eval=10 inflates step counter, num_timesteps target=20M).
PPO eval/episode_reward curve: 109 -> 365 (1.6M) -> 700 (19.7M) -> 1072 (29.5M)
  -> 1384 (32.8M final). Sharp LATE jump 735->1384 = grasping emerging late.
SAC (unofficial) running GPU1: 1M reward 562, ep_box_target(time-summed) 4.34.
Infra fixes needed (jax 0.10.2 vs brax 0.14.2):
  - jax.device_put_replicated removed -> shimmed (pmap-identity) in jax_compat.py
  - brax checkpoint.load_config crashes on KERNEL_INITIALIZER[None] + bad
    observation_size dict -> patched in jax_compat.py
Eval pipeline validated on 1.6M ckpt: succ 0 / reached 0 / maxbt 0 (early, expected).
PPO full ckpt eval (n=256, 1000-step peak-box_target protocol) running on GPU0.

[RESULT - PPO] Official PPO SOLVES PandaPickCube on the strict box_target>=0.9 metric,
but only at FULL budget and LATE. Eval n=256, 1000-step peak protocol:
  step 8.2M : succ 0.0  reached 0.24  mean_max_bt 0.013
  step 13.1M: succ 0.0  reached 0.85  mean_max_bt 0.073  <- grasps but never places
  step 18.0M: succ 0.0  reached 0.996 mean_max_bt 0.105
  step 27.9M: succ 0.0  reached 0.996 mean_max_bt 0.250
  step 29.5M: succ 0.074 reached 0.996 mean_max_bt 0.638  <- threshold first crossed
  step 31.1M: succ 0.527 reached 0.996 mean_max_bt 0.852
  step 32.8M: succ 0.660 reached 0.996 mean_max_bt 0.887  <- FINAL/PEAK
PPO FINAL = PEAK = 66.0% real success @ 32.77M env steps (config target 20M;
brax overshoots). reached(grasp)=99.6%, mean max box_target=0.887.
IMPLICATION: TD-MPC2/jumpy 0% @1-1.5M is mostly a BUDGET failure -- PPO is ALSO 0%
until ~28M (>18x TD-MPC2 budget); grasp emerges ~13M, placement only ~29M+.
RESULTS.json + learning_curve.png written and copied to EC2 demo_videos/.
SAC (unofficial) still running GPU1.

[RESULT - SAC] Unofficial brax SAC (20M, num_envs=256) DOES NOT solve the strict metric.
Eval n=256, 1000-step peak protocol, ALL checkpoints 1M..20M:
  success_rate = 0.0 at EVERY checkpoint (final 20M and peak both 0%).
  reached(grasp) = 0.99 by 5M (grasps reliably & EARLY), mean max box_target stuck ~0.077,
  p90 ~0.167, max-over-eps capped 0.276. Classic reward-hack: grasp/hover, never lift+place.
SAC FINAL/PEAK = 0.0% real success @ 20M. reached=99.2%.
[DONE] Both baselines complete. RESULTS.json + learning_curve.png regenerated w/ both, copied to EC2.
