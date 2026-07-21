# SAC (unofficial) on PandaPickCube — real-success result

Box b3060. Metric = real success: max over a 1000-step rollout of metrics["box_target"] >= 0.9
(gated by grasp). SAME protocol as PPO / TD-MPC2 scoring. Eval n=256 episodes per checkpoint.
Reward = STOCK (PICKCUBE_W_* unset; lift/grasp bonus 0.0).
SAC config = UNOFFICIAL (no official tuned manipulation SAC config exists in mujoco_playground;
brax SAC with sensible off-policy defaults, hidden (256,256,256)+LN, 256 envs, 20M steps).
Source numbers READ FROM: sac_eval_curve.json (eval_ckpts.py --algo sac), sac_gpu1.log.

## RESULT
- PEAK real success  = 0.0%  (0/256 episodes ever reached box_target>=0.9 at ANY checkpoint)
- FINAL real success = 0.0%  @ 20.0M env steps
- reached (grasp) rate FINAL = 99.2% (peaks 99.6%) — SAC reliably reaches/grasps the cube
- mean max box_target = 0.077 (max over all ckpts 0.0775); best single-episode max box_target = 0.2763
- All 20 checkpoints (1M..20M) scored: success_rate = 0.0 at every single one.

## INTERPRETATION
Unofficial SAC LEARNS TO GRASP (reached ~99%) but NEVER LIFTS the cube toward the target —
max box_target saturates at ~0.28, far below the 0.9 place threshold. So SAC solves the
reaching/grasping sub-task but fails the place/lift objective entirely under the stock reward
at 20M steps. Contrast: official PPO crosses the threshold only very late (~29M+) and reaches
66% by 32.8M. SAC at 20M = a grasp-only plateau, 0% real success.

## Files
- sac_eval_curve.json (full 20-point curve, peak+final), sac_gpu1.log (training), sac_eval.log.
