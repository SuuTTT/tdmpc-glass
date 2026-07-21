# Part 33 — Orientation-AWARE abstraction (status)

Date: 2026-06-24. Box: b3060b (217.171.200.22:62684, 4x RTX 3060).

## Hypothesis under test
Part 31 found the abstraction-in-loop residual LOSES on PandaPickCubeOrientation
(peak success 0.320/0.344 vs PPO 0.805/0.836). Part 31's controller does a FIXED
top-down grasp at a FIXED yaw — the wrong prior for a random-orientation target.
Part 33 makes the analytic controller ORIENTATION-AWARE and asks: does that close
the gap toward PPO (~0.82)? If YES -> limitation is prior-FIXEDNESS, parametrized
priors are the route to reusable abstractions. If NO (~0.33) -> deeper problem.

## The code change (orientation-aware)
New files (Part 31 files untouched):
- helios-rl/hl_pickcube/param_controller_oriaware.py — copy of param_controller.py.
  With env RES_ORI_AWARE=1, in ParamPick.act() the desired EE rotation becomes
  R_des = R_target @ R_home (was fixed R_home), applied from GRASP phase onward,
  where R_target = quat_to_mat(data.mocap_quat[mocap_target]) is the per-episode
  target orientation. So the rigidly-grasped cube ends rotated to match the target.
  Stores self.mocap_target in __init__.
- helios-rl/hl_pickcube/residual_env_oriaware.py — copy of residual_env.py importing
  param_controller_oriaware. With RES_ORI_AWARE=1 it appends the per-episode target
  quaternion (4 dims) to the residual obs (Markov conditioning on absolute target
  orientation; base obs only had the relative rot-error). observation_size 77 -> 81.
- exp/.../residual_patch_orientation_oriaware.py, run_ppo_orientation_residual_oriaware.py,
  eval_residual_orientation_oriaware.py, eval_controller_alone_oriaware.py,
  launch_oriaware_b3060b.sh.

alpha kept FIXED at 1.0 (same as Part 31). Protocol matched: NT=35M nominal
(->~55.7M brax-rounded), NE=18, eval n=256 steps=1000 success=box_target>=0.9.

## What is running (started ~2026-06-24 20:31 UTC)
- GPU0: oriaware residual seed 1  (residual_logs_s1/)
- GPU1: oriaware residual seed 2  (residual_logs_s2/)
- GPU3: oriaware residual seed 3  (residual_logs_s3/)
- GPU2: controller-alone oriaware sanity -> DONE (see below)

## Early / done numbers (REAL, from JSON)
- Controller-ALONE oriaware (alpha=0, n=256, 1000 steps):
  success_rate=0.0234, reached_rate=0.8281, mean_max_box_target=0.3336,
  p90=0.8459. (vs fixed-controller-alone smoke n=16: success 0.0.)
  -> the orientation-aware control MECHANISM produces nonzero orientation success
  on its own; the learned residual on top is what the 3 seeds test.

## Where final evals land
- exp/tdmpc_glass/generalization_PandaPickCubeOrientation_oriaware/residual_eval_s{1,2,3}.json
  (written automatically by the launcher after each seed's training).

## ETA
Part 31 residual train ~16 min + eval ~10 min. With 3 concurrent + long initial
JIT (~140s), expect training done ~20:50-21:05 UTC, evals by ~21:15 UTC.

## Safety
Mahjong (tmux moyuHarv) untouched; only our own pids. No --save_full_state.
Per-run logs+ckpts ~19M each. Disk >8G free; disk_guard cron */5 active.
