# EXP#13 — Orientation-aware grasp + in-loop residual (PandaPickCube)

GOAL: break the 0.79–0.81 ceiling by fixing the NAMED failure mode — the analytic
controller only does TOP-DOWN grasps, so the ~20% far-reach tail (box x up to 0.9)
where a top-down grip is infeasible never succeeds. Fix = reach-dependent grasp
ORIENTATION (tilt the gripper toward the base at far reach), then the same in-loop
residual on top of it (alpha=1.0, the #5b best).

## Orientation scheme (LOCKED, validated)
`param_controller_orient.py`: analytic IK grasp orientation =
`R_des = reach_tilt_R(R_home, box_xy)`:
  ang = clip(TILT_GAIN*(reach - R0), 0, MAX), pitch R_home about -perp (mode1),
  reach = ||box_xy - base||, base=(0,0).
  LOCKED: TILT_GAIN=2.0, R0=0.55, MAX=0.9 (rad).
RELEVEL_FROM=4 (TRANSPORT): blend orientation back to top-down R_home from the
TRANSPORT phase so the held cube RE-FLATTENS for placement (rot_err drops).
TILT_GAIN=0.0 recovers the exact top-down controller (fair baseline).

Validation chain:
1. IK feasibility (orient_sweep.py, 17x17 grid over box-sample range): far-reach
   (r>0.55) analytic-IK feasibility 0.748 (top-down) -> 0.934 (orient g2 mode1).
2. Controller-ALONE sim sanity (eval_residual_curve_orient.py --controller_only,
   n=256, 1000-step, real box_target>=0.9, far split r0=0.78):
   - top-down:                  overall 0.066, FAR 0.0233, near 0.088, reached_far 0.674
   - orient g2 NO relevel:      overall 0.012, FAR 0.000, near 0.018, reached_far 0.884
       (tilt helps REACH but tilted grasp ruins placement -> rot_err caps box_target)
   - orient g2 relevel@TRANSPORT: overall 0.098, FAR 0.093, near 0.100, reached_far 0.860  <-- BEST
   - orient g2 relevel@LIFT(3):  FAR 0.070
   - orient g2 relevel@PLACE(5): FAR 0.023 (too late to re-level)
   - orient g3/g2.5 relevel@4:   FAR 0.000 (over-tilt breaks the grasp, reached_far ~0.72)
   => SANITY (a) PASSES: orientation-aware controller raises FAR-reach controller-alone
      success 4x (0.0233 -> 0.093) over top-down. (NB Part 28's "0.12" was the geometric
      fraction of far configs admitting a top-down grasp, not realized success; realized
      top-down far success is only 0.023, so 0.093 is a clear real lift on the named mode.)

## Arms (45M env-steps each, brax PPO, alpha=1.0 fixed, like #5b)
- GPU0 orient_a1_s1  : orient g2 relevel@4 + residual, seed1   (headline)
- GPU1 orient_a1_s2  : orient g2 relevel@4 + residual, seed2   (headline, 2nd seed)
- GPU2 topdown_a1_s1 : top-down + residual, seed1 = fair baseline (reproduce #5b 0.78)
Launch: launch_13.sh. Eval (do NOT skip): eval_all_13.sh -> curve_*.json then RESULTS.json.

## Status
- [done] orientation scheme designed + IK-feasibility validated + controller-alone sanity (PASS)
- [running] 3 residual training arms launched (45M steps)
- [pending] eval checkpoints (eval_all_13.sh) -> uniform+far real success vs 0.81 -> RESULTS.json verdict
