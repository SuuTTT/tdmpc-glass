# HL PickCube — Knowledge Base

Task: mujoco_playground **PandaPickCube** — grasp a 4x4x6cm cube and place it at an
elevated target. SUCCESS = env metric `box_target >= 0.9` (== `reached_box` AND box
within ~2cm of target AND cube ORIENTATION matched). reached_box = `norm(box_center -
gripper_site) < 0.012`. box_target = reached_box * (1 - tanh(5*(0.9*pos_err + 0.1*rot_err))).

**STATUS: SOLVED the 0% wall.** v9 controller gets success_rate ~0.059 (12-15/256),
reached_box ~0.78, place_ok ~0.18, max_lift 0.54 — REAL grasp+lift+place to the
elevated target (verified by single-env replay + video). Up from the neural policy's
0% (reward-hack hover) and the seed IK's 0% (no grasp hold). Remaining ceilings below.

## § Interface facts (from pick.py / mjx_panda.xml — verified)
- **8 position actuators.** `ctrl[0:7]` = arm joint-angle targets (rad), kp 300..1000.
  `ctrl[7]` = gripper finger-width setpoint in `[0, 0.04]` (0=closed, 0.04=open),
  force actuator gainprm=350, biasprm=(0,-350,-10); two fingers tied by equality.
- **step:** `ctrl = clip(prev_ctrl + action*0.04, lowers, uppers)`. Per step arm joint
  targets move <=0.04 rad, gripper setpoint <=0.0016 m. `action ∈ [-1,1]^8` = per-step
  DELTA. Only ~150 steps/episode -> the phase TIME BUDGET is tight (see Current best).
- **Geometry:** box geom size (0.02,0.02,0.03) => 4x4cm footprint, 6cm tall, rest
  center z=0.03. box body xpos = box CENTER. Targets sampled ELEVATED: z∈[0.23,0.43],
  xy ±0.2 around (0.5,0); box starts ±0.2 xy at z~0.03. ALL targets are IK-reachable
  (analytic IK reach err ~2e-4 m even at radius 0.98) — placement failures are
  CONTROL/DYNAMICS, never workspace.
- **gripper SITE == compute_franka_fk(q_arm) endpoint with d7e=0.2104** (verified err
  4mm; site is at the grasp/fingertip level, NOT the wrist). => put the SITE at box
  CENTER to grasp; reached_box is then satisfiable. pure-JAX FK (panda_kinematics.py).
- **ANALYTIC IK exists**: `panda_kinematics.compute_franka_ik(T_ee, q7, q_actual)` —
  vmappable, exact pose. Sweep the redundant q7 over candidates and pick a valid one.
- reached_box LATCHES once it ever fires.

## § Harness design
- JAX-NATIVE controller (pure-JAX phase machine + analytic/DLS FK IK) -> vmaps to 256
  envs / 1 GPU in ~95s. Telemetry read from real rollouts (per-phase pass-rates +
  box_target decomposition). regression.py locks won phases. Knob sweeps fan across 4
  GPUs (one harness per GPU). The seed used CPU-MuJoCo IK in a python loop (serial).

## § What works (the winning mechanisms)
1. **Site-at-box-center grasp** (approach/descend/grasp closing): 1.0/1.0/0.98.
2. **Full-close grip (grasp_width=0)** holds best — wider commanded widths hold WORSE
   (the cube is held by FRICTION; firmest squeeze wins). [iter 2]
3. **Split LIFT(straight up) -> TRANSPORT(translate at carry height) -> PLACE.** Lift
   straight up to the TARGET height first so transport is purely lateral (no
   climb-while-translate sag). Lateral shear during lift slips the cube. [iters 3-4]
4. **Frozen grasp offset + up_bias droop compensation.** PLACE goal = target + up_bias
   + frozen(site-box). Commanding the SITE (always reachable) to a fixed point relative
   to the target — NOT chasing the live (sagging) box — plus up_bias≈0.02-0.04 to
   counter the position-controller droop at extended reach. up_bias is a BIG lever:
   place_ok 0.01 -> 0.41. [iter 5]
5. **ANALYTIC level-gripper IK from GRASP (use_aik=1, aik_from=2)** — THE thing that
   cracked the 0% wall. Keeps the gripper level + yaw-fixed so the square cube stays
   axis-aligned (rot_err~0). First real successes. [iter 8]
6. **Long grasp-hold (grasp_hold_steps=18)** — let the cube SETTLE into the level grip
   on the table before lifting; secures marginal grasps. ~DOUBLED success 0.03->0.06.
   [iter 9]

## § What fails & why (dead-ends — do NOT retry)
- **grasp_width > 0** (loose clamp): holds WORSE than full close. [iter 2]
- **collapsing lift+transport into one diagonal move**: shears the cube out. [iter 3]
- **small place_cart_step / damped place_gain**: the box loses the race to gravity and
  sags away; needs a FIRM place step (0.06) + up_bias. [iters 5, sweeps]
- **ITERATIVE orientation IK (6-DOF DLS, ANY weight; AND nullspace-projected
  orientation)**: dynamically fights position during carry, wrecks placement
  (mean_box_target 0.37->0.06, arm flings cube up to ~1m). Orientation MUST be solved
  EXACTLY (analytic IK), not iteratively traded. [iter 7] The `_so3_log` arccos/sin
  also NaNs under autodiff — use the smooth skew-vee residual if ever needed.
- **engaging analytic IK LATE (aik_from=4 TRANSPORT / 5 PLACE)**: recovers lift_ok
  (0.72-0.78) but jolts the cube out of level right at placement -> place_ok crashes to
  0.04. Must level from GRASP. [iter 8 sweep]
- **tighter xy_tol=0.005 alone / lower grasp_z / slower descend**: do NOT raise
  reached_box above ~0.80 (a hard grasp-precision ceiling). [iter 9 sweep]

## § Remaining ceilings (the localized obstacles)
- **reached_box ~0.78-0.86** (grasp precision): ~15-22% of envs never get site within
  1.2cm of box center; the level grasp leaves the site slightly off. This caps max
  possible success at ~0.80.
- **lift_ok ~0.59** (cube slips during/after the level-grasp lift on marginal grips).
- **rot_err residual**: even good placements sit at box_target ~0.9 (just crossing);
  the level pose is approximate at off-home configs, so rot_err isn't exactly 0.
  Combined, these multiply to ~6% success. Pushing higher needs a tighter grasp
  (e.g. servo the site onto box center harder before closing) and/or a level pose
  that better matches the cube's identity orientation at all xy.

## § Current best — v9 (versions/controller_v9.py, BEST=v9)
- Knobs vs default: xy_tol=0.005, grasp_hold_steps=18, grasp_max=28, use_aik=1,
  aik_from=2, up_bias=0.02.
- Metrics (256 envs, seed 0): success_rate **0.0586**, reached_box 0.781,
  place_ok 0.176, max_lift 0.536, mean_max_box_target 0.297; per-phase
  approach 1.0 / descend 1.0 / grasp 0.98 / lift 0.59.
- Success videos: videos/HL_v9_SUCCESS_env6.mp4 (maxbt 0.915),
  HL_v9_SUCCESS_env21.mp4 (maxbt 0.912) — mirrored to EC2.
