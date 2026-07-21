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

## § UPDATE (session 2, resume) — Goal A pushed 0.031 -> ~0.09 success
- **Current BEST = config in versions/v9_r_hold.json (also controller.py defaults now):**
  grasp_hold_steps=50, grasp_max=55, grasp_close_rate=0.04 (FULL close, width 0),
  xy_tol=0.006, everything else as before (up_bias 0.02, place_tol 0.02,
  place_cart_step 0.06, aik on from GRASP).
  seed0 (steps360,N256): success 0.10, grasp 0.99, lift 0.69, place 0.26,
  reached_box 0.78, mean_max_box_target 0.38. seed7: success 0.074 (distribution
  variance) -> honest success ~0.08-0.10. ~3x the resume baseline (0.031).
- **THE WALL is now twofold and STRUCTURAL for this heuristic:**
  1. reached_box_rate caps at ~0.78 (it needs norm(box-site)<0.012, very tight).
     56/256 envs never get the site within 12mm; their miss is ~12.5mm in xy
     (horizontal), NOT vertical. Success can never exceed reached_box.
  2. Among reached near-miss envs (0.6<=box_target<0.9, n=63): at best moment
     pos_err p50=17mm, rot_err p50=0.18 (cube tilts ~10deg in the grasp). The
     box_target gap is ~47% pos / 53% rot. **If rot_err were 0, 39/63 would
     cross 0.9.** rot_err is the single biggest lever; pos_err second.
- **What MOVED the needle (kept):** longer grasp_hold + grasp_max + tighter
  xy_tol -> cube settles flatter in the gripper -> lower rot_err -> lift 0.62->0.69,
  success 0.0625->0.10. Full close (grasp_width=0) beats partial (0.012).
- **DEAD-ENDS (do not retry):**
  * negative grasp_z_offset (descend below box center): HURTS everything
    (grasp 0.93->0.86, lift 0.6->0.4) — pushes cube/floor, destabilizes grasp.
  * tighter place_tol / smaller place_cart_step: no gain (limiter isn't PLACE
    servo precision, it's reached_box + grasp tilt).
  * up_bias>0.03 or =0: 0.03 worse than 0.02; the 0.02 default is near-optimal.
  * partial grasp_width 0.012: lift collapses to 0.56.
- **NEXT levers if resumed (untried):** (a) drive rot_err directly — re-orient
  cube flat during HOLD via wrist roll/pitch nullspace; (b) fix the 56 xy-miss
  envs (workspace-edge IK failures?) to lift reached_box past 0.78; (c) a true
  closed-loop xy place servo on live box-target error (frozen-offset leaves
  ~17mm steady-state pos_err).

## § GOAL B — return-vs-success (VERDICT)
Per 149-step episode, weights gripper_box=4, box_target=8 (gated by reached_box),
no_floor_collision=0.25, robot_target_qpos=0.3. All from real rollouts.
| regime                         | return | maxbt | gripper_box(w) | box_target(w) |
|--------------------------------|--------|-------|----------------|---------------|
| 1 HOVER (TD-MPC2 bare pi)       | 314.9  | 0.000 | 279.4          | 0.0           |
| 2 scripted GRASP-NO-PLACE       | 673.7  | 0.484 | 460.6          | 174.1         |
| 3 scripted SUCCESS              | 965.3  | 0.911 | 478.1          | 445.1         |
- ckpt: exp/tdmpc_glass/PandaPickCube_realsucc_vanilla_s2/seed_2 best_mppi.pkl
  (the prompt's phasei30_jumk2 ckpt does not exist on box; this realsucc vanilla
  policy is a pure hover: maxbt 0.0, reached_box 0.0, lift 0.03m over 8 eps).
- **VERDICT: reward ORDERING is WELL-SPECIFIED** — real success earns 3.1x the
  hover-hack's return (965 vs 315); grasp-no-place sits cleanly between (674).
  Return rises monotonically with real task progress.
- **BUT it is POORLY-SHAPED for exploration, which explains the reward-hacking:**
  the hover banks 280/ep from gripper_box ALONE (89% of its return) just by
  hovering near the cube, never grasping — a dense, immediate, risk-free floor.
  box_target (the real-task term) is GATED behind reached_box (norm<0.012) +
  grasp + lift + place, so its gradient is sparse and locked. A learner climbs
  the easy gripper_box hill into a hover local optimum and never discovers the
  far larger box_target reward. Mis-shaping (not mis-ordering) drives the hack.
