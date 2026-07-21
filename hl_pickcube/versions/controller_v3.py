#!/usr/bin/env python3
"""HEURISTIC pick-and-place controller for mujoco_playground PandaPickCube.

JAX-NATIVE / VMAPPABLE phase machine. Unlike the seed (CPU-MuJoCo IK in a Python
loop, serial only), this controller's policy is pure JAX so 256 envs run in
parallel under vmap on one GPU.

ACTION INTERFACE (verified from pick.py / mjx_panda.xml):
  - 8 position actuators. ctrl[0:7] = target joint angles (rad), kp=300..1000.
    ctrl[7] = gripper finger target via a force actuator gainprm=350,biasprm=
    (0,-350,-10): ctrl is a finger-width setpoint in [0,0.04] (0=closed,
    0.04=open). The two fingers are tied by an equality constraint.
  - env.step: ctrl = clip(prev_ctrl + action*0.04, lowers, uppers).
    => per step the arm joint targets move <=0.04 rad, the gripper setpoint
    moves <=0.0016 m. Gripper open<->close setpoint takes ~25 steps; physical
    close (force-closing on cube) is faster once setpoint passes contact width.
  - action = per-step DELTA in [-1,1]^8.

GEOMETRY (verified):
  - box geom size (0.02,0.02,0.03) => 4cm x 4cm footprint, 6cm tall, center z=0.03
    at rest (bottom on floor z=0). box body xpos is the box center.
  - gripper SITE == compute_franka_fk(q_arm) endpoint with d7e=0.2104 (verified
    err 4mm). The grasp pads sit ~at the site. So to grasp, put the SITE at the
    box center; reached_box (norm(box-site)<0.012) is then satisfiable.
  - fingers open span ~0.08m; cube is 0.04 wide => ~2cm clearance per side.

SUCCESS = metrics box_target>=0.9 == reached_box AND box within ~2cm of target.

KNOBS (exposed for sweeps): see Knobs dataclass.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from dataclasses import dataclass, asdict
import jax
import jax.numpy as jp
from mujoco_playground._src.manipulation.franka_emika_panda import (
    panda_kinematics as pk,
)

# Phase ids
APPROACH, DESCEND, GRASP, LIFT, TRANSPORT, PLACE, HOLD = 0, 1, 2, 3, 4, 5, 6


@dataclass
class Knobs:
    hover_height: float = 0.10      # APPROACH: site z above box top during align
    grasp_z_offset: float = 0.005   # DESCEND target: box_center_z + this (site at center)
    xy_tol: float = 0.010           # APPROACH: xy align tol before DESCEND
    descend_z_tol: float = 0.010    # DESCEND: |site_z - grasp_z| done tol
    close_threshold: float = 0.030  # gripper setpoint considered "closed enough" (<= this)
    grasp_width: float = 0.0        # closed setpoint target. 0=full close (EJECTS the
                                    #   4cm cube). Set ~0.012-0.02 to clamp without ejecting.
    grasp_hold_steps: int = 12      # GRASP: steps to hold closing before LIFT
    grasp_close_rate: float = 0.04  # gripper setpoint delta per step (cmd space pre-scale)
    lift_clear: float = 0.12        # LIFT: z to raise (above box start) before transport
    lift_done_z: float = 0.10       # LIFT->TRANSPORT: box lifted this much => transport
    transport_xy_tol: float = 0.03  # TRANSPORT->PLACE: box xy within this of target xy
    place_tol: float = 0.020        # PLACE->HOLD: box-target distance to call placed
    ik_gain: float = 0.6            # IK step damping/gain on cartesian->joint
    ik_iters: int = 6               # IK refinement iterations per control step
    cart_step: float = 0.05         # max cartesian goal move per control step (m)
    approach_max: int = 40          # phase timeouts (steps)
    descend_max: int = 40
    grasp_max: int = 30


def site_fk(q_arm):
    """Gripper site position from arm joints (pure JAX)."""
    return pk.compute_franka_fk(q_arm)[:3, 3]


def ik_step(q_arm, target_xyz, lowers, uppers, gain, iters):
    """Damped-least-squares position IK toward target_xyz (vmappable).

    Uses jax.jacobian of the analytic FK; iterates a few Gauss-Newton steps.
    """
    jac_fn = jax.jacobian(site_fk)

    def body(_, q):
        cur = site_fk(q)
        err = target_xyz - cur
        J = jac_fn(q)                       # (3,7)
        lam = 0.05
        JJt = J @ J.T + lam * lam * jp.eye(3)
        dq = J.T @ jp.linalg.solve(JJt, err)
        q = q + gain * dq
        return jp.clip(q, lowers, uppers)

    return jax.lax.fori_loop(0, iters, body, q_arm)


class HeuristicPick:
    """Vmappable phase-machine controller.

    Per-env carry state is a flat dict of arrays so it vmaps cleanly. Build one
    instance per env (it just holds constants); call .init_state() then .act().
    """

    def __init__(self, env, knobs: Knobs = None):
        self.knobs = knobs or Knobs()
        self.arm_qadr = jp.array(env._robot_arm_qposadr)
        self.obj_body = env._obj_body
        self.gid = env._gripper_site
        self.lowers7 = jp.array(env._lowers)[:7]
        self.uppers7 = jp.array(env._uppers)[:7]
        self.grip_lo = float(env._lowers[7])
        self.grip_hi = float(env._uppers[7])
        self.scale = float(env._action_scale)

    def init_state(self):
        return {
            "phase": jp.array(APPROACH, dtype=jp.int32),
            "timer": jp.array(0, dtype=jp.int32),
            "lock_xy": jp.zeros(2),
            "grip_cmd": jp.array(self.grip_hi),  # start open
        }

    def act(self, state, cstate):
        """Return (action[8], new_cstate). Pure JAX; vmappable over leading env axis."""
        k = self.knobs
        d = state.data
        box = d.xpos[self.obj_body]
        site = d.site_xpos[self.gid]
        target = state.info["target_pos"]
        ctrl = d.ctrl
        q_arm = d.qpos[self.arm_qadr]
        fw = d.qpos[7]  # actual finger width

        phase = cstate["phase"]
        timer = cstate["timer"]
        lock_xy = cstate["lock_xy"]
        grip_cmd = cstate["grip_cmd"]

        box_top = box[2] + 0.03
        horiz_err = jp.linalg.norm(site[:2] - box[:2])

        # ----- per-phase cartesian goal + gripper setpoint -----
        # APPROACH: hover above box, gripper open, align xy.
        g_app = jp.array([box[0], box[1], box_top + k.hover_height])
        # DESCEND: straight down on locked xy to grasp z (site at box center+off).
        grasp_z = box[2] + k.grasp_z_offset
        g_des = jp.array([lock_xy[0], lock_xy[1], grasp_z])
        # GRASP: hold position, close gripper.
        g_grasp = jp.array([lock_xy[0], lock_xy[1], grasp_z])
        err_bt = target - box
        # site-to-box offset (so we can command the BOX to a point via the site).
        site_box = site - box
        # LIFT: straight up over the grasp xy to a safe height (NO lateral move —
        #   lateral shear during lift is what slips the cube). Keep grip closed.
        safe_z = 0.03 + k.lift_clear
        g_lift = jp.array([lock_xy[0], lock_xy[1], safe_z + site_box[2]])
        # TRANSPORT: at safe height, move the BOX xy over the target xy. Drive the
        #   site to (target_xy + site_box_xy, safe_z). Hold height; only translate.
        g_trans = jp.array([target[0] + site_box[0], target[1] + site_box[1],
                            jp.maximum(target[2], safe_z) + site_box[2]])
        # PLACE: closed-loop drive the BOX to the target (site = target + site_box).
        g_place = target + site_box
        g_hold = target + site_box

        goal = jp.select(
            [phase == APPROACH, phase == DESCEND, phase == GRASP,
             phase == LIFT, phase == TRANSPORT, phase == PLACE, phase == HOLD],
            [g_app, g_des, g_grasp, g_lift, g_trans, g_place, g_hold],
            default=g_app,
        )

        # gripper setpoint per phase: open in APPROACH/DESCEND, close otherwise.
        opening = (phase == APPROACH) | (phase == DESCEND)
        closed_setpoint = jp.maximum(self.grip_lo, k.grasp_width)
        target_grip = jp.where(opening, self.grip_hi, closed_setpoint)
        # rate-limit the setpoint toward target_grip
        dgrip = jp.clip(target_grip - grip_cmd, -k.grasp_close_rate, k.grasp_close_rate)
        grip_cmd_new = jp.clip(grip_cmd + dgrip, self.grip_lo, self.grip_hi)

        # rate-limit the cartesian goal (smoother, less cube knock)
        gv = goal - site
        gn = jp.linalg.norm(gv) + 1e-9
        goal_rl = jp.where(gn > k.cart_step, site + gv * (k.cart_step / gn), goal)

        # IK to joint targets, emit clipped delta action.
        q_des = ik_step(q_arm, goal_rl, self.lowers7, self.uppers7, k.ik_gain, k.ik_iters)
        ctrl_des = jp.concatenate([q_des, grip_cmd_new[None]])
        action = jp.clip((ctrl_des - ctrl) / self.scale, -1.0, 1.0)

        # ----- transitions -----
        aligned = (horiz_err < k.xy_tol) & (jp.abs(site[2] - g_app[2]) < 0.03)
        to_descend = (phase == APPROACH) & (aligned | (timer > k.approach_max))
        new_lock = jp.where(to_descend, box[:2], lock_xy)

        descended = (site[2] - box[2]) < (k.grasp_z_offset + k.descend_z_tol)
        to_grasp = (phase == DESCEND) & (descended | (timer > k.descend_max))

        closed = (fw < k.close_threshold)
        held = closed | (grip_cmd_new <= closed_setpoint + 1e-4)
        to_lift = (phase == GRASP) & (((timer > k.grasp_hold_steps) & held)
                                      | (timer > k.grasp_max))

        lifted = (box[2] - 0.03) > k.lift_done_z
        to_transport = (phase == LIFT) & (lifted | (timer > 40))

        over_target = jp.linalg.norm(box[:2] - target[:2]) < k.transport_xy_tol
        to_place = (phase == TRANSPORT) & (over_target | (timer > 50))

        placed = jp.linalg.norm(err_bt) < k.place_tol
        to_hold = (phase == PLACE) & placed

        next_phase = jp.select(
            [to_descend, to_grasp, to_lift, to_transport, to_place, to_hold],
            [jp.int32(DESCEND), jp.int32(GRASP), jp.int32(LIFT),
             jp.int32(TRANSPORT), jp.int32(PLACE), jp.int32(HOLD)],
            default=phase,
        )
        advanced = next_phase != phase
        next_timer = jp.where(advanced, jp.int32(0), timer + 1)

        new_cstate = {
            "phase": next_phase,
            "timer": next_timer,
            "lock_xy": new_lock,
            "grip_cmd": grip_cmd_new,
        }
        return action.astype(jp.float32), new_cstate


def make_controller(env, knobs: Knobs = None):
    return HeuristicPick(env, knobs)
