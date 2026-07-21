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
    xy_tol: float = 0.005           # APPROACH: xy align tol before DESCEND (tighter
                                    #   centering improves grasp -> reached_box)
    descend_z_tol: float = 0.010    # DESCEND: |site_z - grasp_z| done tol
    close_threshold: float = 0.030  # gripper setpoint considered "closed enough" (<= this)
    grasp_width: float = 0.0        # closed setpoint target. 0=full close (EJECTS the
                                    #   4cm cube). Set ~0.012-0.02 to clamp without ejecting.
    grasp_hold_steps: int = 18      # GRASP: steps to hold closing before LIFT. Longer
                                    #   hold lets the cube SETTLE into the level grip
                                    #   (analytic IK reorients it) before lifting, so
                                    #   marginal grasps don't slip mid-air. Doubled
                                    #   success 0.03->0.06.
    grasp_close_rate: float = 0.04  # gripper setpoint delta per step (cmd space pre-scale)
    lift_clear: float = 0.12        # LIFT: z to raise (above box start) before transport
    lift_done_z: float = 0.10       # LIFT->TRANSPORT: box lifted this much => transport
    transport_xy_tol: float = 0.03  # TRANSPORT->PLACE: box xy within this of target xy
    place_tol: float = 0.020        # PLACE->HOLD: box-target distance to call placed
    ik_gain: float = 1.0            # IK step damping/gain on cartesian->joint
    ik_iters: int = 12              # IK refinement iterations per control step
    ori_w: float = 0.0              # (DLS path) orientation lock weight. Unstable;
                                    #   prefer use_aik=1 instead.
    aik_from: int = 2              # phase id from which analytic IK engages (2=GRASP,
                                    #   3=LIFT). GRASP (2) wins: a level grasp from the
                                    #   start aligns the cube before lifting.
    use_aik: int = 1               # 1 => use the ANALYTIC level-gripper IK (the
                                    #   mechanism that cracked rot_err -> first successes)
                                    #   (aik_solve) instead of the iterative DLS.
                                    #   Keeps the gripper level+yaw-fixed exactly so
                                    #   the grasped cube stays flat (rot_err~0) and
                                    #   box_target can cross 0.9. This is the Goal-A
                                    #   mechanism.
    cart_step: float = 0.06         # max cartesian goal move per control step (m)
    place_cart_step: float = 0.06   # cart step in PLACE/HOLD. With the frozen-offset
                                    #   site->target goal a firmer step holds the box
                                    #   AT target against sag (small steps lost the
                                    #   race to gravity and the box fell away).
    place_gain: float = 1.0         # PLACE goal = box + place_gain*(target-box).
    up_bias: float = 0.02           # PLACE/HOLD: command the box this far ABOVE target
                                    #   z to counteract gravity sag of the held cube at
                                    #   extended (near-singular) reach. Tune via sweep.
    approach_max: int = 25          # phase timeouts (steps) - approach aligns fast
    descend_max: int = 25
    grasp_max: int = 28


def site_fk(q_arm):
    """Gripper site position from arm joints (pure JAX)."""
    return pk.compute_franka_fk(q_arm)[:3, 3]


def site_fk_R(q_arm):
    """Gripper site rotation matrix (3x3) from arm joints."""
    return pk.compute_franka_fk(q_arm)[:3, :3]


def _so3_err(R):
    """Smooth orientation residual ~ rotation vector of R. Uses the vee of the
    skew part: 0.5*(R - R^T)^vee. This is differentiable everywhere (no arccos
    gradient blowup near identity) and equals the rotation vector to 2nd order;
    sign/direction are correct, which is all the DLS IK needs to descend."""
    return 0.5 * jp.array([R[2, 1] - R[1, 2],
                           R[0, 2] - R[2, 0],
                           R[1, 0] - R[0, 1]])


_Q7_CANDS = jp.linspace(-2.8, 2.8, 15)


def aik_solve(target_xyz, R_des, q_actual, lowers, uppers):
    """ANALYTIC level-gripper IK (vmappable). Builds the EE transform with the
    desired (level, yaw-fixed) rotation R_des at target_xyz, sweeps the redundant
    q7 over a fixed candidate set, and picks the FIRST candidate that is non-NaN,
    within joint limits, and reproduces the pose. Returns (q_arm, ok).

    This gives an EXACT level grasp/carry orientation (perr,oerr ~1e-7) so the
    grasped square cube stays axis-aligned -> rot_err ~ 0 -> box_target reaches
    0.9. Falls back to q_actual when no candidate is valid (unreachable pose)."""
    T = jp.eye(4).at[:3, :3].set(R_des).at[:3, 3].set(target_xyz)

    def one(q7):
        q = pk.compute_franka_ik(T, q7, q_actual)
        finite = jp.all(jp.isfinite(q))
        inlim = jp.all(q >= lowers - 1e-3) & jp.all(q <= uppers + 1e-3)
        # pose check
        p = pk.compute_franka_fk(jp.where(finite, q, q_actual))[:3, 3]
        perr = jp.linalg.norm(target_xyz - p)
        ok = finite & inlim & (perr < 0.02)
        return q, ok

    qs, oks = jax.vmap(one)(_Q7_CANDS)          # (15,7),(15,)
    # pick first ok; if none, pick the one with q closest to q_actual that's finite
    any_ok = jp.any(oks)
    idx_ok = jp.argmax(oks)                       # first True
    # fallback: smallest joint move among finite solutions
    finite = jp.all(jp.isfinite(qs), axis=1)
    dist = jp.where(finite, jp.linalg.norm(qs - q_actual[None], axis=1), 1e9)
    idx_fb = jp.argmin(dist)
    idx = jp.where(any_ok, idx_ok, idx_fb)
    q_sel = qs[idx]
    ok = oks[idx]
    q_sel = jp.where(jp.all(jp.isfinite(q_sel)), q_sel, q_actual)
    return jp.clip(q_sel, lowers, uppers), ok


def ik_step(q_arm, target_xyz, lowers, uppers, gain, iters,
            R_des=None, ori_w=0.0):
    """Damped-least-squares IK toward target_xyz (and optionally R_des).

    Position-only when ori_w==0. When ori_w>0, also drives the gripper rotation
    toward R_des with weight ori_w (a 6-DOF DLS solve) — used to KEEP THE
    GRIPPER LEVEL + YAW-FIXED so the grasped square cube stays axis-aligned
    (rot_err ~ 0 against the identity target). Vmappable.
    """
    jac_p = jax.jacobian(site_fk)

    # `ori_w` may be a traced per-env scalar (0 during APPROACH/DESCEND, >0 during
    # carry). We ALWAYS build the 6-DOF system but scale the orientation block by
    # ori_w, so the branch is data-dependent without a Python `if` (vmappable, and
    # the position-only case is recovered exactly at ori_w=0).
    def ori_res(qq):
        Rq = site_fk_R(qq)
        return _so3_err(R_des @ Rq.T)

    def body(_, q):
        cur = site_fk(q)
        ep = target_xyz - cur                 # (3,)
        Jp = jac_p(q)                         # (3,7)
        lam = 0.05
        JJt = Jp @ Jp.T + lam * lam * jp.eye(3)
        Jp_pinv = Jp.T @ jp.linalg.solve(JJt, jp.eye(3))   # (7,3) damped pinv
        dq_pos = Jp_pinv @ ep                 # primary: position
        # SECONDARY (nullspace): reduce orientation error WITHOUT disturbing
        # position. dq_ori projected through (I - Jp^+ Jp) so it lives in the
        # position-nullspace -> precise placement is preserved while the wrist
        # de-twists the cube. ori_w may be a traced per-env scalar (0 pre-grasp).
        eo = ori_res(q)                       # (3,)
        Jo = jax.jacobian(ori_res)(q)         # (3,7)
        Jo_pinv = Jo.T @ jp.linalg.solve(Jo @ Jo.T + lam * lam * jp.eye(3),
                                         jp.eye(3))
        dq_ori = Jo_pinv @ eo
        N = jp.eye(7) - Jp_pinv @ Jp          # nullspace projector of position
        dq = dq_pos + ori_w * (N @ dq_ori)
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
        # Desired gripper rotation: the HOME FK orientation (level, top-down,
        # yaw such that the closed fingers leave the square cube axis-aligned ->
        # rot_err ~ 0 vs the identity target quat). Holding this fixed through
        # grasp/carry/place is what keeps box_target from being capped by rot_err.
        home_q = jp.array(env._init_q)[self.arm_qadr]
        self.R_home = site_fk_R(home_q)

    def init_state(self):
        return {
            "phase": jp.array(APPROACH, dtype=jp.int32),
            "timer": jp.array(0, dtype=jp.int32),
            "lock_xy": jp.zeros(2),
            "grip_cmd": jp.array(self.grip_hi),  # start open
            # site-minus-box offset FROZEN at grasp time. Using the LIVE offset
            # creates a feedback trap: as the held box sags, the goal chases it
            # down. Freezing at GRASP->LIFT lets carry/place command the SITE to a
            # fixed point relative to the (always-reachable) target.
            "grasp_off": jp.zeros(3),
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
        grasp_off = cstate["grasp_off"]

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
        # FROZEN grasp offset (set at GRASP->LIFT). Carry/place use this fixed
        # offset to command the SITE to a point relative to the target, instead of
        # the live (sagging) box -> no chase-down feedback trap.
        off = grasp_off
        up = jp.array([0.0, 0.0, k.up_bias])
        # LIFT: straight up over the grasp xy, all the way to the TARGET height
        #   (so TRANSPORT is purely lateral at constant z -> no climb-while-translate
        #   sag). NO lateral move during lift (shear slips the cube). Grip closed.
        safe_z = 0.03 + k.lift_clear
        carry_z = jp.maximum(safe_z, target[2])         # box height during carry
        g_lift = jp.array([lock_xy[0], lock_xy[1], carry_z + off[2]])
        # TRANSPORT: hold carry_z, move the BOX xy over the target xy (pure translate).
        g_trans = jp.array([target[0] + off[0], target[1] + off[1],
                            carry_z + off[2]])
        # PLACE: drive the SITE straight to (target + frozen_offset + up_bias). The
        #   target is always IK-reachable; commanding the site there firmly holds
        #   the box AT the target instead of letting it sag away.
        g_place = target + up + off
        # HOLD: park at target (+ up_bias to hold against sag).
        g_hold = target + up + off

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

        # rate-limit the cartesian goal (smoother, less cube knock). Use a SMALLER
        # step in PLACE/HOLD for precise, swing-free settling.
        step_lim = jp.where((phase == PLACE) | (phase == HOLD),
                            k.place_cart_step, k.cart_step)
        gv = goal - site
        gn = jp.linalg.norm(gv) + 1e-9
        goal_rl = jp.where(gn > step_lim, site + gv * (step_lim / gn), goal)

        # IK to joint targets, emit clipped delta action. Orientation lock engages
        # only from GRASP onward (position must dominate during APPROACH/DESCEND so
        # the gripper can reach down onto the cube). Keeping the wrist level+yaw-
        # fixed during grasp/carry stops the cube from twisting (rot_err was the
        # cap on box_target).
        # Position-only DLS (always available, used pre-grasp and as aik fallback).
        q_pos = ik_step(q_arm, goal_rl, self.lowers7, self.uppers7, k.ik_gain,
                        k.ik_iters, R_des=self.R_home, ori_w=0.0)
        if k.use_aik:
            # ANALYTIC level-gripper IK from GRASP onward: exact level+yaw-fixed
            # pose so the held cube stays axis-aligned (rot_err~0). Pre-grasp keeps
            # the position-only solve so the descent isn't over-constrained.
            q_aik, aik_ok = aik_solve(goal_rl, self.R_home, q_arm,
                                      self.lowers7, self.uppers7)
            use = (phase >= k.aik_from) & aik_ok
            q_des = jp.where(use, q_aik, q_pos)
        else:
            q_des = q_pos
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
        # freeze the site-box offset at the moment we commit to lifting.
        new_grasp_off = jp.where(to_lift, site - box, grasp_off)

        # LIFT done when box near carry height (within tol) or it has climbed enough.
        lifted = (box[2] > (carry_z - 0.04)) | ((box[2] - 0.03) > k.lift_done_z + 0.15)
        to_transport = (phase == LIFT) & (lifted | (timer > 35))

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
            "grasp_off": new_grasp_off,
        }
        return action.astype(jp.float32), new_cstate


def make_controller(env, knobs: Knobs = None):
    return HeuristicPick(env, knobs)
