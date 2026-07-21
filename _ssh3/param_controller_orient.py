#!/usr/bin/env python3
"""EXP#13 ORIENTATION-AWARE parametric HL PickCube controller.

Identical to param_controller.py EXCEPT the analytic-IK grasp/carry ORIENTATION is
chosen as a function of target REACH instead of a fixed top-down R_home: near
targets stay top-down; FAR targets TILT the gripper toward the base (pitch the
approach axis back) so a valid analytic IK / grasp exists at far reach where the
top-down level grip does not.

Tilt scheme (validated by orient_sweep.py to lift far-reach IK feasibility
0.748 -> 0.934): rotate R_home about the WORLD horizontal axis perpendicular to
the reach direction (mode1, sign = -perp), by
    ang = clip(TILT_GAIN * (reach - TILT_R0), 0, TILT_MAX)
where reach = ||box_xy - base_xy||. Env-var tunable (ORIENT_TILT_GAIN/R0/MAX);
TILT_GAIN=0 recovers the exact top-down controller (fair-baseline arm).

PARAMETRIC version of the HL PickCube phase controller (exp#2 base).

This is the OPTION-EXECUTION engine: the SAME analytic phase machine as
controller.py (reach->descend->grasp->lift->transport->place->hold, analytic
level-gripper IK), but the continuous Knobs are read from a per-env JAX VECTOR
`kv` instead of the Python `Knobs` dataclass, so a learned high-level policy
pi_hi(context)->kv can set the option parameters PER ENV / PER SITUATION.

We learn at the OPTION (phase) level: pi_hi chooses the abstraction's parameters
(knob values + small per-phase target nudges). The low-level execution is the
unchanged analytic controller. This is the CONTRAST to experiment #1 which learns
a raw per-step action residual.

STRUCTURAL knobs that select code paths (use_aik, aik_from) are FIXED to the v9
BEST values (use_aik=1, aik_from=GRASP) — only the continuous, value-carrying
knobs are made learnable, because those are the abstraction's tunable degrees of
freedom. Integer timing knobs are treated as continuous and rounded inside the
transitions via float comparisons (timer is int, compared against a float knob).

KV LAYOUT (index -> knob), see KV_NAMES / KV_BASE / KV_SCALE below. pi_hi emits a
bounded delta d in [-1,1]^K; the effective knob = clip(base + scale*d, lo, hi).
The per-phase target offsets (descend/grasp/place xyz nudges) are appended; they
default to 0 so kv==base reproduces the v9 controller (~6% success) at init.
"""
import os
import jax
import jax.numpy as jp
from mujoco_playground._src.manipulation.franka_emika_panda import (
    panda_kinematics as pk,
)

# Phase ids (mirror controller.py)
APPROACH, DESCEND, GRASP, LIFT, TRANSPORT, PLACE, HOLD = 0, 1, 2, 3, 4, 5, 6

# ---- exp#13 reach-dependent grasp-orientation tilt ----
# ang = clip(TILT_GAIN*(reach - TILT_R0), 0, TILT_MAX); pitch about -perp axis.
ORIENT_TILT_GAIN = float(os.environ.get("ORIENT_TILT_GAIN", "2.0"))
ORIENT_TILT_R0 = float(os.environ.get("ORIENT_TILT_R0", "0.55"))
ORIENT_TILT_MAX = float(os.environ.get("ORIENT_TILT_MAX", "0.9"))
# RELEVEL: from this phase id onward, blend the grasp orientation back toward the
# top-down R_home (so the held cube re-flattens for placement -> rot_err drops).
# 99 = never relevel (default = tilt held through place). Set e.g. 4 (TRANSPORT)
# or 5 (PLACE) to relevel late.
ORIENT_RELEVEL_FROM = int(os.environ.get("ORIENT_RELEVEL_FROM", "99"))
ORIENT_BASE_XY = jp.array([0.0, 0.0])  # panda base at world origin (x,y)


def _rodrigues(axis, ang):
    n = jp.linalg.norm(axis) + 1e-9
    a = axis / n
    K = jp.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0.0]])
    return jp.eye(3) + jp.sin(ang) * K + (1 - jp.cos(ang)) * (K @ K)


def reach_tilt_R(R_home, box_xy):
    """Reach-dependent grasp orientation. Near -> R_home (top-down); far -> pitch
    the gripper back toward the base by ang(reach). Pure JAX / vmappable."""
    d = box_xy - ORIENT_BASE_XY
    reach = jp.linalg.norm(d) + 1e-9
    dhat = d / reach
    ang = jp.clip(ORIENT_TILT_GAIN * (reach - ORIENT_TILT_R0), 0.0, ORIENT_TILT_MAX)
    perp = jp.array([-dhat[1], dhat[0], 0.0])  # horizontal perp to reach (mode1)
    return _rodrigues(-perp, ang) @ R_home

# ---- learnable continuous knob vector spec ----
# name, base (v9 BEST), scale (delta range for d in [-1,1]), lo, hi
_KNOB_SPEC = [
    ("hover_height",    0.10,   0.05,  0.04,  0.20),
    ("grasp_z_offset",  0.005,  0.015, -0.02, 0.03),
    ("xy_tol",          0.006,  0.006,  0.002, 0.02),
    ("descend_z_tol",   0.010,  0.008,  0.003, 0.03),
    ("grasp_width",     0.0,    0.02,   0.0,  0.03),
    ("grasp_hold_steps",50.0,   25.0,   10.0, 90.0),
    ("grasp_close_rate",0.04,   0.03,   0.01, 0.10),
    ("lift_clear",      0.12,   0.06,   0.05, 0.25),
    ("transport_xy_tol",0.03,   0.02,   0.01, 0.08),
    ("place_tol",       0.020,  0.012,  0.008,0.05),
    ("up_bias",         0.02,   0.03,  -0.02, 0.08),
    ("cart_step",       0.06,   0.04,   0.02, 0.15),
    ("place_cart_step", 0.06,   0.04,   0.02, 0.15),
    ("approach_max",    25.0,   15.0,   10.0, 50.0),
    ("descend_max",     25.0,   15.0,   10.0, 50.0),
    ("grasp_max",       55.0,   25.0,   20.0, 90.0),
]
# per-phase target nudges (xyz) appended: descend(3), grasp(3), place(3)
_OFFSET_SCALE = 0.02  # +-2cm nudge range
_N_KNOBS = len(_KNOB_SPEC)
_N_OFF = 9
KV_DIM = _N_KNOBS + _N_OFF  # total learnable option-param dim

KV_NAMES = [n for (n, _, _, _, _) in _KNOB_SPEC] + [
    "des_dx", "des_dy", "des_dz", "grasp_dx", "grasp_dy", "grasp_dz",
    "place_dx", "place_dy", "place_dz",
]
KV_BASE = jp.array([b for (_, b, _, _, _) in _KNOB_SPEC] + [0.0] * _N_OFF)
KV_SCALE = jp.array([s for (_, _, s, _, _) in _KNOB_SPEC] + [_OFFSET_SCALE] * _N_OFF)
KV_LO = jp.array([lo for (_, _, _, lo, _) in _KNOB_SPEC] + [-_OFFSET_SCALE] * _N_OFF)
KV_HI = jp.array([hi for (_, _, _, _, hi) in _KNOB_SPEC] + [_OFFSET_SCALE] * _N_OFF)

# fixed structural knobs (v9 BEST) — these select code paths, not learned.
GRASP_Z_OFF_FALLBACK = 0.005
FIX_use_aik = 1
FIX_aik_from = GRASP
FIX_close_threshold = 0.030
FIX_lift_done_z = 0.10
FIX_ik_gain = 1.0
FIX_ik_iters = 12


def d_to_knobs(d):
    """Map a bounded delta vector d in [-1,1]^KV_DIM to effective knob values."""
    return jp.clip(KV_BASE + KV_SCALE * d, KV_LO, KV_HI)


def context_of(box0, target):
    """High-level policy context = initial box xyz + target xyz (6-dim)."""
    return jp.concatenate([box0, target])


# import the FK / analytic-IK primitives unchanged from controller.py
def site_fk(q_arm):
    return pk.compute_franka_fk(q_arm)[:3, 3]


def site_fk_R(q_arm):
    return pk.compute_franka_fk(q_arm)[:3, :3]


def _so3_err(R):
    return 0.5 * jp.array([R[2, 1] - R[1, 2],
                           R[0, 2] - R[2, 0],
                           R[1, 0] - R[0, 1]])


_Q7_CANDS = jp.linspace(-2.8, 2.8, 15)


def aik_solve(target_xyz, R_des, q_actual, lowers, uppers):
    T = jp.eye(4).at[:3, :3].set(R_des).at[:3, 3].set(target_xyz)

    def one(q7):
        q = pk.compute_franka_ik(T, q7, q_actual)
        finite = jp.all(jp.isfinite(q))
        inlim = jp.all(q >= lowers - 1e-3) & jp.all(q <= uppers + 1e-3)
        p = pk.compute_franka_fk(jp.where(finite, q, q_actual))[:3, 3]
        perr = jp.linalg.norm(target_xyz - p)
        ok = finite & inlim & (perr < 0.02)
        return q, ok

    qs, oks = jax.vmap(one)(_Q7_CANDS)
    any_ok = jp.any(oks)
    idx_ok = jp.argmax(oks)
    finite = jp.all(jp.isfinite(qs), axis=1)
    dist = jp.where(finite, jp.linalg.norm(qs - q_actual[None], axis=1), 1e9)
    idx_fb = jp.argmin(dist)
    idx = jp.where(any_ok, idx_ok, idx_fb)
    q_sel = qs[idx]
    ok = oks[idx]
    q_sel = jp.where(jp.all(jp.isfinite(q_sel)), q_sel, q_actual)
    return jp.clip(q_sel, lowers, uppers), ok


def ik_step_pos(q_arm, target_xyz, lowers, uppers, gain, iters):
    """Position-only DLS IK (pre-grasp / aik fallback)."""
    jac_p = jax.jacobian(site_fk)

    def body(_, q):
        cur = site_fk(q)
        ep = target_xyz - cur
        Jp = jac_p(q)
        lam = 0.05
        JJt = Jp @ Jp.T + lam * lam * jp.eye(3)
        Jp_pinv = Jp.T @ jp.linalg.solve(JJt, jp.eye(3))
        dq = Jp_pinv @ ep
        q = q + gain * dq
        return jp.clip(q, lowers, uppers)

    return jax.lax.fori_loop(0, iters, body, q_arm)


class ParamPick:
    """Vmappable phase machine whose continuous knobs come from a per-env vector
    kv (KV_DIM,). Mirrors controller.HeuristicPick.act exactly except knobs are
    read from kv instead of the Python dataclass."""

    def __init__(self, env):
        self.arm_qadr = jp.array(env._robot_arm_qposadr)
        self.obj_body = env._obj_body
        self.gid = env._gripper_site
        self.lowers7 = jp.array(env._lowers)[:7]
        self.uppers7 = jp.array(env._uppers)[:7]
        self.grip_lo = float(env._lowers[7])
        self.grip_hi = float(env._uppers[7])
        self.scale = float(env._action_scale)
        home_q = jp.array(env._init_q)[self.arm_qadr]
        self.R_home = site_fk_R(home_q)

    def init_state(self):
        return {
            "phase": jp.array(APPROACH, dtype=jp.int32),
            "timer": jp.array(0, dtype=jp.int32),
            "lock_xy": jp.zeros(2),
            "grip_cmd": jp.array(self.grip_hi),
            "grasp_off": jp.zeros(3),
        }

    def act(self, state, cstate, kv):
        """kv: effective knob values (KV_DIM,) from d_to_knobs. Per-env."""
        kn = kv[:_N_KNOBS]
        off_des = kv[_N_KNOBS + 0:_N_KNOBS + 3]
        off_grasp = kv[_N_KNOBS + 3:_N_KNOBS + 6]
        off_place = kv[_N_KNOBS + 6:_N_KNOBS + 9]
        # unpack named knobs
        (hover_height, grasp_z_offset, xy_tol, descend_z_tol, grasp_width,
         grasp_hold_steps, grasp_close_rate, lift_clear, transport_xy_tol,
         place_tol, up_bias, cart_step, place_cart_step, approach_max,
         descend_max, grasp_max) = [kn[i] for i in range(_N_KNOBS)]

        d = state.data
        box = d.xpos[self.obj_body]
        site = d.site_xpos[self.gid]
        target = state.info["target_pos"]
        ctrl = d.ctrl
        q_arm = d.qpos[self.arm_qadr]
        fw = d.qpos[7]

        phase = cstate["phase"]
        timer = cstate["timer"]
        lock_xy = cstate["lock_xy"]
        grip_cmd = cstate["grip_cmd"]
        grasp_off = cstate["grasp_off"]

        box_top = box[2] + 0.03
        horiz_err = jp.linalg.norm(site[:2] - box[:2])

        g_app = jp.array([box[0], box[1], box_top + hover_height])
        grasp_z = box[2] + grasp_z_offset
        # DESCEND/GRASP targets get learned xyz nudges
        g_des = jp.array([lock_xy[0] + off_des[0], lock_xy[1] + off_des[1],
                          grasp_z + off_des[2]])
        g_grasp = jp.array([lock_xy[0] + off_grasp[0], lock_xy[1] + off_grasp[1],
                            grasp_z + off_grasp[2]])
        err_bt = target - box
        off = grasp_off
        up = jp.array([0.0, 0.0, up_bias])
        safe_z = 0.03 + lift_clear
        carry_z = jp.maximum(safe_z, target[2])
        g_lift = jp.array([lock_xy[0], lock_xy[1], carry_z + off[2]])
        g_trans = jp.array([target[0] + off[0], target[1] + off[1],
                            carry_z + off[2]])
        # PLACE target gets a learned xyz nudge
        g_place = target + up + off + off_place
        g_hold = target + up + off + off_place

        goal = jp.select(
            [phase == APPROACH, phase == DESCEND, phase == GRASP,
             phase == LIFT, phase == TRANSPORT, phase == PLACE, phase == HOLD],
            [g_app, g_des, g_grasp, g_lift, g_trans, g_place, g_hold],
            default=g_app,
        )

        opening = (phase == APPROACH) | (phase == DESCEND)
        closed_setpoint = jp.maximum(self.grip_lo, grasp_width)
        target_grip = jp.where(opening, self.grip_hi, closed_setpoint)
        dgrip = jp.clip(target_grip - grip_cmd, -grasp_close_rate, grasp_close_rate)
        grip_cmd_new = jp.clip(grip_cmd + dgrip, self.grip_lo, self.grip_hi)

        step_lim = jp.where((phase == PLACE) | (phase == HOLD),
                            place_cart_step, cart_step)
        gv = goal - site
        gn = jp.linalg.norm(gv) + 1e-9
        goal_rl = jp.where(gn > step_lim, site + gv * (step_lim / gn), goal)

        q_pos = ik_step_pos(q_arm, goal_rl, self.lowers7, self.uppers7,
                            FIX_ik_gain, FIX_ik_iters)
        # EXP#13: reach-dependent grasp orientation. Use the FROZEN grasp xy
        # (lock_xy, set at APPROACH->DESCEND from the box xy) so grasp/carry/place
        # all hold the SAME tilted orientation (the cube is gripped in it). Before
        # lock is set (APPROACH), lock_xy==0 -> falls to top-down, fine pre-grasp.
        R_tilt = reach_tilt_R(self.R_home, lock_xy)
        # relevel from ORIENT_RELEVEL_FROM phase onward (cube re-flattens for place)
        relevel = phase >= ORIENT_RELEVEL_FROM
        R_grasp = jp.where(relevel, self.R_home, R_tilt)
        # use_aik fixed on:
        q_aik, aik_ok = aik_solve(goal_rl, R_grasp, q_arm,
                                  self.lowers7, self.uppers7)
        use = (phase >= FIX_aik_from) & aik_ok
        q_des = jp.where(use, q_aik, q_pos)
        ctrl_des = jp.concatenate([q_des, grip_cmd_new[None]])
        action = jp.clip((ctrl_des - ctrl) / self.scale, -1.0, 1.0)

        # transitions (timer is int, compared against float knobs)
        aligned = (horiz_err < xy_tol) & (jp.abs(site[2] - g_app[2]) < 0.03)
        to_descend = (phase == APPROACH) & (aligned | (timer > approach_max))
        new_lock = jp.where(to_descend, box[:2], lock_xy)

        descended = (site[2] - box[2]) < (grasp_z_offset + descend_z_tol)
        to_grasp = (phase == DESCEND) & (descended | (timer > descend_max))

        closed = (fw < FIX_close_threshold)
        held = closed | (grip_cmd_new <= closed_setpoint + 1e-4)
        to_lift = (phase == GRASP) & (((timer > grasp_hold_steps) & held)
                                      | (timer > grasp_max))
        new_grasp_off = jp.where(to_lift, site - box, grasp_off)

        lifted = (box[2] > (carry_z - 0.04)) | ((box[2] - 0.03) > FIX_lift_done_z + 0.15)
        to_transport = (phase == LIFT) & (lifted | (timer > 35))

        over_target = jp.linalg.norm(box[:2] - target[:2]) < transport_xy_tol
        to_place = (phase == TRANSPORT) & (over_target | (timer > 50))

        placed = jp.linalg.norm(err_bt) < place_tol
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


def make_param_controller(env):
    return ParamPick(env)
