#!/usr/bin/env python3
"""Exp#13 orientation probe + hard-config miner.

Goal: confirm the gripper-orientation convention from R_home, then test whether a
"tilt-toward-base proportional to reach" grasp orientation admits a valid analytic
IK on FAR-REACH box configs where the fixed top-down orientation does NOT.

We measure, per box xy in a grid over the env's box-sample range, whether
aik_solve(box_center, R_des, q_actual) returns ok=True (an in-limit, pose-matching
analytic IK solution) at the GRASP z. R_des is either:
  - R_home (top-down, the current controller), or
  - R_tilt(box) = tilt R_home about the axis perpendicular to the reach direction,
    by angle = clip(tilt_gain*(reach - tilt_r0), 0, tilt_max), so near targets stay
    top-down and far targets tilt the approach axis toward the base.

This is a pure reachability/feasibility test of the IK with an orientation target;
it is the cheap sanity that motivates the controller change. Writes JSON.
"""
import os, sys, json, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MJPG_IMPL", "jax")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")
sys.path.insert(0, "/root/tdmpc_glass/helios-rl/hl_pickcube")
import numpy as np
import jax, jax.numpy as jp
from mujoco_playground._src.manipulation.franka_emika_panda.pick import (
    PandaPickCube, default_config as _pick_default_config)
from mujoco_playground._src.manipulation.franka_emika_panda import panda_kinematics as pk

import param_controller as PC  # reuse aik_solve / site_fk_R


def rot_x(a):
    c, s = jp.cos(a), jp.sin(a)
    return jp.array([[1,0,0],[0,c,-s],[0,s,c]])

def rot_y(a):
    c, s = jp.cos(a), jp.sin(a)
    return jp.array([[c,0,s],[0,1,0],[-s,0,c]])

def rot_z(a):
    c, s = jp.cos(a), jp.sin(a)
    return jp.array([[c,-s,0],[s,c,0],[0,0,1]])


def tilt_R(R_home, base_xy, box_xy, tilt_gain, tilt_r0, tilt_max):
    """Tilt R_home toward the base, by angle proportional to reach beyond tilt_r0.

    Reach direction = horizontal vector from base to box. We rotate the gripper
    about the WORLD horizontal axis perpendicular to the reach direction (so the
    approach axis pitches from straight-down toward leaning-back-over-the-base),
    by `ang`. This keeps yaw aligned with the cube (square) while letting the
    wrist lean for far reach.
    """
    d = box_xy - base_xy                       # reach vector (2,)
    reach = jp.linalg.norm(d) + 1e-9
    ang = jp.clip(tilt_gain * (reach - tilt_r0), 0.0, tilt_max)
    # horizontal axis perpendicular to reach dir, in world frame
    dhat = d / reach
    axis = jp.array([-dhat[1], dhat[0], 0.0])  # perp, horizontal
    # rotation about `axis` by `ang` (Rodrigues)
    K = jp.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    Rrot = jp.eye(3) + jp.sin(ang)*K + (1-jp.cos(ang))*(K@K)
    return Rrot @ R_home, ang


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=21)
    ap.add_argument("--tilt_gain", type=float, default=2.5)
    ap.add_argument("--tilt_r0", type=float, default=0.55)
    ap.add_argument("--tilt_max", type=float, default=1.0)  # rad (~57deg)
    ap.add_argument("--grasp_z_offset", type=float, default=0.005)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    env = PandaPickCube(config_overrides={"impl": "jax"})
    home_q = jp.array(env._init_q)[jp.array(env._robot_arm_qposadr)]
    R_home = PC.site_fk_R(home_q)
    lowers = jp.array(env._lowers)[:7]; uppers = jp.array(env._uppers)[:7]
    init_obj = np.asarray(env._init_obj_pos)
    base_xy = jp.array([0.0, 0.0])  # panda base at world origin (x,y)

    print("R_home=\n", np.asarray(R_home))
    print("init_obj_pos=", init_obj)
    # approach axis of R_home = its 3rd column (gripper local z, points toward fingers)
    print("R_home approach-axis (col2, world)=", np.asarray(R_home[:, 2]))

    # box sample range: init_obj + uniform([-0.2,-0.2,0],[0.2,0.2,0])
    xs = np.linspace(init_obj[0]-0.2, init_obj[0]+0.2, args.grid)
    ys = np.linspace(init_obj[1]-0.2, init_obj[1]+0.2, args.grid)
    grasp_z = float(init_obj[2]) + args.grasp_z_offset

    def feas_topdown(bx, by):
        tgt = jp.array([bx, by, grasp_z])
        _, ok = PC.aik_solve(tgt, R_home, home_q, lowers, uppers)
        return ok

    def feas_tilt(bx, by):
        tgt = jp.array([bx, by, grasp_z])
        box_xy = jp.array([bx, by])
        R_des, ang = tilt_R(R_home, base_xy, box_xy, args.tilt_gain,
                            args.tilt_r0, args.tilt_max)
        _, ok = PC.aik_solve(tgt, R_des, home_q, lowers, uppers)
        return ok, ang

    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    pts = jp.array(np.stack([XX.ravel(), YY.ravel()], axis=1))
    reach = np.asarray(jp.linalg.norm(pts - base_xy[None], axis=1))

    td = np.asarray(jax.vmap(lambda p: feas_topdown(p[0], p[1]))(pts))
    tl_ok, tl_ang = jax.vmap(lambda p: feas_tilt(p[0], p[1]))(pts)
    tl = np.asarray(tl_ok); ang = np.asarray(tl_ang)

    # FAR region = reach > 0.55 (where the diagnosis says top-down fails)
    far = reach > args.tilt_r0
    near = ~far
    out = {
        "params": {"tilt_gain": args.tilt_gain, "tilt_r0": args.tilt_r0,
                   "tilt_max": args.tilt_max, "grasp_z_offset": args.grasp_z_offset,
                   "grid": args.grid},
        "R_home_approach_axis": [round(float(v),4) for v in np.asarray(R_home[:,2])],
        "init_obj_pos": [round(float(v),4) for v in init_obj],
        "n_grid": int(pts.shape[0]),
        "reach_range": [round(float(reach.min()),3), round(float(reach.max()),3)],
        "topdown_feasible_all": round(float(td.mean()),4),
        "tilt_feasible_all": round(float(tl.mean()),4),
        "topdown_feasible_far": round(float(td[far].mean()),4),
        "tilt_feasible_far": round(float(tl[far].mean()),4),
        "topdown_feasible_near": round(float(td[near].mean()),4),
        "tilt_feasible_near": round(float(tl[near].mean()),4),
        "n_far": int(far.sum()), "n_near": int(near.sum()),
        "mean_tilt_angle_far_rad": round(float(ang[far].mean()),4),
        "max_tilt_angle_rad": round(float(ang.max()),4),
    }
    print(json.dumps(out, indent=2))
    json.dump(out, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
