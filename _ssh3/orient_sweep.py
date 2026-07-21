#!/usr/bin/env python3
"""Sweep tilt sign/axis/params to find an orientation scheme that INCREASES
analytic-IK feasibility at far reach over the fixed top-down R_home.

Tries pitch about +/- perp-axis and about the reach-axis (roll), over a range of
gains, reporting far-region feasibility for each. The winner feeds the controller.
"""
import os, sys, json, argparse, itertools
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("MJPG_IMPL", "jax")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")
sys.path.insert(0, "/root/tdmpc_glass/helios-rl/hl_pickcube")
import numpy as np, jax, jax.numpy as jp
from mujoco_playground._src.manipulation.franka_emika_panda.pick import (
    PandaPickCube)
import param_controller as PC


def rodrigues(axis, ang):
    n = jp.linalg.norm(axis) + 1e-9
    a = axis / n
    K = jp.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0.0]])
    return jp.eye(3) + jp.sin(ang)*K + (1-jp.cos(ang))*(K@K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=17)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    env = PandaPickCube(config_overrides={"impl": "jax"})
    home_q = jp.array(env._init_q)[jp.array(env._robot_arm_qposadr)]
    R_home = PC.site_fk_R(home_q)
    lowers = jp.array(env._lowers)[:7]; uppers = jp.array(env._uppers)[:7]
    init_obj = np.asarray(env._init_obj_pos)
    grasp_z = float(init_obj[2]) + 0.005
    base_xy = jp.array([0.0, 0.0])
    xs = np.linspace(init_obj[0]-0.2, init_obj[0]+0.2, args.grid)
    ys = np.linspace(init_obj[1]-0.2, init_obj[1]+0.2, args.grid)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    pts = jp.array(np.stack([XX.ravel(), YY.ravel()], axis=1))
    reach = np.asarray(jp.linalg.norm(pts - base_xy[None], axis=1))
    r0 = 0.55
    far = reach > r0

    def feas(pt, mode, gain, tmax):
        bx, by = pt[0], pt[1]
        d = pt - base_xy; rr = jp.linalg.norm(d)+1e-9; dhat = d/rr
        ang = jp.clip(gain*(rr - r0), 0.0, tmax)
        perp = jp.array([-dhat[1], dhat[0], 0.0])   # horizontal perp to reach
        # mode: 0 pitch about +perp, 1 pitch about -perp, 2 roll about reach dir,
        # 3 pitch about world +y, 4 pitch about world -y
        axis = jp.select(
            [mode==0, mode==1, mode==2, mode==3, mode==4],
            [perp, -perp, jp.array([dhat[0],dhat[1],0.0]),
             jp.array([0.,1.,0.]), jp.array([0.,-1.,0.])], perp)
        R_des = rodrigues(axis, ang) @ R_home
        tgt = jp.array([bx, by, grasp_z])
        _, ok = PC.aik_solve(tgt, R_des, home_q, lowers, uppers)
        return ok

    # baseline top-down
    td = np.asarray(jax.vmap(lambda p: feas(p, 0, 0.0, 0.0))(pts))
    results = [{"scheme":"topdown","far_feas":round(float(td[far].mean()),4),
               "all_feas":round(float(td.mean()),4)}]
    for mode in range(5):
        for gain in [1.0, 2.0, 3.0, 4.0]:
            for tmax in [0.5, 0.9, 1.3]:
                ok = np.asarray(jax.vmap(lambda p: feas(p, mode, gain, tmax))(pts))
                results.append({"scheme":f"mode{mode}_g{gain}_tmax{tmax}",
                                "mode":mode,"gain":gain,"tmax":tmax,
                                "far_feas":round(float(ok[far].mean()),4),
                                "all_feas":round(float(ok.mean()),4)})
    results.sort(key=lambda r:-r["far_feas"])
    print("TOP 10 by far_feas:")
    for r in results[:10]:
        print(r)
    print("baseline topdown:", results_td:= [r for r in results if r["scheme"]=="topdown"][0])
    json.dump({"n_far":int(far.sum()),"results":results}, open(args.out,"w"), indent=2)


if __name__ == "__main__":
    main()
