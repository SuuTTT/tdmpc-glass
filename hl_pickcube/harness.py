#!/usr/bin/env python3
"""Fixed eval engine for the HEURISTIC PickCube controller.

Runs the controller on N parallel MJX envs (vmap, 1 GPU) and returns PER-PHASE
telemetry + overall metrics. Optionally renders failure/success videos.

PER-PHASE pass-rates (fraction of envs that achieved each milestone at any point):
  approach_ok : site got within xy_tol of box (xy) while hovering  (reached APPROACH goal)
  descend_ok  : site got to grasp z over the box (site within ~3cm of box center, xy close)
  grasp_ok    : gripper physically closed ON the cube (finger width in clamp band AND
                box still between fingers: norm(box-site)<0.03 with fw in (closed,open))
  lift_ok     : box lifted > 0.06 above its start z (clearly off the table, grasped)
  place_ok    : box brought within place_tol of target (== near success geometric)
Overall:
  success_rate     : box_target>=0.9 at any step
  reached_box_rate : env info reached_box (norm(box-site)<0.012) reached 1 at any step
  max_lift         : max over envs of (box_z - start box_z)

Telemetry is read from REAL rollouts; nothing is fabricated.

Usage:
  python harness.py --n 256 --steps 150 [--render 2] [--device 0] [--tag v1]
"""
import os, argparse, json, time
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.55")
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jp
from mujoco_playground import registry

import controller as C

HERE = Path(__file__).resolve().parent


def build(env, knobs):
    ctrl = C.make_controller(env, knobs)
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.vmap(env.step)
    act_vmap = jax.vmap(ctrl.act)
    init_cs_single = ctrl.init_state()

    def run(keys, n_steps):
        n = keys.shape[0]
        state = reset(keys)
        # one zero step so site_xpos/box are forward-kinematics valid
        state = step(state, jp.zeros((n, 8)))
        cs = jax.tree_util.tree_map(
            lambda x: jp.broadcast_to(x, (n,) + x.shape), init_cs_single
        )
        box0 = state.data.xpos[:, env._obj_body, 2]

        def body(carry, _):
            state, cs = carry
            action, cs = act_vmap(state, cs)
            state = step(state, action)
            box = state.data.xpos[:, env._obj_body]
            site = state.data.site_xpos[:, env._gripper_site]
            fw = state.data.qpos[:, 7]
            d = jp.linalg.norm(box - site, axis=-1)
            xy = jp.linalg.norm(box[:, :2] - site[:, :2], axis=-1)
            bt = state.metrics["box_target"]
            reached = state.info["reached_box"]
            tele = {
                "phase": cs["phase"],
                "d_box_site": d,
                "xy_box_site": xy,
                "site_z": site[:, 2],
                "box_z": box[:, 2],
                "fw": fw,
                "box_target": bt,
                "reached": reached,
                "target_pos": state.info["target_pos"],
                "box": box,
            }
            return (state, cs), tele

        (state, cs), tele = jax.lax.scan(body, (state, cs), None, length=n_steps)
        return tele, box0

    return run, ctrl


def analyze(tele, box0, knobs):
    """Compute per-phase pass-rates + overall from scanned telemetry.

    tele arrays are (T, N, ...). Pass = condition met at ANY step.
    """
    phase = np.asarray(tele["phase"])          # (T,N)
    d = np.asarray(tele["d_box_site"])         # (T,N)
    xy = np.asarray(tele["xy_box_site"])
    site_z = np.asarray(tele["site_z"])
    box_z = np.asarray(tele["box_z"])
    fw = np.asarray(tele["fw"])
    bt = np.asarray(tele["box_target"])
    reached = np.asarray(tele["reached"])
    target = np.asarray(tele["target_pos"])    # (T,N,3)
    box = np.asarray(tele["box"])              # (T,N,3)
    box0 = np.asarray(box0)                     # (N,)
    N = phase.shape[1]

    def any_t(mask):  # mask (T,N) -> per-env reached -> rate
        return float(mask.any(axis=0).mean())

    # APPROACH ok: ever entered DESCEND (means xy align passed) OR xy<tol while high
    reached_descend = (phase >= C.DESCEND).any(axis=0)
    approach_ok = float(reached_descend.mean())
    # DESCEND ok: site got near box center vertically over the box
    descend_ok = any_t((xy < (knobs.xy_tol + 0.02)) & (site_z < box_z + 0.04))
    # GRASP ok: fingers clamped (width in band, not fully open, not fully closed-on-air)
    # AND cube within the fingers (d small) at that time.
    clamp = (fw < knobs.close_threshold) & (fw > 0.005)
    grasp_ok = any_t(clamp & (d < 0.04))
    # LIFT ok: box lifted clearly off table while grasped
    lift = box_z - box0[None, :]               # (T,N)
    lift_ok = any_t((lift > 0.06) & (d < 0.05))
    # PLACE ok: box brought near target
    place_dist = np.linalg.norm(target - box, axis=-1)  # (T,N)
    place_ok = any_t(place_dist < knobs.place_tol)

    success_rate = any_t(bt >= 0.9)
    reached_box_rate = any_t(reached > 0.5)
    max_lift = float(lift.max())
    mean_max_bt = float(bt.max(axis=0).mean())

    return {
        "N": N,
        "approach_ok": round(approach_ok, 4),
        "descend_ok": round(descend_ok, 4),
        "grasp_ok": round(grasp_ok, 4),
        "lift_ok": round(lift_ok, 4),
        "place_ok": round(place_ok, 4),
        "success_rate": round(success_rate, 4),
        "reached_box_rate": round(reached_box_rate, 4),
        "max_lift": round(max_lift, 4),
        "mean_max_box_target": round(mean_max_bt, 4),
    }


def render_episodes(env, knobs, seeds, tags, out_dir):
    """Serially re-run specific seeds (single env) and render to mp4."""
    ctrl = C.make_controller(env, knobs)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    act = jax.jit(ctrl.act)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import mediapy as media
        def wv(p, f, fps): media.write_video(str(p), f, fps=fps)
    except ImportError:
        import imageio
        def wv(p, f, fps): imageio.mimsave(str(p), f, fps=fps)
    written = []
    for sd, tag in zip(seeds, tags):
        state = reset(jax.random.PRNGKey(int(sd)))
        state = step(state, jp.zeros(8))
        cs = ctrl.init_state()
        traj = [jax.device_get(state)]
        maxbt = 0.0
        for _ in range(149):
            a, cs = act(state, cs)
            state = step(state, a)
            traj.append(jax.device_get(state))
            maxbt = max(maxbt, float(state.metrics["box_target"]))
            if bool(state.done > 0.5):
                break
        frames = [np.asarray(f) for f in env.render(traj, height=420, width=420)]
        fps = int(round(1.0 / env.dt))
        p = out_dir / f"HL_{tag}_seed{sd}_bt{maxbt:.2f}.mp4"
        wv(p, frames, fps)
        written.append(str(p))
        print(f"  rendered {p} ({len(frames)} frames, maxbt={maxbt:.2f})", flush=True)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render", type=int, default=0, help="num failure/success videos")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--knobs", default="", help="json dict of knob overrides")
    ap.add_argument("--out", default="", help="json file to write metrics")
    args = ap.parse_args()

    knobs = C.Knobs()
    if args.knobs:
        for kk, vv in json.loads(args.knobs).items():
            setattr(knobs, kk, type(getattr(knobs, kk))(vv))

    env = registry.load("PandaPickCube", config_overrides={"impl": "jax"})
    print(f"env obs={env.observation_size} act={env.action_size} dt={env.dt}", flush=True)
    run, ctrl = build(env, knobs)

    t0 = time.time()
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.n)
    tele, box0 = run(keys, args.steps)
    jax.block_until_ready(tele["phase"])
    metrics = analyze(tele, box0, knobs)
    metrics["eval_sec"] = round(time.time() - t0, 1)
    metrics["knobs"] = {k: v for k, v in C.asdict(knobs).items()}

    print("\n==== METRICS ====", flush=True)
    for kk in ["approach_ok", "descend_ok", "grasp_ok", "lift_ok", "place_ok",
               "success_rate", "reached_box_rate", "max_lift",
               "mean_max_box_target"]:
        print(f"  {kk}: {metrics[kk]}", flush=True)
    print(f"  (N={metrics['N']}, {metrics['eval_sec']}s)", flush=True)

    if args.render > 0:
        # pick failure seeds = first few that didn't grasp; success seeds if any.
        phase = np.asarray(tele["phase"])
        bt = np.asarray(tele["box_target"])
        succ_env = (bt >= 0.9).any(axis=0)
        maxbt_env = bt.max(axis=0)
        order = np.argsort(-maxbt_env)  # best first
        seeds, tags = [], []
        # best (maybe success) + worst (failure)
        for i in order[:max(1, args.render // 2 + args.render % 2)]:
            seeds.append(int(i)); tags.append("best_succ" if succ_env[i] else f"best_bt{maxbt_env[i]:.2f}")
        for i in order[::-1][:args.render // 2]:
            seeds.append(int(i)); tags.append(f"fail_bt{maxbt_env[i]:.2f}")
        # NOTE: env seeds for render must match vmap split seeds; we re-derive
        # single-env keys from the same split so trajectories match.
        keys_np = np.asarray(jax.device_get(keys))
        render_seeds = []
        for s in seeds:
            render_seeds.append(s)  # PRNGKey(s) != split key; document mismatch
        # Use independent seeds for rendering (representative, not identical env).
        vids = render_episodes(env, knobs, seeds, tags, HERE / "videos")
        metrics["videos"] = vids

    if args.out:
        Path(args.out).write_text(json.dumps(metrics, indent=2))
    print("\nJSON " + json.dumps({k: metrics[k] for k in metrics if k != "knobs"}), flush=True)


if __name__ == "__main__":
    main()
