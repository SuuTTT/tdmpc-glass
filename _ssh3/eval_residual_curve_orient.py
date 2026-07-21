#!/usr/bin/env python3
"""EXP#13 real-success curve eval — orientation-aware residual env, with a NEAR/FAR
reach split and an alpha=0 controller-ALONE mode.

Same protocol as eval_residual_curve.py (brax-wrap episode_length=STEPS, n>=256,
success = max box_target >= 0.9), but:
  * uses residual_env_orient (orientation-aware option controller), via --env_mod;
  * reports success_far / success_near split by initial box reach (||box_xy|| > r0);
  * --controller_only: loads NO checkpoint, runs the analytic controller alone
    (alpha hard 0, random residual *0) -> the controller-alone sanity number.

ORIENT_TILT_GAIN env var selects orient (default 2.0) vs topdown (0.0).
"""
import os, sys, json, argparse, time, glob, importlib
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("MJPG_IMPL", "jax")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")
sys.path.insert(0, "/root/tdmpc_glass/exp/tdmpc_glass/baselines_ppo_sac")
sys.path.insert(0, "/root/tdmpc_glass/exp/tdmpc_glass/hl_orient")
sys.path.insert(0, "/root/tdmpc_glass/helios-rl/hl_pickcube")
import jax_compat  # noqa
import numpy as np
import jax, jax.numpy as jp
from mujoco_playground import wrapper
from brax.training.agents.ppo import checkpoint as PC_CKPT
from brax.training.agents.ppo import networks as PNET


def alpha_of(step, anneal_total, amin, amax):
    frac = min(max(step / anneal_total, 0.0), 1.0)
    return amin + (amax - amin) * frac


def build_env(alpha, env_mod, steps):
    os.environ["RES_ALPHA_FIXED"] = str(alpha)
    os.environ.pop("RES_GATE_MODE", None)
    RE = importlib.import_module(env_mod)
    importlib.reload(RE)
    base = RE.ResidualPickCube(config_overrides={"impl": "jax"})
    env = wrapper.wrap_for_brax_training(base, episode_length=steps, action_repeat=1)
    return base, env, jax.jit(env.reset), jax.jit(env.step), base.action_size


def rollout(reset, step, base, infer, keys, n, steps, seed, act_size,
            controller_only):
    state = reset(keys)
    # box xpos is not forward-kinematics'd at reset; read box xy from qpos.
    qa = base._obj_qposadr
    box0_xy = state.data.qpos[:, qa:qa + 2]
    reach0 = jp.linalg.norm(box0_xy, axis=1)

    def body(carry, k):
        state, mbt, mreach = carry
        if controller_only:
            act = jp.zeros((n, act_size))  # residual 0 -> pure controller
        else:
            act = jax.vmap(infer)(state.obs, jax.random.split(k, n))[0]
        state = step(state, act)
        return (state, jp.maximum(mbt, state.metrics["box_target"]),
                jp.maximum(mreach, state.info["reached_box"])), None
    z = jp.zeros((n,))
    ks = jax.random.split(jax.random.PRNGKey(seed + 1), steps)
    (state, mbt, mreach), _ = jax.lax.scan(body, (state, z, z), ks)
    return mbt, mreach, reach0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", default="")
    ap.add_argument("--env_mod", default="residual_env_orient")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--far_r0", type=float, default=0.78)
    ap.add_argument("--anneal_total", type=float, default=20_480_000.0)
    ap.add_argument("--alpha_min", type=float, default=0.0)
    ap.add_argument("--alpha_max", type=float, default=1.0)
    ap.add_argument("--alpha_override", type=float, default=-1.0)
    ap.add_argument("--controller_only", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.n)

    def summarize(mbt, mreach, reach0, step, alpha):
        mbt = np.asarray(jax.block_until_ready(mbt)); mreach = np.asarray(mreach)
        reach0 = np.asarray(reach0)
        far = reach0 > args.far_r0; near = ~far
        succ = (mbt >= 0.9)
        def m(mask, arr):
            return round(float(arr[mask].mean()), 4) if mask.sum() else 0.0
        return {
            "step": step, "alpha": round(float(alpha), 4),
            "success_rate": round(float(succ.mean()), 4),
            "success_far": m(far, succ), "success_near": m(near, succ),
            "reached_rate": round(float((mreach > 0.5).mean()), 4),
            "reached_far": m(far, mreach > 0.5),
            "mean_max_box_target": round(float(mbt.mean()), 4),
            "mbt_far": m(far, mbt), "mbt_near": m(near, mbt),
            "n_far": int(far.sum()), "n_near": int(near.sum()), "n_eps": args.n,
        }

    if args.controller_only:
        alpha = 0.0
        base, env, reset, step, act_size = build_env(alpha, args.env_mod, args.steps)
        t0 = time.time()
        mbt, mreach, reach0 = jax.jit(
            lambda ks: rollout(reset, step, base, None, ks, args.n, args.steps,
                               args.seed, act_size, True))(keys)
        row = summarize(mbt, mreach, reach0, -1, alpha)
        row["eval_sec"] = round(time.time() - t0, 1)
        out = {"protocol_steps": args.steps, "controller_only": True,
               "tilt_gain": float(os.environ.get("ORIENT_TILT_GAIN", "2.0")),
               "result": row}
        print(json.dumps(out, indent=2)); json.dump(out, open(args.out, "w"), indent=2)
        return

    ckpts = [d for d in glob.glob(os.path.join(args.ckpt_root, "*"))
             if os.path.isdir(d) and os.path.basename(d).isdigit()]
    ckpts.sort(key=lambda d: int(os.path.basename(d)))
    print(f"Found {len(ckpts)} checkpoints under {args.ckpt_root}", flush=True)
    load_policy = lambda p: PC_CKPT.load_policy(
        p, network_factory=PNET.make_ppo_networks, deterministic=True)
    curve = []; env_cache = {}
    for cp in ckpts:
        st = int(os.path.basename(cp))
        alpha = (args.alpha_override if args.alpha_override >= 0
                 else alpha_of(st, args.anneal_total, args.alpha_min, args.alpha_max))
        key = round(alpha, 4)
        if key not in env_cache:
            env_cache[key] = build_env(alpha, args.env_mod, args.steps)
        base, env, reset, step, act_size = env_cache[key]
        t0 = time.time()
        infer = jax.jit(load_policy(cp))
        mbt, mreach, reach0 = jax.jit(
            lambda ks: rollout(reset, step, base, infer, ks, args.n, args.steps,
                               args.seed, act_size, False))(keys)
        row = summarize(mbt, mreach, reach0, st, alpha)
        row["eval_sec"] = round(time.time() - t0, 1)
        curve.append(row); print("STEP", st, row, flush=True)
        json.dump({"protocol_steps": args.steps, "curve": curve},
                  open(args.out, "w"), indent=2)
    if curve:
        final = curve[-1]
        peak = max(curve, key=lambda r: r["success_rate"])
        peak_far = max(curve, key=lambda r: r["success_far"])
        cross = next((r["step"] for r in curve if r["success_rate"] >= 0.66), None)
        out = {"protocol_steps": args.steps, "n_eps": args.n, "far_r0": args.far_r0,
               "final": final, "peak_by_success": peak,
               "peak_by_success_far": peak_far, "steps_to_cross_0.66": cross,
               "curve": curve}
        json.dump(out, open(args.out, "w"), indent=2)
        print("\n=== SUMMARY ===\nFINAL:", final, "\nPEAK :", peak,
              "\nPEAK_FAR:", peak_far, "\ncross0.66:", cross)


if __name__ == "__main__":
    main()
