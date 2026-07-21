#!/usr/bin/env python3
"""Part 34 real-success eval for the fuller-parametrization residual.

Identical to eval_residual_orientation_oriaware.py but builds the env from
residual_env_oriaware_v2 so RES_ROT_FROM / RES_ROTERR_OBS levers take effect.
success = max metrics[box_target] >= 0.9, n>=256, deterministic policy.
"""
import os, sys, json, argparse, time, glob, importlib
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MJPG_IMPL", "jax")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")
sys.path.insert(0, "/root/tdmpc_glass/exp/tdmpc_glass/baselines_ppo_sac")
sys.path.insert(0, "/root/tdmpc_glass/helios-rl/hl_pickcube")
import jax_compat  # noqa
import numpy as np
import jax, jax.numpy as jp
from mujoco_playground import wrapper
from brax.training.agents.ppo import checkpoint as PC_CKPT
from brax.training.agents.ppo import networks as PNET


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--alpha_override", type=float, default=1.0)
    ap.add_argument("--residual", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpts = [d for d in glob.glob(os.path.join(args.ckpt_root, "*"))
             if os.path.isdir(d) and os.path.basename(d).isdigit()]
    ckpts.sort(key=lambda d: int(os.path.basename(d)))
    print(f"Found {len(ckpts)} checkpoints under {args.ckpt_root}", flush=True)

    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.n)
    load_policy = lambda p: PC_CKPT.load_policy(
        p, network_factory=PNET.make_ppo_networks, deterministic=True)

    if args.residual:
        os.environ["RES_ALPHA_FIXED"] = str(args.alpha_override)
        import residual_env_oriaware_v2 as RE
        importlib.reload(RE)
        base = RE.ResidualPickCube(config_overrides={"impl": "jax"},
                                   sample_orientation=True)
    else:
        from mujoco_playground import registry
        base = registry.load("PandaPickCubeOrientation",
                             config_overrides={"impl": "jax"})
    env = wrapper.wrap_for_brax_training(base, episode_length=args.steps,
                                         action_repeat=1)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    curve = []
    for cp in ckpts:
        st = int(os.path.basename(cp))
        t0 = time.time()
        infer = jax.jit(load_policy(cp))

        def run(keys):
            state = reset(keys)
            def body(carry, k):
                state, mbt, mreach = carry
                act = jax.vmap(infer)(state.obs, jax.random.split(k, args.n))[0]
                state = step(state, act)
                reached = state.info.get("reached_box", jp.zeros((args.n,)))
                return (state, jp.maximum(mbt, state.metrics["box_target"]),
                        jp.maximum(mreach, reached)), None
            z = jp.zeros((args.n,))
            ks = jax.random.split(jax.random.PRNGKey(args.seed + 1), args.steps)
            (state, mbt, mreach), _ = jax.lax.scan(body, (state, z, z), ks)
            return mbt, mreach
        mbt, mreach = jax.jit(run)(keys)
        mbt = np.asarray(jax.block_until_ready(mbt)); mreach = np.asarray(mreach)
        row = {
            "step": st,
            "success_rate": round(float((mbt >= 0.9).mean()), 4),
            "reached_rate": round(float((mreach > 0.5).mean()), 4),
            "mean_max_box_target": round(float(mbt.mean()), 4),
            "p90_max_box_target": round(float(np.percentile(mbt, 90)), 4),
            "n_eps": args.n, "eval_sec": round(time.time() - t0, 1),
        }
        curve.append(row)
        print("STEP", st, row, flush=True)
        json.dump({"task": "PandaPickCubeOrientation",
                   "residual": bool(args.residual),
                   "alpha": args.alpha_override if args.residual else None,
                   "protocol_steps": args.steps, "curve": curve},
                  open(args.out, "w"), indent=2)

    if curve:
        final = curve[-1]
        peak = max(curve, key=lambda r: r["success_rate"])
        cross66 = next((r["step"] for r in curve if r["success_rate"] >= 0.66), None)
        cross40 = next((r["step"] for r in curve if r["success_rate"] >= 0.40), None)
        summary = {"final": final, "peak": peak,
                   "steps_to_0.40": cross40, "steps_to_0.66": cross66}
        out = json.load(open(args.out))
        out["summary"] = summary
        json.dump(out, open(args.out, "w"), indent=2)
        print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
