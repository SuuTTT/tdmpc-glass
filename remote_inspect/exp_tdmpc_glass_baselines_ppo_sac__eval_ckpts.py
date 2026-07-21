"""Dedicated real-success eval for brax PPO/SAC checkpoints on PandaPickCube.
Protocol = SAME as eval_curve.py / TD-MPC2 scoring:
  brax-wrapped env episode_length=STEPS (default 1000), action_repeat=1;
  per parallel episode track MAX metrics["box_target"] and MAX info["reached_box"];
  success = (max box_target >= 0.9); reached = (max reached_box > 0.5).
Evaluates every checkpoint dir to produce a learning curve, reports PEAK + FINAL.
Usage: eval_ckpts.py --algo ppo --ckpt_root <logdir>/checkpoints --n 256 --steps 1000 --out x.json
"""
import os, sys, json, argparse, time, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax_compat  # noqa
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")
import numpy as np
import jax, jax.numpy as jp
from mujoco_playground import registry, wrapper

def get_loader(algo):
    if algo == "ppo":
        from brax.training.agents.ppo import checkpoint as C
        from brax.training.agents.ppo import networks as N
        return lambda p: C.load_policy(p, network_factory=N.make_ppo_networks, deterministic=True)
    else:
        from brax.training.agents.sac import checkpoint as C
        from brax.training.agents.sac import networks as N
        return lambda p: C.load_policy(p, network_factory=N.make_sac_networks, deterministic=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=["ppo","sac"])
    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    load_policy = get_loader(args.algo)
    base = registry.load("PandaPickCube", config_overrides={"impl":"jax"})
    env = wrapper.wrap_for_brax_training(base, episode_length=args.steps, action_repeat=1)
    reset, step = jax.jit(env.reset), jax.jit(env.step)

    # checkpoint dirs are integer-named subdirs
    ckpts = [d for d in glob.glob(os.path.join(args.ckpt_root, "*")) if os.path.isdir(d) and os.path.basename(d).isdigit()]
    ckpts.sort(key=lambda d: int(os.path.basename(d)))
    print(f"Found {len(ckpts)} checkpoints under {args.ckpt_root}", flush=True)

    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.n)
    curve = []
    for cp in ckpts:
        st_steps = int(os.path.basename(cp))
        t0 = time.time()
        infer = load_policy(cp)
        jinfer = jax.jit(infer)
        def run(keys):
            state = reset(keys)
            def body2(carry, k):
                state, mbt, mreach = carry
                rng = k
                act = jax.vmap(jinfer)(state.obs, jax.random.split(rng, args.n))[0]
                state = step(state, act)
                bt = state.metrics["box_target"]
                rb = state.info["reached_box"]
                return (state, jp.maximum(mbt, bt), jp.maximum(mreach, rb)), None
            z = jp.zeros((args.n,))
            ks = jax.random.split(jax.random.PRNGKey(args.seed+1), args.steps)
            (state, mbt, mreach), _ = jax.lax.scan(body2, (state, z, z), ks)
            return mbt, mreach
        mbt, mreach = jax.jit(run)(keys)
        mbt = np.asarray(jax.block_until_ready(mbt)); mreach = np.asarray(mreach)
        row = {
            "step": st_steps,
            "success_rate": round(float((mbt >= 0.9).mean()), 4),
            "reached_rate": round(float((mreach > 0.5).mean()), 4),
            "mean_max_box_target": round(float(mbt.mean()), 4),
            "p90_max_box_target": round(float(np.percentile(mbt, 90)), 4),
            "max_box_target_over_eps": round(float(mbt.max()), 4),
            "n_eps": args.n, "eval_sec": round(time.time()-t0,1),
        }
        curve.append(row)
        print("STEP", st_steps, row, flush=True)
        json.dump({"algo":args.algo,"steps":args.steps,"curve":curve}, open(args.out,"w"), indent=2)

    # PEAK + FINAL
    if curve:
        final = curve[-1]
        peak = max(curve, key=lambda r: r["success_rate"])
        summary = {"algo":args.algo, "steps_protocol":args.steps, "n_eps":args.n,
                   "final": final, "peak_by_success": peak, "curve": curve}
        json.dump(summary, open(args.out,"w"), indent=2)
        print("\n=== SUMMARY", args.algo, "===")
        print("FINAL:", final)
        print("PEAK :", peak)

if __name__ == "__main__":
    main()
