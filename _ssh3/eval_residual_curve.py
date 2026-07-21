#!/usr/bin/env python3
"""Exp#5 real-success curve eval with TRAINING-MATCHED alpha.

For each PPO checkpoint at total env-step S, the residual policy was trained under
alpha(S) = clip(S / ANNEAL_TOTAL, ALPHA_MIN, ALPHA_MAX) where ANNEAL_TOTAL is the
alpha-anneal horizon expressed in TOTAL env-steps (= per-env ANNEAL_STEPS *
num_envs). We therefore evaluate each checkpoint with RES_ALPHA_FIXED = alpha(S),
so executed = clip(a_option + alpha(S)*pi_res, -1, 1) matches the regime the
policy was optimized in. This makes the escape-speed (crosses-0.66) and ceiling
(>=0.82) numbers honest.

Protocol identical to eval_ckpts.py: brax-wrap episode_length=STEPS (1000),
n>=256 parallel eps, success = max box_target >= 0.9.

Rebuilds the residual env per distinct alpha (cheap relative to ckpt eval). All
numbers from real rollouts; PEAK + FINAL reported; written to --out JSON.
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


def alpha_of(step, anneal_total, amin, amax):
    frac = min(max(step / anneal_total, 0.0), 1.0)
    return amin + (amax - amin) * frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--anneal_total", type=float, default=20_480_000.0,
                    help="alpha-anneal horizon in TOTAL env-steps (per-env*num_envs)")
    ap.add_argument("--alpha_min", type=float, default=0.0)
    ap.add_argument("--alpha_max", type=float, default=1.0)
    ap.add_argument("--alpha_override", type=float, default=-1.0,
                    help="if >=0, use this constant alpha for all ckpts (for fixed-alpha runs)")
    ap.add_argument("--gate_mode", action="store_true",
                    help="learned per-state gate: action has +1 gate dim; alpha=sigmoid(gate). Reports per-phase mean alpha.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpts = [d for d in glob.glob(os.path.join(args.ckpt_root, "*"))
             if os.path.isdir(d) and os.path.basename(d).isdigit()]
    ckpts.sort(key=lambda d: int(os.path.basename(d)))
    print(f"Found {len(ckpts)} checkpoints under {args.ckpt_root}", flush=True)

    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.n)
    load_policy = lambda p: PC_CKPT.load_policy(
        p, network_factory=PNET.make_ppo_networks, deterministic=True)

    curve = []
    env_cache = {}

    def get_env(alpha):
        key = "gate" if args.gate_mode else round(alpha, 4)
        if key not in env_cache:
            if args.gate_mode:
                os.environ["RES_GATE_MODE"] = "1"
                os.environ.pop("RES_ALPHA_FIXED", None)
            else:
                os.environ["RES_ALPHA_FIXED"] = str(alpha)
            import residual_env as RE
            importlib.reload(RE)
            base = RE.ResidualPickCube(config_overrides={"impl": "jax"})
            env = wrapper.wrap_for_brax_training(base, episode_length=args.steps,
                                                 action_repeat=1)
            env_cache[key] = (env, jax.jit(env.reset), jax.jit(env.step),
                              base.action_size)
        return env_cache[key]

    for cp in ckpts:
        st = int(os.path.basename(cp))
        alpha = (args.alpha_override if args.alpha_override >= 0
                 else alpha_of(st, args.anneal_total, args.alpha_min, args.alpha_max))
        env, reset, step, act_size = get_env(alpha)
        t0 = time.time()
        infer = jax.jit(load_policy(cp))

        n_phases = 7
        def run(keys):
            state = reset(keys)
            def body(carry, k):
                state, mbt, mreach, asum, acnt = carry
                act = jax.vmap(infer)(state.obs, jax.random.split(k, args.n))[0]
                state = step(state, act)
                a = state.info["alpha"] * jp.ones((args.n,))  # per-env alpha this step
                ph = state.info["phase"].astype(jp.int32)     # (n,)
                oh = jax.nn.one_hot(ph, n_phases)             # (n, P)
                asum = asum + (oh * a[:, None]).sum(axis=0)   # (P,)
                acnt = acnt + oh.sum(axis=0)                  # (P,)
                return (state, jp.maximum(mbt, state.metrics["box_target"]),
                        jp.maximum(mreach, state.info["reached_box"]),
                        asum, acnt), None
            z = jp.zeros((args.n,))
            zp = jp.zeros((n_phases,))
            ks = jax.random.split(jax.random.PRNGKey(args.seed + 1), args.steps)
            (state, mbt, mreach, asum, acnt), _ = jax.lax.scan(
                body, (state, z, z, zp, zp), ks)
            return mbt, mreach, asum, acnt
        mbt, mreach, asum, acnt = jax.jit(run)(keys)
        mbt = np.asarray(jax.block_until_ready(mbt)); mreach = np.asarray(mreach)
        asum = np.asarray(asum); acnt = np.asarray(acnt)
        row = {
            "step": st, "alpha": round(float(alpha), 4),
            "success_rate": round(float((mbt >= 0.9).mean()), 4),
            "reached_rate": round(float((mreach > 0.5).mean()), 4),
            "mean_max_box_target": round(float(mbt.mean()), 4),
            "p90_max_box_target": round(float(np.percentile(mbt, 90)), 4),
            "n_eps": args.n, "eval_sec": round(time.time() - t0, 1),
        }
        if args.gate_mode:
            phase_names = ["APPROACH","DESCEND","GRASP","LIFT","TRANSPORT","PLACE","HOLD"]
            with np.errstate(invalid="ignore", divide="ignore"):
                mean_alpha_phase = np.where(acnt > 0, asum / np.maximum(acnt, 1.0), 0.0)
            row["mean_alpha_overall"] = round(float(asum.sum() / max(acnt.sum(),1.0)), 4)
            row["mean_alpha_by_phase"] = {phase_names[i]: round(float(mean_alpha_phase[i]),4) for i in range(7)}
            row["phase_step_frac"] = {phase_names[i]: round(float(acnt[i]/max(acnt.sum(),1.0)),4) for i in range(7)}
        curve.append(row)
        print("STEP", st, row, flush=True)
        json.dump({"protocol_steps": args.steps, "anneal_total": args.anneal_total,
                   "curve": curve}, open(args.out, "w"), indent=2)

    if curve:
        final = curve[-1]
        peak = max(curve, key=lambda r: r["success_rate"])
        # steps to first crossing 0.66
        cross = next((r["step"] for r in curve if r["success_rate"] >= 0.66), None)
        summary = {"protocol_steps": args.steps, "n_eps": args.n,
                   "anneal_total": args.anneal_total,
                   "alpha_schedule": {"min": args.alpha_min, "max": args.alpha_max,
                                      "override": args.alpha_override},
                   "final": final, "peak_by_success": peak,
                   "steps_to_cross_0.66": cross, "curve": curve}
        json.dump(summary, open(args.out, "w"), indent=2)
        print("\n=== SUMMARY ===")
        print("FINAL:", final)
        print("PEAK :", peak)
        print("steps_to_cross_0.66:", cross)


if __name__ == "__main__":
    main()
