#!/usr/bin/env python3
"""Planning-only COMPOSITIONAL-PLANNING-DEPTH eval (standalone; does NOT touch run_benchmark hot path).

THE IDEA. jumpy-MPPI plans n_macro macro-steps, each a k-step jump via the jumpy head jdyn —
i.e. it COMPOSES jdyn n_macro times (effective horizon = k*n_macro). Standard runs use n_macro=6
(horizon 24). The mechanism-check found composing a converged k4 head is MORE accurate than a
dedicated longer-k head. THIS SCRIPT answers: on a CONVERGED model, does DEEPER compositional
planning (larger n_macro) raise EVAL RETURN, or does compounding error eventually hurt? Pure
planning-time inference — NO training, NO new dynamics — just re-evaluate ONE trained checkpoint at
several n_macro depths.

PROTOCOL. Load a trained jumpy ckpt (enc/pi/q/jdyn, plus jrew — exactly as
scripts/p3_macroq_decomp.py / value_probe.py load it + construct the env). For each depth n_macro
in the sweep, build the SAME eval planner run_benchmark builds at eval —
tdmpc2.make_jumpy_mppi_fn (see run_benchmark.py:893-898) — with k fixed at the ckpt's jumpy_k and
every other MPPI knob (n_samples, num_elites, n_iter, min/max std, gamma) held at run_benchmark's
eval values, varying ONLY n_macro. Run N eval episodes per depth with the SAME per-episode seeds
across depths (paired comparison; the eval loop is copied verbatim from run_benchmark's eval_jumpy,
lines 1216-1239). Record mean/std/per-seed return, success rate (full-reward fraction, the
_episode_diag 'full' proxy from run_benchmark line 1147-1160), effective horizon k*n_macro, and
per-episode wall-time.

VERDICT. GO iff some depth's mean return exceeds the n_macro=6 baseline by >= +10% (deeper
composition helps); report the argmax depth and its effective horizon. Honest caveats persisted in
JSON: single training checkpoint / single seed-of-training, and planning compute grows ~linearly
with n_macro — so a 'win' that merely costs 2-3x compute is FLAGGED via per-episode wall-time.

REUSE NOTE. The MPPI itself is NOT reimplemented here: make_jumpy_mppi_fn (tdmpc2.py:871) is the
exact eval planner run_benchmark uses, and the rollout loop mirrors eval_jumpy. Deterministic given
--seed. Runs on a worker / ssh7 (EC2 has no jax); JSON output = read-from-JSON discipline.

  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \
    python scripts/plan_depth_eval.py --ckpt <best_mppi.pkl> --task PandaPickCubeOrientation \
      --jumpy_k 4 --n_macros "3,6,9,12,16" --n_ep 20 --mppi_horizon 8 --out r.json
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
# ssh7 is SHARED — never preallocate the whole GPU. Caller should also export these, but set a
# safe default here so a forgotten env var can't OOM the co-tenant Mahjong job.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")

# match run_benchmark / value_probe / p3 sys.path so mujoco_playground + helios resolve on a worker
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wiki/learn_mujoco_playground/repo"))

import jax
import jax.numpy as jnp
from helios.algorithms.tdmpc2 import (
    Encoder, Pi, QEnsemble, JumpyDynamics, JumpyReward, DEFAULTS, make_jumpy_mppi_fn,
)
from mujoco_playground import registry, wrapper


def _episode_diag(rewards_list, episode_length):
    """Copy of run_benchmark.py:_episode_diag (lines 1147-1160): success proxy = 'full' =
    fraction of steps with reward > 0.5 (standing AND fast / task-success per step)."""
    if not rewards_list:
        return {"full": 0.0, "stand": 0.0, "falls": 0, "ttf": episode_length}
    r = np.asarray(rewards_list, dtype=np.float32)
    full = (r > 0.5)
    stand = (r > 0.01)
    falls = int(np.sum(stand[:-1] & (~stand[1:]))) if len(stand) > 1 else 0
    ttf = int(np.argmax(full)) if full.any() else episode_length
    return {"full": float(full.mean()), "stand": float(stand.mean()),
            "falls": falls, "ttf": ttf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", default="PandaPickCubeOrientation")
    ap.add_argument("--jumpy_k", type=int, default=4,
                    help="k-step jump of the ckpt's trained jdyn head (must match training).")
    ap.add_argument("--n_macros", default="3,6,9,12,16",
                    help="comma-sep depth sweep; effective horizon per depth = jumpy_k*n_macro.")
    ap.add_argument("--n_ep", type=int, default=20, help="eval episodes per depth (paired seeds).")
    ap.add_argument("--mppi_horizon", type=int, default=8,
                    help="training mppi_horizon (informational; jumpy-MPPI horizon = k*n_macro, "
                         "NOT this H). Recorded for provenance — match training.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episode_length", type=int, default=1000)
    # arch hyper-params (DEFAULTS in helios.algorithms.tdmpc2 — match p3 / value_probe)
    ap.add_argument("--latent_dim", type=int, default=512)
    ap.add_argument("--num_bins", type=int, default=101)
    ap.add_argument("--V", type=int, default=8)
    ap.add_argument("--dyn_arch", default="mlp", choices=["mlp", "attn", "resmlp"],
                    help="iter-27 backbone of the ckpt (phasei27_jum runs = mlp).")
    # MPPI knobs: defaults = run_benchmark eval values (DEFAULTS dict). Held CONSTANT across depths.
    ap.add_argument("--baseline_n_macro", type=int, default=6,
                    help="the standard-run depth the verdict compares against (horizon 24 at k=4).")
    ap.add_argument("--win_margin", type=float, default=0.10,
                    help="fractional mean-return margin over baseline required for a GO verdict.")
    args = ap.parse_args()

    hidden = (512, 512)
    k = int(args.jumpy_k)
    assert k > 0, "--jumpy_k must be > 0 (the ckpt must contain a trained jdyn head)"
    depths = [int(x) for x in str(args.n_macros).split(",") if str(x).strip()]
    assert depths, "--n_macros parsed to an empty list"
    d = DEFAULTS
    # MPPI eval settings — run_benchmark uses DEFAULTS for the jumpy planner (run_benchmark.py:893):
    #   n_samples=NS(512), num_elites(64), n_iter=NI(6), min_std(0.05), max_std(2.0)
    NS = int(d["NS"]); ELITES = int(d["NUM_ELITES"]); NI = int(d["NI"])
    MIN_STD = float(d["MIN_STD"]); MAX_STD = float(d["MAX_STD"])
    GAMMA = float(d["gamma"])       # 0.99 (the tdmpc2 path value used in run_benchmark)
    al, ah = -1.0, 1.0

    # ── env (mirror run_benchmark/value_probe/p3: registry.load -> wrap_for_brax_training,
    #    single env via split-1)
    env = registry.load(args.task)
    env = wrapper.wrap_for_brax_training(env, episode_length=args.episode_length, action_repeat=1)
    obs_dim, act_dim = env.observation_size, env.action_size
    single_env_reset = jax.jit(lambda kk: env.reset(jax.random.split(kk, 1)))   # run_benchmark:1138
    single_env_step = jax.jit(lambda st, a: env.step(st, a[None]))              # run_benchmark:1134

    # ── nets + checkpoint params (keys enc/pi/q/jdyn/jrew — as init'd in run_benchmark)
    enc_net = Encoder(latent_dim=args.latent_dim, hidden=hidden, V=args.V, arch=args.dyn_arch)
    pi_net = Pi(action_dim=act_dim, hidden=hidden)
    q_net = QEnsemble(hidden=hidden, num_bins=args.num_bins)
    jumpy_net = JumpyDynamics(latent_dim=args.latent_dim, hidden=hidden, V=args.V, arch=args.dyn_arch)
    jumpy_rew_net = JumpyReward(hidden=hidden, num_bins=args.num_bins)
    with open(args.ckpt, "rb") as f:
        params = pickle.load(f)["params"]
    assert "jdyn" in params, (
        f"ckpt has no 'jdyn' params (keys: {sorted(params.keys())}) — not a jumpy ckpt?")
    assert "jrew" in params, (
        f"ckpt has no 'jrew' params (keys: {sorted(params.keys())}) — jumpy-MPPI needs the macro "
        "reward head; was this trained with --jumpy_k>0?")

    def run_depth(n_macro):
        """Build the eval planner at this depth (only n_macro varies) and run the SAME N episodes
        with paired per-episode seeds. Returns (per_seed_returns, per_seed_success, per_ep_walltime,
        compile_walltime)."""
        # === REUSED EVAL PLANNER (identical call to run_benchmark.py:893-898) =================
        plan_jumpy = make_jumpy_mppi_fn(
            enc_net, jumpy_net, jumpy_rew_net, q_net, pi_net,
            k=k, n_macro=n_macro, n_samples=NS, num_elites=ELITES,
            n_iter=NI, min_std=MIN_STD, max_std=MAX_STD,
            act_low=al, act_high=ah, act_dim=act_dim, gamma=GAMMA,
        )
        # ====================================================================================
        rets, succ, walls = [], [], []
        t_compile0 = time.time()
        compile_t = None
        for ep in range(args.n_ep):
            # PAIRED seeds: episode ep uses the SAME PRNG sub-stream at every depth, so depths are
            # compared on identical initial states + planner noise streams.
            ep_key = jax.random.PRNGKey((args.seed * 1_000_003) ^ (ep * 7919 + 1))
            rk2, pk_seed = jax.random.split(ep_key)
            state = single_env_reset(rk2)
            obs = jnp.asarray(state.obs[0])
            # warm-start buffers — shapes mirror run_benchmark.py:1224-1225 (eval_jumpy)
            mu = jnp.zeros((n_macro, k, act_dim))
            std = jnp.full((n_macro, k, act_dim), MAX_STD)
            er = 0.0; ep_rew = []; t0_j = jnp.bool_(True)
            pk = pk_seed
            t_ep0 = time.time()
            for _t in range(args.episode_length):
                act, mu, std = plan_jumpy(params, obs, mu, std, pk, t0_j)
                t0_j = jnp.bool_(False)
                pk, _ = jax.random.split(pk)
                state = single_env_step(state, act)
                r = float(state.reward[0]); er += r; ep_rew.append(r)
                if bool(state.done[0] > 0.5):
                    break
                obs = jnp.asarray(state.obs[0])
            if compile_t is None:
                # first episode includes JIT compile of plan_jumpy at this depth
                compile_t = time.time() - t_compile0
            walls.append(time.time() - t_ep0)
            rets.append(er)
            succ.append(_episode_diag(ep_rew, args.episode_length)["full"])
        return rets, succ, walls, compile_t

    # ── sweep
    per_depth = {}
    for nm in depths:
        rets, succ, walls, compile_t = run_depth(nm)
        rets_a = np.asarray(rets, np.float64)
        succ_a = np.asarray(succ, np.float64)
        # walls[0] includes JIT compile; report steady-state median over the rest when available
        walls_a = np.asarray(walls, np.float64)
        steady = walls_a[1:] if walls_a.size > 1 else walls_a
        per_depth[nm] = {
            "n_macro": nm,
            "effective_horizon": k * nm,
            "return_mean": round(float(rets_a.mean()), 4),
            "return_std": round(float(rets_a.std()), 4),
            "return_per_seed": [round(float(x), 4) for x in rets_a],
            "success_rate_mean": round(float(succ_a.mean()), 4),  # _episode_diag 'full' proxy
            "success_rate_std": round(float(succ_a.std()), 4),
            "success_per_seed": [round(float(x), 4) for x in succ_a],
            "walltime_per_ep_s": round(float(steady.mean()), 3),
            "walltime_per_ep_median_s": round(float(np.median(steady)), 3),
            "first_ep_compile_s": round(float(compile_t), 3),
            "n_ep": int(len(rets)),
        }
        print(f"  n_macro={nm:>3} effH={k*nm:>3}  ret={per_depth[nm]['return_mean']:9.2f}"
              f" +/- {per_depth[nm]['return_std']:7.2f}  succ={per_depth[nm]['success_rate_mean']:.3f}"
              f"  wall/ep={per_depth[nm]['walltime_per_ep_s']:.2f}s", flush=True)

    # ── verdict: GO iff some depth beats the baseline n_macro by >= win_margin on mean return
    baseline = args.baseline_n_macro
    baseline_present = baseline in per_depth
    base_mean = per_depth[baseline]["return_mean"] if baseline_present else None
    # argmax depth by mean return
    best_depth = max(per_depth, key=lambda nm: per_depth[nm]["return_mean"])
    best_mean = per_depth[best_depth]["return_mean"]
    rel_gain = None
    go = False
    if baseline_present:
        denom = abs(base_mean) if abs(base_mean) > 1e-9 else 1e-9
        rel_gain = (best_mean - base_mean) / denom
        go = (best_depth != baseline) and (rel_gain >= args.win_margin)
    verdict = "GO" if go else "NO-GO"

    # compute-cost flag: did the winning depth cost materially more wall-time than baseline?
    compute_flag = None
    if baseline_present and best_depth != baseline:
        b_wall = per_depth[baseline]["walltime_per_ep_s"]
        w_wall = per_depth[best_depth]["walltime_per_ep_s"]
        ratio = (w_wall / b_wall) if b_wall > 1e-9 else float("inf")
        compute_flag = {
            "winner_wall_per_ep_s": w_wall,
            "baseline_wall_per_ep_s": b_wall,
            "wall_ratio_winner_over_baseline": round(float(ratio), 3),
            "flagged_costly_win": bool(go and ratio >= 1.5),
            "note": ("a GO whose winner costs >=1.5x baseline wall-time per episode is flagged: the "
                     "return gain may be bought with planning compute, not a free depth win"),
        }

    out = {
        "probe": "plan_depth_eval",
        "config": {
            "ckpt": args.ckpt, "task": args.task, "jumpy_k": k, "n_macros": depths,
            "n_ep": args.n_ep, "seed": args.seed, "mppi_horizon_train": args.mppi_horizon,
            "episode_length": args.episode_length, "latent_dim": args.latent_dim,
            "num_bins": args.num_bins, "V": args.V, "dyn_arch": args.dyn_arch,
            "obs_dim": int(obs_dim), "act_dim": int(act_dim),
            "mppi_eval_settings": {
                "n_samples": NS, "num_elites": ELITES, "n_iter": NI,
                "min_std": MIN_STD, "max_std": MAX_STD, "gamma": GAMMA,
                "source": "tdmpc2.DEFAULTS — identical to run_benchmark eval (held constant across depths)",
            },
            "planner": "tdmpc2.make_jumpy_mppi_fn (same call as run_benchmark.py:893-898); "
                       "eval loop mirrors run_benchmark eval_jumpy (lines 1216-1239)",
            "baseline_n_macro": baseline, "win_margin": args.win_margin,
        },
        "per_depth": [per_depth[nm] for nm in depths],
        "verdict": verdict,
        "verdict_block": {
            "verdict": verdict,
            "rule": (f"GO iff some depth's mean return exceeds the n_macro={baseline} baseline by "
                     f">= {args.win_margin*100:.0f}% (deeper compositional planning helps); "
                     "NO-GO means deeper composition does not help (compounding error / plateau)"),
            "baseline_n_macro": baseline,
            "baseline_return_mean": base_mean,
            "baseline_present": baseline_present,
            "argmax_depth": best_depth,
            "argmax_effective_horizon": k * best_depth,
            "argmax_return_mean": best_mean,
            "relative_gain_over_baseline": (round(float(rel_gain), 4) if rel_gain is not None else None),
            "compute_cost_flag": compute_flag,
        },
        "caveats": [
            "single training checkpoint / single seed-of-training — no across-training-seed variance",
            "planning compute grows ~linearly with n_macro (n_macro model applies per MPPI sample); "
            "per-episode wall-time is reported per depth so a return 'win' that just costs more "
            "compute is visible and flagged in verdict_block.compute_cost_flag",
            "k is FIXED at the ckpt's jumpy_k and ALL other MPPI knobs are held constant across "
            "depths — only n_macro varies, so this isolates compositional depth",
            "success_rate is the _episode_diag 'full' proxy (fraction of steps with reward>0.5), "
            "the same per-step proxy run_benchmark logs; it is NOT a task-defined terminal success",
            "deeper n_macro stresses jdyn composition further out-of-distribution than training "
            "(which trained 1- and 2-jump horizon-consistency); accuracy there is not guaranteed",
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["verdict_block"], indent=2), flush=True)


if __name__ == "__main__":
    main()
