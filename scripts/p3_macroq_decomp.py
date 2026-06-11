#!/usr/bin/env python3
"""iter-30 P3 mechanism-check (standalone; does NOT touch run_benchmark hot path).

THE QUESTION (cheap kill-test BEFORE training anything): P3 proposes retraining the jumpy
k-step head d_k to be VALUE-equivalent (preserve macro-Q) instead of state-faithful. That
lever only has headroom if d_k's CURRENT prediction errors actually COST value — i.e. the
latent-space error must translate into Q-error. If d_k's errors live in value-irrelevant
directions (Q(ẑ_k) ≈ Q(z_true_k) even when ||ẑ_k − z_true_k|| is large), a value-equivalent
loss has nothing to fix → NO-GO without training (the redundancy criterion, condition C1,
applied to the macro head).

PROTOCOL: load a trained jumpy ckpt, roll out the pi policy for --n_ep episodes; at each
step t with t+k inside the episode compute
    z_t = enc(o_t);  z_true = enc(o_{t+k});  ẑ = jdyn(z_t, a_{t:t+k})
    latent_err = ||ẑ − z_true||_2
    q_true = Q(z_true, a_{t+k});  q_pred = Q(ẑ, a_{t+k});  q_err = |q_pred − q_true|
(Q = reduction over the 2-head ensemble after two_hot_inv; default mean, --q_reduce min
matches value_probe's min-head convention.)

METRICS + PRE-REGISTERED VERDICT:
    spearman = rank-corr(latent_err, q_err)
    value_cost_ratio = median(q_err) / (median(|q_true − median(q_true)|) + eps)
        (q_err relative to the natural Q spread, MAD)
    value_irrelevant_error_frac = P(latent_err in top quartile AND q_err in bottom half)
        (joint; independence baseline 0.125 — also reported conditionally, baseline 0.5)
GO  iff value_cost_ratio >= 0.3 AND spearman >= 0.4  (errors are big in value terms AND
systematically cost value → a value-equivalent loss has a signal to exploit).
NO-GO otherwise: d_k errors are already value-benign → value-equivalent training has
nothing to gain.

Mirrors scripts/value_probe.py for ckpt loading / env construction / Q apply, and the
SE_DUMP block of run_benchmark.py (≈ line 1386-1450) for the jumpy-net apply. Runs on a
worker (/root/helios-rl, /root/venv/bin/python); EC2 has no jax. Output = JSON
(read-from-JSON discipline).
  JAX_PLATFORMS=cpu python scripts/p3_macroq_decomp.py --ckpt <best_mppi.pkl> \
      --task PandaPickCubeOrientation --jumpy_k 4 --n_ep 8 --out r.json
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# match run_benchmark / value_probe sys.path so mujoco_playground + helios resolve on a worker
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wiki/learn_mujoco_playground/repo"))

import jax
import jax.numpy as jnp
from helios.algorithms.tdmpc2 import Encoder, JumpyDynamics, Pi, QEnsemble, two_hot_inv
from mujoco_playground import registry, wrapper


def spearman(x, y):
    """Spearman rank correlation without scipy (average ranks via argsort-of-argsort
    is fine here: float errs make ties measure-zero)."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    if x.size < 3:
        return float("nan")
    rx = np.empty(x.size)
    rx[np.argsort(x)] = np.arange(x.size)
    ry = np.empty(y.size)
    ry[np.argsort(y)] = np.arange(y.size)
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / max(den, 1e-12))


def pearson(x, y):
    x = np.asarray(x, np.float64) - np.mean(x)
    y = np.asarray(y, np.float64) - np.mean(y)
    den = np.sqrt((x ** 2).sum() * (y ** 2).sum())
    return float((x * y).sum() / max(den, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", default="PandaPickCubeOrientation")
    ap.add_argument("--jumpy_k", type=int, default=4)
    ap.add_argument("--n_ep", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episode_length", type=int, default=1000)
    # arch hyper-params (DEFAULTS in helios.algorithms.tdmpc2 — match value_probe)
    ap.add_argument("--latent_dim", type=int, default=512)
    ap.add_argument("--num_bins", type=int, default=101)
    ap.add_argument("--V", type=int, default=8)
    ap.add_argument("--dyn_arch", default="mlp", choices=["mlp", "attn", "resmlp"],
                    help="iter-27 backbone of the ckpt (phasei27_jum runs = mlp)")
    ap.add_argument("--q_reduce", default="mean", choices=["mean", "min"],
                    help="ensemble reduction for Q (spec: mean; value_probe used min)")
    # pre-registered gates
    ap.add_argument("--gate_ratio", type=float, default=0.3)
    ap.add_argument("--gate_spearman", type=float, default=0.4)
    args = ap.parse_args()
    hidden = (512, 512)
    k = int(args.jumpy_k)
    assert k > 0, "--jumpy_k must be > 0 (the ckpt must contain a trained jdyn head)"

    # ── env (mirror run_benchmark/value_probe: registry.load -> wrap_for_brax_training,
    #    single env via split-1)
    env = registry.load(args.task)
    env = wrapper.wrap_for_brax_training(env, episode_length=args.episode_length, action_repeat=1)
    obs_dim, act_dim = env.observation_size, env.action_size
    reset = jax.jit(lambda kk: env.reset(jax.random.split(kk, 1)))
    step = jax.jit(lambda st, a: env.step(st, a[None]))

    # ── nets + checkpoint params (keys: enc / pi / q / jdyn — as init'd in run_benchmark)
    enc_net = Encoder(latent_dim=args.latent_dim, hidden=hidden, V=args.V, arch=args.dyn_arch)
    pi_net = Pi(action_dim=act_dim, hidden=hidden)
    q_net = QEnsemble(hidden=hidden, num_bins=args.num_bins)
    jumpy_net = JumpyDynamics(latent_dim=args.latent_dim, hidden=hidden, V=args.V, arch=args.dyn_arch)
    with open(args.ckpt, "rb") as f:
        params = pickle.load(f)["params"]
    assert "jdyn" in params, (
        f"ckpt has no 'jdyn' params (keys: {sorted(params.keys())}) — not a jumpy ckpt?")

    @jax.jit
    def enc(obs):            # (N,obs)->(N,latent)
        return enc_net.apply(params["enc"], obs)

    @jax.jit
    def act_of(z):           # deterministic policy action tanh(mu)
        mu, _ = pi_net.apply(params["pi"], z)
        return jnp.tanh(mu)

    @jax.jit
    def q_of(z, a):          # (N,latent),(N,act)->(N,) scalar Q, ensemble-reduced
        q = q_net.apply(params["q"], z, a)                     # (N,2,num_bins)
        qs = two_hot_inv(q, num_bins=args.num_bins)            # (N,2)
        if args.q_reduce == "min":
            return jnp.min(qs, axis=-1)
        return jnp.mean(qs, axis=-1)

    @jax.jit
    def jdyn(z, a_cat):      # (N,latent),(N,k*act)->(N,latent)  [SE_DUMP pattern]
        return jumpy_net.apply(params["jdyn"], z, a_cat)

    # ── roll out pi for n_ep episodes, collect (z_t, a_{t:t+k}, z_{t+k}, a_{t+k}) tuples
    key = jax.random.PRNGKey(args.seed)
    Zt, Acat, Ztk, Atk = [], [], [], []
    ep_lens = []
    for _ep in range(args.n_ep):
        key, rk = jax.random.split(key)
        st = reset(rk)
        obs = st.obs[0]
        zseq, aseq = [], []
        for _t in range(args.episode_length):
            z = enc(obs[None])
            a = act_of(z)[0]
            zseq.append(np.asarray(z).reshape(-1))
            aseq.append(np.asarray(a).reshape(-1))
            st = step(st, a)
            if bool(st.done[0] > 0.5):
                break
            obs = st.obs[0]
        zseq = np.stack(zseq).astype(np.float32)               # (T,latent)
        aseq = np.stack(aseq).astype(np.float32)               # (T,act)
        T = len(zseq)
        ep_lens.append(T)
        if T <= k:
            continue
        for t in range(T - k):                                 # t+k <= T-1, a_{t+k} exists
            Zt.append(zseq[t])
            Acat.append(aseq[t:t + k].reshape(-1))             # (k*act,)
            Ztk.append(zseq[t + k])
            Atk.append(aseq[t + k])
    assert Zt, "no (t, t+k) pairs collected — episodes shorter than k?"
    Zt = np.stack(Zt)
    Acat = np.stack(Acat)
    Ztk = np.stack(Ztk)
    Atk = np.stack(Atk)
    N = Zt.shape[0]

    # ── batched jumpy prediction + Q decomposition
    def batched2(fn, X, Y, bs=2048):
        return np.concatenate(
            [np.asarray(fn(jnp.asarray(X[i:i + bs]), jnp.asarray(Y[i:i + bs])))
             for i in range(0, N, bs)], 0)
    Zhat = batched2(jdyn, Zt, Acat)                            # (N,latent) ẑ = d_k(z_t, a_{t:t+k})
    latent_err = np.sqrt(((Zhat - Ztk) ** 2).sum(-1))          # (N,)
    q_true = batched2(q_of, Ztk, Atk)                          # (N,) Q(z_true_{t+k}, a_{t+k})
    q_pred = batched2(q_of, Zhat, Atk)                         # (N,) Q(ẑ, a_{t+k})
    q_err = np.abs(q_pred - q_true)                            # (N,)

    # ── metrics
    eps = 1e-9
    sp = spearman(latent_err, q_err)
    pr = pearson(latent_err, q_err)
    q_med = float(np.median(q_true))
    q_mad = float(np.median(np.abs(q_true - q_med)))           # natural Q spread (MAD)
    value_cost_ratio = float(np.median(q_err) / (q_mad + eps))
    # value-irrelevant error fraction: big latent error, small value error
    le_q3 = float(np.quantile(latent_err, 0.75))
    qe_med = float(np.median(q_err))
    top_latent = latent_err >= le_q3
    low_q = q_err <= qe_med
    vif_joint = float((top_latent & low_q).mean())             # independence baseline 0.125
    vif_cond = float(low_q[top_latent].mean()) if top_latent.any() else float("nan")  # baseline 0.5

    # per-quartile decomposition: median q_err within each latent_err quartile
    qedges = np.quantile(latent_err, [0.0, 0.25, 0.5, 0.75, 1.0])
    per_quartile = []
    for qi in range(4):
        lo, hi = qedges[qi], qedges[qi + 1]
        m = (latent_err >= lo) & (latent_err <= hi if qi == 3 else latent_err < hi)
        per_quartile.append({
            "latent_err_range": [round(float(lo), 4), round(float(hi), 4)],
            "n": int(m.sum()),
            "q_err_median": round(float(np.median(q_err[m])), 4) if m.any() else None,
            "q_err_mean": round(float(q_err[m].mean()), 4) if m.any() else None,
        })

    go = (value_cost_ratio >= args.gate_ratio) and (sp >= args.gate_spearman)
    verdict = "GO" if go else "NO-GO"

    qtile = lambda a, qs: {f"p{int(q*100)}": round(float(np.quantile(a, q)), 4)
                           for q in qs}
    out = {
        "probe": "p3_macroq_decomp",
        "config": {
            "ckpt": args.ckpt, "task": args.task, "jumpy_k": k, "n_ep": args.n_ep,
            "seed": args.seed, "episode_length": args.episode_length,
            "latent_dim": args.latent_dim, "num_bins": args.num_bins, "V": args.V,
            "dyn_arch": args.dyn_arch, "q_reduce": args.q_reduce,
            "obs_dim": int(obs_dim), "act_dim": int(act_dim),
            "gates": {"value_cost_ratio_min": args.gate_ratio,
                      "spearman_min": args.gate_spearman},
        },
        "n_pairs": int(N), "ep_lens": ep_lens,
        "latent_err": {**qtile(latent_err, [0.05, 0.25, 0.5, 0.75, 0.95]),
                       "mean": round(float(latent_err.mean()), 4)},
        "q_err": {**qtile(q_err, [0.05, 0.25, 0.5, 0.75, 0.95]),
                  "mean": round(float(q_err.mean()), 4)},
        "q_true_stats": {"median": round(q_med, 4), "mad": round(q_mad, 4),
                         "std": round(float(q_true.std()), 4),
                         **qtile(q_true, [0.05, 0.5, 0.95])},
        "correlations": {"spearman_latent_err_vs_q_err": round(sp, 4),
                         "pearson_latent_err_vs_q_err": round(pr, 4)},
        "value_cost_ratio": round(value_cost_ratio, 4),
        "value_irrelevant_error_frac": {
            "joint": round(vif_joint, 4), "joint_baseline_if_independent": 0.125,
            "conditional_on_top_latent_quartile": round(vif_cond, 4),
            "conditional_baseline_if_independent": 0.5,
        },
        "per_latent_err_quartile": per_quartile,
        "verdict": verdict,
        "verdict_rule": (
            f"GO iff value_cost_ratio >= {args.gate_ratio} AND spearman >= {args.gate_spearman}; "
            "NO-GO means d_k errors are already value-benign -> value-equivalent training has "
            "nothing to gain (redundancy criterion C1 applied to the macro head)"),
        "caveats": [
            "single checkpoint / single seed — no across-seed variance estimate",
            "pi-policy state-action distribution only; MPPI's planning distribution (where the "
            "jumpy head is actually consumed) may stress d_k differently",
            "Q is used as its own judge: q_err is measured with the ckpt's own Q ensemble, so "
            "Q-approximation error and Q's own smoothness both contaminate the decomposition",
            "deterministic pi (tanh(mu)) rollouts — no exploration-noise coverage",
            f"q_reduce={args.q_reduce} over the 2-head ensemble (value_probe used min)",
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
