#!/usr/bin/env python3
"""V(z)-decode vs return-to-go-decode probe (Paper-A criterion metric test).

On a trained checkpoint, roll out pi and fit TWO held-out ridge linear probes from the latent z:
  (a) target = V(z)   = min-head two_hot_inv(Q(z, pi(z)))   -- the value NETWORK (low variance)
  (b) target = discounted Monte-Carlo return-to-go          -- the confounded metric we used
Report held-out test R² for both, plus mean return + V spread. The question: does V(z)-decode
AVOID the return-variance confound (high R2 even when policy collapsed) and DISCRIMINATE across
policies? Supports distractor envs so it matches distractor-trained checkpoints.

  JAX_PLATFORMS=cpu python scripts/vz_probe.py --ckpt <best_mppi.pkl> --task CheetahRun \
      --distractor_dims 0 --out r.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import jax, jax.numpy as jnp
from helios.algorithms.tdmpc2 import Encoder, Pi, QEnsemble, two_hot_inv
from mujoco_playground import registry, wrapper


def add_distractors(env, n_dims, scale=1.0, rho=0.95):
    if not n_dims or n_dims <= 0:
        return env
    class _D:
        def __init__(s, inner): s.__dict__["_inner"] = inner
        def __getattr__(s, k): return getattr(s.__dict__["_inner"], k)
        @property
        def observation_size(s): return s.__dict__["_inner"].observation_size + n_dims
        def reset(s, rng):
            k_env, k_noise = jax.random.split(rng)
            st = s.__dict__["_inner"].reset(k_env)
            z = scale * jax.random.normal(k_noise, (n_dims,))
            st.info["_distract_z"] = z; st.info["_distract_k"] = k_noise
            return st.replace(obs=jnp.concatenate([st.obs, z], -1))
        def step(s, st, a):
            z = st.info["_distract_z"]; k = st.info["_distract_k"]
            k, sub = jax.random.split(k)
            z = rho * z + scale * jnp.sqrt(1 - rho * rho) * jax.random.normal(sub, z.shape)
            s2 = s.__dict__["_inner"].step(st, a)
            s2.info["_distract_z"] = z; s2.info["_distract_k"] = k
            return s2.replace(obs=jnp.concatenate([s2.obs, z], -1))
    return _D(env)


def heldout_ridge_r2(Z, y, lam=10.0):
    Z = np.asarray(Z, np.float64); y = np.asarray(y, np.float64)
    n = len(Z)
    if n < 100:
        return float("nan")
    rng = np.random.default_rng(0)
    idx = rng.permutation(n); cut = int(n * 0.8)
    tr, te = idx[:cut], idx[cut:]
    Xtr, Xte = Z[tr], Z[te]; ytr, yte = y[tr], y[te]
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    ym = ytr.mean(); d = Xtr.shape[1]
    w = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(d), Xtr.T @ (ytr - ym))
    pred = Xte @ w + ym
    sr = float(((yte - pred) ** 2).sum()); stt = float(((yte - yte.mean()) ** 2).sum())
    return float(1.0 - sr / stt) if stt > 1e-8 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", default="CheetahRun")
    ap.add_argument("--distractor_dims", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_ep", type=int, default=8)
    ap.add_argument("--episode_length", type=int, default=1000)
    ap.add_argument("--latent_dim", type=int, default=512)
    ap.add_argument("--num_bins", type=int, default=101)
    ap.add_argument("--V", type=int, default=8)
    ap.add_argument("--gamma", type=float, default=0.99)
    args = ap.parse_args()
    hidden = (512, 512)

    env = registry.load(args.task, config_overrides={"impl": "jax"})
    env = add_distractors(env, args.distractor_dims)
    env = wrapper.wrap_for_brax_training(env, episode_length=args.episode_length, action_repeat=1)
    obs_dim, act_dim = env.observation_size, env.action_size
    reset = jax.jit(lambda k: env.reset(jax.random.split(k, 1)))
    step = jax.jit(lambda st, a: env.step(st, a[None]))

    enc_net = Encoder(latent_dim=args.latent_dim, hidden=hidden, V=args.V)
    pi_net = Pi(action_dim=act_dim, hidden=hidden)
    q_net = QEnsemble(hidden=hidden, num_bins=args.num_bins)
    import pickle
    with open(args.ckpt, "rb") as f:
        params = pickle.load(f)["params"]

    @jax.jit
    def enc(o): return enc_net.apply(params["enc"], o)
    @jax.jit
    def act_of(z):
        mu, _ = pi_net.apply(params["pi"], z); return jnp.tanh(mu)
    @jax.jit
    def value_of_z(z):
        q = q_net.apply(params["q"], z, act_of(z))
        return jnp.min(two_hot_inv(q, num_bins=args.num_bins), -1)

    key = jax.random.PRNGKey(0)
    OBS, REW, EPLENS = [], [], []
    rets = []
    for ep in range(args.n_ep):
        key, rk = jax.random.split(key)
        st = reset(rk); obs = st.obs[0]; er = 0.0; ep_obs = []; ep_rew = []
        for _t in range(args.episode_length):
            ep_obs.append(np.asarray(obs))
            z = enc(obs[None]); a = act_of(z)[0]
            st = step(st, a); r = float(st.reward[0]); er += r; ep_rew.append(r)
            if bool(st.done[0] > 0.5): break
            obs = st.obs[0]
        rets.append(er)
        # discounted return-to-go aligned with ep_obs
        g = 0.0; rtg = [0.0] * len(ep_rew)
        for t in range(len(ep_rew) - 1, -1, -1):
            g = ep_rew[t] + args.gamma * g; rtg[t] = g
        n = min(len(ep_obs), len(rtg))
        OBS.extend(ep_obs[:n]); REW.extend(rtg[:n])
    OBS = np.stack(OBS).astype(np.float32); RTG = np.asarray(REW, np.float32)

    def batched(fn, X, bs=2048):
        return np.concatenate([np.asarray(fn(jnp.asarray(X[i:i+bs]))) for i in range(0, len(X), bs)], 0)
    Z = batched(enc, OBS); Vv = batched(value_of_z, Z)

    out = {
        "ckpt": args.ckpt, "task": args.task, "distractor_dims": args.distractor_dims,
        "n_states": int(len(Z)), "n_ep": args.n_ep, "obs_dim": int(obs_dim),
        "mean_return": round(float(np.mean(rets)), 2),
        "r2_Vz_decode_heldout": round(heldout_ridge_r2(Z, Vv), 4),
        "r2_returntogo_decode_heldout": round(heldout_ridge_r2(Z, RTG), 4),
        "V_mean": round(float(Vv.mean()), 3), "V_std": round(float(Vv.std()), 3),
        "rtg_mean": round(float(RTG.mean()), 3), "rtg_std": round(float(RTG.std()), 3),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
