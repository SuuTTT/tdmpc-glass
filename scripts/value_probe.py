#!/usr/bin/env python3
"""iter-28 mechanism-check (standalone; does NOT touch run_benchmark hot path).

Tests the "value-organized abstraction" thesis BEFORE any multi-week build, by measuring two
quantities on a *trained* TD-MPC2 / jumpy checkpoint — the cheap kill-tests for the two levers:

PROBE 1 — value-irrelevant latent capacity (gates the value-equivalence lever, iter-26 `ve`):
  How much of the encoder latent's variance lives in directions that do NOT change the value V?
  V(z)=min-head two_hot_inv(Q(z, pi(z))). We compute dV/dz over the visited-state distribution,
  get the value-relevant subspace (participation ratio of the gradient set), and report the
  fraction of latent variance ORTHOGONAL to it. HIGH value-irrelevant fraction => the
  self-predictive latent wastes capacity on value-irrelevant state => a value-equivalent macro
  model has headroom. LOW => latent is already value-aligned => value-equivalence is redundant
  (the same redundancy that killed the Glass family — predicts a `ve` null).

PROBE 2 — value-criticality variation (gates the value-critical adaptive-horizon lever):
  The killed adaptive-k gated on MODEL ERROR, which is uniform in-distribution (nothing to gate).
  Decision-relevance need not be uniform: at each state, criticality c(z) = spread of Q(z,a) over
  candidate actions (how much the action choice matters). We report the coefficient of variation
  of c across states + the flat-state fraction. HIGH CV / clear flat fraction => a value-critical
  horizon has something to gate on; uniform c => dead on arrival like error-gated adaptive-k.

Reads latents by rolling out the checkpoint's pi policy in the task env (mirrors run_benchmark's
single-env eval exactly). Output = JSON (read-from-JSON discipline). CPU-friendly:
  JAX_PLATFORMS=cpu python scripts/value_probe.py --ckpt <best_mppi.pkl> --task PandaPickCube --out r.json
"""
import argparse, json, os, pickle, sys
from pathlib import Path

import numpy as np

# match run_benchmark's sys.path so mujoco_playground + helios resolve on a worker
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wiki/learn_mujoco_playground/repo"))

import jax
import jax.numpy as jnp
from helios.algorithms.tdmpc2 import Encoder, Pi, QEnsemble, two_hot_inv
from mujoco_playground import registry, wrapper


def participation_ratio(eigs):
    """Effective dimensionality of a set of non-negative spectrum values:
    (sum λ)^2 / sum λ^2 — 1 if one direction dominates, N if uniform."""
    eigs = np.asarray(eigs, np.float64)
    eigs = eigs[eigs > 0]
    if eigs.size == 0:
        return 0.0
    return float((eigs.sum() ** 2) / (eigs ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", default="PandaPickCube")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_ep", type=int, default=16)
    ap.add_argument("--episode_length", type=int, default=1000)
    ap.add_argument("--latent_dim", type=int, default=512)
    ap.add_argument("--num_bins", type=int, default=101)
    ap.add_argument("--V", type=int, default=8)
    ap.add_argument("--n_actions", type=int, default=16, help="probe-2 candidate actions per state")
    ap.add_argument("--action_sigma", type=float, default=0.3, help="probe-2 perturbation std")
    args = ap.parse_args()
    hidden = (512, 512)

    # ── env (mirror run_benchmark: registry.load -> wrap_for_brax_training, single env via split-1)
    env = registry.load(args.task)
    env = wrapper.wrap_for_brax_training(env, episode_length=args.episode_length, action_repeat=1)
    obs_dim, act_dim = env.observation_size, env.action_size
    reset = jax.jit(lambda k: env.reset(jax.random.split(k, 1)))
    step = jax.jit(lambda st, a: env.step(st, a[None]))

    # ── nets + checkpoint params
    enc_net = Encoder(latent_dim=args.latent_dim, hidden=hidden, V=args.V)
    pi_net = Pi(action_dim=act_dim, hidden=hidden)
    q_net = QEnsemble(hidden=hidden, num_bins=args.num_bins)
    with open(args.ckpt, "rb") as f:
        params = pickle.load(f)["params"]

    @jax.jit
    def enc(obs):            # (N,obs)->(N,latent)
        return enc_net.apply(params["enc"], obs)

    @jax.jit
    def act_of(z):           # deterministic policy action tanh(mu)
        mu, _ = pi_net.apply(params["pi"], z)
        return jnp.tanh(mu)

    @jax.jit
    def value_of_z(z):       # (N,latent)->(N,) V=min-head two_hot_inv(Q(z,pi(z)))
        q = q_net.apply(params["q"], z, act_of(z))            # (N,2,num_bins)
        return jnp.min(two_hot_inv(q, num_bins=args.num_bins), axis=-1)

    @jax.jit
    def q_of(z, a):          # (N,latent),(N,act)->(N,) min-head scalar Q
        q = q_net.apply(params["q"], z, a)
        return jnp.min(two_hot_inv(q, num_bins=args.num_bins), axis=-1)

    # gradient of scalar V wrt a single latent (for the value-relevant subspace)
    _gradV = jax.jit(jax.vmap(jax.grad(lambda z: value_of_z(z[None])[0])))

    # ── roll out pi, collect visited observations
    key = jax.random.PRNGKey(0)
    OBS = []
    for ep in range(args.n_ep):
        key, rk = jax.random.split(key)
        st = reset(rk)
        obs = st.obs[0]
        for _t in range(args.episode_length):
            OBS.append(np.asarray(obs))
            z = enc(obs[None])
            a = act_of(z)[0]
            st = step(st, a)
            if bool(st.done[0] > 0.5):
                break
            obs = st.obs[0]
    OBS = np.stack(OBS).astype(np.float32)                     # (N,obs)
    N = OBS.shape[0]

    # batch encode + value
    def batched(fn, X, bs=2048):
        return np.concatenate([np.asarray(fn(jnp.asarray(X[i:i + bs]))) for i in range(0, len(X), bs)], 0)
    Z = batched(enc, OBS)                                      # (N,latent)
    Vv = batched(value_of_z, Z)                               # (N,)

    # ── PROBE 1: value-irrelevant latent capacity
    Zc = Z - Z.mean(0, keepdims=True)
    cov_z = (Zc.T @ Zc) / max(N - 1, 1)
    z_eigs = np.linalg.eigvalsh(cov_z)
    d_z = participation_ratio(z_eigs)                          # effective dim of the latent
    # value-relevant subspace = principal directions of the gradient set dV/dz
    G = batched(lambda z: _gradV(z), Z)                       # (N,latent)
    Gc = G - G.mean(0, keepdims=True)
    gram_g = (Gc.T @ Gc) / max(N - 1, 1)
    g_eigs, g_vecs = np.linalg.eigh(gram_g)                    # ascending
    g_eigs = g_eigs[::-1]; g_vecs = g_vecs[:, ::-1]
    d_val = participation_ratio(g_eigs)                        # effective dim of value-relevant subspace
    r = max(1, int(round(d_val)))
    top = g_vecs[:, :r]                                        # (latent,r) value-relevant basis
    var_total = float(np.trace(cov_z))
    var_in_val = float(np.trace(top.T @ cov_z @ top))
    value_irrelevant_frac = 1.0 - var_in_val / max(var_total, 1e-9)
    # linear decodability of V from Z (sanity)
    A = np.concatenate([Zc, np.ones((N, 1), np.float32)], 1)
    coef, *_ = np.linalg.lstsq(A, Vv - Vv.mean(), rcond=None)
    pred = A @ coef
    ss_res = float(((Vv - Vv.mean() - pred) ** 2).sum())
    ss_tot = float(((Vv - Vv.mean()) ** 2).sum())
    r2_lin = 1.0 - ss_res / max(ss_tot, 1e-9)

    # ── PROBE 2: value-criticality variation
    rng = np.random.default_rng(0)
    A0 = batched(act_of, Z)                                    # (N,act) deterministic action
    crit = np.zeros(N, np.float32)
    for i in range(0, N, 2048):
        zb = Z[i:i + 2048]; ab = A0[i:i + 2048]
        nb = zb.shape[0]
        noise = rng.normal(size=(nb, args.n_actions, act_dim)).astype(np.float32) * args.action_sigma
        cand = np.clip(ab[:, None, :] + noise, -1.0, 1.0)      # (nb,K,act)
        zr = np.repeat(zb, args.n_actions, axis=0)
        qr = np.asarray(q_of(jnp.asarray(zr), jnp.asarray(cand.reshape(-1, act_dim))))
        qr = qr.reshape(nb, args.n_actions)
        crit[i:i + nb] = qr.max(1) - qr.min(1)                 # advantage spread per state
    c_mean = float(crit.mean()); c_std = float(crit.std())
    c_cv = c_std / max(abs(c_mean), 1e-9)
    flat_frac = float((crit < 0.1 * crit.max()).mean())
    v_spread = float(Vv.max() - Vv.min())

    out = {
        "ckpt": args.ckpt, "task": args.task, "n_states": int(N), "n_ep": args.n_ep,
        "obs_dim": int(obs_dim), "act_dim": int(act_dim), "latent_dim": args.latent_dim,
        "probe1_value_equivalence": {
            "effective_dim_latent": round(d_z, 2),
            "effective_dim_value_subspace": round(d_val, 2),
            "value_irrelevant_variance_frac": round(value_irrelevant_frac, 4),
            "linear_V_decode_r2": round(r2_lin, 4),
            "interpretation": "HIGH irrelevant_frac (>~0.5) => headroom for value-equivalence; LOW => latent already value-aligned (predicts ve null)",
        },
        "probe2_value_criticality": {
            "crit_mean": round(c_mean, 4), "crit_std": round(c_std, 4),
            "crit_cv": round(c_cv, 4), "flat_state_frac": round(flat_frac, 4),
            "value_range_over_states": round(v_spread, 3),
            "interpretation": "HIGH cv (>~0.5) + nonzero flat_frac => headroom for value-critical adaptive horizon; uniform => dead like error-gated adaptive-k",
        },
        "value_stats": {"V_mean": round(float(Vv.mean()), 3), "V_min": round(float(Vv.min()), 3), "V_max": round(float(Vv.max()), 3)},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
