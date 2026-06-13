#!/usr/bin/env python3
"""Compounding-error curve: where compositional temporal abstraction stays accurate vs saturates.

THE FIGURE (understanding paper, temporal-axis section): two earlier results need reconciling —
  (i) pyramid mechanism-check: composed-d4 (jdyn applied twice) BEAT a dedicated d8 at h=8;
  (ii) depth-sweep: controller RETURN is flat out to horizon 64.
This script produces the curve that explains both at once: the latent prediction error as a
function of prediction HORIZON h, measured two ways over pi-policy rollouts —

  (A) COMPOSED:        err_compose(h) = || (jdyn ∘ jdyn ∘ ... )(z_t, a_{t:t+h}) − enc(o_{t+h}) ||
        the k=4 jumpy head composed floor(h/4) times over consecutive 4-action chunks.
  (B) ITERATED-1-STEP: err_iter(h)    = || (dyn ∘ dyn ∘ ... )(z_t, a_{t:t+h}) − enc(o_{t+h}) ||
        the 1-step dynamics rolled h times (the SE_DUMP `_iter1` pattern, generalised to h steps).

Both are NORMALISED at each horizon by the median h-step latent DISPLACEMENT
||enc(o_{t+h}) − enc(o_t)|| (so the curve is scale-free per horizon); the raw (unnormalised)
curves are also recorded. The advantage ratio err_iter/err_compose per h is the
"composition advantage vs horizon" line; the crossover horizon is where err_compose_norm first
exceeds a threshold (default 0.5) — the read-as "composition keeps error sub-linear out to
horizon X" point.

Reuses scripts/p3_macroq_decomp.py for ckpt loading / env construction / enc+pi apply, and the
SE_DUMP block of scripts/run_benchmark.py (≈ line 1412-1416, the `_iter1` helper rolling
dyn_net k steps) for path (B). For path (A) jdyn is applied with consecutive (N, k*act) chunks,
exactly as run_benchmark line 1439 (`jumpy_net.apply(params["jdyn"], zt, a_pi.reshape(1, -1))`).

Runs on a worker / ssh7 (/root/venv or the ssh7 venv); EC2 has no jax. Output = JSON
(read-from-JSON discipline). HONEST CAVEATS: single ckpt / single seed; pi-distribution only
(MPPI's planning distribution, where jdyn is actually consumed, may stress the heads differently).

  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \
  python scripts/compounding_curve.py --ckpt <best_mppi.pkl> --task PandaPickCubeOrientation \
      --jumpy_k 4 --horizons "4,8,12,16,24,32,48,64" --n_ep 8 --out r.json
"""
from functools import partial
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# match run_benchmark / p3 sys.path so mujoco_playground + helios resolve on a worker
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wiki/learn_mujoco_playground/repo"))

import jax
import jax.numpy as jnp
from helios.algorithms.tdmpc2 import Dynamics, Encoder, JumpyDynamics, Pi
from mujoco_playground import registry, wrapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", default="PandaPickCubeOrientation")
    ap.add_argument("--jumpy_k", type=int, default=4)
    ap.add_argument("--horizons", default="4,8,12,16,24,32,48,64",
                    help="comma-separated h values; each must be a multiple of --jumpy_k")
    ap.add_argument("--n_ep", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episode_length", type=int, default=1000)
    # arch hyper-params (DEFAULTS in helios.algorithms.tdmpc2 — match p3_macroq_decomp)
    ap.add_argument("--latent_dim", type=int, default=512)
    ap.add_argument("--num_bins", type=int, default=101)
    ap.add_argument("--V", type=int, default=8)
    ap.add_argument("--dyn_arch", default="mlp", choices=["mlp", "attn", "resmlp"],
                    help="iter-27 backbone of the ckpt (phasei27_jum runs = mlp)")
    ap.add_argument("--crossover_thresh", type=float, default=0.5,
                    help="err_compose_norm value defining the crossover horizon")
    args = ap.parse_args()
    hidden = (512, 512)
    k = int(args.jumpy_k)
    assert k > 0, "--jumpy_k must be > 0 (the ckpt must contain a trained jdyn head)"
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    assert horizons, "--horizons parsed empty"
    for h in horizons:
        assert h % k == 0, f"horizon {h} is not a multiple of jumpy_k={k} (composed path needs h/k chunks)"
    Hmax = max(horizons)

    # ── env (mirror run_benchmark / p3: registry.load -> wrap_for_brax_training, single env via split-1)
    env = registry.load(args.task)
    env = wrapper.wrap_for_brax_training(env, episode_length=args.episode_length, action_repeat=1)
    obs_dim, act_dim = env.observation_size, env.action_size
    reset = jax.jit(lambda kk: env.reset(jax.random.split(kk, 1)))
    step = jax.jit(lambda st, a: env.step(st, a[None]))

    # ── nets + checkpoint params (keys: enc / pi / dyn / jdyn — as init'd in run_benchmark)
    enc_net = Encoder(latent_dim=args.latent_dim, hidden=hidden, V=args.V, arch=args.dyn_arch)
    pi_net = Pi(action_dim=act_dim, hidden=hidden)
    dyn_net = Dynamics(latent_dim=args.latent_dim, hidden=hidden, V=args.V, arch=args.dyn_arch)
    jumpy_net = JumpyDynamics(latent_dim=args.latent_dim, hidden=hidden, V=args.V, arch=args.dyn_arch)
    with open(args.ckpt, "rb") as f:
        params = pickle.load(f)["params"]
    assert "jdyn" in params, (
        f"ckpt has no 'jdyn' params (keys: {sorted(params.keys())}) — not a jumpy ckpt?")
    assert "dyn" in params, (
        f"ckpt has no 'dyn' params (keys: {sorted(params.keys())}) — cannot run the iterated-1-step path")

    @jax.jit
    def enc(obs):            # (N,obs)->(N,latent)
        return enc_net.apply(params["enc"], obs)

    @jax.jit
    def act_of(z):           # deterministic policy action tanh(mu)
        mu, _ = pi_net.apply(params["pi"], z)
        return jnp.tanh(mu)

    @jax.jit
    def jdyn(z, a_cat):      # (N,latent),(N,k*act)->(N,latent)  [run_benchmark line 1439 pattern]
        return jumpy_net.apply(params["jdyn"], z, a_cat)

    @jax.jit
    def dyn1(z, a):          # (N,latent),(N,act)->(N,latent)  [SE_DUMP _iter1 single step, line 1415]
        return dyn_net.apply(params["dyn"], z, a)

    # ── COMPOSED path (A): apply jdyn floor(h/k) times over consecutive k-action chunks.
    #    a_win is (N, h, act); reshape each k-chunk to (N, k*act) and feed jdyn (line 1439 pattern).
    @partial(jax.jit, static_argnums=(2,))
    def compose_jdyn(z0, a_win, n_chunks):     # a_win (N,h,act); n_chunks = h//k -> (N,latent)
        def body(z, j):
            chunk = jax.lax.dynamic_slice_in_dim(a_win, j * k, k, axis=1)   # (N,k,act)
            return jdyn(z, chunk.reshape(chunk.shape[0], -1)), None
        z, _ = jax.lax.scan(body, z0, jnp.arange(n_chunks))
        return z

    # ── ITERATED-1-STEP path (B): roll dyn_net h times — the SE_DUMP `_iter1` helper
    #    (run_benchmark lines 1412-1416), generalised from kk to h single steps.
    @partial(jax.jit, static_argnums=(2,))
    def iter1(z0, a_win, n_steps):              # a_win (N,h,act); n_steps = h -> (N,latent)
        def body(z, j):
            a_j = jax.lax.dynamic_index_in_dim(a_win, j, axis=1, keepdims=False)  # (N,act)
            return dyn1(z, a_j), None
        z, _ = jax.lax.scan(body, z0, jnp.arange(n_steps))
        return z

    # ── roll out pi for n_ep episodes; store full per-episode z/a sequences, evaluate curves offline.
    key = jax.random.PRNGKey(args.seed)
    episodes = []   # list of (zseq (T,latent), aseq (T,act))
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
        ep_lens.append(int(len(zseq)))
        episodes.append((zseq, aseq))
    assert any(len(z) > min(horizons) for z, _ in episodes), (
        "no episode longer than the smallest horizon — increase --episode_length or --n_ep")

    # ── per-horizon: gather (z_t, action window a_{t:t+h}, z_true_{t+h}) over all eps, identical tuples
    #    feed both paths. Only t where t+h < episode end (z_{t+h} exists).
    eps = 1e-12
    per_h = []
    for h in horizons:
        n_chunks = h // k
        Zt, Awin, Ztrue, Zt0 = [], [], [], []
        for zseq, aseq in episodes:
            T = len(zseq)
            for t in range(T - h):                             # t+h <= T-1
                Zt.append(zseq[t])
                Awin.append(aseq[t:t + h])                     # (h,act)
                Ztrue.append(zseq[t + h])
                Zt0.append(zseq[t])
            # (Zt and Zt0 identical here; kept separate for clarity of the displacement term)
        if not Zt:
            per_h.append({
                "h": h, "n_samples": 0,
                "err_compose_norm": {"median": None, "mean": None},
                "err_iter_norm": {"median": None, "mean": None},
                "err_compose_raw": {"median": None, "mean": None},
                "err_iter_raw": {"median": None, "mean": None},
                "displacement_raw": {"median": None, "mean": None},
                "advantage_ratio_iter_over_compose": {"median": None, "mean": None},
            })
            continue
        Zt = np.stack(Zt).astype(np.float32)                   # (n,latent)
        Awin = np.stack(Awin).astype(np.float32)               # (n,h,act)
        Ztrue = np.stack(Ztrue).astype(np.float32)             # (n,latent)
        n = Zt.shape[0]

        # batched prediction over both paths (scan is jitted; loop batches to bound device memory)
        def predict(fn, bs=1024):
            outs = []
            for i in range(0, n, bs):
                z0 = jnp.asarray(Zt[i:i + bs])
                aw = jnp.asarray(Awin[i:i + bs])
                if fn == "compose":
                    z = compose_jdyn(z0, aw, n_chunks)
                else:
                    z = iter1(z0, aw, h)
                outs.append(np.asarray(z))
            return np.concatenate(outs, 0)

        Zc = predict("compose")                                # (n,latent) composed-d4
        Zi = predict("iter")                                   # (n,latent) iterated-1-step
        err_c = np.sqrt(((Zc - Ztrue) ** 2).sum(-1))           # (n,) raw composed error
        err_i = np.sqrt(((Zi - Ztrue) ** 2).sum(-1))           # (n,) raw iterated error
        disp = np.sqrt(((Ztrue - Zt) ** 2).sum(-1))            # (n,) h-step latent displacement
        disp_med = float(np.median(disp))                      # normaliser at this horizon
        denom = disp_med + eps
        errc_norm = err_c / denom
        erri_norm = err_i / denom
        # advantage of composition vs naive iteration, per-sample then summarised
        adv = err_i / (err_c + eps)

        per_h.append({
            "h": int(h),
            "n_samples": int(n),
            "err_compose_norm": {"median": round(float(np.median(errc_norm)), 6),
                                 "mean": round(float(errc_norm.mean()), 6)},
            "err_iter_norm": {"median": round(float(np.median(erri_norm)), 6),
                              "mean": round(float(erri_norm.mean()), 6)},
            "err_compose_raw": {"median": round(float(np.median(err_c)), 6),
                                "mean": round(float(err_c.mean()), 6)},
            "err_iter_raw": {"median": round(float(np.median(err_i)), 6),
                             "mean": round(float(err_i.mean()), 6)},
            "displacement_raw": {"median": round(disp_med, 6),
                                 "mean": round(float(disp.mean()), 6)},
            "advantage_ratio_iter_over_compose": {"median": round(float(np.median(adv)), 6),
                                                  "mean": round(float(adv.mean()), 6)},
        })

    # ── summary: crossover horizon (first h where err_compose_norm median > thresh; None if never)
    crossover = None
    for r in per_h:
        m = r["err_compose_norm"]["median"]
        if m is not None and m > args.crossover_thresh:
            crossover = r["h"]
            break
    advantage_by_h = {str(r["h"]): r["advantage_ratio_iter_over_compose"]["median"] for r in per_h}

    out = {
        "probe": "compounding_curve",
        "config": {
            "ckpt": args.ckpt, "task": args.task, "jumpy_k": k, "horizons": horizons,
            "n_ep": args.n_ep, "seed": args.seed, "episode_length": args.episode_length,
            "latent_dim": args.latent_dim, "num_bins": args.num_bins, "V": args.V,
            "dyn_arch": args.dyn_arch, "crossover_thresh": args.crossover_thresh,
            "obs_dim": int(obs_dim), "act_dim": int(act_dim),
        },
        "ep_lens": ep_lens,
        "curves": per_h,
        "summary": {
            "crossover_horizon": crossover,
            "crossover_rule": (
                f"first h where median(err_compose_norm) > {args.crossover_thresh}; "
                "None = composition stays below threshold across all measured horizons "
                "(error sub-linear / scale-free out to max horizon)"),
            "composition_advantage_ratio_iter_over_compose_by_h": advantage_by_h,
            "advantage_note": (
                "ratio > 1 means the iterated-1-step model accumulates MORE error than the "
                "composed jumpy model at that horizon (composition advantage); ratio trending "
                "toward 1 with h means the advantage shrinks as both saturate"),
        },
        "interpretation": (
            "err_compose_norm is the scale-free latent error of the k=4 jumpy head composed "
            "h/k times; err_iter_norm is the 1-step model rolled h times. Reconciles the pyramid "
            "result (composed-d4 beats dedicated d8 at h=8) with the flat return-vs-depth sweep: "
            "composition keeps error sub-linear out to the crossover horizon, but the controller's "
            "return saturates earlier because the policy does not need horizon beyond that."),
        "caveats": [
            "single checkpoint / single seed — no across-seed variance estimate",
            "pi-policy state-action distribution only; MPPI's planning distribution (where the "
            "jumpy head is actually consumed) may stress the heads differently",
            "deterministic pi (tanh(mu)) rollouts — no exploration-noise coverage",
            "normaliser is the median h-step displacement on the SAME pi distribution; horizons "
            "with few long-episode samples have a noisier normaliser",
            "composed path requires each horizon to be a multiple of jumpy_k=4; iterated path "
            "rolls the 1-step head h times so any error in dyn compounds fully",
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["summary"], indent=2), flush=True)
    print(f"[compounding_curve] wrote {args.out} ({len(per_h)} horizons, "
          f"{sum(r['n_samples'] for r in per_h)} total samples)", flush=True)


if __name__ == "__main__":
    main()
