#!/usr/bin/env python3
"""Pyramid World Models mechanism-check (standalone; does NOT touch run_benchmark hot path).

THE QUESTION (cheap kill-test BEFORE building any hierarchy): a pyramid world model
composes short jumps into long ones (d4∘d4 covers 8 steps). That architecture only has a
foundation if COMPOSING two 4-step jumps is more accurate at horizon 8 than one monolithic
8-step jump:
    err4comp = || d4(d4(z_t, a_{t:t+4}), a_{t+4:t+8}) − enc4(o_{t+8}) ||      (k4 ckpt)
    err8     = || d8(z_t, a_{t:t+8})                  − enc8(o_{t+8}) ||      (k8 ckpt)
If composition wins → GO (pyramid has a foundation). If monolithic wins or ties → NO-GO,
the idea dies free.

CROSS-MODEL SUBTLETY: the two ckpts were trained separately, so their encoders define
DIFFERENT latent spaces — each model is evaluated against ITS OWN encoder's target
(enc4(o_{t+8}) vs enc8(o_{t+8})), and errors are NORMALIZED by that latent's natural
8-step scale: median ||enc(o_{t+8}) − enc(o_t)|| over the eval stream. Unnormalized values
are also reported. SimNorm bounds both latents identically (V-simplex blocks), which makes
the normalized comparison reasonable, but it remains a cross-space comparison — caveated.

SHARED BEHAVIOR STREAM: the k4 ckpt's deterministic policy (tanh(mu)) generates --n_ep
episodes once; observations+actions are recorded and BOTH models are evaluated offline on
the identical (o_t, a_{t:t+8}, o_{t+8}) tuples.

METRICS + PRE-REGISTERED VERDICT:
    rho      = median(err4comp_norm) / median(err8_norm)
    win_rate = P(err4comp_norm < err8_norm)  per-state
    same stats restricted to top-quartile 8-step-displacement states (fast/contact states)
GO  iff rho <= 0.9 AND win_rate >= 0.6  (composition at least 10% more accurate AND wins
on most states). NO-GO otherwise.
ASYMMETRY (stated in JSON): d4∘d4 was never trained for 8-step consistency while d8 was
trained exactly for it — if composition wins anyway, a trained pyramid should do even
better (strong GO); if it loses, cross-level training could still rescue it (softer NO-GO).

Ckpt loading / env construction / jdyn apply mirror scripts/p3_macroq_decomp.py
(params keys enc/pi/jdyn; jdyn.apply(params['jdyn'], z, a_concat), a_concat=(N, k*act_dim)).
Runs on a worker (/root/helios-rl, /root/venv/bin/python); EC2 has no jax. Output = JSON
(read-from-JSON discipline). ssh7 is SHARED — keep the XLA env vars:
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \
  /root/venv/bin/python scripts/pyramid_mechcheck.py \
      --ckpt_k4 <phasei27_jum best_mppi.pkl> --ckpt_k8 <phasei30_jumk8 best_mppi.pkl> \
      --task PandaPickCubeOrientation --n_ep 8 --out r.json
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
# ssh7 is SHARED with the Mahjong RL project — never preallocate, cap at 35% of 16GB.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")

# match run_benchmark / p3_macroq_decomp sys.path so mujoco_playground + helios resolve
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wiki/learn_mujoco_playground/repo"))

import jax
import jax.numpy as jnp
from helios.algorithms.tdmpc2 import Encoder, JumpyDynamics, Pi
from mujoco_playground import registry, wrapper


def qtile(a, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    return {f"p{int(q * 100)}": round(float(np.quantile(a, q)), 5) for q in qs}


def stats(a):
    return {**qtile(a), "mean": round(float(np.mean(a)), 5), "n": int(np.size(a))}


def spearman(x, y):
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


def build_model(ckpt_path, k, act_dim, latent_dim, hidden, V, dyn_arch):
    """Load one ckpt and return jitted enc / pi-action / jdyn fns + its params (p3 pattern)."""
    with open(ckpt_path, "rb") as f:
        params = pickle.load(f)["params"]
    assert "jdyn" in params, (
        f"{ckpt_path}: no 'jdyn' params (keys: {sorted(params.keys())}) — not a jumpy ckpt?")
    enc_net = Encoder(latent_dim=latent_dim, hidden=hidden, V=V, arch=dyn_arch)
    pi_net = Pi(action_dim=act_dim, hidden=hidden)
    jumpy_net = JumpyDynamics(latent_dim=latent_dim, hidden=hidden, V=V, arch=dyn_arch)
    # sanity: jdyn input width must match latent_dim + k*act_dim (catches wrong --k / ckpt mixup)
    leaf = jax.tree_util.tree_leaves(params["jdyn"])
    in_widths = {l.shape[0] for l in leaf if hasattr(l, "ndim") and l.ndim == 2}
    expect = latent_dim + k * act_dim
    assert expect in in_widths, (
        f"{ckpt_path}: no jdyn kernel with input width {expect} (latent {latent_dim} + "
        f"{k}*act {act_dim}); 2D widths found: {sorted(in_widths)} — wrong k for this ckpt?")

    enc = jax.jit(lambda obs: enc_net.apply(params["enc"], obs))

    @jax.jit
    def act_of(z):  # deterministic policy action tanh(mu)
        mu, _ = pi_net.apply(params["pi"], z)
        return jnp.tanh(mu)

    jdyn = jax.jit(lambda z, a_cat: jumpy_net.apply(params["jdyn"], z, a_cat))
    return enc, act_of, jdyn


def batched1(fn, X, bs=2048):
    return np.concatenate(
        [np.asarray(fn(jnp.asarray(X[i:i + bs]))) for i in range(0, len(X), bs)], 0)


def batched2(fn, X, Y, bs=2048):
    return np.concatenate(
        [np.asarray(fn(jnp.asarray(X[i:i + bs]), jnp.asarray(Y[i:i + bs])))
         for i in range(0, len(X), bs)], 0)


def l2(a, b):
    return np.sqrt(((a - b) ** 2).sum(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_k4", required=True, help="phasei27/i30 jumpy k=4 best_mppi.pkl")
    ap.add_argument("--ckpt_k8", required=True, help="phasei30_jumk8 k=8 best_mppi.pkl (same task)")
    ap.add_argument("--task", default="PandaPickCubeOrientation")
    ap.add_argument("--n_ep", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episode_length", type=int, default=1000)
    # arch hyper-params (defaults match phasei27_jum / phasei30_jumk8 queue configs: mlp, 512, V8)
    ap.add_argument("--k4", type=int, default=4)
    ap.add_argument("--k8", type=int, default=8)
    ap.add_argument("--latent_dim", type=int, default=512)
    ap.add_argument("--V", type=int, default=8)
    ap.add_argument("--dyn_arch_k4", default="mlp", choices=["mlp", "attn", "resmlp"])
    ap.add_argument("--dyn_arch_k8", default="mlp", choices=["mlp", "attn", "resmlp"])
    # pre-registered gates
    ap.add_argument("--gate_rho", type=float, default=0.9)
    ap.add_argument("--gate_winrate", type=float, default=0.6)
    args = ap.parse_args()
    hidden = (512, 512)
    k4, k8 = int(args.k4), int(args.k8)
    assert 2 * k4 == k8, f"design assumes two k4 jumps == one k8 jump (got k4={k4}, k8={k8})"
    H = k8  # 8-step horizon

    # ── env (mirror p3_macroq_decomp: registry.load -> wrap_for_brax_training, split-1)
    env = registry.load(args.task)
    env = wrapper.wrap_for_brax_training(env, episode_length=args.episode_length, action_repeat=1)
    obs_dim, act_dim = env.observation_size, env.action_size
    reset = jax.jit(lambda kk: env.reset(jax.random.split(kk, 1)))
    step = jax.jit(lambda st, a: env.step(st, a[None]))

    # ── both models (each evaluated against its OWN encoder's targets)
    enc4, act4, jdyn4 = build_model(
        args.ckpt_k4, k4, act_dim, args.latent_dim, hidden, args.V, args.dyn_arch_k4)
    enc8, _, jdyn8 = build_model(
        args.ckpt_k8, k8, act_dim, args.latent_dim, hidden, args.V, args.dyn_arch_k8)

    # ── ONE shared behavior stream: roll the K4 ckpt's deterministic policy, record raw
    #    (obs, action) sequences; both models are then evaluated offline on identical tuples.
    key = jax.random.PRNGKey(args.seed)
    Ot, O4, O8, Acat = [], [], [], []   # o_t, o_{t+4}, o_{t+8}, a_{t:t+8} concat (8*act,)
    ep_lens = []
    for _ep in range(args.n_ep):
        key, rk = jax.random.split(key)
        st = reset(rk)
        obs = st.obs[0]
        oseq, aseq = [], []
        for _t in range(args.episode_length):
            a = act4(enc4(obs[None]))[0]
            oseq.append(np.asarray(obs, np.float32).reshape(-1))
            aseq.append(np.asarray(a, np.float32).reshape(-1))
            st = step(st, a)
            if bool(st.done[0] > 0.5):
                break
            obs = st.obs[0]
        oseq = np.stack(oseq)                                  # (T,obs)
        aseq = np.stack(aseq)                                  # (T,act)
        T = len(oseq)
        ep_lens.append(T)
        for t in range(T - H):                                 # o_{t+8} exists (t+H <= T-1)
            Ot.append(oseq[t])
            O4.append(oseq[t + k4])
            O8.append(oseq[t + H])
            Acat.append(aseq[t:t + H].reshape(-1))             # (8*act,)
    assert Ot, f"no (t, t+{H}) tuples collected — episodes shorter than {H}?"
    Ot, O4, O8, Acat = np.stack(Ot), np.stack(O4), np.stack(O8), np.stack(Acat)
    N = Ot.shape[0]
    A03, A47 = Acat[:, :k4 * act_dim], Acat[:, k4 * act_dim:]  # first / second 4-action block

    # ── offline eval, k4 model in ITS latent space
    Z4t, Z4_4, Z4_8 = batched1(enc4, Ot), batched1(enc4, O4), batched1(enc4, O8)
    Z4mid = batched2(jdyn4, Z4t, A03)                          # d4(z_t, a_{0:4})
    Z4hat = batched2(jdyn4, Z4mid, A47)                        # d4(d4(...), a_{4:8})
    err4comp = l2(Z4hat, Z4_8)                                 # composed 8-step error
    err4_leg1 = l2(Z4mid, Z4_4)                                # 4-step error, leg 1 (reference)
    err4_leg2 = l2(batched2(jdyn4, Z4_4, A47), Z4_8)           # 4-step error from TRUE midpoint
    disp4 = l2(Z4_8, Z4t)                                      # natural 8-step scale, k4 latent
    disp4_4 = l2(Z4_4, Z4t)                                    # 4-step scale (for leg refs)

    # ── offline eval, k8 model in ITS latent space (same tuples)
    Z8t, Z8_8 = batched1(enc8, Ot), batched1(enc8, O8)
    err8 = l2(batched2(jdyn8, Z8t, Acat), Z8_8)                # monolithic 8-step error
    disp8 = l2(Z8_8, Z8t)                                      # natural 8-step scale, k8 latent
    # NOTE: d8 has no 4-step mode (fixed 8*act_dim input) — no 4-step error for the k8 model.

    # ── normalize by each latent's own median 8-step displacement
    eps = 1e-9
    scale4 = float(np.median(disp4)) + eps
    scale8 = float(np.median(disp8)) + eps
    e4n, e8n = err4comp / scale4, err8 / scale8

    def compare(mask, label):
        a, b = e4n[mask], e8n[mask]
        return {
            "subset": label, "n": int(mask.sum()),
            "err4comp_norm": stats(a), "err8_norm": stats(b),
            "rho_median_ratio": round(float(np.median(a) / max(np.median(b), eps)), 4),
            "mean_ratio": round(float(np.mean(a) / max(np.mean(b), eps)), 4),
            "win_rate_composed_lt_monolithic": round(float((a < b).mean()), 4),
        }

    all_mask = np.ones(N, bool)
    overall = compare(all_mask, "all_states")
    rho = overall["rho_median_ratio"]
    win = overall["win_rate_composed_lt_monolithic"]

    # top-quartile 8-step-displacement states (fast/contact — matter most for planning);
    # subset defined in each latent space separately + rank agreement between the two
    top4 = disp4 >= np.quantile(disp4, 0.75)
    top8 = disp8 >= np.quantile(disp8, 0.75)
    disp_rank_agreement = spearman(disp4, disp8)

    go = (rho <= args.gate_rho) and (win >= args.gate_winrate)
    verdict = "GO" if go else "NO-GO"

    out = {
        "probe": "pyramid_mechcheck",
        "question": ("at 8-step latent prediction, is composing two 4-step jumps "
                     "(d4∘d4, k4 ckpt) more accurate than one monolithic 8-step jump "
                     "(d8, k8 ckpt), each judged in its own latent space, normalized by "
                     "its own median 8-step displacement?"),
        "config": {
            "ckpt_k4": args.ckpt_k4, "ckpt_k8": args.ckpt_k8, "task": args.task,
            "k4": k4, "k8": k8, "n_ep": args.n_ep, "seed": args.seed,
            "episode_length": args.episode_length, "latent_dim": args.latent_dim,
            "V": args.V, "dyn_arch_k4": args.dyn_arch_k4, "dyn_arch_k8": args.dyn_arch_k8,
            "obs_dim": int(obs_dim), "act_dim": int(act_dim),
            "behavior_policy": "k4 ckpt deterministic tanh(mu); both models evaluated "
                               "offline on the identical recorded (o_t, a_{t:t+8}, o_{t+8})",
            "gates": {"rho_max": args.gate_rho, "win_rate_min": args.gate_winrate},
        },
        "n_tuples": int(N), "ep_lens": ep_lens,
        "normalization": {
            "rule": "err / median ||enc(o_{t+8}) - enc(o_t)|| in the model's OWN latent space",
            "scale4_median_8step_disp_k4latent": round(scale4, 5),
            "scale8_median_8step_disp_k8latent": round(scale8, 5),
            "disp4_stats": stats(disp4), "disp8_stats": stats(disp8),
            "disp_rank_agreement_spearman_k4_vs_k8_latent": round(disp_rank_agreement, 4),
        },
        "unnormalized": {
            "err4comp_composed_8step_k4latent": stats(err4comp),
            "err8_monolithic_8step_k8latent": stats(err8),
            "err4_leg1_4step_from_z_t_k4latent": stats(err4_leg1),
            "err4_leg2_4step_from_true_midpoint_k4latent": stats(err4_leg2),
            "err4_leg1_norm_by_median_4step_disp": stats(
                err4_leg1 / (float(np.median(disp4_4)) + eps)),
            "k8_model_4step_error": ("not computable: d8 takes a fixed 8*act_dim action "
                                     "block, it has no 4-step mode"),
        },
        "comparison_overall": overall,
        "comparison_top_quartile_8step_displacement": {
            "by_k4_latent_disp": compare(top4, "top25pct_disp_k4latent"),
            "by_k8_latent_disp": compare(top8, "top25pct_disp_k8latent"),
        },
        "rho": rho, "win_rate": win,
        "verdict": verdict,
        "verdict_rule": (
            f"GO iff rho <= {args.gate_rho} (composition >= "
            f"{round((1 - args.gate_rho) * 100)}% more accurate at the median) AND "
            f"win_rate >= {args.gate_winrate}; NO-GO otherwise — pyramid idea dies free"),
        "training_distribution_caveat": (
            "ASYMMETRIC test: the two ckpts were trained with DIFFERENT k losses. d4∘d4 was "
            "NEVER trained for 8-step consistency (composition compounds two un-coordinated "
            "4-step heads), while d8 was trained exactly on the 8-step objective. If "
            "composition wins anyway, a pyramid trained WITH cross-level consistency should "
            "do even better -> strong GO. If composition loses, cross-level training could "
            "still rescue it -> the NO-GO is softer than the gate suggests (it kills "
            "'pyramid wins for free', not 'pyramid can never work')."),
        "caveats": [
            "cross-latent-space comparison: each model judged in its own encoder's geometry; "
            "median-8-step-displacement normalization makes scales comparable but cannot "
            "remove all geometry differences (both are SimNorm V-simplex latents, which helps)",
            "encoders trained separately -> targets enc4(o_{t+8}) and enc8(o_{t+8}) differ; "
            "neither is ground truth, each is only self-consistent",
            "behavior stream from the k4 ckpt's deterministic policy: on-distribution for the "
            "k4 model, slightly off-distribution for the k8 model (shared stream was required "
            "for identical tuples; favors k4 — counted against GO honesty when rho is near gate)",
            "single ckpt pair / single seed — no across-seed variance estimate",
            "pi-policy distribution only; MPPI's planning distribution may stress the heads "
            "differently",
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
