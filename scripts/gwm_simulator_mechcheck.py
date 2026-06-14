#!/usr/bin/env python3
"""GWM-as-SIMULATOR mechanism check (the decisive one).

============================================================================
THE QUESTION
============================================================================
"Graph World Model as Simulator" (GWM survey, arXiv 2604.27895) claims a
graph/entity-factored latent beats a monolithic latent on the two things that
actually matter for a learned simulator:

  (a) COMPOSITIONAL-OOD VALUE-DECODABILITY: can you linearly read the return out
      of the latent when the object count is OUT of the training distribution?
  (b) CONTACT-CONDITIONED PREDICTION: is next-state prediction better AT the
      relational/contact events (the timesteps where the structure bites)?

This script tests both on a CONTACT-RICH controlled world
(``helios.envs.contact_entities``: elastic disk collisions + walls, sparse
push-to-goal reward, ground-truth per-step contact graph). It trains BOTH models
on the SAME data at N_train:
  * GRAPH WM   = ``helios.dynamics.entity_wm.EntityWM`` (transformer over entities)
  * MONOLITHIC = ``helios.dynamics.monolithic_wm.MonolithicWM`` (flat MLP),
    param-count matched to the graph WM at N_train.
and measures + persists to JSON:

  (i)   value-decodability linear R² of return from each latent, in-dist AND at
        held-out OOD object counts (the survey's headroom regime).
  (ii)  next-state self-pred error SPLIT by contact vs non-contact timesteps, per
        model (in-dist and OOD).
  (iii) overall next-state self-pred error + reward R², both models, in/OOD.

============================================================================
PRE-REGISTERED VERDICT (stated before running)
============================================================================
GWM-as-simulator GO iff EITHER:
  (A) monolithic OOD value-R² drops >= 0.15 BELOW graph's OOD value-R²
      (averaged over the OOD N), OR
  (B) graph contact-step self-pred error <= 0.8x monolithic's contact-step error
      (at N_train).
The margin is stated in the verdict block. NO-GO otherwise -> the graph latent is
redundant on contacts too (consistent with this campaign's prior nulls).

HONEST CAVEATS (persisted in the JSON):
  * synthetic disks are NOT real manipulation; this is a controlled proxy.
  * single seed (--seed); no multi-seed error bars.
  * param-match is APPROXIMATE (monolithic flat layers grow with N; matched at
    N_train only) — exact counts are reported.

Runs on a GPU/CPU worker with jax+flax+optax. NOT on the control box (no GPU).

CLI:
    --n_train 5 --n_ood 7,9 --steps 6000 --episodes 256 --seed 0 \
    --out exp/tdmpc_glass/mechcheck/gwm_simulator.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helios.dynamics.entity_wm import EntityWM, entity_wm_loss
from helios.dynamics.monolithic_wm import (
    MonolithicWM,
    matched_hidden,
    monolithic_wm_loss,
    _count_params,
)
from helios.envs import contact_entities as ce

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = "exp/tdmpc_glass/mechcheck/gwm_simulator.json"


# ---------------------------------------------------------------------------
# Data collection (mirrors run_synthetic_gate, plus the contact graph)
# ---------------------------------------------------------------------------


def collect_dataset(env, key, n_episodes, ep_len, gamma=0.99):
    """Collect transitions over many episodes from the contact env.

    Returns a dict of jax arrays flattened over (episodes * ep_len):
        ent:        (M, N, d) current per-entity state
        action:     (M, 2)
        next_ent:   (M, N, d) ground-truth next per-entity state
        reward:     (M,)
        ret:        (M,) discounted return-to-go (Monte-Carlo) — Q target + the
                    value-decodability regression target.
        any_contact:(M,) 0/1 whether any disk-disk contact occurred that step
    """
    roll = jax.jit(lambda k: ce.rollout(env, k, ep_len))

    def returns_to_go(rewards):
        def body(carry, r):
            g = r + gamma * carry
            return g, g
        _, g_rev = jax.lax.scan(body, 0.0, rewards[::-1])
        return g_rev[::-1]

    rtg = jax.jit(returns_to_go)

    ents, acts, nents, rews, rets, contacts = [], [], [], [], [], []
    keys = jax.random.split(key, n_episodes)
    for k in keys:
        traj = roll(k)
        next_ent = jax.vmap(
            lambda o: o.reshape(env.n_entities, ce.ENTITY_DIM)
        )(traj["next_obs"])
        g = rtg(traj["reward"])
        ents.append(traj["ent"])
        acts.append(traj["action"])
        nents.append(next_ent)
        rews.append(traj["reward"])
        rets.append(g)
        contacts.append(traj["any_contact"])

    cat = lambda xs: jnp.concatenate(xs, axis=0)
    return {
        "ent": cat(ents),
        "action": cat(acts),
        "next_ent": cat(nents),
        "reward": cat(rews),
        "ret": cat(rets),
        "any_contact": cat(contacts),
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def make_graph_model(args, n_entities):
    return EntityWM(
        entity_dim=ce.ENTITY_DIM,
        action_dim=2,
        n_entities=n_entities,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_entities=max(64, max([n_entities] + args.n_ood) + 1),
    )


def mono_max_entities(args):
    """Fixed slot count for the 'pad' monolithic baseline. Matches the graph
    model's max_entities so both support the same OOD N range."""
    return max(64, max([args.n_train] + args.n_ood) + 1)


def make_mono_model(args, n_entities, hidden):
    return MonolithicWM(
        entity_dim=ce.ENTITY_DIM,
        action_dim=2,
        n_entities=n_entities,
        hidden=hidden,
        n_layers=args.mono_layers,
        mode=args.mono_mode,
        max_entities=mono_max_entities(args),
    )


def train_model(model, loss_fn, data, key, args):
    """Generic trainer: returns (params, last_metrics, n_params)."""
    k_init, key = jax.random.split(key)
    params = model.init(k_init, data["ent"][:2], data["action"][:2])["params"]
    n_params = _count_params(params)

    opt = optax.adam(args.lr)
    opt_state = opt.init(params)
    M = data["ent"].shape[0]

    @jax.jit
    def train_step(params, opt_state, batch):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, model.apply, batch
        )
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, metrics

    last = {}
    for _ in range(args.steps):
        key, bk = jax.random.split(key)
        idx = jax.random.randint(bk, (args.batch,), 0, M)
        batch = {
            "ent": data["ent"][idx],
            "action": data["action"][idx],
            "next_ent": data["next_ent"][idx],
            "reward": data["reward"][idx],
            "q_target": data["ret"][idx],
        }
        params, opt_state, metrics = train_step(params, opt_state, batch)
        last = {k: float(v) for k, v in metrics.items()}
    return params, last, n_params


# ---------------------------------------------------------------------------
# Latent extraction (the value-decodability feature) for each model
# ---------------------------------------------------------------------------


def graph_latent(model, params, ent, action):
    """Flat entity-concat latent = final per-entity tokens flattened (B, N*D)."""
    out = model.apply({"params": params}, ent, action, return_attn=True)
    tokens = out["tokens"][-1]              # (B, N, D) final layer
    B = tokens.shape[0]
    return np.asarray(tokens.reshape(B, -1))


def mono_latent(model, params, ent, action):
    """Monolithic latent (B, hidden) from the flat trunk."""
    out = model.apply({"params": params}, ent, action, return_attn=True)
    tok = out["tokens"]                     # (1, B, 1, hidden)
    B = tok.shape[1]
    return np.asarray(tok.reshape(B, -1))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def linear_r2(features, targets):
    """Closed-form linear regression R² (numpy). features (M,F), targets (M,)."""
    f = np.asarray(features)
    y = np.asarray(targets)
    X = np.concatenate([f, np.ones((f.shape[0], 1))], axis=1)
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ w
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def self_pred_errors(model, params, data):
    """Per-sample next-state MSE (mean over entities*dims). Returns (M,) numpy."""
    out = model.apply({"params": params}, data["ent"], data["action"])
    err = jnp.mean((out["next_ent"] - data["next_ent"]) ** 2, axis=(1, 2))  # (M,)
    return np.asarray(err)


def reward_r2(model, params, data):
    out = model.apply({"params": params}, data["ent"], data["action"])
    return linear_r2(np.asarray(out["reward"]).reshape(-1, 1), np.asarray(data["reward"]))


def split_contact(err, any_contact):
    """Split a per-sample error vector by contact / non-contact timesteps."""
    ac = np.asarray(any_contact).astype(bool)
    contact_err = float(np.mean(err[ac])) if ac.any() else float("nan")
    noncontact_err = float(np.mean(err[~ac])) if (~ac).any() else float("nan")
    return {
        "contact_step_error": contact_err,
        "noncontact_step_error": noncontact_err,
        "n_contact_steps": int(ac.sum()),
        "n_noncontact_steps": int((~ac).sum()),
        "frac_contact_steps": float(ac.mean()),
        "overall_error": float(np.mean(err)),
    }


# ---------------------------------------------------------------------------
# Per-N evaluation of both models on fresh data
# ---------------------------------------------------------------------------


def eval_at_n(graph_model, graph_params, mono_model, mono_params, env, key,
              n_episodes, ep_len):
    data = collect_dataset(env, key, n_episodes, ep_len)

    # (i) value-decodability R²
    g_feat = graph_latent(graph_model, graph_params, data["ent"], data["action"])
    m_feat = mono_latent(mono_model, mono_params, data["ent"], data["action"])
    g_vr2 = linear_r2(g_feat, data["ret"])
    m_vr2 = linear_r2(m_feat, data["ret"])
    raw = np.asarray(
        jnp.concatenate(
            [data["ent"].reshape(data["ent"].shape[0], -1), data["action"]], axis=-1
        )
    )
    raw_vr2 = linear_r2(raw, data["ret"])

    # (ii) contact-conditioned self-pred error
    g_err = self_pred_errors(graph_model, graph_params, data)
    m_err = self_pred_errors(mono_model, mono_params, data)
    g_split = split_contact(g_err, data["any_contact"])
    m_split = split_contact(m_err, data["any_contact"])

    # (iii) reward R²
    g_rr2 = reward_r2(graph_model, graph_params, data)
    m_rr2 = reward_r2(mono_model, mono_params, data)

    return {
        "n_samples": int(data["ent"].shape[0]),
        "frac_contact_steps": g_split["frac_contact_steps"],
        "value_decodability_r2": {
            "graph": g_vr2,
            "monolithic": m_vr2,
            "raw_obs": raw_vr2,
        },
        "contact_conditioned_selfpred": {
            "graph": g_split,
            "monolithic": m_split,
        },
        "overall_selfpred_error": {
            "graph": g_split["overall_error"],
            "monolithic": m_split["overall_error"],
        },
        "reward_r2": {"graph": g_rr2, "monolithic": m_rr2},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="GWM-as-simulator mechanism check.")
    p.add_argument("--n_train", type=int, default=5)
    p.add_argument("--n_ood", type=str, default="7,9")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--ep_len", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--mono_layers", type=int, default=2,
                   help="trunk layers for the monolithic model")
    p.add_argument("--mono_mode", type=str, default="pool",
                   choices=["pool", "pad"],
                   help="monolithic encoder input: 'pool' (mean+max, N-invariant "
                        "width, default) or 'pad' (zero-pad to max_entities, "
                        "fixed per-entity slots — the fairness control)")
    p.add_argument("--out", type=str, default=DEFAULT_OUT)
    args = p.parse_args()
    args.n_ood = [int(x) for x in str(args.n_ood).split(",") if x.strip()]
    # If a non-default mono_mode is requested but the user kept the default --out,
    # tag the filename with the mode suffix so pad/pool runs don't clobber each
    # other. An explicit --out is respected as-is.
    if args.mono_mode != "pool" and args.out == DEFAULT_OUT:
        stem, ext = os.path.splitext(args.out)
        args.out = f"{stem}_{args.mono_mode}{ext}"
    return args


def main():
    args = parse_args()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    key = jax.random.PRNGKey(args.seed)

    # --- collect training data at N_train ---
    env_train = ce.make_env(n_entities=args.n_train, seed=args.seed)
    k_data, k_gtrain, k_mtrain, key = jax.random.split(key, 4)
    data = collect_dataset(env_train, k_data, args.episodes, args.ep_len)

    # Standardize the Q/return target (so the Q head is well-conditioned; raw MC
    # returns produce a useless Q surface — same fix as value_coupling_probe).
    ret = np.asarray(data["ret"])
    ret_mean, ret_std = float(ret.mean()), float(ret.std() + 1e-8)
    data = dict(data)
    data["ret"] = (data["ret"] - ret_mean) / ret_std

    # --- build + train the GRAPH model ---
    graph_model = make_graph_model(args, args.n_train)
    graph_params, g_metrics, g_nparams = train_model(
        graph_model, entity_wm_loss, data, k_gtrain, args
    )

    # --- size the MONOLITHIC model to match graph param count at N_train ---
    hidden, mono_target_nparams = matched_hidden(
        g_nparams, ce.ENTITY_DIM, 2, args.n_train, n_layers=args.mono_layers,
        mode=args.mono_mode, max_entities=mono_max_entities(args),
    )
    mono_model = make_mono_model(args, args.n_train, hidden)
    mono_params, m_metrics, m_nparams = train_model(
        mono_model, monolithic_wm_loss, data, k_mtrain, args
    )

    # --- evaluate both models at N_train and OOD N on fresh data ---
    per_n = {}
    for n in [args.n_train] + args.n_ood:
        env_n = ce.make_env(n_entities=n, seed=args.seed)
        k_eval, key = jax.random.split(key)
        per_n[str(n)] = eval_at_n(
            graph_model, graph_params, mono_model, mono_params, env_n, k_eval,
            args.episodes, args.ep_len,
        )
        per_n[str(n)]["value_relevant_entities"] = list(env_n.value_relevant_entities)

    # --- pre-registered verdict ---
    # (A) monolithic OOD value-R² drops >= 0.15 below graph's (avg over OOD N).
    g_ood = np.mean([per_n[str(n)]["value_decodability_r2"]["graph"] for n in args.n_ood])
    m_ood = np.mean([per_n[str(n)]["value_decodability_r2"]["monolithic"] for n in args.n_ood])
    value_gap = float(g_ood - m_ood)            # positive => monolithic worse
    crit_value = bool(value_gap >= 0.15)

    # (B) graph contact-step error <= 0.8x monolithic's contact-step error (N_train).
    g_ce = per_n[str(args.n_train)]["contact_conditioned_selfpred"]["graph"]["contact_step_error"]
    m_ce = per_n[str(args.n_train)]["contact_conditioned_selfpred"]["monolithic"]["contact_step_error"]
    contact_ratio = float(g_ce / m_ce) if (m_ce and m_ce == m_ce) else float("nan")
    crit_contact = bool(contact_ratio == contact_ratio and contact_ratio <= 0.8)

    go = bool(crit_value or crit_contact)
    verdict = {
        "gwm_simulator_go": go,
        "criteria_preregistered": (
            "GO iff (monolithic OOD value-R2 drops >=0.15 below graph's) OR "
            "(graph contact-step error <=0.8x monolithic's). NO-GO otherwise -> "
            "graph latent redundant on contacts too."
        ),
        "criterion_A_value_decodability": {
            "graph_ood_value_r2": float(g_ood),
            "monolithic_ood_value_r2": float(m_ood),
            "graph_minus_monolithic_gap": value_gap,
            "threshold": 0.15,
            "passed": crit_value,
            "margin_over_threshold": float(value_gap - 0.15),
        },
        "criterion_B_contact_prediction": {
            "graph_contact_step_error": g_ce,
            "monolithic_contact_step_error": m_ce,
            "ratio_graph_over_mono": contact_ratio,
            "threshold": 0.8,
            "passed": crit_contact,
            "margin_under_threshold": float(0.8 - contact_ratio)
            if contact_ratio == contact_ratio else float("nan"),
        },
        "interpretation": (
            "GO: entity/graph latent beats the param-matched monolithic control "
            "on compositional-OOD value-decodability and/or contact prediction "
            "-> the GWM-as-simulator bet has mechanism support. NO-GO: graph "
            "latent is redundant on contacts too -> consistent with the campaign."
        ),
    }

    caveats = [
        "Synthetic elastic disks are NOT real manipulation; controlled proxy only.",
        f"Single seed (seed={args.seed}); no multi-seed error bars.",
        "Param-match is APPROXIMATE: monolithic flat layers grow with N, matched "
        "at N_train only; exact counts reported in param_counts.",
    ]

    out = {
        "tag": "gwm_simulator_mechcheck",
        "config": vars(args),
        "ret_standardization": {"mean": ret_mean, "std": ret_std},
        "param_counts": {
            "graph_entity_wm": g_nparams,
            "monolithic_wm": m_nparams,
            "monolithic_hidden": hidden,
            "monolithic_target_at_init": mono_target_nparams,
            "match_ratio_mono_over_graph": float(m_nparams / g_nparams),
        },
        "train_metrics": {"graph": g_metrics, "monolithic": m_metrics},
        "per_n": per_n,
        "verdict": verdict,
        "caveats": caveats,
        "wall_time_sec": time.time() - t0,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[gwm_simulator_mechcheck] wrote {out_path}")
    print(json.dumps(verdict, indent=2))
    print(json.dumps(out["param_counts"], indent=2))
    for n in [args.n_train] + args.n_ood:
        r = per_n[str(n)]
        print(
            f"N={n}: value-R2 graph={r['value_decodability_r2']['graph']:.3f} "
            f"mono={r['value_decodability_r2']['monolithic']:.3f} | "
            f"contact-err graph={r['contact_conditioned_selfpred']['graph']['contact_step_error']:.4f} "
            f"mono={r['contact_conditioned_selfpred']['monolithic']['contact_step_error']:.4f}"
        )


if __name__ == "__main__":
    main()
