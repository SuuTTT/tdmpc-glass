#!/usr/bin/env python3
"""Synthetic mechanism-check GATE: train an entity-factored WM on the synthetic
multi-entity world and run the value-decodability probe IN-DIST and at OOD
object counts. Persist ALL results to JSON (read-from-JSON discipline).

This is INFRASTRUCTURE. The scientific *value-coupling graph* probe is added by
the main session; here we run probe (i) only: LINEAR value-decodability R² of
the return from the flat entity-concat latent (the WM's per-entity tokens),
measured at N_train and at OOD N (6, 8).

Runs on a GPU worker (needs jax+flax+optax). NOT runnable on the control box.

Output:
    exp/synthetic_gate/<tag>.json   (config + per-N R² + train metrics)

CLI / env (all optional):
    --n_train 4        (or env SG_N_TRAIN)
    --n_ood 6,8        (or env SG_N_OOD, comma-separated)
    --steps 2000       (or env SG_STEPS)   train steps
    --episodes 256     (or env SG_EPISODES) episodes collected per N
    --ep_len 100       (or env SG_EP_LEN)  steps per episode
    --seed 0           (or env SG_SEED)
    --d_model 64 --n_layers 2 --n_heads 4
    --tag synthetic_gate   output json basename

Example (on a worker, repo synced to /root/helios-rl):
    PYTHONPATH=/root/helios-rl/src python3 \\
        /root/helios-rl/scripts/run_synthetic_gate.py \\
        --n_train 4 --n_ood 6,8 --steps 2000 --episodes 256 --seed 0
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
from helios.envs import synthetic_entities as se

EXP_DIR = Path(__file__).resolve().parents[1] / "exp" / "synthetic_gate"


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def collect_dataset(env, key, n_episodes, ep_len, gamma=0.99):
    """Collect (ent, action, next_ent, reward, return) over many episodes.

    Returns a dict of numpy/jax arrays flattened over (episodes * ep_len):
        ent:     (M, N, d)
        action:  (M, 2)
        next_ent:(M, N, d)
        reward:  (M,)
        ret:     (M,)  discounted return-to-go (Monte-Carlo) — used both as the
                 Q target AND as the value-decodability regression target.
    """
    roll = jax.jit(lambda k: se.rollout(env, k, ep_len))
    discounts = (gamma ** jnp.arange(ep_len))

    def returns_to_go(rewards):  # rewards (ep_len,)
        # G_t = sum_{u>=t} gamma^{u-t} r_u  (computed via reverse cumsum)
        def body(carry, r):
            g = r + gamma * carry
            return g, g
        _, g_rev = jax.lax.scan(body, 0.0, rewards[::-1])
        return g_rev[::-1]

    rtg = jax.jit(returns_to_go)

    ents, acts, nents, rews, rets = [], [], [], [], []
    keys = jax.random.split(key, n_episodes)
    for k in keys:
        traj = roll(k)
        next_ent = jax.vmap(lambda o: o.reshape(env.n_entities, se.ENTITY_DIM))(
            traj["next_obs"]
        )
        g = rtg(traj["reward"])
        ents.append(traj["ent"])
        acts.append(traj["action"])
        nents.append(next_ent)
        rews.append(traj["reward"])
        rets.append(g)

    cat = lambda xs: jnp.concatenate(xs, axis=0)
    return {
        "ent": cat(ents),
        "action": cat(acts),
        "next_ent": cat(nents),
        "reward": cat(rews),
        "ret": cat(rets),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def make_model(args, n_entities):
    return EntityWM(
        entity_dim=se.ENTITY_DIM,
        action_dim=2,
        n_entities=n_entities,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_entities=max(64, max([n_entities] + args.n_ood) + 1),
    )


def train(args, env, data, key):
    model = make_model(args, env.n_entities)
    k_init, key = jax.random.split(key)
    dummy_ent = data["ent"][:2]
    dummy_act = data["action"][:2]
    params = model.init(k_init, dummy_ent, dummy_act)["params"]

    opt = optax.adam(args.lr)
    opt_state = opt.init(params)

    M = data["ent"].shape[0]

    @jax.jit
    def train_step(params, opt_state, batch):
        (loss, metrics), grads = jax.value_and_grad(entity_wm_loss, has_aux=True)(
            params, model.apply, batch
        )
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, metrics

    last = {}
    for step in range(args.steps):
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
    return model, params, last


# ---------------------------------------------------------------------------
# Probe (i): linear value-decodability R²
# ---------------------------------------------------------------------------


def linear_r2(features, targets):
    """Closed-form ridge-free linear regression R² (numpy).

    features: (M, F), targets: (M,). Adds a bias column. Returns R^2.
    """
    f = np.asarray(features)
    y = np.asarray(targets)
    X = np.concatenate([f, np.ones((f.shape[0], 1))], axis=1)
    # least squares
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ w
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def latent_features(model, params, ent, action):
    """Flat entity-concat latent = final per-entity tokens flattened (B, N*D)."""
    out = model.apply({"params": params}, ent, action, return_attn=True)
    tokens = out["tokens"][-1]              # (B, N, D) final layer
    B = tokens.shape[0]
    return np.asarray(tokens.reshape(B, -1))


def value_decodability(model, params, env, key, n_episodes, ep_len):
    """Collect fresh data at this N, extract latent, regress return -> R²."""
    data = collect_dataset(env, key, n_episodes, ep_len)
    feats = latent_features(model, params, data["ent"], data["action"])
    r2 = linear_r2(feats, data["ret"])
    # Baseline: R² from raw flat observation (sanity reference).
    raw = np.asarray(
        jnp.concatenate(
            [data["ent"].reshape(data["ent"].shape[0], -1), data["action"]], axis=-1
        )
    )
    r2_raw = linear_r2(raw, data["ret"])
    return {"r2_latent": r2, "r2_raw_obs": r2_raw, "n_samples": int(feats.shape[0])}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    def envdef(name, default):
        return os.environ.get(name, default)

    p = argparse.ArgumentParser(description="Synthetic mechanism-check gate.")
    p.add_argument("--n_train", type=int, default=int(envdef("SG_N_TRAIN", 4)))
    p.add_argument("--n_ood", type=str, default=envdef("SG_N_OOD", "6,8"))
    p.add_argument("--steps", type=int, default=int(envdef("SG_STEPS", 2000)))
    p.add_argument("--episodes", type=int, default=int(envdef("SG_EPISODES", 256)))
    p.add_argument("--ep_len", type=int, default=int(envdef("SG_EP_LEN", 100)))
    p.add_argument("--seed", type=int, default=int(envdef("SG_SEED", 0)))
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--tag", type=str, default=envdef("SG_TAG", "synthetic_gate"))
    args = p.parse_args()
    args.n_ood = [int(x) for x in str(args.n_ood).split(",") if x.strip()]
    return args


def main():
    args = parse_args()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    key = jax.random.PRNGKey(args.seed)

    # --- train at N_train ---
    env_train = se.make_env(n_entities=args.n_train, seed=args.seed)
    k_data, k_train, key = jax.random.split(key, 3)
    data = collect_dataset(env_train, k_data, args.episodes, args.ep_len)
    model, params, train_metrics = train(args, env_train, data, k_train)

    # --- probe at N_train and OOD N ---
    results = {}
    for n in [args.n_train] + args.n_ood:
        env_n = se.make_env(n_entities=n, seed=args.seed)
        k_probe, key = jax.random.split(key)
        res = value_decodability(
            model, params, env_n, k_probe, args.episodes, args.ep_len
        )
        res["value_relevant_entities"] = list(env_n.value_relevant_entities)
        res["value_pair"] = list(env_n.value_pair)
        res["coupling_nnz"] = int(np.count_nonzero(np.asarray(env_n.coupling)))
        results[str(n)] = res

    out = {
        "tag": args.tag,
        "config": {
            "n_train": args.n_train,
            "n_ood": args.n_ood,
            "steps": args.steps,
            "episodes": args.episodes,
            "ep_len": args.ep_len,
            "seed": args.seed,
            "batch": args.batch,
            "lr": args.lr,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
        },
        "train_metrics": train_metrics,
        "probe_value_decodability": results,
        "wall_time_sec": time.time() - t0,
    }

    out_path = EXP_DIR / f"{args.tag}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[run_synthetic_gate] wrote {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
