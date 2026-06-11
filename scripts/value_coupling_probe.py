#!/usr/bin/env python3
"""B1 — the VALUE-COUPLING GRAPH probe (the heart of the VG-SE bet).

============================================================================
WHAT THIS TESTS (instrument validation, on the synthetic world)
============================================================================
The VG-SE proposal's core move is to build the abstraction graph from
VALUE-COUPLING rather than SIMILARITY. Before betting on a real multi-object
env, we must show the instrument WORKS: on a world whose value-coupling is known
by construction (helios.envs.synthetic_entities), can a value-coupling graph
*recover the known structure*, and does it differ from (beat) a similarity graph?

Ground truth of the synthetic world: reward = -dist(agent=0, goal=1)
- w_pair * dist(p=2, q=3). So the value depends JOINTLY (non-separably) on the
entity pairs {(0,1), (2,3)} and on NO other pair. Every other entity is a
distractor whose state the reward never reads.

We define the value-coupling between entities i,j as how much the value depends
*jointly* on their states — the cross-Hessian block norm

    w_value[i,j] = E_states || d^2 f / d s_i d s_j ||_F ,   f in {reward, Q}

which is exactly zero for an additively-separable f and nonzero only where f
genuinely couples i and j. A correct instrument yields w_value hot at (0,1) and
(2,3), cold everywhere else.

We compare against the SIMILARITY graph (cosine of entity tokens), which is what
every prior SE-on-WM attempt (and our own 16 nulls) implicitly used. The bet is
that w_value recovers the task-relevant pairs and the similarity graph does not.

============================================================================
VERDICT (instrument-validation GO/NO-GO for the synthetic stage)
============================================================================
GO (instrument works -> earn the right to build the real-env experiment) iff:
  (1) w_value (reward head) recovers {(0,1),(2,3)}: average-precision AP_value
      well above chance AND above a degree-preserving shuffle null; AND
  (2) value-gating adds signal over similarity: AP_value > AP_similarity by a
      margin (else the similarity graph already carries it -> value-gating is
      redundant, consistent with the redundancy criterion); AND
  (3) the structure persists OOD (probe at held-out object counts N=6,8).
NO-GO otherwise -> the value-coupling instrument can't even recover known
structure, so VG-SE on a real env is hopeless; fold into the principle paper.

Persists everything to exp/synthetic_gate/<tag>_vcoupling.json (read-from-JSON).
Runs on a GPU/CPU worker with jax+flax+optax. NOT on the control box.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for run_synthetic_gate

from helios.envs import synthetic_entities as se
from run_synthetic_gate import collect_dataset, make_model, train  # type: ignore

EXP_DIR = Path(__file__).resolve().parents[1] / "exp" / "synthetic_gate"


# ---------------------------------------------------------------------------
# Cross-Hessian value-coupling graph
# ---------------------------------------------------------------------------


def _scalar_head(model, params, head, action_dim):
    """Return f(ent_Nd) -> scalar for head in {'reward','q'} at a single state.

    ent_Nd: (N, d). action is held at zero (the coupling structure of the reward
    is action-independent; for Q we still evaluate at a fixed reference action so
    the cross-state structure is well-defined)."""

    def f(ent_Nd, action_A):
        out = model.apply({"params": params}, ent_Nd[None], action_A[None])
        return out[head][0]

    return f


def value_coupling_graph(model, params, ent_batch, action_batch, head):
    """w[i,j] = mean over states of ||d^2 f/ d s_i d s_j||_F (i != j), f=head.

    ent_batch: (B, N, d), action_batch: (B, A). Returns (N, N) numpy, symmetric,
    zero diagonal.
    """
    B, N, d = ent_batch.shape
    A = action_batch.shape[-1]
    f = _scalar_head(model, params, head, A)

    # Hessian of a scalar wrt ent (N,d) -> (N,d,N,d). vmap over the batch.
    hess = jax.vmap(jax.hessian(lambda e, a: f(e, a), argnums=0))(
        ent_batch, action_batch
    )  # (B, N, d, N, d)
    # Frobenius norm of each (d,d) cross block -> (B, N, N)
    block_fro = jnp.sqrt(jnp.sum(hess ** 2, axis=(2, 4)) + 1e-18)
    w = jnp.mean(block_fro, axis=0)  # (N, N)
    w = 0.5 * (w + w.T)
    w = w.at[jnp.diag_indices(N)].set(0.0)
    return np.asarray(w)


def similarity_graph(model, params, ent_batch, action_batch):
    """w_sim[i,j] = mean cosine similarity of final-layer entity tokens."""
    out = model.apply(
        {"params": params}, ent_batch, action_batch, return_attn=True
    )
    tok = out["tokens"][-1]  # (B, N, D)
    tok = tok / (jnp.linalg.norm(tok, axis=-1, keepdims=True) + 1e-8)
    sim = jnp.einsum("bid,bjd->bij", tok, tok)  # (B, N, N) cosine
    w = np.asarray(jnp.mean(sim, axis=0))
    N = w.shape[0]
    w = 0.5 * (w + w.T)
    np.fill_diagonal(w, 0.0)
    return w


def attention_graph(model, params, ent_batch, action_batch):
    """w_attn[i,j] = mean over layers/heads/states of attention weight."""
    out = model.apply(
        {"params": params}, ent_batch, action_batch, return_attn=True
    )
    attn = out["attn"]  # (L, B, H, N, N)
    w = np.asarray(jnp.mean(attn, axis=(0, 1, 2)))  # (N, N)
    w = 0.5 * (w + w.T)
    np.fill_diagonal(w, 0.0)
    return w


# ---------------------------------------------------------------------------
# Recovery metrics (does the graph rank the true value pairs on top?)
# ---------------------------------------------------------------------------


def _upper_pairs(N):
    return [(i, j) for i in range(N) for j in range(i + 1, N)]


def average_precision(scores, positives):
    """AP of ranking `scores` (dict pair->float) against the positive set."""
    ranked = sorted(scores, key=lambda p: -scores[p])
    hits, ap = 0, 0.0
    for k, p in enumerate(ranked, 1):
        if p in positives:
            hits += 1
            ap += hits / k
    return ap / max(1, len(positives))


def roc_auc(scores, positives):
    pos = [scores[p] for p in scores if p in positives]
    neg = [scores[p] for p in scores if p not in positives]
    if not pos or not neg:
        return float("nan")
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def shuffle_null_ap(w, positives, N, n_shuffle=2000, seed=0):
    """Degree-preserving-ish null: permute the node labels, recompute AP.

    Permuting labels destroys which pair carries the weight while preserving the
    weight distribution exactly -> a fair 'no structure' baseline."""
    rng = np.random.default_rng(seed)
    pairs = _upper_pairs(N)
    aps = []
    for _ in range(n_shuffle):
        perm = rng.permutation(N)
        wp = w[np.ix_(perm, perm)]
        sc = {p: float(wp[p]) for p in pairs}
        aps.append(average_precision(sc, positives))
    aps = np.array(aps)
    return float(aps.mean()), float(aps.std())


def evaluate_graph(w, N, positives, tag, seed=0):
    pairs = _upper_pairs(N)
    scores = {p: float(w[p]) for p in pairs}
    ap = average_precision(scores, positives)
    auc = roc_auc(scores, positives)
    null_mean, null_std = shuffle_null_ap(w, positives, N, seed=seed)
    z = (ap - null_mean) / (null_std + 1e-9)
    ranked = sorted(scores, key=lambda p: -scores[p])
    return {
        "graph": tag,
        "ap": ap,
        "auc": auc,
        "ap_null_mean": null_mean,
        "ap_null_std": null_std,
        "ap_z_over_null": float(z),
        "chance_ap": len(positives) / len(pairs),
        "top5_pairs": [list(p) for p in ranked[:5]],
        "top5_scores": [round(scores[p], 6) for p in ranked[:5]],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    def envdef(name, default):
        return os.environ.get(name, default)

    p = argparse.ArgumentParser(description="Value-coupling graph probe (B1).")
    p.add_argument("--n_train", type=int, default=int(envdef("SG_N_TRAIN", 4)))
    p.add_argument("--n_ood", type=str, default=envdef("SG_N_OOD", "6,8"))
    p.add_argument("--steps", type=int, default=int(envdef("SG_STEPS", 4000)))
    p.add_argument("--episodes", type=int, default=int(envdef("SG_EPISODES", 256)))
    p.add_argument("--ep_len", type=int, default=int(envdef("SG_EP_LEN", 100)))
    p.add_argument("--seed", type=int, default=int(envdef("SG_SEED", 0)))
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--probe_states", type=int, default=512,
                   help="# states for the cross-Hessian average")
    p.add_argument("--tag", type=str, default=envdef("SG_TAG", "gate0"))
    args = p.parse_args()
    args.n_ood = [int(x) for x in str(args.n_ood).split(",") if x.strip()]
    return args


def main():
    args = parse_args()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    key = jax.random.PRNGKey(args.seed)

    # --- train at N_train, with a STANDARDIZED Q target (so the Q head is
    #     meaningful; raw MC returns gave q_loss~30 and a useless Q surface) ---
    env_train = se.make_env(n_entities=args.n_train, seed=args.seed)
    k_data, k_train, key = jax.random.split(key, 3)
    data = collect_dataset(env_train, k_data, args.episodes, args.ep_len)
    ret = np.asarray(data["ret"])
    ret_mean, ret_std = float(ret.mean()), float(ret.std() + 1e-8)
    data = dict(data)
    data["ret"] = (data["ret"] - ret_mean) / ret_std  # standardized Q target

    targs = SimpleNamespace(
        steps=args.steps, batch=args.batch, lr=args.lr, d_model=args.d_model,
        n_layers=args.n_layers, n_heads=args.n_heads, n_ood=args.n_ood,
        n_train=args.n_train,
    )
    model, params, train_metrics = train(targs, env_train, data, k_train)

    # --- probe value-coupling vs similarity at N_train and OOD N ---
    per_n = {}
    for n in [args.n_train] + args.n_ood:
        env_n = se.make_env(n_entities=n, seed=args.seed)
        k_probe, key = jax.random.split(key)
        d = collect_dataset(env_n, k_probe, max(8, args.probe_states // args.ep_len + 1),
                            args.ep_len)
        # subsample states for the (expensive) Hessian
        M = d["ent"].shape[0]
        k_sub, key = jax.random.split(key)
        idx = np.asarray(jax.random.randint(k_sub, (min(args.probe_states, M),), 0, M))
        ent_b = d["ent"][idx]
        act_b = d["action"][idx]

        positives = {tuple(sorted(env_n.value_pair)), (0, 1)}  # {(2,3),(0,1)}

        w_rew = value_coupling_graph(model, params, ent_b, act_b, "reward")
        w_q = value_coupling_graph(model, params, ent_b, act_b, "q")
        w_sim = similarity_graph(model, params, ent_b, act_b)
        w_attn = attention_graph(model, params, ent_b, act_b)

        ev = {
            "value_reward_head": evaluate_graph(w_rew, n, positives, "value_reward", args.seed),
            "value_q_head": evaluate_graph(w_q, n, positives, "value_q", args.seed),
            "similarity": evaluate_graph(w_sim, n, positives, "similarity", args.seed),
            "attention": evaluate_graph(w_attn, n, positives, "attention", args.seed),
        }
        # value-gating margin: does the value graph beat similarity at recovery?
        ev["value_minus_similarity_ap"] = ev["value_reward_head"]["ap"] - ev["similarity"]["ap"]
        ev["positives"] = [list(p) for p in sorted(positives)]
        ev["coupling_nnz"] = int(np.count_nonzero(np.asarray(env_n.coupling)))
        per_n[str(n)] = ev

    # --- instrument verdict (synthetic-stage GO/NO-GO) ---
    base = per_n[str(args.n_train)]["value_reward_head"]
    ood_ok = all(
        per_n[str(n)]["value_reward_head"]["ap_z_over_null"] >= 3.0
        for n in args.n_ood
    )
    go = bool(
        base["ap_z_over_null"] >= 3.0
        and base["ap"] >= 0.8
        and per_n[str(args.n_train)]["value_minus_similarity_ap"] > 0.1
        and ood_ok
    )
    verdict = {
        "instrument_go": go,
        "criteria": {
            "reward_ap_in_dist": base["ap"],
            "reward_ap_z_over_null_in_dist": base["ap_z_over_null"],
            "value_minus_similarity_ap_in_dist":
                per_n[str(args.n_train)]["value_minus_similarity_ap"],
            "ood_z_over_null_ge_3": ood_ok,
        },
        "interpretation": (
            "GO: value-coupling graph recovers known pairs above null AND beats "
            "similarity AND persists OOD -> build the real multi-object env. "
            "NO-GO: instrument cannot recover known structure -> fold into the "
            "redundancy-principle paper."
        ),
    }

    out = {
        "tag": args.tag,
        "config": vars(args),
        "ret_standardization": {"mean": ret_mean, "std": ret_std},
        "train_metrics": train_metrics,
        "per_n": per_n,
        "verdict": verdict,
        "wall_time_sec": time.time() - t0,
    }
    out_path = EXP_DIR / f"{args.tag}_vcoupling.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[value_coupling_probe] wrote {out_path}")
    print(json.dumps(verdict, indent=2))
    for n, ev in per_n.items():
        print(f"N={n}: reward AP={ev['value_reward_head']['ap']:.3f} "
              f"(z={ev['value_reward_head']['ap_z_over_null']:.1f}) "
              f"sim AP={ev['similarity']['ap']:.3f} "
              f"attn AP={ev['attention']['ap']:.3f} "
              f"top2={ev['value_reward_head']['top5_pairs'][:2]}")


if __name__ == "__main__":
    main()
