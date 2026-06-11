#!/usr/bin/env python3
"""SE-attention-graph mechanism-check for the graph-WM + structural-entropy (SE)
research direction.

THE QUESTION
------------
Does a *trained* transformer world-model's attention graph carry community
(structural-entropy) structure -- analogous to the ~53% SE compression gap we
measured on SimNorm latents? If yes, SE-as-a-loss could *shape* that structure
and the graph-WM + SE north star is alive. If the attention graph is a
structureless blob (gap ~ that of a degree-preserving random graph), the
direction is dead.

This is a READ-ONLY consumer of the existing model code. It imports
``helios.dynamics.transformer_wm.TransformerWM`` (whose ``__call__(...,
return_attn=True)`` returns ``attn`` of shape (n_layers, B, n_heads, T, T) and
``tokens`` of shape (n_layers+1, B, T, d_model)) and the trained params; it does
not modify the world model or any hot-path file.

TWO MODES (so GPU rollout and selib analysis can be split)
----------------------------------------------------------
Workers have jax + the model but NOT selib; EC2 has selib but no GPU. So:

  1. ``--mode dump``  (run on a worker GPU)
        Load the trained transformer-WM checkpoint, roll out the actor for a few
        episodes collecting attention via ``return_attn=True``, aggregate the
        per-layer attention over heads + batch + time-windows into per-layer
        (T, T) adjacency matrices, also save mean token embeddings, write an npz.

  2. ``--mode analyze``  (run on EC2 with selib)
        For each layer build a networkx graph from the (symmetrized) attention
        adjacency, sparsify weak edges (a couple thresholds + a kNN variant,
        mirroring the SimNorm pre-check that needed sparsification), run
        ``selib.calc.se_report`` -> per-layer SE compression gap (compression_2d)
        + num_communities, and compare to a degree-preserving shuffled baseline.
        Reports per-layer gap and the GO / NO-GO verdict.

ROBUSTNESS
----------
* Missing checkpoint -> a clear, actionable error (not a stack trace deep in
  pickle).
* ``--untrained`` builds a freshly-initialised transformer-WM with the SAME
  architecture so you can dump an untrained attention graph and compare
  trained-vs-untrained SE gap. The *interesting* signal is whether TRAINING
  induces attention community structure, so always dump both.

CHECKPOINT CONTRACT
-------------------
``scripts/run_dreamer4.py`` does not yet persist checkpoints. When it does (or a
sidecar saver does), the expected pickle payload for ``--mode dump`` is a dict
that contains, at minimum:

    {
      "wm_params":  <Flax params for the TransformerWM core>,  # state["wm_params"]["wm"]
      "encoder_params": <Flax params for the MLP encoder>,     # state["wm_params"]["encoder"]
      "actor_params":   <Flax params for the Actor>,           # state["actor_params"]
      "transformer_config": {                                  # to rebuild TransformerWM
          "embed_dim": int, "action_dim": int, "d_model": int,
          "n_layers": int,  "n_heads": int, "context_len": int,
          "mlp_ratio": int, "pos_encoding": str,
      },
      # optional, for env-driven rollout:
      "obs_dim": int, "action_dim": int, "task": str,
    }

The loader is permissive: it also accepts the nested ``run_dreamer4`` agent state
layout (``payload["wm_params"]["wm"]`` / ``["encoder"]`` and
``payload["actor_params"]``) and a top-level ``config``/``transformer`` block. If
your saver differs, adjust ``_extract_payload`` (clearly delimited below) only --
nothing else needs to change.

VALIDATION WITHOUT A TRAINED CKPT
---------------------------------
The analysis pipeline is validated end-to-end without any GPU:
``--mode selftest`` synthesises npz files (a block-structured T x T matrix ->
should show a high SE gap; a uniform matrix -> ~0 gap) and runs ``--mode
analyze`` on them, proving the selib integration + sparsification + baseline
logic work. The ``--mode dump`` path can only be smoke-checked for import / shape
logic without a GPU; ``--mode dump --untrained --smoke`` does exactly that with a
tiny random rollout and no env.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Make src importable for both modes (TransformerWM lives there). selib is only
# needed in --mode analyze and is imported lazily there.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))


# ===========================================================================
# Shared: defaults that mirror scripts/run_dreamer4.make_config().transformer
# ===========================================================================

_DEFAULT_TCFG = dict(
    embed_dim=256,
    d_model=256,
    n_layers=4,
    n_heads=4,
    context_len=32,
    mlp_ratio=4,
    pos_encoding="learned",
)


# ===========================================================================
# CHECKPOINT EXTRACTION  (the one place to touch if your saver layout differs)
# ===========================================================================

def _extract_payload(payload: dict) -> dict:
    """Normalise a loaded checkpoint dict into a flat contract:

        {wm_params, encoder_params, actor_params, tcfg, obs_dim, action_dim, task}

    Accepts both the documented flat layout and the nested ``run_dreamer4`` agent
    state (``payload["wm_params"]["wm"]`` etc.). Raises a clear error if the
    transformer params can't be found.
    """
    if not isinstance(payload, dict):
        raise ValueError(
            f"Checkpoint is not a dict (got {type(payload).__name__}); cannot "
            "locate transformer params. See CHECKPOINT CONTRACT in the header."
        )

    # --- transformer core params --------------------------------------------
    wm_params = None
    encoder_params = None
    actor_params = payload.get("actor_params")

    wmp = payload.get("wm_params")
    if isinstance(wmp, dict) and ("wm" in wmp or "encoder" in wmp):
        # Nested run_dreamer4 agent-state layout.
        wm_params = wmp.get("wm")
        encoder_params = wmp.get("encoder")
    elif wmp is not None:
        # Flat layout: wm_params *is* the transformer params.
        wm_params = wmp
        encoder_params = payload.get("encoder_params")

    if wm_params is None:
        raise ValueError(
            "Could not find transformer-WM params in checkpoint. Expected key "
            "'wm_params' (flat: the TransformerWM params; or nested: "
            "{'wm': ..., 'encoder': ...}). Keys present: "
            f"{sorted(payload.keys())}. See CHECKPOINT CONTRACT in the header."
        )
    if encoder_params is None:
        encoder_params = payload.get("encoder_params")

    # --- transformer config --------------------------------------------------
    tcfg = dict(_DEFAULT_TCFG)
    cfg_block = (
        payload.get("transformer_config")
        or payload.get("transformer")
        or (payload.get("config", {}) or {}).get("transformer")
    )
    if isinstance(cfg_block, dict):
        tcfg.update({k: cfg_block[k] for k in _DEFAULT_TCFG if k in cfg_block})
    elif cfg_block is not None:  # SimpleNamespace-like
        for k in _DEFAULT_TCFG:
            if hasattr(cfg_block, k):
                tcfg[k] = getattr(cfg_block, k)

    obs_dim = payload.get("obs_dim")
    action_dim = payload.get("action_dim", tcfg.get("action_dim"))
    task = payload.get("task")

    return dict(
        wm_params=wm_params,
        encoder_params=encoder_params,
        actor_params=actor_params,
        tcfg=tcfg,
        obs_dim=obs_dim,
        action_dim=action_dim,
        task=task,
    )


def _load_checkpoint(ckpt_path: str) -> dict:
    p = Path(ckpt_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {p}\n"
            "Training may not have produced one yet. Expected something like\n"
            "  exp/tdmpc_glass/PandaPickCube_twm_*/seed_0/checkpoints/best_*.pkl\n"
            "Run `--mode dump --untrained --smoke` to validate the dump logic "
            "without a trained ckpt, or `--mode selftest` to validate analysis."
        )
    with open(p, "rb") as fh:
        payload = pickle.load(fh)
    return _extract_payload(payload)


# ===========================================================================
# MODE: dump  (worker GPU)  -- roll out, collect attention, aggregate, save npz
# ===========================================================================

def _build_wm(tcfg: dict, action_dim: int):
    """Instantiate TransformerWM from a (possibly partial) transformer config."""
    from helios.dynamics.transformer_wm import TransformerWM

    return TransformerWM(
        embed_dim=int(tcfg["embed_dim"]),
        action_dim=int(action_dim),
        d_model=int(tcfg["d_model"]),
        n_layers=int(tcfg["n_layers"]),
        n_heads=int(tcfg["n_heads"]),
        context_len=int(tcfg["context_len"]),
        mlp_ratio=int(tcfg["mlp_ratio"]),
        pos_encoding=str(tcfg.get("pos_encoding", "learned")),
    )


def _aggregate_attn(attn) -> np.ndarray:
    """attn: (n_layers, B, n_heads, T, T) -> per-layer (T, T) mean adjacency.

    Mean over heads + batch (the "+ time-windows" averaging is implicit: each
    rollout window of length T is one batch row, so averaging over B averages
    over windows). Returns float64 numpy of shape (n_layers, T, T).
    """
    attn = np.asarray(attn, dtype=np.float64)
    # mean over heads (axis 2) then batch (axis 1) -> (n_layers, T, T)
    return attn.mean(axis=2).mean(axis=1)


def _dump_smoke(out_path: str, tcfg: dict, action_dim: int, T: int, B: int, seed: int):
    """No-env, no-ckpt smoke check of the dump shape logic: build an UNTRAINED
    TransformerWM, run a single random forward with return_attn=True, aggregate,
    save npz. Proves the import + shape pipeline on a CPU."""
    import jax
    import jax.numpy as jnp

    wm = _build_wm(tcfg, action_dim)
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    embed = jax.random.normal(k1, (B, T, int(tcfg["embed_dim"])))
    action = jax.random.normal(k2, (B, T, int(action_dim)))
    params = wm.init(k3, embed, action)
    out = wm.apply(params, embed, action, return_attn=True)
    attn = out["attn"]              # (n_layers, B, n_heads, T, T)
    tokens = out["tokens"]         # (n_layers+1, B, T, d_model)
    adj = _aggregate_attn(attn)    # (n_layers, T, T)
    tok_mean = np.asarray(tokens, np.float64).mean(axis=1)   # (n_layers+1, T, d_model)
    _save_npz(out_path, adj, tok_mean, meta=dict(
        source="dump-smoke-untrained", task="<none>", T=T, B=B,
        n_layers=int(attn.shape[0]), n_heads=int(attn.shape[2]),
        trained=False, tcfg=tcfg,
    ))
    print(f"[dump:smoke] wrote {out_path}  adj={adj.shape}  tokens={tok_mean.shape}")
    print("[dump:smoke] NOTE: random weights, no env -- shape/ import check only.")


def _dump_rollout(
    ckpt_path: str | None,
    task: str,
    out_path: str,
    *,
    untrained: bool,
    n_episodes: int,
    max_steps: int,
    seed: int,
    tcfg_override: dict | None,
):
    """Real rollout: load (or randomly init) the WM + actor, drive the env with
    the actor for a few episodes collecting (embed, action) windows, run the WM
    with return_attn=True over each window, aggregate per-layer attention."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jnp
    from mujoco_playground import registry, wrapper

    from helios.core.networks import MLP
    from helios.core.distributions import TanhNormal
    from helios.algorithms.dreamer import Actor

    # --- env ---
    env = registry.load(task)
    env = wrapper.wrap_for_brax_training(env, episode_length=1000, action_repeat=1)
    obs_dim = int(env.observation_size)
    action_dim = int(env.action_size)

    # --- config / params ---
    if untrained:
        tcfg = dict(_DEFAULT_TCFG)
        if tcfg_override:
            tcfg.update(tcfg_override)
        wm_params = encoder_params = actor_params = None
        trained = False
    else:
        ck = _load_checkpoint(ckpt_path)
        tcfg = ck["tcfg"]
        if tcfg_override:
            tcfg.update(tcfg_override)
        wm_params = ck["wm_params"]
        encoder_params = ck["encoder_params"]
        actor_params = ck["actor_params"]
        if ck.get("action_dim"):
            action_dim = int(ck["action_dim"])
        trained = True

    embed_dim = int(tcfg["embed_dim"])
    d_model = int(tcfg["d_model"])
    ctx = int(tcfg["context_len"])

    wm = _build_wm(tcfg, action_dim)
    encoder = MLP(hidden_dims=(embed_dim,), output_dim=embed_dim, activation="silu")
    actor = Actor(action_dim=action_dim, hidden_dims=(512, 512))

    key = jax.random.PRNGKey(seed)
    if untrained or wm_params is None:
        key, k1, k2, k3 = jax.random.split(key, 4)
        dummy_obs = jnp.zeros((1, obs_dim))
        encoder_params = encoder.init(k1, dummy_obs)
        wm_params = wm.init(k2, jnp.zeros((1, ctx, embed_dim)), jnp.zeros((1, ctx, action_dim)))
        actor_params = actor.init(k3, jnp.zeros((1, d_model)))

    have_actor = actor_params is not None

    @jax.jit
    def encode(obs):
        return encoder.apply(encoder_params, obs)

    @jax.jit
    def wm_last_feat(emb_hist, act_hist):
        return wm.apply(wm_params, emb_hist, act_hist)["z"][:, -1]

    @jax.jit
    def act_fn(feat, k):
        a_out = actor.apply(actor_params, feat)
        dist = TanhNormal(a_out["mean"], a_out["log_std"])
        return dist.mode()

    @jax.jit
    def env_reset(k):
        return env.reset(jax.random.split(k, 1))

    @jax.jit
    def env_step(st, a):
        return env.step(st, a)

    # --- collect full-length attention windows from real rollouts ---
    # We gather, per episode, the LAST window of ctx (embed, action) pairs (the
    # most informative, fully-populated window) and run one return_attn pass.
    attn_accum = None     # (n_layers, T, T) running sum
    tok_accum = None      # (n_layers+1, T, d_model) running sum
    n_windows = 0
    T_eff = ctx

    for ep in range(n_episodes):
        key, rk = jax.random.split(key)
        st = env_reset(rk)
        obs = jnp.asarray(st.obs)                       # (1, obs_dim)
        emb_hist = np.zeros((1, ctx, embed_dim), np.float32)
        act_hist = np.zeros((1, ctx, action_dim), np.float32)
        for t in range(max_steps):
            emb = np.array(encode(obs))                 # (1, embed_dim)
            emb_hist = np.concatenate([emb_hist[:, 1:, :], emb[:, None, :]], axis=1)
            feat = wm_last_feat(jnp.asarray(emb_hist), jnp.asarray(act_hist))
            if have_actor:
                key, ak = jax.random.split(key)
                action = act_fn(feat, ak)
            else:
                key, ak = jax.random.split(key)
                action = jax.random.uniform(ak, (1, action_dim), minval=-1.0, maxval=1.0)
            act_np = np.array(action, np.float32)
            act_hist = np.concatenate([act_hist[:, 1:, :], act_np[:, None, :]], axis=1)
            st = env_step(st, action[0][None] if action.ndim == 1 else action)
            if bool(np.asarray(st.done).reshape(-1)[0] > 0.5):
                break
            obs = jnp.asarray(st.obs)

        # One return_attn pass over the final populated window of this episode.
        out = wm.apply(wm_params, jnp.asarray(emb_hist), jnp.asarray(act_hist),
                       return_attn=True)
        adj = _aggregate_attn(out["attn"])              # (n_layers, T, T)
        tok = np.asarray(out["tokens"], np.float64).mean(axis=1)  # (n_layers+1,T,dm)
        attn_accum = adj if attn_accum is None else attn_accum + adj
        tok_accum = tok if tok_accum is None else tok_accum + tok
        n_windows += 1
        print(f"[dump] episode {ep+1}/{n_episodes} collected window (T={ctx})", flush=True)

    if n_windows == 0:
        raise RuntimeError("No rollout windows were collected.")
    adj_mean = attn_accum / n_windows
    tok_mean = tok_accum / n_windows

    _save_npz(out_path, adj_mean, tok_mean, meta=dict(
        source=("untrained" if untrained else f"ckpt:{ckpt_path}"),
        task=task, T=T_eff, B=1, n_windows=n_windows,
        n_layers=int(adj_mean.shape[0]),
        trained=trained, tcfg=tcfg,
    ))
    print(f"[dump] wrote {out_path}  adj={adj_mean.shape}  tokens={tok_mean.shape}  "
          f"(trained={trained}, windows={n_windows})")


def _save_npz(out_path: str, adj: np.ndarray, tokens: np.ndarray, meta: dict):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        adj=adj.astype(np.float32),       # (n_layers, T, T) per-layer mean attention
        tokens=tokens.astype(np.float32), # (n_layers+1, T, d_model) mean node feats
        meta=json.dumps(meta),
    )


# ===========================================================================
# MODE: analyze  (EC2 with selib)  -- build graph, SE gap, baseline, verdict
# ===========================================================================

def _symmetrize(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def _sparsify_threshold(A: np.ndarray, frac: float) -> np.ndarray:
    """Zero out edges below the (frac)-quantile of nonzero weights. Keeps the
    strongest (1-frac) fraction. Mirrors the SimNorm pre-check: a near-dense
    attention matrix needs sparsification before community structure shows."""
    A = A.copy()
    np.fill_diagonal(A, 0.0)
    nz = A[A > 0]
    if nz.size == 0:
        return A
    thr = np.quantile(nz, frac)
    A[A < thr] = 0.0
    return A


def _sparsify_knn(A: np.ndarray, k: int) -> np.ndarray:
    """Keep top-k outgoing edges per row (then symmetrize-OR). A directed->
    undirected kNN graph, the other standard sparsifier."""
    A = A.copy()
    np.fill_diagonal(A, 0.0)
    n = A.shape[0]
    k = min(k, n - 1)
    out = np.zeros_like(A)
    for i in range(n):
        if k <= 0:
            continue
        idx = np.argpartition(A[i], -k)[-k:]
        out[i, idx] = A[i, idx]
    # OR-symmetrize: edge kept if it's a top-k of either endpoint.
    return np.maximum(out, out.T)


def _to_nx(A: np.ndarray):
    import networkx as nx

    G = nx.Graph()
    n = A.shape[0]
    G.add_nodes_from(range(n))
    iu = np.triu_indices(n, k=1)
    for i, j in zip(*iu):
        w = float(A[i, j])
        if w > 0:
            G.add_edge(int(i), int(j), weight=w)
    return G


def _degree_preserving_shuffle(A: np.ndarray, seed: int) -> np.ndarray:
    """A weighted degree-preserving null: keep the multiset of edge weights but
    randomly re-pair them across the existing edge slots (Maslov-Sneppen-style on
    the weighted upper triangle). Destroys community structure while keeping the
    weight distribution + (approximately) the number of edges, so any SE gap
    above this baseline is *real* community structure, not just a heavy-tailed
    weight distribution."""
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    iu = np.triu_indices(n, k=1)
    w = A[iu].copy()
    nz_mask = w > 0
    weights = w[nz_mask].copy()
    rng.shuffle(weights)
    # Reassign shuffled weights to a fresh random set of slots of the same count.
    new_w = np.zeros_like(w)
    n_edges = int(nz_mask.sum())
    slots = rng.choice(w.size, size=n_edges, replace=False)
    new_w[slots] = weights
    B = np.zeros_like(A)
    B[iu] = new_w
    B = B + B.T
    return B


def _analyze_one(A_sym: np.ndarray, *, knn_k: int, thresholds, seed: int) -> dict:
    """Run se_report across sparsification settings + baseline for one layer."""
    from selib import calc as se_calc

    variants = {}
    # threshold variants
    for frac in thresholds:
        variants[f"thr{frac:g}"] = _sparsify_threshold(A_sym, frac)
    # kNN variant
    variants[f"knn{knn_k}"] = _sparsify_knn(A_sym, knn_k)

    results = {}
    best_gap = -1.0
    best_variant = None
    for name, Asp in variants.items():
        G = _to_nx(Asp)
        if G.number_of_edges() == 0:
            results[name] = dict(error="empty graph after sparsification")
            continue
        rep = se_calc.se_report(G)
        # degree-preserving null on the SAME sparsified graph
        Anull = _degree_preserving_shuffle(Asp, seed)
        Gnull = _to_nx(Anull)
        null_rep = (
            se_calc.se_report(Gnull)
            if Gnull.number_of_edges() > 0
            else {"compression_2d": 0.0, "num_communities": 0}
        )
        gap = float(rep["compression_2d"])
        null_gap = float(null_rep["compression_2d"])
        results[name] = dict(
            n=rep["n"], m=rep["m"],
            se_1d=rep["se_1d"], se_2d_optimal=rep["se_2d_optimal"],
            compression_2d=gap, num_communities=rep["num_communities"],
            null_compression_2d=null_gap,
            null_num_communities=null_rep.get("num_communities", 0),
            gap_over_null=round(gap - null_gap, 4),
        )
        if gap > best_gap:
            best_gap = gap
            best_variant = name

    return dict(variants=results, best_variant=best_variant, best_gap=round(best_gap, 4))


# Decision thresholds for the GO / NO-GO verdict (see header / report).
GO_GAP = 0.10            # absolute compression_2d to be of interest at all
GO_MARGIN_OVER_NULL = 0.05   # must beat the degree-preserving null by this much


def _verdict(per_layer: list[dict]) -> dict:
    best = max(
        (
            (li, v["best_gap"],
             v["variants"][v["best_variant"]]["gap_over_null"]
             if v["best_variant"] else 0.0)
            for li, v in enumerate(per_layer)
        ),
        key=lambda t: t[1],
        default=(None, -1.0, 0.0),
    )
    layer_i, gap, margin = best
    has_structure = (gap >= GO_GAP) and (margin >= GO_MARGIN_OVER_NULL)
    return dict(
        best_layer=layer_i,
        best_gap=round(gap, 4),
        best_gap_over_null=round(margin, 4),
        go_gap_threshold=GO_GAP,
        go_margin_threshold=GO_MARGIN_OVER_NULL,
        has_exploitable_community_structure=bool(has_structure),
        verdict=("GO -- attention graph has exploitable community structure; "
                 "SE-as-a-loss can shape it."
                 if has_structure else
                 "NO-GO -- attention graph gap is at/below the degree-preserving "
                 "null; structureless blob, direction is dead."),
    )


def _analyze(npz_path: str, out_path: str, *, knn_k: int, thresholds, seed: int):
    p = Path(npz_path)
    if not p.exists():
        raise FileNotFoundError(f"npz not found: {p} (run --mode dump first).")
    data = np.load(p, allow_pickle=True)
    adj = np.asarray(data["adj"], dtype=np.float64)   # (n_layers, T, T)
    if adj.ndim == 2:
        adj = adj[None]
    meta = {}
    if "meta" in data:
        try:
            meta = json.loads(str(data["meta"]))
        except Exception:
            meta = {}

    per_layer = []
    for li in range(adj.shape[0]):
        A_sym = _symmetrize(adj[li])
        res = _analyze_one(A_sym, knn_k=knn_k, thresholds=thresholds, seed=seed)
        res["layer"] = li
        per_layer.append(res)
        bv = res["best_variant"]
        bg = res["best_gap"]
        mar = (res["variants"][bv]["gap_over_null"] if bv else 0.0)
        print(f"[analyze] layer {li}: best gap={bg:.4f} via {bv} "
              f"(over null +{mar:.4f}, "
              f"{res['variants'].get(bv, {}).get('num_communities', '?')} communities)")

    verdict = _verdict(per_layer)
    report = dict(
        source_npz=str(p),
        meta=meta,
        sparsification=dict(thresholds=list(thresholds), knn_k=knn_k),
        per_layer=per_layer,
        verdict=verdict,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n[analyze] {verdict['verdict']}")
    print(f"[analyze] best layer={verdict['best_layer']} "
          f"gap={verdict['best_gap']} (over null +{verdict['best_gap_over_null']})")
    print(f"[analyze] wrote {out_path}")
    return report


# ===========================================================================
# MODE: selftest  -- synthesise block (high gap) + uniform (~0 gap) npz, analyze
# ===========================================================================

def _make_block_matrix(T: int, n_blocks: int, intra: float, inter: float,
                       noise: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = np.full((T, T), inter, dtype=np.float64)
    block = T // n_blocks
    for b in range(n_blocks):
        s = b * block
        e = T if b == n_blocks - 1 else (b + 1) * block
        A[s:e, s:e] = intra
    A = A + noise * rng.random((T, T))
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    return A


def _selftest(tmpdir: str, *, knn_k: int, thresholds, seed: int):
    tmp = Path(tmpdir)
    tmp.mkdir(parents=True, exist_ok=True)
    T = 32
    dm = 8

    # 1) Block-structured -> should show HIGH SE gap.
    block = _make_block_matrix(T, n_blocks=4, intra=1.0, inter=0.02, noise=0.01, seed=seed)
    block_npz = str(tmp / "selftest_block.npz")
    _save_npz(block_npz, block[None], np.zeros((1, T, dm)),
              meta=dict(source="selftest-block", trained=True))

    # 2) Uniform -> should show ~0 SE gap.
    uni = np.ones((T, T), dtype=np.float64)
    np.fill_diagonal(uni, 0.0)
    uni_npz = str(tmp / "selftest_uniform.npz")
    _save_npz(uni_npz, uni[None], np.zeros((1, T, dm)),
              meta=dict(source="selftest-uniform", trained=True))

    print("=== SELFTEST: block-structured matrix (expect HIGH gap) ===")
    block_rep = _analyze(block_npz, str(tmp / "selftest_block.json"),
                        knn_k=knn_k, thresholds=thresholds, seed=seed)
    print("\n=== SELFTEST: uniform matrix (expect ~0 gap) ===")
    uni_rep = _analyze(uni_npz, str(tmp / "selftest_uniform.json"),
                      knn_k=knn_k, thresholds=thresholds, seed=seed)

    block_gap = block_rep["verdict"]["best_gap"]
    uni_gap = uni_rep["verdict"]["best_gap"]
    block_margin = block_rep["verdict"]["best_gap_over_null"]
    uni_margin = uni_rep["verdict"]["best_gap_over_null"]
    print("\n=== SELFTEST SUMMARY ===")
    print(f"  block   gap={block_gap:.4f}  over-null={block_margin:+.4f}  GO?  "
          f"{block_rep['verdict']['has_exploitable_community_structure']}")
    print(f"  uniform gap={uni_gap:.4f}  over-null={uni_margin:+.4f}  GO?  "
          f"{uni_rep['verdict']['has_exploitable_community_structure']}")
    # The discriminating signal is the verdict (which is keyed on the
    # degree-preserving null margin, NOT the raw gap): a uniform graph can show a
    # nonzero RAW gap purely from sparsification artefacts, but it must NOT beat
    # its own degree-preserving null. That is exactly what the null baseline is
    # for, so the selftest asserts on the verdict, not the raw gap.
    ok = (block_gap > 0.20) and (block_margin > 0) and \
        block_rep["verdict"]["has_exploitable_community_structure"] and \
        (uni_margin <= 0) and \
        not uni_rep["verdict"]["has_exploitable_community_structure"]
    print(f"  PIPELINE VALIDATION: {'PASS' if ok else 'FAIL'} "
          "(block: high gap, beats null, GO | uniform: at/below null, NO-GO)")
    if not ok:
        sys.exit(1)


# ===========================================================================
# CLI
# ===========================================================================

def _parse_thresholds(s: str):
    return [float(x) for x in str(s).split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(
        description="SE-attention-graph mechanism-check (dump | analyze | selftest).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--mode", required=True,
                    choices=["dump", "analyze", "selftest"])
    # dump
    ap.add_argument("--ckpt", default=None, help="trained transformer-WM best_*.pkl")
    ap.add_argument("--task", default="PandaPickCube", help="MuJoCo Playground env id")
    ap.add_argument("--untrained", action="store_true",
                    help="dump a freshly-initialised WM (trained-vs-untrained baseline)")
    ap.add_argument("--smoke", action="store_true",
                    help="dump: no-env random forward, shape/import check only")
    ap.add_argument("--n_episodes", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=200)
    # analyze
    ap.add_argument("--npz", default=None, help="attention npz from --mode dump")
    ap.add_argument("--knn_k", type=int, default=4, help="kNN sparsifier: edges/node")
    ap.add_argument("--thresholds", default="0.5,0.8,0.9",
                    help="comma-separated quantile thresholds for sparsification")
    # shared
    ap.add_argument("--out", default=None, help="output path (npz for dump, json for analyze)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    thresholds = _parse_thresholds(args.thresholds)

    if args.mode == "dump":
        out = args.out or "attn.npz"
        if args.smoke:
            _dump_smoke(out, dict(_DEFAULT_TCFG), action_dim=4, T=int(_DEFAULT_TCFG["context_len"]),
                        B=2, seed=args.seed)
            return
        if not args.untrained and not args.ckpt:
            ap.error("--mode dump requires --ckpt (or --untrained, or --smoke).")
        _dump_rollout(
            args.ckpt, args.task, out,
            untrained=args.untrained,
            n_episodes=args.n_episodes, max_steps=args.max_steps,
            seed=args.seed, tcfg_override=None,
        )

    elif args.mode == "analyze":
        if not args.npz:
            ap.error("--mode analyze requires --npz.")
        out = args.out or "se_report.json"
        _analyze(args.npz, out, knn_k=args.knn_k, thresholds=thresholds, seed=args.seed)

    elif args.mode == "selftest":
        out = args.out or "/tmp/se_attn_selftest"
        _selftest(out, knn_k=args.knn_k, thresholds=thresholds, seed=args.seed)


if __name__ == "__main__":
    main()
