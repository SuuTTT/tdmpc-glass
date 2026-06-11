"""Differentiable structural entropy (SE) in pure JAX, cross-checked vs selib.

This is a NEW, validated implementation of 1D and 2D structural entropy (Li &
Pan, "Structural information and dynamical complexity of networks", IEEE TIT
2016), written for the graph-world-model + SE research direction. It is
differentiable in both the adjacency ``A`` and the (soft) community assignment
``S``, so it can be used directly as a loss on a learned graph / transformer
world model.

It is numerically validated against the official ``selib`` implementation
(``selib.metrics.structural_entropy_2d`` and ``selib.calc.one_dimensional``);
see ``tests/test_se_jax.py``. Agreement is to < 1e-4 on karate, les_miserables,
and planted-SBM graphs for hard (one-hot) partitions, and the soft relaxation
converges to the hard value as the softmax temperature grows.

--------------------------------------------------------------------------------
Directed vs. undirected (READ THIS)
--------------------------------------------------------------------------------
Canonical Li-Pan / selib SE is **undirected**. The degree of a node is its
(weighted) degree, and the cut ``g_j`` of a module counts each crossing edge for
*both* incident modules. On an undirected graph ``A == A.T`` and ``g_j`` equals
the total weight of edges with exactly one endpoint in ``j``.

A transition-like / world-model graph is typically **directed and asymmetric**
(``A != A.T``): ``A[i, j]`` is the probability/weight of going from state-cluster
``i`` to ``j``. Computing SE on such an ``A`` *without symmetrizing* uses
out-degree (``d_v = row-sum``) and an out-cut-only ``g_j``. That is a perfectly
well-defined quantity, but it is **not** the canonical undirected SE: in general
it differs from ``selib`` on the same partition.

This module therefore exposes ``symmetrize`` (default ``True``):

  * ``symmetrize=True``  -> compute SE on ``A_sym = (A + A.T) / 2``. This matches
    selib's undirected semantics EXACTLY (verified to < 1e-4). The 1/2 factor is
    the correct one: an undirected edge {u, v} of weight w appears in a symmetric
    adjacency as ``A[u,v] = A[v,u] = w``; ``nx.to_numpy_array`` of an undirected
    graph produces exactly that symmetric matrix, and ``(A + A.T)/2`` is the
    identity on an already-symmetric matrix, so feeding either the symmetric
    matrix or any matrix whose symmetric part equals it reproduces selib.

  * ``symmetrize=False`` -> compute the directed (out-degree / out-cut) quantity
    on ``A`` as given. Use this only if you specifically want the directed
    variant; it is the quantity the OLD Glass code computed (and the reason the
    old Glass SE disagreed with selib on asymmetric transition graphs).

All quantities are in bits (log base 2).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

_LOG2 = jnp.log(2.0)


def _log2(x: jax.Array) -> jax.Array:
    return jnp.log(x) / _LOG2


def se_1d(A: jax.Array, *, symmetrize: bool = True, eps: float = 1e-12) -> jax.Array:
    """1D structural entropy H^1(G) = -sum_v (d_v/2m) log2(d_v/2m).

    The positioning entropy of the degree-stationary random walk, with no
    partition. This is the partition-free upper bound on every H^k.

    Args:
      A: (N, N) (weighted) adjacency. Self-loops on the diagonal contribute to
        degree exactly as in selib's ``G.degree(weight=...)`` (a self-loop of
        weight w adds 2w to the degree in an undirected graph; with
        ``symmetrize=True`` a diagonal entry A[v,v]=w yields d_v += 2w, matching
        networkx).
      symmetrize: if True, use A_sym = (A + A.T)/2 (canonical undirected SE).
      eps: numerical floor for logs.

    Returns:
      scalar H^1 in bits.
    """
    if symmetrize:
        A = 0.5 * (A + A.swapaxes(-1, -2))
    d = jnp.sum(A, axis=-1)
    two_m = jnp.sum(d)
    p = d / (two_m + eps)
    # 0 * log(0) -> 0; mask zero-degree nodes.
    mask = (d > 0).astype(A.dtype)
    return -jnp.sum(mask * p * _log2(p + eps))


def se_2d(
    A: jax.Array,
    S: jax.Array,
    *,
    symmetrize: bool = True,
    hard: bool = False,
    eps: float = 1e-12,
) -> jax.Array:
    """2D structural entropy of a (soft) partition S over graph A.

    H^2 = - sum_v (d_v/2m) log2(d_v / V_{j(v)})        # node-within-module term
          - sum_j (g_j/2m) log2(V_j / 2m)              # module-cut term

    where d_v = degree(v), V_j = sum of degrees in module j (volume), g_j = cut
    of module j (weight of edges leaving j), 2m = sum of all degrees.

    Args:
      A: (N, N) (weighted) adjacency.
      S: (N, K) assignment. Soft (a probability over modules per node, i.e.
        softmax already applied, rows sum to 1) or one-hot. Differentiable in S.
      symmetrize: if True, SE is computed on A_sym = (A+A.T)/2 (canonical
        undirected SE, matches selib). If False, the directed out-degree/out-cut
        variant on A is computed (the OLD Glass quantity).
      hard: if True, S is straight-through hardened to one-hot (argmax) for the
        forward value while keeping soft gradients. Use to get the exact
        discrete SE value while remaining differentiable.
      eps: numerical floor.

    Returns:
      scalar H^2 in bits. Differentiable in A and S.
    """
    if symmetrize:
        A = 0.5 * (A + A.swapaxes(-1, -2))
    if hard:
        idx = jnp.argmax(S, axis=-1)
        S_hard = jax.nn.one_hot(idx, S.shape[-1], dtype=S.dtype)
        S = S_hard + (S - jax.lax.stop_gradient(S))  # straight-through

    d = jnp.sum(A, axis=-1)                 # (N,) degree
    two_m = jnp.sum(d)                      # 2m
    inv2m = 1.0 / (two_m + eps)

    V = S.T @ d                             # (K,) module volumes  V_j = sum_{v in j} d_v
    AS = A @ S                              # (N, K)  AS[v, j] = sum_u A[v,u] S[u,j]
    # internal degree of v toward each module: S-weighted neighbours.
    # cut of module j = sum_{v in j} (d_v - edges from v staying in j)
    g = jnp.sum(S * (d[:, None] - AS), axis=0)   # (K,) cut per module

    # --- node-within-module term: -sum_v (d_v/2m) log2(d_v/V_{j(v)}) ---------
    # V_{j(v)} = sum_j S[v,j] V_j (exact for one-hot; the soft generalization).
    V_of_v = S @ V                          # (N,)
    node_mask = (d > 0).astype(A.dtype)
    term_node = -jnp.sum(
        node_mask * (d * inv2m) * _log2((d + eps) / (V_of_v + eps))
    )

    # --- module-cut term: -sum_j (g_j/2m) log2(V_j/2m) -----------------------
    cut_mask = (g > 0).astype(A.dtype)
    term_cut = -jnp.sum(cut_mask * (g * inv2m) * _log2((V * inv2m) + eps))

    return term_node + term_cut


def se_2d_gap(
    A: jax.Array,
    S: jax.Array,
    *,
    symmetrize: bool = True,
    hard: bool = False,
    eps: float = 1e-12,
) -> jax.Array:
    """Compression / structural-gap metric: (H^1 - H^2) / H^1, in [0, 1].

    Higher = the partition S explains more of the graph's structure (more bits
    saved relative to the partition-free 1D entropy). 0 for a trivial partition,
    -> 1 for a perfectly modular partition. Differentiable in A and S; a natural
    *reward*-shaped term (maximize gap == minimize H^2).
    """
    h1 = se_1d(A, symmetrize=symmetrize, eps=eps)
    h2 = se_2d(A, S, symmetrize=symmetrize, hard=hard, eps=eps)
    return (h1 - h2) / (h1 + eps)
