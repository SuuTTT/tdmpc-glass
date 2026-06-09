"""iter-19 — Community-detection skill discovery for TD-MPC2.

The Glass machinery already builds a prototype TRANSITION graph P (KxK, how often the
agent moves prototype i -> j) and partitions it by structural entropy. Here we reuse that
graph for TEMPORAL abstraction: communities in the transition graph are densely
intra-connected latent regions separated by bottlenecks -> natural subgoals/options.

A skill := "reach community c". This is navigational and ACTION-CONDITIONAL, so it passes
the iter-15 controllability law (I(a; next | state) > 0) that killed the reward-equivalence
quotient planner. We expose:

  detect_communities(P)        -> community label per prototype (Louvain / 2D-SE)
  community_structure(P)       -> labels, modularity, bottleneck prototypes, sizes
  community_centroids(...)     -> subgoal latent per community (for goal-conditioned skills)

Pure numpy/networkx; no JAX dependency so it runs offline on glass_diag npz dumps.
"""
from __future__ import annotations
import numpy as np

try:
    import networkx as nx
    from networkx.algorithms.community import louvain_communities, modularity
    _HAVE_NX = True
except Exception:  # pragma: no cover
    _HAVE_NX = False


def _symmetrize(P: np.ndarray, eps: float = 1e-9, keep_frac: float | None = 0.2) -> np.ndarray:
    """Undirected weighted adjacency from a (possibly row-stochastic) transition matrix.

    keep_frac (iter-19 Stage-1 finding): the raw Glass P is too dense — the 1e-4 graph
    smoothing + diffuse SimNorm dynamics flood it (40-76% effective edge density) so Louvain
    sees one blob (modularity ~0). Sparsifying to the top `keep_frac` of edges by weight
    exposes the real community structure (CartpoleSparse: 0.02 -> 0.705 modularity at 0.2).
    Set keep_frac=None to disable (raw graph)."""
    P = np.asarray(P, dtype=np.float64)
    A = 0.5 * (P + P.T)
    np.fill_diagonal(A, 0.0)          # self-loops carry no community signal
    A[A < eps] = 0.0
    if keep_frac is not None and 0.0 < keep_frac < 1.0:
        pos = np.sort(A[A > 0])[::-1]
        if len(pos):
            thr = pos[min(int(keep_frac * len(pos)), len(pos) - 1)]
            A[A < thr] = 0.0
    return A


def detect_communities(P: np.ndarray, resolution: float = 1.0, seed: int = 0) -> np.ndarray:
    """Louvain community labels for each prototype from transition graph P.

    Returns int array shape (K,) of community ids in [0, n_comm). Falls back to a
    spectral 2-cut if networkx is unavailable. Isolated nodes (no edges) get their own
    singleton community.
    """
    A = _symmetrize(P)
    K = A.shape[0]
    if _HAVE_NX:
        G = nx.from_numpy_array(A)
        comms = louvain_communities(G, weight="weight", resolution=resolution, seed=seed)
        lab = np.full(K, -1, dtype=int)
        for cid, nodes in enumerate(comms):
            for n in nodes:
                lab[n] = cid
        return lab
    # fallback: sign of Fiedler vector (2 communities)
    d = A.sum(1)
    L = np.diag(d) - A
    w, v = np.linalg.eigh(L)
    fiedler = v[:, 1] if K > 1 else np.zeros(K)
    return (fiedler > 0).astype(int)


def bottleneck_prototypes(P: np.ndarray, labels: np.ndarray, top: int = 3) -> list[int]:
    """Prototypes whose transitions most often CROSS communities = subgoal/bottleneck states.

    Score_i = (mass of i's edges going to a different community) / (total edge mass of i).
    These are the states a skill should aim THROUGH to switch communities.
    """
    A = _symmetrize(P)
    cross = np.zeros(A.shape[0])
    tot = A.sum(1) + 1e-12
    for i in range(A.shape[0]):
        cross[i] = A[i, labels != labels[i]].sum()
    score = cross / tot
    return list(np.argsort(-score)[:top])


def community_structure(P: np.ndarray, resolution: float = 1.0, seed: int = 0) -> dict:
    """Full report: labels, #communities, modularity, sizes, bottlenecks, separation."""
    A = _symmetrize(P)
    labels = detect_communities(P, resolution=resolution, seed=seed)
    n_comm = int(labels.max() + 1) if labels.size else 0
    sizes = np.bincount(labels, minlength=n_comm).tolist()
    mod = None
    if _HAVE_NX and n_comm > 1:
        G = nx.from_numpy_array(A)
        groups = [set(np.where(labels == c)[0].tolist()) for c in range(n_comm)]
        mod = float(modularity(G, groups, weight="weight"))
    # separation = fraction of edge mass that stays WITHIN communities (1=perfect skills)
    within = sum(A[i, labels == labels[i]].sum() for i in range(A.shape[0]))
    sep = float(within / (A.sum() + 1e-12))
    return {
        "labels": labels,
        "n_communities": n_comm,
        "sizes": sizes,
        "modularity": mod,
        "within_fraction": sep,
        "bottlenecks": bottleneck_prototypes(P, labels),
    }


def community_centroids(prototypes: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Mean latent of each community's prototypes = subgoal embedding for a reach-skill.
    prototypes: (K, latent_dim). Returns (n_comm, latent_dim)."""
    n_comm = int(labels.max() + 1)
    return np.stack([prototypes[labels == c].mean(0) for c in range(n_comm)])


# --- Stage-2 primitives: community membership + goal-reaching reward -----------
# A skill = "reach community c". The low-level policy is conditioned on a one-hot
# target community and rewarded for ENTERING it. These are framework-agnostic
# (work on numpy or jax arrays via duck-typed ops) so they drop into the TD-MPC2
# collection loop as an intrinsic reward.

def assign_community(z, prototypes, labels, temperature: float = 1.0):
    """Hard community id for latent(s) z: label of the nearest (cosine) prototype.
    z: (..., L); prototypes: (K, L); labels: (K,). Returns (...,) int community ids.
    numpy implementation (call inside the host-side collection loop, like the
    existing cluster_id_batch path)."""
    z = np.asarray(z); P = np.asarray(prototypes)
    zn = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    pn = P / (np.linalg.norm(P, axis=-1, keepdims=True) + 1e-8)
    sim = zn @ pn.T                                   # (..., K)
    nearest = np.argmax(sim, axis=-1)                 # (...,)
    return np.asarray(labels)[nearest]


def skill_reward(z_next, target_comm, prototypes, labels, centroids=None,
                 shaped: bool = True):
    """Intrinsic reward for the reach-community skill, per env.
    +1 when z_next is IN target_comm; optional dense shaping = negative cosine
    distance to the target community centroid (guides before arrival).
    z_next: (N, L); target_comm: (N,) int; returns (N,) float."""
    z = np.asarray(z_next)
    comm = assign_community(z, prototypes, labels)
    hit = (comm == np.asarray(target_comm)).astype(np.float32)
    if not shaped:
        return hit
    if centroids is None:
        centroids = community_centroids(np.asarray(prototypes), np.asarray(labels))
    cen = np.asarray(centroids)[np.asarray(target_comm)]          # (N, L)
    zn = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    cn = cen / (np.linalg.norm(cen, axis=-1, keepdims=True) + 1e-8)
    cos = np.sum(zn * cn, axis=-1)                                # (N,) in [-1,1]
    return hit + 0.1 * cos                                        # sparse hit + light shaping
