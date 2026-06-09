#!/usr/bin/env python3
"""iter-23 SE pre-check: does the SimNorm latent graph have exploitable community
structure (2-D structural entropy materially below 1-D)?  The viability gate for the
structural-entropy temporal-abstraction lever.

Tier-1 (this script): runs on saved prototype transition graphs / prototype latents
(proxy for the jumpy substrate; same SimNorm encoder family). Tier-2 (TODO): repeat on
real jumpy-rollout latents (>=128 nodes, k-step transitions) loaded from a checkpoint.

Li & Pan structural entropy:
  H^1 = -sum_i (d_i/2m) log2(d_i/2m)                            (1-D, no partition)
  H^2(P) = -sum_c [ sum_{i in c}(d_i/2m)log2(d_i/V_c) + (g_c/2m)log2(V_c/2m) ]
where d_i=degree, V_c=community volume, g_c=community cut, 2m=total volume.
gap = H1 - H2;  gap/H1 >~15% => exploitable structure;  ~0 => "blob" (lever dead).

Finding (2026-06-09): with SE-OPTIMAL Louvain + top-frac sparsification (or kNN on the
512-d SimNorm latents) the gap reaches 22-47% — PASS. Raw (unsparsified) graph ~0% ("blob",
matches iter-19). So the DR-flagged SimNorm-density risk is real but MITIGABLE via sparsify/kNN.
"""
from __future__ import annotations
import numpy as np

try:
    import networkx as nx
    from networkx.algorithms.community import louvain_communities
    _HAVE_NX = True
except Exception:
    _HAVE_NX = False


def symmetrize(P, eps=1e-9, keep_frac=0.2):
    P = np.asarray(P, float); A = 0.5 * (P + P.T); np.fill_diagonal(A, 0.0); A[A < eps] = 0.0
    if keep_frac and 0 < keep_frac < 1:
        pos = np.sort(A[A > 0])[::-1]
        if len(pos):
            A[A < pos[min(int(keep_frac * len(pos)), len(pos) - 1)]] = 0.0
    return A


def H1(A):
    d = A.sum(1); m2 = d.sum(); p = d[d > 0] / m2
    return float(-(p * np.log2(p)).sum()) if m2 > 0 else 0.0


def H2(A, lab):
    d = A.sum(1); m2 = d.sum(); H = 0.0
    if m2 <= 0:
        return 0.0
    for c in np.unique(lab):
        idx = np.where(lab == c)[0]; Vc = d[idx].sum()
        if Vc <= 0:
            continue
        gc = A[np.ix_(idx, lab != c)].sum(); di = d[idx]; di = di[di > 0]
        H += -(di / m2 * np.log2(di / Vc)).sum()
        if gc > 0:
            H += -(gc / m2) * np.log2(Vc / m2)
    return float(H)


def louvain(A, res=1.0, seed=0):
    G = nx.from_numpy_array(A); cs = louvain_communities(G, weight="weight", resolution=res, seed=seed)
    lab = np.full(A.shape[0], -1)
    for cid, ns in enumerate(cs):
        for n in ns:
            lab[n] = cid
    return lab


def best_gap(A, resolutions=(0.5, 1.0, 2.0, 4.0)):
    """SE-optimal gap over a resolution sweep (best case for the lever)."""
    h1 = H1(A); best = 0.0; best_lab = None
    for r in resolutions:
        lab = louvain(A, res=r); g = (h1 - H2(A, lab)) / h1 if h1 > 0 else 0.0
        if g > best:
            best, best_lab = g, lab
    return best, best_lab, h1


def knn_graph(X, k=5):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    S = Xn @ Xn.T; np.fill_diagonal(S, 0.0); A = np.zeros_like(S)
    for i in range(S.shape[0]):
        nn = np.argsort(-S[i])[:k]; A[i, nn] = S[i, nn]
    A = 0.5 * (A + A.T); A[A < 0] = 0.0
    return A


def precheck_transition(P, name="P"):
    assert _HAVE_NX, "needs networkx for SE-optimal partition"
    print(f"=== {name}: transition graph SE gap (sparsify sweep, SE-optimal partition) ===")
    overall = 0.0
    for keep in (0.1, 0.2, 0.3):
        A = symmetrize(P, keep_frac=keep)
        if A.sum() == 0:
            print(f"  keep={keep}: empty"); continue
        g, lab, h1 = best_gap(A); overall = max(overall, g)
        print(f"  keep={keep}: ncomm={lab.max()+1:2d} H1={h1:.2f} best_gap={100*g:5.1f}%")
    print(f"  -> BEST transition SE gap = {100*overall:.1f}%  ({'PASS' if overall>=0.15 else 'weak'})")
    return overall


def precheck_latents(X, name="latents"):
    assert _HAVE_NX
    print(f"=== {name}: kNN-on-latent-geometry SE gap ===")
    overall = 0.0
    for k in (3, 5, 8):
        g, _, h1 = best_gap(knn_graph(X, k=k)); overall = max(overall, g)
        print(f"  kNN k={k}: H1={h1:.2f} best_gap={100*g:5.1f}%")
    print(f"  -> BEST kNN SE gap = {100*overall:.1f}%  ({'PASS' if overall>=0.15 else 'weak'})")
    return overall


if __name__ == "__main__":
    import sys
    base = "exp/tdmpc_glass/skill_substrate"
    P = np.load(f"{base}/cartsparse_geoglass_s0_P.npz")["P"]
    protos = np.load(f"{base}/skill_substrate_cartsparse_communities.npz")["prototypes"]
    precheck_transition(P, "geoglass CartpoleSparse (32-node proto)")
    precheck_latents(protos, "geoglass 512-d SimNorm protos")
