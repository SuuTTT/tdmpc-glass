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
    h1 = H1(A); best = -1.0; best_lab = louvain(A, res=resolutions[0])
    for r in resolutions:
        lab = louvain(A, res=r); g = (h1 - H2(A, lab)) / h1 if h1 > 0 else 0.0
        if g > best:
            best, best_lab = g, lab
    return max(best, 0.0), best_lab, h1


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


def _dist2(X, C):
    """Squared euclidean (N,K) via matmul identity — O(N*K) memory, not O(N*K*D)."""
    return (X * X).sum(1)[:, None] + (C * C).sum(1)[None, :] - 2.0 * (X @ C.T)


def _kmeans(X, n, iters=25, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), n, replace=False)].copy()
    for _ in range(iters):
        lab = _dist2(X, C).argmin(1)
        for c in range(n):
            m = lab == c
            if m.any():
                C[c] = X[m].mean(0)
    return lab, C


def analyze_real(npz_path, n_nodes=128, sub=8000, seed=0):
    """Tier-2: real rollout latents -> cluster to n_nodes -> k-step transition graph + kNN
    geometry graph -> SE gap. The actual go/no-go for the SE-k lever on the jumpy substrate."""
    assert _HAVE_NX
    d = np.load(npz_path, allow_pickle=True)
    Z = d["Z"].astype(np.float32); Zt = d["Zt"].astype(np.float32); Ztk = d["Ztk"].astype(np.float32)
    print(f"=== tier-2 real latents: {npz_path}  Z={Z.shape} k={int(d['k'])} env={d.get('env')} ===")
    rng = np.random.default_rng(seed)
    if len(Z) > sub:
        Z = Z[rng.choice(len(Z), sub, replace=False)]
    # node assignment via kmeans on the full latent cloud
    allz = np.concatenate([Z, Zt, Ztk], 0)
    if len(allz) > sub:
        allz = allz[rng.choice(len(allz), sub, replace=False)]
    _, C = _kmeans(allz, n_nodes, seed=seed)

    def nearest(x):
        return _dist2(x, C).argmin(1)
    # k-step transition graph over nodes
    a, b = nearest(Zt), nearest(Ztk)
    P = np.zeros((n_nodes, n_nodes))
    for i, j in zip(a, b):
        P[i, j] += 1.0
    print("  -- k-step transition graph --")
    g_t = precheck_transition(P, f"jumpy {d.get('env')} k-step ({n_nodes} nodes)")
    # kNN geometry graph over node centroids
    print("  -- kNN on node centroids --")
    g_k = precheck_latents(C, f"jumpy {d.get('env')} centroids")
    verdict = max(g_t, g_k)
    print(f"\n  TIER-2 VERDICT: best SE gap = {100*verdict:.1f}%  -> "
          f"{'PASS (build SE-k)' if verdict>=0.15 else 'FAIL (fall back to F)'}")
    return verdict


def _boundary_scores(A, lab):
    """Per-node fraction of edge mass crossing to a DIFFERENT community (= bottleneck/phase-boundary
    score). High b = the node sits on a community boundary (contact / phase transition)."""
    tot = A.sum(1) + 1e-12
    b = np.array([A[i, lab != lab[i]].sum() for i in range(A.shape[0])]) / tot
    return b


def mechcheck(npz_path, n_nodes=128, sub=8000, seed=0):
    """SE-k KILL-TEST: does the SE community-boundary score b(z_t) correlate with the jumpy model's
    TRUE k-step prediction error e_t? If yes, boundary-gated jump-length has a real signal to exploit
    (long k in-community where error is low, short k at boundaries where error is high). If corr~=0,
    SE-k is dead -> fall back to F. Needs an npz with a non-empty `err` array (SE_DUMP mech mode)."""
    assert _HAVE_NX
    d = np.load(npz_path, allow_pickle=True)
    if "err" not in d or len(d["err"]) == 0:
        print("no err array (run SE_DUMP with --jumpy_k matching the ckpt for mech mode)"); return None
    Zt = d["Zt"].astype(np.float32); err = d["err"].astype(np.float32)
    n = min(len(Zt), len(err)); Zt, err = Zt[:n], err[:n]
    print(f"=== SE-k mechcheck: {npz_path}  N={n} k={int(d['k'])} env={d.get('env')} ===")
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, min(n, sub), replace=False)
    Zs, es = Zt[idx], err[idx]
    _, C = _kmeans(Zs, n_nodes, seed=seed)
    node = _dist2(Zs, C).argmin(1)
    # transition graph for partition (use the sampled z_t -> need successors; approximate the partition
    # from kNN geometry of centroids, which the pre-check showed carries the structure).
    A = knn_graph(C, k=5)
    lab = best_gap(A)[1]
    bnode = _boundary_scores(A, lab)
    b = bnode[node]                      # boundary score per sampled transition
    # correlations
    bs, es2 = b - b.mean(), es - es.mean()
    pear = float((bs * es2).sum() / (np.sqrt((bs**2).sum() * (es2**2).sum()) + 1e-9))
    rb = np.argsort(np.argsort(b)); re = np.argsort(np.argsort(es))
    rb, re = rb - rb.mean(), re - re.mean()
    spear = float((rb * re).sum() / (np.sqrt((rb**2).sum() * (re**2).sum()) + 1e-9))
    hi = es[b >= np.quantile(b, 0.66)]; lo = es[b <= np.quantile(b, 0.33)]
    print(f"  Pearson(b,err)={pear:+.3f}  Spearman={spear:+.3f}")
    print(f"  k-step err: high-boundary tertile={hi.mean():.3f}  low-boundary tertile={lo.mean():.3f}  "
          f"ratio={hi.mean()/(lo.mean()+1e-9):.2f}x")
    ok = spear > 0.15 and hi.mean() > 1.10 * lo.mean()
    print(f"  VERDICT: {'PASS — boundary score tracks k-step error -> build SE-k gate' if ok else 'WEAK/FAIL — boundary score does NOT track error -> fall back to F (uncertainty gate)'}")
    return dict(pearson=pear, spearman=spear, hi=float(hi.mean()), lo=float(lo.mean()), ok=ok)


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / (np.sqrt((ra**2).sum() * (rb**2).sum()) + 1e-9))


def fcheck(npz_path):
    """F (uncertainty-gated horizon) SALVAGE-TEST. Uses ensemble-free disagreement =
    ||jumpy_pred - iterated_1step_pred||. (1) Does disagreement track TRUE k-step error on pi actions?
    (2) Under MPPI-perturbed actions, is disagreement high-variance (=> hard regions a horizon-gate can
    exploit)? PASS if disc tracks err (spearman>0.3) AND perturbed disagreement has real spread
    (CV>0.5 and max>>pi-action disc)."""
    d = np.load(npz_path, allow_pickle=True)
    err = d["err"].astype(np.float64); disc = d["disc"].astype(np.float64)
    dpm = d.get("discp_max"); dpme = d.get("discp_mean")
    dpm = dpm.astype(np.float64) if dpm is not None else np.zeros(0)
    dpme = dpme.astype(np.float64) if dpme is not None else np.zeros(0)
    n = min(len(err), len(disc)); err, disc = err[:n], disc[:n]
    print(f"=== F salvage-test: {npz_path}  N={n} k={int(d['k'])} env={d.get('env')} ===")
    sp = _spearman(disc, err)
    print(f"  (1) disagreement vs TRUE err: spearman={sp:+.3f}  (signal valid if >0.3)")
    print(f"      err  mean={err.mean():.4f} std={err.std():.4f} CV={err.std()/(err.mean()+1e-9):.2f}")
    print(f"      disc(pi) mean={disc.mean():.4f} std={disc.std():.4f} CV={disc.std()/(disc.mean()+1e-9):.2f}")
    cv_p = float(dpm.std() / (dpm.mean() + 1e-9)) if len(dpm) else 0.0
    infl = float(dpm.mean() / (disc.mean() + 1e-9)) if len(dpm) else 0.0
    print(f"  (2) perturbed disagreement: mean_max={dpm.mean() if len(dpm) else 0:.4f} "
          f"CV={cv_p:.2f}  inflation(max/pi)={infl:.2f}x")
    ok = sp > 0.3 and cv_p > 0.5 and infl > 1.3
    print(f"  VERDICT: {'PASS — disagreement tracks error AND perturbation creates hard regions -> build F (disagreement-gated horizon, ensemble-free)' if ok else 'WEAK/FAIL — adaptive-k has no exploitable signal even OOD -> abandon adaptive-k family'}")
    return dict(spearman=sp, err_cv=float(err.std()/(err.mean()+1e-9)), perturb_cv=cv_p, inflation=infl, ok=ok)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "fcheck" and sys.argv[2].endswith(".npz"):
        fcheck(sys.argv[2]); sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "mech" and sys.argv[2].endswith(".npz"):
        mechcheck(sys.argv[2], n_nodes=int(sys.argv[3]) if len(sys.argv) > 3 else 128)
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1].endswith(".npz"):
        analyze_real(sys.argv[1], n_nodes=int(sys.argv[2]) if len(sys.argv) > 2 else 128)
        sys.exit(0)
    base = "exp/tdmpc_glass/skill_substrate"
    P = np.load(f"{base}/cartsparse_geoglass_s0_P.npz")["P"]
    protos = np.load(f"{base}/skill_substrate_cartsparse_communities.npz")["prototypes"]
    precheck_transition(P, "geoglass CartpoleSparse (32-node proto)")
    precheck_latents(protos, "geoglass 512-d SimNorm protos")
