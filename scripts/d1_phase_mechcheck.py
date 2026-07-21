#!/usr/bin/env python3
"""D1 phase-gated switching-WM mechanism check (offline, CPU).

Tests whether SimNorm discrete-code "phases" carry structure a switching world model
could exploit, BEYOND the raw motion-magnitude / disagreement signals that the dead
lever-11 / F-salvage uncertainty-gate already failed to convert.

Inputs: archival se_dump *.npz with Zt (latent at t), Ztk (true latent at t+k),
err (true jumpy k-step error at t), disc (jumpy-vs-iterated disagreement), k.
SimNorm latent = 8 groups x 64-way softmax; phase code = 8 per-group argmaxes.

Three legs (all pre-registered in docs/CHANGELOG.md 2026-06-15):
  L1 phase-stratified error  : eta^2(err ~ phase-id) vs shuffled-label control.
  L2 boundary beyond motion  : partial Spearman rho(err, code-churn | displacement).
  L3 beyond uncertainty      : partial Spearman rho(err, code-churn | disp, disc).
Discriminative: stronger on Pick/Ori (jumpy-win) than Cab (jumpy-null).
"""
import numpy as np, json, sys, os, glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DUMP = ROOT / "exp" / "tdmpc_glass" / "se_dump"
OUT  = ROOT / "exp" / "tdmpc_glass" / "mechcheck"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(0)
N_BOOT = 2000
N_EP = 12              # dumps always use n_ep=12 (verified: Ori N=11904=12*992)
G, V = 8, 64           # SimNorm grouping (verified)

def rankdata(x):
    order = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[order] = np.arange(len(x))
    # average ties
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); starts = csum - cnt
    avg = (starts + csum - 1) / 2.0
    return avg[inv]

def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra*ra).sum() * (rb*rb).sum())
    return float((ra*rb).sum()/d) if d > 0 else 0.0

def partial_spearman(y, x, controls):
    """Spearman partial corr of y,x given controls (list of arrays): residualize ranks linearly."""
    ry = rankdata(y); rx = rankdata(x)
    if controls:
        C = np.column_stack([rankdata(c) for c in controls])
        C = np.column_stack([np.ones(len(y)), C])
        # residualize
        coef_y, *_ = np.linalg.lstsq(C, ry, rcond=None); ry = ry - C@coef_y
        coef_x, *_ = np.linalg.lstsq(C, rx, rcond=None); rx = rx - C@coef_x
    ry = ry - ry.mean(); rx = rx - rx.mean()
    d = np.sqrt((ry*ry).sum()*(rx*rx).sum())
    return float((ry*rx).sum()/d) if d > 0 else 0.0

def detect_episodes(Zt):
    """Return list of (start,end) index ranges per episode by top-(N_EP-1) latent jumps."""
    dist = np.linalg.norm(np.diff(Zt, axis=0), axis=1)
    cuts = np.sort(np.argsort(dist)[-(N_EP-1):]) + 1   # boundary starts
    bounds = [0, *cuts.tolist(), len(Zt)]
    eps = [(bounds[i], bounds[i+1]) for i in range(len(bounds)-1)]
    med = float(np.median(dist)); jump = float(np.median(dist[cuts-1]))
    return eps, med, jump

def kmeans(X, K, iters=50):
    idx = RNG.choice(len(X), K, replace=False); C = X[idx].copy()
    for _ in range(iters):
        d = ((X[:,None,:]-C[None,:,:])**2).sum(-1)
        lab = d.argmin(1)
        newC = np.array([X[lab==j].mean(0) if (lab==j).any() else C[j] for j in range(K)])
        if np.allclose(newC, C): C = newC; break
        C = newC
    return lab

def eta_squared(err, labels):
    grand = err.mean(); ss_tot = ((err-grand)**2).sum()
    ss_between = sum(((err[labels==j].mean()-grand)**2)*np.sum(labels==j)
                     for j in np.unique(labels) if (labels==j).any())
    return float(ss_between/ss_tot) if ss_tot > 0 else 0.0

def analyze(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    Zt = d["Zt"].astype(np.float32); Ztk = d["Ztk"].astype(np.float32)
    err = d["err"].astype(np.float32); k = int(d["k"])
    disc = d["disc"].astype(np.float32) if ("disc" in d.files and d["disc"].size==len(err)) else None
    n = min(len(Zt), len(err)); Zt, Ztk, err = Zt[:n], Ztk[:n], err[:n]
    if disc is not None: disc = disc[:n]

    eps, med_jump, reset_jump = detect_episodes(Zt)
    # discrete phase code + per-step code-flip (within episode)
    code = Zt.reshape(n, G, V).argmax(-1)                       # (n,8)
    flip = np.zeros(n)                                          # flips entering step i (within ep)
    valid_step = np.ones(n, bool)
    for (s,e) in eps:
        if e-s >= 2:
            flip[s+1:e] = (code[s+1:e] != code[s:e-1]).sum(1)
        # mark steps whose k-window stays inside this episode
    # upcoming code churn over the k-window [i, i+k); valid only if window within episode
    churn = np.full(n, np.nan); disp = np.linalg.norm(Ztk - Zt, axis=1)
    win_ok = np.zeros(n, bool)
    for (s,e) in eps:
        for i in range(s, e):
            if i+k <= e:
                churn[i] = flip[i+1:min(i+k, e)].sum() if i+1 < e else 0.0
                win_ok[i] = True
    m = win_ok & np.isfinite(churn) & np.isfinite(err) & (err>0)
    y, x, dp = err[m], churn[m], disp[m]
    dc = disc[m] if disc is not None else None

    # ---- L2 / L3 partial correlations + episode-block bootstrap ----
    ep_id = np.zeros(n, int)
    for j,(s,e) in enumerate(eps): ep_id[s:e]=j
    ep_id = ep_id[m]; uniq_ep = np.unique(ep_id)
    def boot_stat(fn):
        vals=[]
        for _ in range(N_BOOT):
            chosen = RNG.choice(uniq_ep, len(uniq_ep), replace=True)
            sel = np.concatenate([np.where(ep_id==c)[0] for c in chosen])
            vals.append(fn(sel))
        return np.percentile(vals,[2.5,50,97.5])
    raw_rho   = spearman(y, x)
    rho_disp  = partial_spearman(y, x, [dp])
    ctrl3 = [dp] + ([dc] if dc is not None else [])
    rho_dd    = partial_spearman(y, x, ctrl3)
    ci_disp = boot_stat(lambda s: partial_spearman(y[s], x[s], [dp[s]]))
    ci_dd   = boot_stat(lambda s: partial_spearman(y[s], x[s], [c[s] for c in ctrl3]))
    # shuffle control for raw coupling (break temporal alignment)
    rho_shuf = np.percentile([spearman(y, RNG.permutation(x)) for _ in range(200)],[2.5,50,97.5])

    # ---- L1 phase-stratified error: eta^2 by KMeans phase vs shuffled labels ----
    l1 = {}
    Xz = Zt[m]
    for K in (4,6,8):
        lab = kmeans(Xz, K)
        eta = eta_squared(y, lab)
        shuf = np.percentile([eta_squared(y, RNG.permutation(lab)) for _ in range(200)],[2.5,50,97.5])
        l1[f"K{K}"] = {"eta2": eta, "shuffled_ci": shuf.tolist(),
                       "pass": bool(eta > shuf[2]*1.5)}

    # ---- magnitude: median err top vs bottom churn quintile ----
    q = np.quantile(x,[0.2,0.8]); lo = y[x<=q[0]]; hi = y[x>=q[1]]
    ratio = float(np.median(hi)/np.median(lo)) if len(lo) and np.median(lo)>0 else float("nan")

    return {
        "file": os.path.basename(npz_path), "env": str(d["env"]), "k": k, "n_used": int(m.sum()),
        "reset_jump_over_median": round(reset_jump/med_jump,2),
        "raw_rho_err_churn": round(raw_rho,4),
        "rho_shuffle_ci": [round(v,4) for v in rho_shuf],
        "L2_partial_rho_given_disp": round(rho_disp,4), "L2_ci": [round(v,4) for v in ci_disp],
        "L3_partial_rho_given_disp_disc": round(rho_dd,4), "L3_ci": [round(v,4) for v in ci_dd],
        "L1_phase_stratified": l1,
        "mag_ratio_hi_lo_churn_quintile": round(ratio,3),
        "mean_flip_per_step": round(float(flip[flip>0].mean()) if (flip>0).any() else 0,3),
    }

def main():
    groups = {
        "Pick": sorted(glob.glob(str(DUMP/"ks_Pick_k8_s*_seed1_k8_mech.npz"))),
        "Ori":  sorted(glob.glob(str(DUMP/"ks_Ori_k8_s*_seed1_k8_mech.npz"))),
        "Cab":  sorted(glob.glob(str(DUMP/"ks_Cab_k8_s*_seed1_k8_mech.npz"))),
    }
    results = {}
    for name, files in groups.items():
        results[name] = [analyze(f) for f in files]
        for r in results[name]:
            print(f"[{name}] {r['file']}: rawρ={r['raw_rho_err_churn']} "
                  f"L2(|disp)={r['L2_partial_rho_given_disp']} CI{r['L2_ci']} "
                  f"L3(|disp,disc)={r['L3_partial_rho_given_disp_disc']} CI{r['L3_ci']} "
                  f"L1K6η²={r['L1_phase_stratified']['K6']['eta2']:.3f}/"
                  f"pass={r['L1_phase_stratified']['K6']['pass']} mag={r['mag_ratio_hi_lo_churn_quintile']}")
    out = OUT/"d1_phase_mechcheck.json"
    json.dump(results, open(out,"w"), indent=2)
    print("\nsaved", out)

if __name__ == "__main__":
    main()
