#!/usr/bin/env python3
"""Persist the tier-2 SimNorm SE-gap numbers to JSON (fills TBD-persist-simnorm-se-json).

Re-runs the exact tier-2 analysis from scripts/se_precheck.py (analyze_real:
kmeans-128 nodes -> k-step transition graph + kNN-on-centroids graph -> SE-optimal
partition gap) on a latent dump npz, and writes the per-graph gaps to JSON instead
of only printing. Defaults match the original run that produced the 53.1%/47.2%
figures cited in docs/research/se-precheck-note.md §6 (cheetah_jumpy_k4.npz,
n_nodes=128, sub=8000, seed=0).

Control-box safe: numpy + networkx only.

Usage:
    python3 scripts/persist_se_precheck.py \
        --npz exp/tdmpc_glass/se_dump/cheetah_jumpy_k4.npz \
        --out exp/tdmpc_glass/mechcheck/se_precheck_simnorm_cheetah.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import se_precheck as sp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="exp/tdmpc_glass/se_dump/cheetah_jumpy_k4.npz")
    ap.add_argument("--out", default="exp/tdmpc_glass/mechcheck/se_precheck_simnorm_cheetah.json")
    ap.add_argument("--n_nodes", type=int, default=128)
    ap.add_argument("--sub", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    Z = d["Z"].astype(np.float32)
    Zt = d["Zt"].astype(np.float32)
    Ztk = d["Ztk"].astype(np.float32)
    rng = np.random.default_rng(args.seed)
    if len(Z) > args.sub:
        Z = Z[rng.choice(len(Z), args.sub, replace=False)]
    allz = np.concatenate([Z, Zt, Ztk], 0)
    if len(allz) > args.sub:
        allz = allz[rng.choice(len(allz), args.sub, replace=False)]
    _, C = sp._kmeans(allz, args.n_nodes, seed=args.seed)

    a = sp._dist2(Zt, C).argmin(1)
    b = sp._dist2(Ztk, C).argmin(1)
    P = np.zeros((args.n_nodes, args.n_nodes))
    for i, j in zip(a, b):
        P[i, j] += 1.0

    gap_transition = sp.precheck_transition(P, f"jumpy {d.get('env')} k-step ({args.n_nodes} nodes)")
    gap_knn = sp.precheck_latents(C, f"jumpy {d.get('env')} centroids")

    out = {
        "npz": args.npz,
        "env": str(d.get("env")),
        "jumpy_k": int(d["k"]),
        "n_latents": int(d["Z"].shape[0]),
        "latent_dim": int(d["Z"].shape[1]),
        "n_nodes": args.n_nodes,
        "sub": args.sub,
        "seed": args.seed,
        "best_gap_kstep_transition": float(gap_transition),
        "best_gap_knn_centroids": float(gap_knn),
        "tier2_verdict_best_gap": float(max(gap_transition, gap_knn)),
        "pass_threshold": 0.15,
        "note": "Persisted re-run of se_precheck.analyze_real on the original dump; "
                "backs the SimNorm tier-2 SE-gap figures in the redundancy paper.",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")
    print(json.dumps({k: v for k, v in out.items() if "gap" in k}, indent=2))


if __name__ == "__main__":
    main()
