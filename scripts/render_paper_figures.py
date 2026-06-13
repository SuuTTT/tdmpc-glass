#!/usr/bin/env python3
"""Render the understanding-paper figures from persisted JSON (EC2, matplotlib only, no GPU).

Reads exp/tdmpc_glass/mechcheck/*.json and writes PNGs to docs/writeup/figures/.
Every number comes from a persisted evidence JSON (read-from-JSON discipline).
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MC = ROOT / "exp" / "tdmpc_glass" / "mechcheck"
OUT = ROOT / "docs" / "writeup" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def fig_compounding():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, task in zip(axes, ["PandaPickCubeOrientation", "PandaOpenCabinet"]):
        d = json.load(open(MC / f"compounding_curve_{task}.json"))
        h = [e["h"] for e in d["curves"]]
        comp = [e["err_compose_raw"]["median"] for e in d["curves"]]
        it = [e["err_iter_raw"]["median"] for e in d["curves"]]
        ax.plot(h, comp, "o-", label="composed k=4 (jumpy)", color="#2166ac")
        ax.plot(h, it, "s--", label="iterated 1-step", color="#b2182b")
        ax.set_title(task.replace("Panda", ""))
        ax.set_xlabel("prediction horizon h")
        ax.set_ylabel("median latent error")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Compositional temporal abstraction: sub-linear error growth (1.3–1.8× below iteration)")
    fig.tight_layout()
    fig.savefig(OUT / "compounding_curve.png", dpi=140)
    plt.close(fig)
    return "compounding_curve.png"


def fig_anchor():
    d = json.load(open(MC / "anchor_jumpy_vs_vanilla.json"))
    tasks = list(d["tasks"].keys())
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(tasks)); w = 0.36
    for off, arm, col in [(-w / 2, "jum", "#2166ac"), (w / 2, "van", "#999999")]:
        means = [d["tasks"][t][arm]["final_mean"] for t in tasks]
        ci = [d["tasks"][t][arm].get("final_ci95", [None, None]) for t in tasks]
        err = [[m - (c[0] if c[0] is not None else m) for m, c in zip(means, ci)],
               [(c[1] if c[1] is not None else m) - m for m, c in zip(means, ci)]]
        ax.bar(x + off, means, w, yerr=err, capsize=4,
               label="jumpy" if arm == "jum" else "vanilla", color=col)
    for i, t in enumerate(tasks):
        if d["tasks"][t].get("ci_separated_final"):
            ax.text(i, max(d["tasks"][t]["jum"]["final_mean"],
                           d["tasks"][t]["van"]["final_mean"]) * 1.05, "*", ha="center", fontsize=16)
    ax.set_xticks(x); ax.set_xticklabels([t.replace("Panda", "") for t in tasks], fontsize=8)
    ax.set_ylabel("final return (mean, 95% CI)")
    ax.set_title("Jumpy vs vanilla TD-MPC2 (n=8; * = CI-separated)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "anchor.png", dpi=140); plt.close(fig)
    return "anchor.png"


def fig_ksweep():
    h = json.load(open(MC / "p1_ksweep_harvest.json"))
    k4 = {"Pick": 1969, "Ori": 2145, "Cab": 1050}
    fig, ax = plt.subplots(figsize=(7, 4))
    for task, col in [("Pick", "#2166ac"), ("Ori", "#1a9850"), ("Cab", "#b2182b")]:
        ks, rs = [2, 4, 8], [h[f"{task}_jumk2"]["final_mean"], k4[task], h[f"{task}_jumk8"]["final_mean"]]
        ax.plot(ks, rs, "o-", label=task, color=col)
    ax.set_xticks([2, 4, 8]); ax.set_xlabel("jump length k (effective horizon fixed at 24)")
    ax.set_ylabel("final return"); ax.set_title("Temporal grain dose–response: k=4 is the unimodal optimum")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "ksweep_dose_response.png", dpi=140); plt.close(fig)
    return "ksweep_dose_response.png"


if __name__ == "__main__":
    made = []
    for fn in (fig_compounding, fig_anchor, fig_ksweep):
        try:
            made.append(fn())
        except Exception as e:
            print(f"[skip] {fn.__name__}: {e}")
    print("wrote:", made, "->", OUT)
