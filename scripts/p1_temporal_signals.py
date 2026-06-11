#!/usr/bin/env python3
"""P1 — candidate signals for WHEN temporal abstraction (jumpy) pays.

Reads the per-task mech dumps (SE_DUMP=1 mech mode: Z, Zt, Ztk, err, disc,
discp_mean, discp_max) for the three anchor tasks and computes candidate
checkpoint-level signals, then compares their cross-task ordering against the
measured jumpy final-return gains:

    Ori +90% (CI-sep)  >  Pick +32% (trend)  >  Cab 0% (null)
    [exp/tdmpc_glass/mechcheck/anchor_jumpy_vs_vanilla.json]

HONESTY NOTE (in the JSON too): with n=3 tasks, ordering consistency is a
HYPOTHESIS GENERATOR, not a test (any monotone signal matches with p~1/6).
The pre-registered TEST is: the surviving candidate(s) predict the jumpy gain
on unseen tasks (PandaPickCubeCartesian + one more) BEFORE those gates run,
and the k-sweep (phasei30_jumk{2,8}) dose-response must be consistent.

Candidate signals (all from a single trained jumpy ckpt's rollout dump):
  err_med          median true k-step prediction error  (model quality at k)
  err_cv           coefficient of variation of err      (heterogeneity)
  err_rel          err_med / median latent step size    (relative difficulty)
  disc_med         median jumpy-vs-iterated disagreement (ensemble-free signal)
  disc_err_gap     disc_med / err_med                   (calibration)
  perturb_inflation discp_mean / disc                   (OOD-robustness of model)
  latent_speed_med  median ||z_{t+1}-z_t|| over policy rollouts (dynamics tempo)
  latent_speed_cv   CV of latent speed                  (burstiness / contact-ness)

Usage (EC2, numpy only):
    python3 scripts/p1_temporal_signals.py \
        --dumps Pick=exp/tdmpc_glass/se_dump/panda_fcheck.npz \
                Ori=exp/tdmpc_glass/se_dump/ori_mech.npz \
                Cab=exp/tdmpc_glass/se_dump/cab_mech.npz \
        --out exp/tdmpc_glass/mechcheck/p1_temporal_signals.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

GAINS = {"Pick": 0.32, "Ori": 0.90, "Cab": 0.00}  # measured jumpy final gains


def signals_from_dump(npz_path: str) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    out = {"npz": npz_path, "env": str(d.get("env")), "k": int(d["k"]),
           "n_states": int(d["Z"].shape[0])}
    Z = d["Z"].astype(np.float32)
    Zt = d["Zt"].astype(np.float32)
    err = np.asarray(d["err"], dtype=np.float32) if "err" in d else None
    disc = np.asarray(d["disc"], dtype=np.float32) if "disc" in d else None

    # latent tempo from consecutive policy-rollout latents (Z is in rollout order)
    speed = np.linalg.norm(np.diff(Z, axis=0), axis=1)
    out["latent_speed_med"] = float(np.median(speed))
    out["latent_speed_cv"] = float(np.std(speed) / (np.mean(speed) + 1e-9))

    if err is not None:
        out["err_med"] = float(np.median(err))
        out["err_cv"] = float(np.std(err) / (np.mean(err) + 1e-9))
        # k-step displacement as the natural scale for err
        kdisp = np.linalg.norm(d["Ztk"].astype(np.float32) - Zt, axis=1)
        out["kstep_disp_med"] = float(np.median(kdisp))
        out["err_rel"] = float(np.median(err) / (np.median(kdisp) + 1e-9))
    if disc is not None:
        out["disc_med"] = float(np.median(disc))
        if err is not None:
            out["disc_err_gap"] = float(np.median(disc) / (np.median(err) + 1e-9))
    if "discp_mean" in d and disc is not None:
        out["perturb_inflation"] = float(
            np.median(np.asarray(d["discp_mean"], dtype=np.float32))
            / (np.median(disc) + 1e-9))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="+", required=True,
                    help="Name=path.npz per task (names must match GAINS keys)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    per_task = {}
    for spec in args.dumps:
        name, path = spec.split("=", 1)
        per_task[name] = signals_from_dump(path)

    tasks = [t for t in ("Ori", "Pick", "Cab") if t in per_task]  # gain-desc order
    gain_order = sorted(tasks, key=lambda t: -GAINS[t])

    # ordering consistency per signal
    sigkeys = sorted({k for v in per_task.values() for k in v
                      if isinstance(v[k], float)})
    candidates = {}
    for k in sigkeys:
        if not all(k in per_task[t] for t in tasks):
            continue
        vals = {t: per_task[t][k] for t in tasks}
        asc = sorted(tasks, key=lambda t: vals[t])
        desc = list(reversed(asc))
        match = "+" if desc == gain_order else ("-" if asc == gain_order else None)
        candidates[k] = {"values": vals, "consistent": match}

    surviving = {k: v for k, v in candidates.items() if v["consistent"]}
    out = {
        "measured_gains": GAINS,
        "per_task_signals": per_task,
        "ordering_candidates": candidates,
        "surviving_candidates": list(surviving),
        "honesty_note": ("n=3 tasks: ordering consistency is hypothesis-generating "
                          "only (chance ~1/3 per signed direction). The TEST is the "
                          "pre-registered prediction on unseen tasks + the "
                          "phasei30_jumk{2,8} dose-response."),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")
    for k, v in candidates.items():
        mark = v["consistent"] or " "
        print(f"  [{mark}] {k}: " + "  ".join(f"{t}={v['values'][t]:.4g}" for t in tasks))
    print("surviving:", list(surviving))


if __name__ == "__main__":
    main()
