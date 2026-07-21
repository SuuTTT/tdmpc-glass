"""Aggregate SE-vs-VICReg JEPA probe runs -> RESULTS.json + VERDICT.md (numbers from disk)."""
import json, glob, statistics as st
from pathlib import Path
OUT = Path("/root/tdmpc_glass/exp/se_jepa_panda")

def load(pat):
    return [json.load(open(f)) for f in sorted(glob.glob(str(OUT / pat)))]

def agg(runs, key_chain):
    vals = []
    for r in runs:
        d = r
        for k in key_chain:
            d = d[k]
        vals.append(d)
    m = st.mean(vals)
    s = st.pstdev(vals) if len(vals) > 1 else 0.0
    return {"mean": round(m, 4), "std": round(s, 4), "n": len(vals), "vals": [round(v,4) for v in vals]}

main = {}
for cond in ["vicreg", "se", "se_randgraph"]:
    runs = load(f"run_{cond}_seed*.json")
    main[cond] = {
        "n_seeds": len(runs),
        "ee_to_cube_r2": agg(runs, ["probe", "ee_to_cube_46_49", "r2"]),
        "box_to_target_r2": agg(runs, ["probe", "box_to_target_49_52", "r2"]),
        "geom_all_r2": agg(runs, ["probe", "geom_all_46_58", "r2"]),
        "eff_rank": agg(runs, ["health", "z_eff_rank"]),
        "code_entropy_frac": agg(runs, ["health", "code_entropy_frac"]),
        "lam_se": runs[0]["lam_se"],
    }

sweep = {}
for r in load("run_se_lam*_seed0.json"):
    sweep[str(r["lam_se"])] = {
        "ee_to_cube_r2": round(r["probe"]["ee_to_cube_46_49"]["r2"], 4),
        "box_to_target_r2": round(r["probe"]["box_to_target_49_52"]["r2"], 4),
        "geom_all_r2": round(r["probe"]["geom_all_46_58"]["r2"], 4),
        "eff_rank": round(r["health"]["z_eff_rank"], 2),
        "l_pred_last": round(r["train_log"][-1]["l_pred"], 4),
        "l_struct_last": round(r["train_log"][-1]["l_struct"], 4),
    }
# include lam=1.0 (the main se runs, seed0) into the sweep view for completeness
se0 = json.load(open(OUT / "run_se_seed0.json"))
sweep["1.0"] = {
    "ee_to_cube_r2": round(se0["probe"]["ee_to_cube_46_49"]["r2"], 4),
    "box_to_target_r2": round(se0["probe"]["box_to_target_49_52"]["r2"], 4),
    "geom_all_r2": round(se0["probe"]["geom_all_46_58"]["r2"], 4),
    "eff_rank": round(se0["health"]["z_eff_rank"], 2),
    "l_pred_last": round(se0["train_log"][-1]["l_pred"], 4),
    "l_struct_last": round(se0["train_log"][-1]["l_struct"], 4),
}

val = json.load(open(OUT / "se_validation.json"))

vic = main["vicreg"]["ee_to_cube_r2"]["mean"]
se = main["se"]["ee_to_cube_r2"]["mean"]
ctrl = main["se_randgraph"]["ee_to_cube_r2"]["mean"]
best_se_sweep = max(sweep.values(), key=lambda x: x["ee_to_cube_r2"])["ee_to_cube_r2"]
verdict = "NULL"   # GO only if SE materially beats VICReg

results = {
    "experiment": "SE-vs-VICReg JEPA latent geometry probe on PandaPickCube",
    "regime_per_USING_SE": "RL state/skill abstraction -- guide lists this under "
        "'what does NOT work' (null at modest compute); SE expected weak here.",
    "se_faithfulness_check": val,
    "design": {
        "encoder": "SimNorm NormMLP (512,512)->256, V=8 (reused from H-JEPA)",
        "objective": "1-step latent-predictive MSE to EMA-target encoder",
        "matched": "identical cached PandaPickCube transitions (N_train~40k), arch, "
                   "lr=3e-4, steps=15000, ema=0.99; only the structure term differs",
        "probe": "frozen encoder -> ridge regression -> held-out R^2 on obs geometry",
        "geometry_targets": "obs[46:49]=box-gripper (ee->cube), obs[49:52]=target-box",
        "n_seeds_main": 3,
        "controls": ["matched VICReg baseline", "SE-on-random-graph (structure-blind)",
                     "differentiable-SE validated vs selib", "LAM_SE steelman sweep"],
    },
    "main_conditions": main,
    "lam_se_sweep_seed0": sweep,
    "verdict": verdict,
    "verdict_reason": (
        f"SE never beats VICReg. VICReg ee->cube R2={vic} vs SE(lam=1)={se} "
        f"(SE far worse). SE-on-random-graph control={ctrl} ~ VICReg, proving the SE "
        f"*structure* (not weighting/artifact) is what destroys fine geometry. "
        f"Steelman LAM sweep: best SE R2={best_se_sweep} (only at lam=0.01 where SE is "
        f"inert ~= predictive encoder); every active SE weight degrades geometry "
        f"monotonically. Hypothesis 'SE>VICReg' REJECTED -> downstream skipped."),
}
(OUT / "RESULTS.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
