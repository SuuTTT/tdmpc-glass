"""CSV discovery, eval summaries, and phase / CI statistics for the dashboard.

Pure data-layer: walks the local exp mirror, parses per-seed CSVs, and produces
the payloads behind /api/curves, /api/phases, and /api/phase_ci. No Flask here.
"""
from __future__ import annotations

import csv
import math
import re
import time
from pathlib import Path

REPO = Path("/home/ubuntu/tdmpc-glass")
LOCAL_EXP = REPO / "exp" / "tdmpc_glass"
MIRROR = LOCAL_EXP / "remote_mirror"

# Keyed by CANONICAL phase name (after canonical_phase() normalization).
PHASE_NOTES: dict[str, str] = {
    # Iter 1 baselines
    "pre_phase1":   "Very early Glass baseline runs",
    "phase1b":      "Glass baseline: [438,526,294,187,562] mean=401",
    "phase1c":      "Act-noise anneal 0.30->0.10; FALSIFIED: hurts winners",
    "phase2":       "Early iteration 2 runs",
    # Iter 2 (phases d-h)
    "phased":       "H=5+noise=0.40 (Warp-901) or H=5-only (plateau 199); FALSIFIED",
    "phasee":       "Q-reset (impl bug: zeroed all opt state); FALSIFIED",
    "phasef":       "Smooth=1e-3; s1=571 first>500 ever; s2-5 plateau 255-284",
    "phaseg":       "Consistency coef=1.0; best 482 (s2), none>500",
    "phaseh":       "Smooth=1e-3 + ccoef=1.0; peak 490, plateau",
    "phasei":       "Smooth=1e-4 (too weak); peak 312",
    # Iter 3 (phases j-o)
    "phasej":       "Curriculum smooth (off<250k); s2=518 winner; 12% hit rate",
    "phasek":       "Smooth + lambda_temporal=0.05 (over-reg); peak 292",
    "phasel":       "tdmpc2 + smooth, Glass zeroed; peak 289; Glass IS needed",
    "phasem":       "Python-cond smooth graph; s4 K=3->K=4 rescued",
    "phasen":       "Proto temp=0.4 sharper; basin still K=3; FALSIFIED",
    "phaseo":       "Glass OFF after 2M hybrid; s3=577 winner; 1/3>500",
    # Iter 4 (paths 1,5,7,9,10,P,Pa)
    "phasep":       "EXPL_UNTIL=500k random explore; s4=538 slow-burn; 1/3>500",
    "phaset":       "Knee penalty reward shaping; 2/3>500, max=612 G2 (benchmark-unfair)",
    "phasev":       "Cluster soft-dist as pi/q obs (Path 7); peak 232; FALSIFIED (drift)",
    "phasex":       "NS=2048 MPPI; s3=523, s8~500; ~40% G1 hit rate",
    "phasex_ns1024":"NS=1024 compromise for 6GB GPU; lower ceiling than 2048",
    "phasey":       "Hierarchical Glass K_super=4 (Path 10); peak 462; 0/3>500",
    "phasePP":      "Cluster entropy intrinsic (static 0.1); FALSIFIED: peak 91 collapsed",
    "phasePa":      "Cluster entropy intrinsic (decayed [500k,3M]->0); FALSIFIED: peak 25",
    # Iter 6
    "phasez":       "Vanilla tdmpc2 baseline (NS=512, no Glass, no shaping)",
    "phaseq_knee":  "Knee penalty + NS=2048 no Glass; 4/12>500 (33%), max=557",
    "phaser1_soft": "Soft stand bonus 0.1 annealed->0@3M; 1/5>500 (20%), max=553",
    "phaser2_gait": "Gait fall penalty + action smooth; 1/4>500, max=510",
    "phaserstack":  "ALL shaping stacked; COLLAPSED (MPPI~5-16, all seeds)",
    "phaserstack_nosmooth": "Stack ablation A: drop action_smooth",
    "phaserstack_nosoft":   "Stack ablation B: drop soft_stand_bonus",
    # Iter 7 - benchmark-fair K_UPDATE sweep
    "phaseaa_codex_tdmpc2_k64":  "K_UPDATE=64 tdmpc2; ~0.25 updates/env-step",
    "phaseaa_codex_tdmpc2_k128": "K_UPDATE=128 tdmpc2; ~0.5 updates/env-step",
    "phaseaa_codex_tdmpc2_k256": "K_UPDATE=256 tdmpc2; ~1.0 update/env-step (official rate)",
    "phaseab_codex_tdmpc2_5seed":"K_UPDATE winner, 5-seed vanilla tdmpc2",
    "phaseac_codex_glass_5seed": "K_UPDATE winner, 5-seed Glass vs tdmpc2 comparison",
    "phasead_codex_explmix":     "Fair expl-mix: random action prob 1->0 over 2M steps",
    "phasear_restart_K128": "Auto-restart low early returns; tests whether bad basin seeds can be rescued by restart",
    "phaseg2_tempstab_0.05": "Glass V2 temp-stability 0.05; tested stronger temporal assignment stability",
    # Iter 9 - quick probes / seed promotion
    "phasei9m": "Phase1b with Glass off after 2M, no temp-stability/no smooth; current off-schedule handoff probe",
    "phasei9g": "Warmup 500k + temp-stability 0.01 + latent smooth; tests delayed Glass pressure",
    "phasei9j": "No latent smooth + temp-stability 0.01; tests whether smoothing was blocking good basins",
    "phasei9l": "Phase1b-style Glass + temp-stability 0.01; tests stabilizing the original Glass winner recipe",
    "phasei9n": "Phase1b K=128 fill/rerun; estimates baseline variance under the current queue code",
    "phasei9q": "Phase1b + temp-stability 0.01 with Glass off after 2M; tests late policy consolidation",
    "phasei9r": "Phase1b with Glass off after 1M; tests earlier handoff from Glass shaping to actor learning",
    "phasei9s": "Phase1b with Glass off after 1.5M; midpoint handoff between i9r and i9q",
    "phasei9t": "Phase1b off after 1.5M on hard seed 4; stress-test midpoint handoff",
    "phasei9u": "Phase1b off after 3M on hard seed 4; tests longer Glass guidance before handoff",
    # Iter 10 - clean confirmations from Iter 9 handoff signal
    "phasei10a": "Clean confirmation: Phase1b Glass handoff, off after 1M; tests 5/5 G1 candidate",
    "phasei10b": "Clean confirmation: Phase1b Glass handoff, off after 1.5M; tests midpoint handoff robustness",
    "phasei10c": "Fresh clean 5-seed confirmation: Phase1b Glass handoff, off after 1M; reruns away from interrupted/unstable boxes",
    "phasei10d": "Long clean confirmation: Phase1b Glass handoff, off after 1M; 12M cap and patience disabled for >8h runs",
    # Smoke tests
    "smoke":        "Smoke test (hardware validation only)",
}


# ─── Phase name normalization ─────────────────────────────────────────────

# Strips device/run-variant suffixes so seeds from the same logical experiment
# but run on different hardware are grouped under one canonical phase name.
# Examples: phasex_local → phasex, phasem_remote → phasem, phasep_remote_3m → phasep
_CANON_DEVICE_RE = re.compile(
    r"(?:_(?:local|remote|baseline))*"   # _local, _remote, _baseline
    r"(?:_(?:\d*x)?\d+(?:ti|lap)?)*"    # _4060, _3060ti, _2x3060, _2080ti
    r"(?:_gpu\d+)*"                       # _gpu0, _gpu1
    r"(?:_3m)?"                           # _3m after _remote
    r"$",
    re.IGNORECASE,
)
_CANON_VERSION_RE = re.compile(
    r"_v\d+(?:_[a-z0-9]+)*$",
    re.IGNORECASE,
)


def canonical_phase(phase: str) -> str:
    """Strip device/hardware and run-variant suffixes to get canonical family name.

    Preserved: _ns1024, _nosmooth, _nosoft, _codex_*, _knee, _soft, _gait, _stack.
    """
    iter_probe = re.match(r"^(phasei\d+[a-z])(?:_|$)", phase, flags=re.IGNORECASE)
    if iter_probe:
        return iter_probe.group(1)
    result = _CANON_DEVICE_RE.sub("", phase)
    result = _CANON_VERSION_RE.sub("", result)
    return result or phase


# ─── CSV discovery ────────────────────────────────────────────────────────

def discover_csvs():
    """Walk LOCAL_EXP (incl. remote_mirror) for HopperHop_phase*/seed_*.csv.

    Deduplicates by (phase, seed): if the same phase+seed appears in both the
    local exp tree and one or more remote mirrors, the one with the latest mtime
    AND the largest file size wins (size is a proxy for "more eval rows logged").
    """
    by_key: dict[tuple, dict] = {}
    for csv_path in LOCAL_EXP.rglob("*_phase*/seed_*.csv"):  # any task: HopperHop/Humanoid/Walker/Cheetah…
        name = csv_path.name
        if re.search(r"_v\d+_|_partial_|_died_|_final_|_done_|_diag\.csv$", name):
            continue
        try:
            st = csv_path.stat()
        except OSError:
            continue
        if st.st_size < 30:
            continue  # < 30 bytes = header only or smaller
        if time.time() - st.st_mtime > 7 * 86400:
            continue
        # Strip the leading task prefix (CamelCase task name + "_") so a curve's `phase`
        # equals its output TAG for ALL tasks — not just HopperHop. This keeps it in sync
        # with the active-runs list (which keys on the tag), so "running only" matches DMC
        # curves too (iter-14 fix: was .replace("HopperHop_","") which only worked for Hopper).
        phase_dir = re.sub(r'^[A-Z][A-Za-z0-9]*_', '', csv_path.parent.name)
        seed = csv_path.stem.replace("seed_", "")
        rel = csv_path.relative_to(LOCAL_EXP)
        box = "local"
        if str(rel).startswith("remote_mirror/"):
            box = str(rel).split("/")[1]
        key = (phase_dir, seed)
        cand = {"phase": phase_dir, "seed": seed, "box": box,
                "path": str(csv_path), "mtime": st.st_mtime, "size": st.st_size}
        prev = by_key.get(key)
        if prev is None or (cand["size"], cand["mtime"]) > (prev["size"], prev["mtime"]):
            by_key[key] = cand
    found = list(by_key.values())
    found.sort(key=lambda r: (r["phase"], int(r["seed"]) if r["seed"].isdigit() else 99999))
    return found


def read_curve(csv_path):
    """Read CSV → list of {step, reward, eval_type}. Returns [] on parse error."""
    points = []
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    step = int(float(row.get("step", 0)))
                    rew = float(row.get("reward", 0))
                    et = row.get("eval_type", "")
                    points.append({"step": step, "reward": rew, "eval_type": et})
                except (ValueError, TypeError):
                    continue
    except Exception:
        pass
    return points


def eval_summary(csv_path):
    """Return per-seed pi/mppi summary derived from the eval-type CSV."""
    stats = {
        "best_pi": -1.0,
        "best_pi_step": -1,
        "best_mppi": -1.0,
        "best_mppi_step": -1,
        "best_any": -1.0,
        "best_any_step": -1,
        "best_any_selector": None,
        "last_pi": -1.0,
        "last_pi_step": -1,
        "last_mppi": -1.0,
        "last_mppi_step": -1,
        "pi_minus_mppi_last": None,
    }
    pairs: dict[int, dict[str, float]] = {}
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                et = row.get("eval_type")
                if et not in ("pi", "mppi"):
                    continue
                try:
                    r = float(row.get("reward", 0))
                    s = int(float(row.get("step", 0)))
                except (ValueError, TypeError):
                    continue
                key_best = f"best_{et}"
                key_best_step = f"best_{et}_step"
                key_last = f"last_{et}"
                key_last_step = f"last_{et}_step"
                if r > stats[key_best]:
                    stats[key_best] = r
                    stats[key_best_step] = s
                stats[key_last] = r
                stats[key_last_step] = s
                pair = pairs.setdefault(s, {})
                pair[et] = r
    except Exception:
        return stats

    if stats["best_pi"] >= 0 or stats["best_mppi"] >= 0:
        if stats["best_pi"] >= stats["best_mppi"]:
            stats["best_any"] = stats["best_pi"]
            stats["best_any_step"] = stats["best_pi_step"]
            stats["best_any_selector"] = "pi"
        else:
            stats["best_any"] = stats["best_mppi"]
            stats["best_any_step"] = stats["best_mppi_step"]
            stats["best_any_selector"] = "mppi"

    last_shared_step = -1
    for step, pair in pairs.items():
        if "pi" in pair and "mppi" in pair and step >= last_shared_step:
            last_shared_step = step
            stats["pi_minus_mppi_last"] = pair["pi"] - pair["mppi"]
    return stats


MIN_COUNTED_RESULT_STEPS = 4_000_000
PROMISING_EARLY_REWARD = 500.0


def eval_is_countable(summary: dict, min_steps: int = MIN_COUNTED_RESULT_STEPS) -> bool:
    """Return True once a run is mature enough for aggregate phase statistics.

    Short interrupted runs are useful for debugging, but they should not change
    phase means, G1 rates, or promising-phase ranking unless they already cross
    the G1 bar. Early G1s are useful signal and should remain visible.
    """
    last_step = max(summary.get("last_pi_step") or -1, summary.get("last_mppi_step") or -1)
    return last_step >= min_steps or (summary.get("best_any") or -1) >= PROMISING_EARLY_REWARD


def fmt_metric(v):
    return round(v, 1) if isinstance(v, (int, float)) and v >= 0 else None


def read_diag_last_mppi(diag_path: str) -> dict | None:
    """Return the last mppi row of a _diag.csv as a dict, or None on any error."""
    result = None
    try:
        with open(diag_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("eval_type") == "mppi":
                    result = {
                        "standing_rate": float(row.get("standing_rate", 0)),
                        "fall_count": float(row.get("fall_count", 0)),
                        "ttf": float(row.get("time_to_first_full", 1000)),
                        "full_reward_rate": float(row.get("full_reward_rate", 0)),
                    }
    except Exception:
        pass
    return result


def find_active_csv_for(box: str, seed: str):
    """Locate the per-box CSV for a given seed that's actively being written.

    Strategy: look at the deduped CSV list (discover_csvs already filters to
    last 7 days). Prefer entries whose mirror box matches; fall back to the
    local exp tree. Return the freshest match, or None.
    """
    if not seed:
        return None
    matches = [c for c in discover_csvs()
               if c["seed"] == str(seed) and (c["box"] == box or box == "local")]
    if not matches:
        return None
    # most recently modified wins
    matches.sort(key=lambda r: r["mtime"], reverse=True)
    return matches[0]


# ─── Phase / CI aggregation (payload builders behind the routes) ───────────

def build_curves(phase_filter: str | None):
    """List of per-seed curve dicts (downsampled to <=200 points)."""
    out = []
    for c in discover_csvs():
        if phase_filter and phase_filter not in c["phase"]:
            continue
        pts = read_curve(c["path"])
        if len(pts) > 200:
            step = len(pts) // 200
            pts = pts[::step]
        out.append({**c, "points": pts})
    return out


def build_phases():
    """Per-CANONICAL-phase aggregated best-any stats."""
    csvs = discover_csvs()
    by_canon: dict[str, dict] = {}
    for c in csvs:
        canon = canonical_phase(c["phase"])
        entry = by_canon.setdefault(canon, {"paths": [], "variants": set()})
        entry["paths"].append(c["path"])
        if c["phase"] != canon:
            entry["variants"].add(c["phase"])
    out = []
    for canon, info in sorted(by_canon.items()):
        paths = info["paths"]
        variants = sorted(info["variants"])
        bests = []
        for path in paths:
            summary = eval_summary(path)
            if not eval_is_countable(summary):
                continue
            best = summary["best_any"]
            if best >= 0:
                bests.append(best)
        n = len(bests)
        mean_b = sum(bests) / n if n else None
        std_b = (sum((b - mean_b) ** 2 for b in bests) / n) ** 0.5 if n > 1 else None
        mx = max(bests) if bests else None
        n_g1 = sum(1 for b in bests if b >= 500)
        out.append({
            "phase": canon,
            "variants": variants,
            "n_seeds": len(paths),
            "n_with_data": n,
            "mean_best": round(mean_b, 1) if mean_b is not None else None,
            "std_best": round(std_b, 1) if std_b is not None else None,
            "max_best": round(mx, 1) if mx is not None else None,
            "n_g1": n_g1,
            "notes": PHASE_NOTES.get(canon, ""),
        })
    out.sort(key=lambda r: -(r["mean_best"] or -1))
    return out


def _phase_matches(canon: str, tokens: list[str]) -> bool:
    """True if canon phase matches any of the filter tokens (substring)."""
    if not tokens:
        return True
    return any(t in canon for t in tokens)


def build_phase_ci(raw_filter: str):
    """Per-CANONICAL-phase 95% CI curves (filtered by comma-separated tokens)."""
    csvs = discover_csvs()
    raw_filter = (raw_filter or "").strip()
    tokens = [t.strip() for t in raw_filter.split(",") if t.strip()] if raw_filter else []

    by_canon: dict[str, list] = {}
    for c in csvs:
        canon = canonical_phase(c["phase"])
        if tokens and not _phase_matches(canon, tokens):
            continue
        pts = [p for p in read_curve(c["path"]) if p["eval_type"] == "mppi"]
        if pts:
            by_canon.setdefault(canon, []).append(pts)

    # When no filter, cap at top-10 by max reward
    if not tokens and len(by_canon) > 10:
        def _max_reward(curves):
            return max(p["reward"] for pts in curves for p in pts)
        ranked = sorted(by_canon.items(), key=lambda kv: -_max_reward(kv[1]))
        by_canon = dict(ranked[:10])

    out = []
    for canon, seed_curves in sorted(by_canon.items()):
        all_steps = sorted({p["step"] for pts in seed_curves for p in pts})
        if len(all_steps) > 150:
            stride = max(1, len(all_steps) // 150)
            all_steps = all_steps[::stride]
        rows = []
        for qs in all_steps:
            vals = []
            for pts in seed_curves:
                v = next((p["reward"] for p in reversed(pts) if p["step"] <= qs), None)
                if v is not None:
                    vals.append(v)
            if not vals:
                continue
            n = len(vals)
            mean = sum(vals) / n
            if n > 1:
                var = sum((v - mean) ** 2 for v in vals) / (n - 1)
                se = math.sqrt(var / n)
                margin = 1.96 * se
            else:
                margin = 0.0
            rows.append({"step": qs, "mean": round(mean, 2),
                         "lower": round(mean - margin, 2),
                         "upper": round(mean + margin, 2), "n": n})
        if rows:
            out.append({"phase": canon, "n_seeds": len(seed_curves), "rows": rows,
                        "notes": PHASE_NOTES.get(canon, "")})
    return out
