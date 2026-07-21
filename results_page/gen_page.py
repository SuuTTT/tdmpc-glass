#!/usr/bin/env python3
"""Generate a live results page for the TD-MPC-Glass DMC benchmark.

Reads benchmark CSVs (task,seed,step,reward) from exp/benchmark/, groups by
(algo, task, seed), and writes index.html + learning-curve PNGs into site/.

Verification discipline: every number comes from a real CSV (or a verified
static source labelled as such). Missing cells render as "-". n is always shown.
"""
import os, glob, csv, json, math, datetime, html
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.environ.get("BENCH_DIR", "/root/helios-rl/exp/benchmark")
SITE = os.path.join(HERE, "site")
CURVES = os.path.join(SITE, "curves")
os.makedirs(CURVES, exist_ok=True)

TASKS = ["AcrobotSwingup", "BallInCup", "CartpoleBalance", "CartpoleSwingup",
         "CheetahRun", "FingerSpin", "FingerTurnEasy", "FingerTurnHard",
         "HopperHop", "HopperStand", "PendulumSwingup", "ReacherEasy",
         "ReacherHard", "WalkerRun", "WalkerStand", "WalkerWalk"]

# World-model algos read live from CSVs.
WM_ALGOS = ["tdmpc2", "tdmpc-glass"]

# ---------------------------------------------------------------------------
# Static / verified-elsewhere columns (NEVER fabricated; sources noted on page)
# ---------------------------------------------------------------------------
# PPO: from exp/tdmpc_glass/dmc_ppo_vs_tdmpc2/RESULTS.json (peak over curve) and
# its logs. Only a handful of DMC tasks were run, at 50-285M env steps. Values
# are PEAK return (robust to PPO terminal collapse); budget labelled.
PPO_STATIC = {
    "AcrobotSwingup": {"val": 268.4, "budget": "285M", "src": "ppo log peak"},
    "CheetahRun":     {"val": 928.3, "budget": "285M", "src": "RESULTS.json peak"},
    "FingerTurnHard": {"val": 967.9, "budget": "285M", "src": "ppo log peak"},
    "HopperHop":      {"val": 33.5,  "budget": "285M", "src": "RESULTS.json peak"},
    "WalkerWalk":     {"val": 970.1, "budget": "79M",  "src": "RESULTS.json peak"},
}
# abstraction (analytic controller + learned residual): only PendulumSwingup
# verified (Part 45, 3 seeds, ckpt-eval). Acrobot abstraction was NOT run.
ABS_STATIC = {
    "PendulumSwingup": {"mean": 835.0, "std": 14.0, "n": 3,
                        "src": "Part 45 residual_pendulum, 3 seeds ~835+/-10"},
}

PAGE_TITLE = "TD-MPC-Glass - DMC Benchmark (live)"


def parse_csv(path):
    """Return list of (seed, step, reward) rows from a benchmark CSV."""
    rows = []
    try:
        with open(path, newline="") as f:
            r = csv.DictReader(f)
            for d in r:
                try:
                    seed = int(float(d["seed"]))
                    step = int(float(d["step"]))
                    rew = float(d["reward"])
                except (KeyError, ValueError, TypeError):
                    continue
                rows.append((seed, step, rew))
    except OSError:
        return []
    return rows


def collect():
    """Build {algo: {task: {seed: [(step,reward),...]}}} from fs_ CSVs only."""
    data = {a: defaultdict(lambda: defaultdict(list)) for a in WM_ALGOS}
    for path in sorted(glob.glob(os.path.join(BENCH, "*.csv"))):
        base = os.path.basename(path)
        # Identify algo by the unambiguous "<algo>_<task>_fs_<algo>_..." pattern.
        algo = None
        for a in WM_ALGOS:
            if base.startswith(a + "_") and ("_fs_" + a + "_") in base:
                algo = a
                break
        if algo is None:
            continue
        # task = the canonical DMC task this file belongs to
        task = None
        for t in TASKS:
            if base.startswith(f"{algo}_{t}_fs_"):
                task = t
                break
        if task is None:
            continue
        for seed, step, rew in parse_csv(path):
            data[algo][task][seed].append((step, rew))
    # sort each seed series by step
    for a in data:
        for t in data[a]:
            for s in data[a][t]:
                data[a][t][s].sort()
    return data


def final_returns(data, algo, task):
    """List of final-checkpoint returns, one per seed (empty if none)."""
    out = []
    seeds = data.get(algo, {}).get(task, {})
    for s in sorted(seeds):
        series = seeds[s]
        if series:
            out.append(series[-1][1])
    return out


def stats(vals):
    n = len(vals)
    if n == 0:
        return None
    arr = np.array(vals, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0)) if n > 1 else 0.0
    ci = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
    return {"n": n, "mean": mean, "std": std, "ci": ci}


def fmt_cell(st):
    if st is None:
        return "&mdash;", ""
    if st["n"] == 1:
        return f"{st['mean']:.1f}", "n=1"
    return (f"{st['mean']:.1f} &plusmn; {st['std']:.1f}",
            f"95% CI &plusmn;{st['ci']:.1f}, n={st['n']}")


# ---------------------------------------------------------------------------
# Learning-curve PNGs: mean across seeds with 95% CI band, per task.
# We interpolate each seed onto a common step grid so seeds with different
# logged steps can be averaged honestly.
# ---------------------------------------------------------------------------
COLORS = {"tdmpc2": "#1f77b4", "tdmpc-glass": "#d62728"}
LABELS = {"tdmpc2": "TD-MPC2", "tdmpc-glass": "tdmpc-glass"}


def make_curve(data, task):
    plt.figure(figsize=(4.6, 3.2))
    any_data = False
    for algo in WM_ALGOS:
        seeds = data.get(algo, {}).get(task, {})
        series = [seeds[s] for s in sorted(seeds) if seeds[s]]
        if not series:
            continue
        any_data = True
        # common grid = union of steps clipped to the shortest seed's max step
        max_common = min(s[-1][0] for s in series)
        grid = np.linspace(0, max_common, 60)
        stacked = []
        for s in series:
            xs = np.array([p[0] for p in s], float)
            ys = np.array([p[1] for p in s], float)
            stacked.append(np.interp(grid, xs, ys))
        M = np.vstack(stacked)
        mean = M.mean(0)
        n = M.shape[0]
        plt.plot(grid / 1e6, mean, color=COLORS[algo],
                 label=f"{LABELS[algo]} (n={n})", lw=1.8)
        if n > 1:
            ci = 1.96 * M.std(0, ddof=0) / math.sqrt(n)
            plt.fill_between(grid / 1e6, mean - ci, mean + ci,
                             color=COLORS[algo], alpha=0.2)
    plt.xlabel("env steps (M)")
    plt.ylabel("return")
    plt.title(task, fontsize=10)
    if any_data:
        plt.legend(fontsize=7, loc="best")
    else:
        plt.text(0.5, 0.5, "no data", ha="center", va="center",
                 transform=plt.gca().transAxes, color="#999")
    plt.tight_layout()
    out = os.path.join(CURVES, f"{task}.png")
    plt.savefig(out, dpi=90)
    plt.close()
    return any_data


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
def build_html(data):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cols = ["PPO", "TD-MPC2", "tdmpc-glass", "abstraction", "TAMP"]

    rows_html = []
    for task in TASKS:
        cells = [f'<th class="task">{task}</th>']

        # PPO (static, partial)
        if task in PPO_STATIC:
            p = PPO_STATIC[task]
            cells.append(f'<td>{p["val"]:.1f}'
                         f'<span class="sub">peak @ {p["budget"]}</span></td>')
        else:
            cells.append('<td class="na">&mdash;</td>')

        # TD-MPC2 + tdmpc-glass (live)
        for algo in ["tdmpc2", "tdmpc-glass"]:
            st = stats(final_returns(data, algo, task))
            main, sub = fmt_cell(st)
            cls = "" if st else "na"
            sub_html = f'<span class="sub">{sub}</span>' if sub else ""
            cells.append(f'<td class="{cls}">{main}{sub_html}</td>')

        # abstraction (static, partial)
        if task in ABS_STATIC:
            a = ABS_STATIC[task]
            ci = 1.96 * a["std"] / math.sqrt(a["n"])
            cells.append(f'<td>{a["mean"]:.1f} &plusmn; {a["std"]:.1f}'
                         f'<span class="sub">95% CI &plusmn;{ci:.1f}, n={a["n"]}</span></td>')
        else:
            cells.append('<td class="na">&mdash;</td>')

        # TAMP (N/A for DMC)
        cells.append('<td class="na">&mdash;</td>')

        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    # curve grid
    curve_imgs = []
    for task in TASKS:
        has = make_curve(data, task)
        curve_imgs.append(
            f'<figure><img src="curves/{task}.png" alt="{task}" loading="lazy">'
            f'</figure>')

    header_cells = "".join(f"<th>{c}</th>" for c in cols)

    # summary counts
    glass_done = sum(1 for t in TASKS if final_returns(data, "tdmpc-glass", t))
    t2_done = sum(1 for t in TASKS if final_returns(data, "tdmpc2", t))
    glass_seeds = {len(final_returns(data, "tdmpc-glass", t)) for t in TASKS}

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="300">
<title>{PAGE_TITLE}</title>
<style>
 body{{font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
   margin:0;background:#0f1115;color:#e6e6e6}}
 .wrap{{max-width:1100px;margin:0 auto;padding:24px 18px 60px}}
 h1{{font-size:22px;margin:0 0 4px}}
 h2{{font-size:17px;margin:34px 0 10px;border-bottom:1px solid #2a2f3a;padding-bottom:6px}}
 .meta{{color:#9aa4b2;font-size:12.5px;margin-bottom:8px}}
 table{{border-collapse:collapse;width:100%;font-size:13px}}
 th,td{{border:1px solid #2a2f3a;padding:7px 9px;text-align:center}}
 thead th{{background:#1a1f2b;position:sticky;top:0}}
 th.task{{text-align:left;background:#161b25;font-weight:600;white-space:nowrap}}
 td.na{{color:#5a6473}}
 .sub{{display:block;color:#8b97a7;font-size:10.5px;margin-top:2px}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:10px}}
 figure{{margin:0;background:#161b25;border:1px solid #2a2f3a;border-radius:6px;padding:6px}}
 figure img{{width:100%;display:block;border-radius:4px}}
 .note{{color:#9aa4b2;font-size:12px;background:#161b25;border:1px solid #2a2f3a;
   border-radius:6px;padding:10px 12px;margin:10px 0}}
 code{{background:#222834;padding:1px 5px;border-radius:4px}}
 a{{color:#6db3ff}}
</style></head><body><div class="wrap">
<h1>{PAGE_TITLE}</h1>
<div class="meta">Last updated <b>{now}</b> &middot; auto-refresh 5 min &middot;
 source <code>{html.escape(BENCH)}/*.csv</code></div>
<div class="meta">Live coverage: tdmpc-glass {glass_done}/16 tasks, TD-MPC2 {t2_done}/16 tasks.
 Seeds-per-task (glass): {sorted(glass_seeds)}. For mean&plusmn;CI you need &ge;3 seeds.</div>

<h2>A. Main results table</h2>
<div class="note">World-model cells = mean &plusmn; std of <b>final-checkpoint</b> return
 across available seeds (95% CI = 1.96&middot;std/&radic;n; n shown). Each TD-MPC2 /
 tdmpc-glass number is read live from a real CSV. <b>Missing = &mdash; (never fabricated).</b><br>
 <b>PPO</b>: partial &mdash; only ~5 DMC tasks were run, at 50&ndash;285M env steps;
 shown value is PEAK return (robust to PPO terminal collapse), budget labelled.<br>
 <b>abstraction</b> (analytic controller + learned residual): only PendulumSwingup
 verified (Part 45, 3 seeds); Acrobot abstraction not run &rarr; &mdash;.<br>
 <b>TAMP</b>: N/A for DMC &mdash; it is an OpenCabinet manipulation controller (Part 43),
 not a DMC method.</div>
<table><thead><tr><th class="task">Task</th>{header_cells}</tr></thead>
<tbody>{''.join(rows_html)}</tbody></table>

<h2>B. Learning curves</h2>
<div class="note">Mean across seeds with 95% CI shaded band, one line per world-model
 method. x = env steps (M), y = return. Regenerated each update.</div>
<div class="grid">{''.join(curve_imgs)}</div>

<h2>C. Rollout GIFs</h2>
<div class="note">Pending &mdash; rendered eval rollouts will be embedded here as they
 are produced (lowest-priority visual; table &amp; curves are not blocked on it).</div>

</div></body></html>"""


def main():
    data = collect()
    out = build_html(data)
    with open(os.path.join(SITE, "index.html"), "w") as f:
        f.write(out)
    print("wrote", os.path.join(SITE, "index.html"))


if __name__ == "__main__":
    main()
