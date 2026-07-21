#!/usr/bin/env python3
"""Live KANBAN dashboard for PandaPickCube experiment branches on b3060.
Self-contained stdlib http.server app. Parses status files fresh on each /data
fetch (JS polls every 15s). Robust: missing/locked files -> "n/a", never crash.
Read-only: NEVER touches training or GPUs (nvidia-smi query only).
"""
import csv
import io
import json
import os
import re
import subprocess
import time
import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BENCH = "/root/helios-rl/exp/benchmark"
BASE = "/root/helios-rl/exp/tdmpc_glass/baselines_ppo_sac"
PORT = 8137

# 1-indexed CSV columns per spec -> 0-indexed
C_STEP = 0
C_PI_SUCCESS = 5
C_MPPI_SUCCESS = 6
C_PI_REACHED = 8
C_PI_BOX_TARGET = 11

ANCHORS = {
    "PPO solves": "66% @ 33M env-steps",
    "TD-MPC2 / jumpy": "0% @ 1.5M (reward-hack hover)",
    "HL heuristic": "9.4%",
}


def safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def fnum(x):
    try:
        v = float(x)
        if v != v:  # nan
            return None
        return v
    except Exception:
        return None


def read_csv_rows(path):
    """Return (header, rows) or (None, []) on any problem."""
    if not os.path.exists(path):
        return None, []
    try:
        with open(path, "r", newline="") as f:
            data = f.read()
        rdr = list(csv.reader(io.StringIO(data)))
        if not rdr:
            return None, []
        header = rdr[0]
        rows = [r for r in rdr[1:] if r and len(r) > C_STEP]
        return header, rows
    except Exception:
        return None, []


def col_max(rows, idx):
    best = None
    for r in rows:
        if len(r) <= idx:
            continue
        v = fnum(r[idx])
        if v is None:
            continue
        if best is None or v > best:
            best = v
    return best


def last_step(rows):
    for r in reversed(rows):
        v = fnum(r[C_STEP]) if len(r) > C_STEP else None
        if v is not None:
            return int(v)
    return None


def mtime_str(path):
    try:
        t = os.path.getmtime(path)
        return datetime.datetime.fromtimestamp(t).strftime("%H:%M:%S"), (time.time() - t)
    except Exception:
        return "n/a", None


def get_gpu_map():
    """Map TDMPC_GLASS_OUTPUT_TAG -> CUDA_VISIBLE_DEVICES by scanning ps.
    Returns {tag: gpu_index}. Robust to failures."""
    out = {}
    try:
        ps = subprocess.run(
            ["ps", "-eo", "args"], capture_output=True, text=True, timeout=8
        ).stdout
    except Exception:
        return out
    for line in ps.splitlines():
        if "run_benchmark.py" not in line:
            continue
        mtag = re.search(r"TDMPC_GLASS_OUTPUT_TAG=(\S+)", line)
        mgpu = re.search(r"CUDA_VISIBLE_DEVICES=(\d+)", line)
        if mtag and mgpu:
            out[mtag.group(1)] = mgpu.group(1)
    return out


def get_running_tags():
    """Tags with a live run_benchmark.py python process."""
    tags = set()
    try:
        ps = subprocess.run(
            ["ps", "-eo", "args"], capture_output=True, text=True, timeout=8
        ).stdout
    except Exception:
        return tags
    for line in ps.splitlines():
        if "run_benchmark.py" not in line:
            continue
        if "python" not in line and "python3" not in line:
            continue
        m = re.search(r"TDMPC_GLASS_OUTPUT_TAG=(\S+)", line)
        if m:
            tags.add(m.group(1))
    return tags


def gpu_header():
    rows = []
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8,
        ).stdout.strip()
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                rows.append({
                    "idx": parts[0], "util": parts[1],
                    "used": parts[2], "total": parts[3],
                })
    except Exception:
        pass
    return rows


def pct(v):
    return "n/a" if v is None else f"{v * 100:.1f}%"


def tdmpc_card(name, tag, label, total_steps, gpu_map, running_tags):
    """Build a card dict for a TD-MPC2 branch read from a realsuccess CSV."""
    path = os.path.join(BENCH, f"tdmpc2_PandaPickCube_{tag}_realsuccess.csv")
    header, rows = read_csv_rows(path)
    upd, age = mtime_str(path) if os.path.exists(path) else ("n/a", None)
    is_running = tag in running_tags
    card = {
        "name": name, "sub": label, "gpu": gpu_map.get(tag),
        "updated": upd, "file_ok": header is not None,
    }
    if header is None:
        card["status"] = "RUNNING" if is_running else "PENDING"
        card["note"] = "no eval logged yet" if is_running else "no data file"
        card["metrics"] = {}
        return card
    step = last_step(rows)
    peak_succ = col_max(rows, C_PI_SUCCESS)
    peak_mppi = col_max(rows, C_MPPI_SUCCESS)
    peak_reach = col_max(rows, C_PI_REACHED)
    peak_box = col_max(rows, C_PI_BOX_TARGET)
    # status
    if is_running:
        status = "RUNNING"
    elif age is not None and age < 600:
        status = "RUNNING"
    elif step is not None and total_steps and step >= total_steps * 0.98:
        status = "DONE"
    else:
        status = "DONE" if not is_running else "RUNNING"
    card["status"] = status
    card["metrics"] = {
        "step": step, "total": total_steps,
        "peak_success": pct(max([v for v in [peak_succ, peak_mppi] if v is not None], default=None)
                            if (peak_succ is not None or peak_mppi is not None) else None),
        "peak_reached": pct(peak_reach),
        "peak_box": "n/a" if peak_box is None else f"{peak_box:.3f}",
    }
    return card


def reweng_card():
    """Summary card across all reweng v1..v9jumpy (all DONE, 0 success)."""
    files = sorted([f for f in os.listdir(BENCH)
                    if re.match(r"tdmpc2_PandaPickCube_reweng_.*_s1_realsuccess\.csv$", f)]) \
        if os.path.exists(BENCH) else []
    versions = []
    best_succ = None
    best_reach = None
    best_box = None
    newest = None
    for f in files:
        m = re.search(r"reweng_(\w+)_s1", f)
        v = m.group(1) if m else f
        path = os.path.join(BENCH, f)
        _, rows = read_csv_rows(path)
        s = col_max(rows, C_PI_SUCCESS)
        sm = col_max(rows, C_MPPI_SUCCESS)
        rch = col_max(rows, C_PI_REACHED)
        bx = col_max(rows, C_PI_BOX_TARGET)
        sc = max([x for x in [s, sm] if x is not None], default=None)
        versions.append({"v": v, "succ": sc, "reach": rch, "box": bx, "step": last_step(rows)})
        for cur, val in (("best_succ", sc), ("best_reach", rch), ("best_box", bx)):
            pass
        if sc is not None and (best_succ is None or sc > best_succ):
            best_succ = sc
        if rch is not None and (best_reach is None or rch > best_reach):
            best_reach = rch
        if bx is not None and (best_box is None or bx > best_box):
            best_box = bx
        _, age = mtime_str(path)
        if age is not None and (newest is None or age < newest):
            newest = age
    upd = "n/a"
    if files:
        latest = max(files, key=lambda f: safe(lambda: os.path.getmtime(os.path.join(BENCH, f)), 0))
        upd = mtime_str(os.path.join(BENCH, latest))[0]
    return {
        "name": "reward_eng v1-v9jumpy", "sub": f"{len(files)} variants (DONE)",
        "status": "DONE" if files else "PENDING", "updated": upd, "gpu": None,
        "metrics": {
            "variants": len(files),
            "peak_success": pct(best_succ),
            "peak_reached": pct(best_reach),
            "peak_box": "n/a" if best_box is None else f"{best_box:.3f}",
        },
        "detail": versions,
        "note": "all 0% real success (reward-hack hover)",
    }


def json_card(name, sub, path, extract):
    """Generic card from a JSON results file. extract(data)->metrics dict."""
    if not os.path.exists(path):
        return {"name": name, "sub": sub, "status": "PENDING",
                "updated": "n/a", "gpu": None, "metrics": {},
                "note": "pending (file not present yet)"}
    try:
        with open(path) as f:
            data = json.load(f)
        m, status, note = extract(data)
        return {"name": name, "sub": sub, "status": status,
                "updated": mtime_str(path)[0], "gpu": None,
                "metrics": m, "note": note}
    except Exception as e:
        return {"name": name, "sub": sub, "status": "FAILED",
                "updated": mtime_str(path)[0], "gpu": None, "metrics": {},
                "note": f"parse error: {e}"}


def md_card(name, sub, path):
    """Card from a markdown/text result file: show first ~6 non-empty lines."""
    if not os.path.exists(path):
        return {"name": name, "sub": sub, "status": "PENDING",
                "updated": "n/a", "gpu": None, "metrics": {},
                "note": "pending (file not present yet)"}
    try:
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]
        snippet = " | ".join(lines[:6])[:400]
        return {"name": name, "sub": sub, "status": "DONE",
                "updated": mtime_str(path)[0], "gpu": None, "metrics": {},
                "note": snippet}
    except Exception as e:
        return {"name": name, "sub": sub, "status": "FAILED",
                "updated": mtime_str(path)[0], "gpu": None, "metrics": {},
                "note": str(e)}


def ppo_extract(data):
    p = data.get("PPO", {})
    if isinstance(p, dict) and "final" in p:
        fin = p["final"]
        m = {
            "step": fin.get("step"),
            "peak_success": pct(p.get("peak_by_success", fin).get("success_rate")),
            "peak_reached": pct(fin.get("reached_rate")),
            "peak_box": f"{fin.get('mean_max_box_target'):.3f}" if fin.get("mean_max_box_target") is not None else "n/a",
        }
        return m, "DONE", "OFFICIAL brax PPO, 256 eps"
    return {}, "PENDING", "no PPO final yet"


def ppo100_extract(data):
    # tolerant: look for success_rate anywhere reasonable
    blk = data.get("PPO", data) if isinstance(data, dict) else {}
    fin = blk.get("final", blk) if isinstance(blk, dict) else {}
    sr = fin.get("success_rate") if isinstance(fin, dict) else None
    m = {
        "step": fin.get("step") if isinstance(fin, dict) else None,
        "peak_success": pct(sr),
        "peak_reached": pct(fin.get("reached_rate")) if isinstance(fin, dict) else "n/a",
        "peak_box": (f"{fin.get('mean_max_box_target'):.3f}"
                     if isinstance(fin, dict) and fin.get("mean_max_box_target") is not None else "n/a"),
    }
    return m, "DONE", "PPO pushed to 100%"


def build_state():
    gpu_map = get_gpu_map()
    running = get_running_tags()
    cards = []
    # RUNNING TD-MPC2 branches
    cards.append(tdmpc_card("scaleup_vanilla_s1", "scaleup_vanilla_s1",
                            "full TD-MPC2, 10M", 10_000_000, gpu_map, running))
    cards.append(tdmpc_card("scaleup_jumpy_s1", "scaleup_jumpy_s1",
                            "jumpy TD-MPC2, 10M", 10_000_000, gpu_map, running))
    cards.append(tdmpc_card("small_vanilla_s1", "small_vanilla_s1",
                            "small TD-MPC2 latent256, 10M", 10_000_000, gpu_map, running))
    # DONE: reward eng
    cards.append(reweng_card())
    # DONE: PPO baseline
    cards.append(json_card("PPO baseline", "brax PPO (66%)",
                           os.path.join(BASE, "RESULTS.json"), ppo_extract))
    # Optional parallel-agent outputs (graceful pending)
    cards.append(md_card("SAC", "off-policy baseline",
                         os.path.join(BASE, "SAC_RESULT.md")))
    cards.append(json_card("PPO-100%", "PPO pushed to solve",
                           os.path.join(BASE, "PPO_100_RESULT.json"), ppo100_extract))
    cards.append(md_card("InFOM-dataset", "dataset build",
                         os.path.join(BASE, "PROGRESS2.md")))
    return {
        "cards": cards,
        "gpus": gpu_header(),
        "now": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "anchors": ANCHORS,
    }


# ----------------------- HTML -----------------------
PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PandaPickCube Experiments - Kanban</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--line:#30363d;--txt:#c9d1d9;--mut:#8b949e;
--run:#1f6feb;--done:#238636;--fail:#da3633;--pend:#9e6a03;--accent:#58a6ff;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);font:14px/1.45 -apple-system,Segoe UI,Roboto,Helvetica,Arial,monospace;}
header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap;gap:16px;align-items:center;}
h1{font-size:18px;margin:0;font-weight:600;letter-spacing:.3px;}
.sub{color:var(--mut);font-size:12px;}
.gpus{display:flex;gap:10px;flex-wrap:wrap;margin-left:auto;}
.gpu{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:5px 9px;font-size:12px;}
.gpu b{color:var(--accent);}
.bar{height:5px;border-radius:3px;background:#21262d;margin-top:3px;overflow:hidden;}
.bar i{display:block;height:100%;background:var(--run);}
.legend{padding:8px 20px;font-size:11px;color:var(--mut);border-bottom:1px solid var(--line);}
.legend b{color:var(--txt);}
.board{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;padding:16px 20px;align-items:start;}
.col{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px;min-height:120px;}
.col h2{font-size:13px;margin:0 0 10px;text-transform:uppercase;letter-spacing:1px;display:flex;align-items:center;gap:8px;}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block;}
.c-RUNNING .dot,.dot.RUNNING{background:var(--run);}
.c-DONE .dot,.dot.DONE{background:var(--done);}
.c-FAILED .dot,.dot.FAILED{background:var(--fail);}
.c-PENDING .dot,.dot.PENDING,.dot.IDLE{background:var(--pend);}
.card{background:#0d1117;border:1px solid var(--line);border-left:3px solid var(--mut);border-radius:6px;padding:9px 11px;margin-bottom:9px;}
.card.RUNNING{border-left-color:var(--run);}
.card.DONE{border-left-color:var(--done);}
.card.FAILED{border-left-color:var(--fail);}
.card.PENDING{border-left-color:var(--pend);}
.card .nm{font-weight:600;font-size:13px;}
.card .lbl{color:var(--mut);font-size:11px;margin-bottom:6px;}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:3px 12px;font-size:12px;}
.grid .k{color:var(--mut);}
.grid .v{text-align:right;font-variant-numeric:tabular-nums;}
.foot{margin-top:6px;font-size:10.5px;color:var(--mut);display:flex;justify-content:space-between;gap:8px;}
.gpubadge{background:#21262d;border-radius:4px;padding:0 5px;color:var(--accent);}
.note{font-size:11px;color:var(--mut);margin-top:5px;font-style:italic;word-break:break-word;}
.empty{color:var(--mut);font-size:12px;padding:8px 2px;}
.flash{animation:f 1s ease;}
@keyframes f{from{background:#10243a}to{background:#0d1117}}
@media(max-width:880px){.board{grid-template-columns:1fr;}}
</style></head>
<body>
<header>
  <div><h1>PandaPickCube &mdash; Experiment Kanban</h1>
  <div class="sub">box b3060 (4x RTX 3060) &middot; auto-refresh 15s &middot; <span id="now">...</span></div></div>
  <div class="gpus" id="gpus"></div>
</header>
<div class="legend">
  Anchors: <b>PPO</b> 66% @33M &middot; <b>TD-MPC2/jumpy</b> 0% @1.5M (reward-hack hover) &middot; <b>HL heuristic</b> 9.4%
  &nbsp;|&nbsp; <span class="dot RUNNING"></span>running <span class="dot DONE"></span>done <span class="dot FAILED"></span>failed <span class="dot PENDING"></span>pending/idle
</div>
<div class="board">
  <div class="col c-RUNNING"><h2><span class="dot RUNNING"></span>Running</h2><div id="col-RUNNING"></div></div>
  <div class="col c-DONE"><h2><span class="dot DONE"></span>Done</h2><div id="col-DONE"></div></div>
  <div class="col c-FAILED"><h2><span class="dot FAILED"></span>Failed / Pending</h2><div id="col-FAILED"></div></div>
</div>
<script>
function esc(s){return (s==null?'':String(s)).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function fmtStep(s,t){if(s==null)return 'n/a';let a=s.toLocaleString();if(t)a+=' / '+t.toLocaleString();return a;}
function card(c){
  let m=c.metrics||{};
  let rows='';
  if('step' in m) rows+=`<div class="k">step</div><div class="v">${esc(fmtStep(m.step,m.total))}</div>`;
  if('variants' in m) rows+=`<div class="k">variants</div><div class="v">${esc(m.variants)}</div>`;
  if('peak_success' in m) rows+=`<div class="k">peak success</div><div class="v">${esc(m.peak_success)}</div>`;
  if('peak_reached' in m) rows+=`<div class="k">peak grasp</div><div class="v">${esc(m.peak_reached)}</div>`;
  if('peak_box' in m) rows+=`<div class="k">peak box_tgt</div><div class="v">${esc(m.peak_box)}</div>`;
  let gpu=(c.gpu!=null)?`<span class="gpubadge">GPU ${esc(c.gpu)}</span>`:'';
  let note=c.note?`<div class="note">${esc(c.note)}</div>`:'';
  return `<div class="card ${esc(c.status)}" data-k="${esc(c.name)}">
    <div class="nm">${esc(c.name)}</div>
    <div class="lbl">${esc(c.sub||'')}</div>
    <div class="grid">${rows}</div>
    ${note}
    <div class="foot"><span>upd ${esc(c.updated)}</span>${gpu}</div>
  </div>`;
}
let prev={};
async function tick(){
  try{
    let r=await fetch('/data?_='+Date.now());
    let d=await r.json();
    document.getElementById('now').textContent='refreshed '+d.now;
    // gpus
    document.getElementById('gpus').innerHTML=(d.gpus||[]).map(g=>{
      let p=g.total>0?Math.round(g.used/g.total*100):0;
      return `<div class="gpu"><b>GPU ${esc(g.idx)}</b> ${esc(g.util)}% util<br>${esc(g.used)}/${esc(g.total)} MiB<div class="bar"><i style="width:${p}%"></i></div></div>`;
    }).join('');
    let cols={RUNNING:[],DONE:[],FAILED:[]};
    (d.cards||[]).forEach(c=>{
      let col=(c.status==='RUNNING')?'RUNNING':(c.status==='DONE')?'DONE':'FAILED';
      cols[col].push(card(c));
    });
    for(let k in cols){
      let el=document.getElementById('col-'+k);
      el.innerHTML=cols[k].length?cols[k].join(''):'<div class="empty">none</div>';
    }
  }catch(e){
    document.getElementById('now').textContent='fetch error: '+e;
  }
}
tick(); setInterval(tick,15000);
</script>
</body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/" or path == "/index.html":
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/data":
            try:
                body = json.dumps(build_state()).encode("utf-8")
            except Exception as e:
                body = json.dumps({"error": str(e), "cards": [], "gpus": [],
                                   "now": time.strftime("%H:%M:%S")}).encode("utf-8")
            self._send(200, body, "application/json")
        elif path == "/health":
            self._send(200, b"ok", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")


def main():
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), H)
    print(f"dashboard serving on :{PORT}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
