#!/usr/bin/env python3
"""Live GPU + training-throughput monitor across the tdmpc fleet.
Flask on :8080, SSH-polls each box every 15s for (a) per-GPU util/mem/temp and
(b) per-run training speed (it/s) + step parsed from /root/ghm/logs/*.log tqdm output.
Exposed via `cloudflared tunnel --url http://localhost:8080`."""
import subprocess, threading, time, re
from flask import Flask

app = Flask(__name__)
BOXES = {
    "3070-laptop (38702735)": "b3070",
    "A4000 (39109169) [SISA-shared]": "a4",
    "4x3060 (156.238.224.242)": "b3060",
}
STATE = {}

GPU_Q = ("nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu "
         "--format=csv,noheader,nounits")
# recently-active GHM logs -> "name|<step>/<total>|<sps>"
LOG_Q = (r'''for f in /root/ghm/logs/*.log; do [ -f "$f" ] || continue; '''
         r'''now=$(date +%s); m=$(stat -c %Y "$f"); if [ $((now-m)) -lt 240 ]; then '''
         r'''t=$(tr "\r" "\n" < "$f" 2>/dev/null | grep -oE "[0-9]+/[0-9]+ \[[^]]*[0-9.]+it/s" | tail -1); '''
         r'''echo "$(basename "$f" .log)|$t"; fi; done''')
# TKDE TSLib log: parse last experiment name + left_time from TSLib stdout
TKDE_Q = (r'''python3 -c "
import re, os
logs = ['/root/nsf_3070.log', '/root/ecl_b64_gpu0.log', '/root/fed_pems_gpu1_nohup.log',
        '/root/nsf_p03_nohup.log', '/root/auto_ecl_nohup.log', '/root/auto_tr_nohup.log']
for lf in logs:
    if not os.path.isfile(lf): continue
    try:
        lines = open(lf).readlines()[-80:]
    except: continue
    text = ''.join(lines)
    done = [m.group(1) for m in re.finditer(r'DONE (NSF_\w+|TN_\w+|FED_\w+|AUTO_\w+)', text)]
    cur = re.findall(r'--- (NSF_\w+|TN_\w+|FED_\w+|AUTO_\w+) \w{3} \w{3} \d+', text)
    lt = re.findall(r'left time: ([0-9.]+)s', text)
    ep = re.findall(r'epoch: (\d+)', text)
    name = cur[-1] if cur else (done[-1]+' DONE' if done else '?')
    left = f'{float(lt[-1])/60:.0f}m' if lt else '—'
    epoch = ep[-1] if ep else '—'
    print(f'{os.path.basename(lf)}|{name}|ep{epoch}|{left} left')
" 2>/dev/null''')

def sh(host, cmd):
    return subprocess.run(["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", host, cmd],
                          capture_output=True, text=True, timeout=24).stdout.strip()

def parse_job(line):
    name, _, rest = line.partition("|")
    m_step = re.search(r"(\d+)/(\d+)", rest)
    m_sps = re.search(r"([0-9.]+)it/s", rest)
    step = f"{int(m_step.group(1)):,}/{int(m_step.group(2)):,}" if m_step else "—"
    sps = f"{float(m_sps.group(1)):.0f}" if m_sps else "—"
    return name, step, sps

# Boxes that also run TKDE TSLib experiments (parsed separately)
TKDE_BOXES = {"3070-laptop (38702735)"}

def parse_tkde(line):
    parts = line.split("|")
    if len(parts) < 4: return None
    return {"log": parts[0], "exp": parts[1], "epoch": parts[2], "left": parts[3]}

def poller():
    while True:
        for label, host in BOXES.items():
            try:
                gpus = [l.split(", ") for l in sh(host, GPU_Q).splitlines() if l.strip()]
                jobs = [parse_job(l) for l in sh(host, LOG_Q).splitlines() if l.strip() and "|" in l]
                tkde = []
                if label in TKDE_BOXES:
                    tkde = [r for r in [parse_tkde(l) for l in sh(host, TKDE_Q).splitlines()
                            if l.strip() and "|" in l] if r]
                STATE[label] = {"ok": True, "gpus": gpus, "jobs": jobs, "tkde": tkde,
                                "t": time.strftime("%H:%M:%S UTC", time.gmtime())}
            except Exception as e:
                STATE[label] = {"ok": False, "err": str(e)[:80],
                                "t": time.strftime("%H:%M:%S UTC", time.gmtime())}
        time.sleep(15)

def bar(pct):
    p = int(float(pct)); c = "#e74c3c" if p > 80 else "#f39c12" if p > 30 else "#2ecc71"
    return (f'<div style="background:#222;width:120px;height:14px;display:inline-block;border-radius:3px;'
            f'vertical-align:middle"><div style="background:{c};width:{p*1.2}px;height:14px;border-radius:3px"></div></div> {p}%')

@app.route("/")
def index():
    blocks = ""
    total_gpus = busy = 0
    for label, s in STATE.items():
        if not s.get("ok"):
            blocks += f'<div class=box><b>{label}</b> — <span style="color:#e67e22">unreachable: {s.get("err","")}</span></div>'
            continue
        rows = ""
        for g in s["gpus"]:
            idx, util, mu, mt, temp = (g + ["?"]*5)[:5]
            total_gpus += 1
            if float(util or 0) > 5: busy += 1
            rows += f'<tr><td>GPU{idx}</td><td>{bar(util)}</td><td>{mu}/{mt} MiB</td><td>{temp}°C</td></tr>'
        jobs = ""
        for n, step, sps in s["jobs"]:
            jobs += f'<tr><td>{n}</td><td>{step}</td><td><b>{sps}</b> it/s</td></tr>'
        jobs_tbl = (f'<table class=j><tr><th>run</th><th>step</th><th>speed</th></tr>{jobs}</table>'
                    if jobs else '<small style="color:#8a93a2">no active GHM runs</small>')
        tkde_rows = ""
        for r in s.get("tkde", []):
            tkde_rows += f'<tr><td>{r["log"]}</td><td>{r["exp"]}</td><td>{r["epoch"]}</td><td>{r["left"]}</td></tr>'
        tkde_tbl = (f'<table class=j><tr><th>log</th><th>experiment</th><th>epoch</th><th>left</th></tr>{tkde_rows}</table>'
                    if tkde_rows else "")
        blocks += (f'<div class=box><b>{label}</b> <small>· {s["t"]}</small>'
                   f'<table class=g><tr><th>GPU</th><th>util</th><th>mem</th><th>temp</th></tr>{rows}</table>'
                   f'{jobs_tbl}{tkde_tbl}</div>')
    hdr = f'{busy}/{total_gpus} GPUs active'
    return f"""<!doctype html><html><head><meta charset=utf-8><title>tdmpc GPU monitor</title>
<meta http-equiv=refresh content=15>
<style>body{{font-family:system-ui,Arial;background:#0f1115;color:#e6e6e6;padding:20px}}
h1{{font-size:19px}} .box{{background:#171a21;border:1px solid #2a2e37;border-radius:8px;padding:12px 16px;margin:10px 0;max-width:760px}}
table{{border-collapse:collapse;margin-top:6px}} td,th{{padding:4px 12px;text-align:left;font-size:14px}}
.g th{{color:#8a93a2;font-weight:500}} .j{{margin-top:8px}} .j th{{color:#5dade2;font-weight:500}}
.j td{{border-top:1px solid #23272f}}</style></head><body>
<h1>🖥️ tdmpc-glass fleet — live · <span style="color:#2ecc71">{hdr}</span> <small>(auto-refresh 15s)</small></h1>
{blocks}
<p><small>SSH-polled every 15s. Speed = training it/s parsed from /root/ghm/logs tqdm.</small></p></body></html>"""

if __name__ == "__main__":
    threading.Thread(target=poller, daemon=True).start()
    app.run(host="0.0.0.0", port=8080)
