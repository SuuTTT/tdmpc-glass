# PandaPickCube Experiment Kanban Dashboard

Live KANBAN web dashboard for the PandaPickCube experiment branches on box **b3060**.
Read-only: parses status files fresh on every `/data` fetch (browser polls every 15s).
Never touches training processes or GPUs (nvidia-smi query only).

## Public URL (Cloudflare quick tunnel)

    https://increase-distant-candidate-bizarre.trycloudflare.com

NOTE: trycloudflare quick-tunnel URLs are ephemeral. If cloudflared restarts you get a NEW
random URL — re-read it from the cloudflared log (see below).

## Where things live (on b3060)

- App:        /root/helios-rl/exp/tdmpc_glass/dashboard/app.py
- App log:    /root/helios-rl/exp/tdmpc_glass/dashboard/dashboard.log
- CF log:     /root/helios-rl/exp/tdmpc_glass/dashboard/cloudflared.log
- Local port: 8137  (http://localhost:8137/  +  /data  +  /health)
- venv:       /root/helios-rl/.venv  (stdlib http.server, no extra deps needed)

## Current PIDs (as launched)

- dashboard app : 815349  (PPID=1, detached)
- cloudflared   : 819919  (own session leader, detached)

(Find them again: `pgrep -f "dashboard/app.py"` and `pgrep -f "cloudflared tunnel"`.)

## What it shows

Three columns: RUNNING / DONE / FAILED(+PENDING). One card per experiment branch:
- scaleup_vanilla_s1  (full TD-MPC2, 10M)      -> GPU 0
- scaleup_jumpy_s1    (jumpy TD-MPC2, 10M)     -> GPU 2
- small_vanilla_s1    (small TD-MPC2 lat256)   -> GPU 3
- reward_eng v1..v9jumpy (DONE, all 0% success)
- PPO baseline (DONE, 66% @33M)
- SAC / PPO-100% / InFOM-dataset (graceful "pending" until parallel agent writes files)

Each card: branch name, status, latest step / total, peak real success, peak grasp(reached),
peak box_target, last-update time, GPU badge. Header shows per-GPU util/mem from nvidia-smi.
Legend anchors: PPO 66%@33M, TD-MPC2/jumpy 0%@1.5M, HL heuristic 9.4%.

Status files parsed:
- TD-MPC2 branches: /root/helios-rl/exp/benchmark/tdmpc2_PandaPickCube_<TAG>_realsuccess.csv
  cols (1-idx): step=1, pi_success=6, mppi_success=7, pi_reached=9, pi_box_target_max=12
- reward_eng:       /root/helios-rl/exp/benchmark/tdmpc2_PandaPickCube_reweng_*_s1_realsuccess.csv
- PPO baseline:     /root/helios-rl/exp/tdmpc_glass/baselines_ppo_sac/RESULTS.json
- optional:         baselines_ppo_sac/{PROGRESS2.md, SAC_RESULT.md, PPO_100_RESULT.json}

Robust: missing/locked files -> "n/a" / "pending", never crashes.

## Restart instructions

If the dashboard app died:

    ssh b3060
    cd /root/helios-rl/exp/tdmpc_glass/dashboard
    pkill -f "dashboard/app.py"          # kill any stale copy first
    nohup setsid /root/helios-rl/.venv/bin/python app.py >dashboard.log 2>&1 </dev/null &
    curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8137/   # expect 200

If cloudflared died (gives a NEW random URL):

    ssh b3060
    cd /root/helios-rl/exp/tdmpc_glass/dashboard
    pkill -f "cloudflared tunnel"
    nohup setsid /opt/instance-tools/bin/cloudflared tunnel --url http://localhost:8137 \
        >cloudflared.log 2>&1 </dev/null &
    # wait ~10s, then read the new URL:
    grep -oE "https://[a-zA-Z0-9-]+\.trycloudflare\.com" cloudflared.log | head -1

cloudflared binary: /opt/instance-tools/bin/cloudflared
