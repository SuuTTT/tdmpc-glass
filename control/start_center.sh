#!/usr/bin/env bash
# start_center.sh — launch the TD-MPC-Glass control plane on this EC2 box.
#
# This box is the CONTROL PLANE ONLY (no GPU). It runs three daemons:
#   1. web_dashboard.py     — Flask UI + REST queue API (port $DASHBOARD_PORT, default 5055)
#   2. task_queue_daemon.py — claims pending tasks, SSH-launches them on idle GPU workers
#   3. iter5_stream_remotes.sh — rsyncs remote experiment CSVs into exp/.../remote_mirror/
#
# NEVER trains locally (no local box in BOXES). All training is on remote vast.ai GPUs.
# One-master rule: do not run a second queue daemon against the same fleet.
set -euo pipefail

REPO=/home/ubuntu/tdmpc-glass
PY=$REPO/.venv/bin/python3
PORT="${DASHBOARD_PORT:-5055}"
LOGD=$REPO/exp/tdmpc_glass/logs/daemons
mkdir -p "$LOGD"
export SSH_IDENTITY_FILE="${SSH_IDENTITY_FILE:-/home/ubuntu/.ssh/vastai_id_ed25519}"
export DASHBOARD_PORT="$PORT"

cd "$REPO"

echo "== web dashboard (Flask, port $PORT) =="
pkill -f 'tdmpc-glass/control/web_dashboard.py' 2>/dev/null || true; sleep 1
nohup setsid "$PY" -u control/web_dashboard.py >> "$LOGD/dashboard.log" 2>&1 < /dev/null & disown
sleep 2

echo "== task queue daemon =="
pkill -f 'tdmpc-glass/control/task_queue_daemon.py' 2>/dev/null || true; sleep 1
nohup setsid "$PY" -u control/task_queue_daemon.py >> "$LOGD/tqd.log" 2>&1 < /dev/null & disown
sleep 1

echo "== remote CSV streamer =="
pkill -f 'tdmpc-glass/control/iter5_stream_remotes.sh' 2>/dev/null || true; sleep 1
nohup setsid bash control/iter5_stream_remotes.sh >> "$LOGD/stream.log" 2>&1 < /dev/null & disown
sleep 1

echo ""
echo "== alive check =="
pgrep -fa 'control/web_dashboard.py'     | head -1 || echo "  dashboard NOT running"
pgrep -fa 'control/task_queue_daemon.py' | head -1 || echo "  queue daemon NOT running"
pgrep -fa 'control/iter5_stream_remotes.sh' | head -1 || echo "  streamer NOT running"
echo ""
echo "Dashboard:  http://localhost:$PORT   (tunnel: ssh -L $PORT:localhost:$PORT <this-ec2>)"
echo "Logs:       tail -f $LOGD/{dashboard,tqd,stream}.log"
