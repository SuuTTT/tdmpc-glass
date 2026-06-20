#!/usr/bin/env bash
# 8h keep-busy SAFETY NET for the GHM-fix run. The fix agent owns the 4 b3060 GPUs and is
# mandated to keep them busy; this only fires if ALL 4 go idle for 2 consecutive checks
# (~30 min) — i.e. the agent stalled/finished — then launches default antmaze GHM runs so the
# GPUs never sit idle. Conservative (2-strike) to avoid racing the agent between waves.
set -u
ROOT=/home/ubuntu/tdmpc-glass
GH=$ROOT/exp/tdmpc_glass/ghm
LOG=$GH/fix_safety.log
echo $$ > "$GH/fix_safety.pid"
strikes=0
say(){ echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }
idle_count(){ # number of b3060 GPUs with <800 MiB used
  timeout 25 ssh -o ConnectTimeout=12 b3060 "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits" 2>/dev/null \
    | awk '{if($1+0 < 800) n++} END{print n+0}'
}
while :; do
  n=$(idle_count)
  if [ "${n:-0}" -ge 4 ]; then
    strikes=$((strikes+1)); say "all 4 b3060 GPUs idle (strike $strikes/2)"
    if [ "$strikes" -ge 2 ]; then
      say "fix agent appears stopped -> launching 4 default antmaze GHM runs to keep GPUs busy"
      for g in 0 1 2 3; do
        t=$((g+1))
        timeout 45 ssh -o ConnectTimeout=20 b3060 "cd /root/ghm/infom && setsid bash -c '. /root/ghm/.venv/bin/activate && export PYTHONPATH=/root/ghm/infom MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g && python main.py --env_name=antmaze-medium-navigate-singletask-task${t}-v0 --agent=agents/ghm.py --enable_wandb=0 --save_dir=/root/ghm/exp_fix --wandb_run_group=safety_amz_t${t}_g${g}' > /root/ghm/logs/safety_amz_t${t}.log 2>&1 < /dev/null &" >/dev/null 2>&1
        sleep 60
      done
      strikes=0
    fi
  else
    [ "$strikes" -ne 0 ] && say "GPUs busy again ($n idle) — reset strikes"
    strikes=0
  fi
  sleep 900
done
