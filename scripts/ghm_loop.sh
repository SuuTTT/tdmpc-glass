#!/usr/bin/env bash
# GHM keep-busy loop v2 (EC2 control-plane). Keeps the 6-GPU fleet pinned with antmaze GHM:
#   b3070 (3070 8GB) + a4 (A4000 16GB, SISA-shared) + b3060 (4x 3060 12GB).
# GPU-idle-driven self-refill from a file-based env rotation (antmaze by default), harvests
# finetuning_eval.csv to EC2, prunes old checkpoints (disk safety), pidfile + watchdog-friendly.
set -u
ROOT=/home/ubuntu/tdmpc-glass
GH=$ROOT/exp/tdmpc_glass/ghm
MIRROR=$GH/mirror
CNT=$GH/counter
LOG=$GH/loop.log
mkdir -p "$MIRROR"
echo $$ > "$GH/loop.pid"
[ -f "$CNT" ] || echo 0 > "$CNT"
SLOTS=( "b3070:0" "b3060:0" "b3060:1" "b3060:2" "b3060:3" )
ENVFILE=$GH/envs.txt
if [ ! -f "$ENVFILE" ]; then
  for t in 1 2 3 4 5; do echo "antmaze-medium-navigate-singletask-task${t}-v0"; done > "$ENVFILE"
fi
# per-box extra args appended to launch_infom.sh (e.g. reduced batch for 8GB b3070; set after smoke).
declare -A EXTRA=( ["b3070"]="" ["a4"]="" ["b3060"]="" )
# per-box env prefix (a4 is SISA-shared -> cap GPU mem so it can't starve SISA).
mem_prefix(){ case "$1" in a4) echo "XLA_PYTHON_CLIENT_MEM_FRACTION=0.5";; *) echo "";; esac; }

say(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
box_of(){ echo "${1%%:*}"; }; gpu_of(){ echo "${1##*:}"; }

gpu_idle(){ local mu; mu=$(timeout 20 ssh -o ConnectTimeout=12 "$1" "nvidia-smi -i $2 --query-gpu=memory.used --format=csv,noheader,nounits" 2>/dev/null | tr -dc '0-9'); [ -n "$mu" ] && [ "$mu" -lt 800 ]; }

harvest(){
  for box in b3070 a4 b3060; do
    timeout 40 rsync -az -m -e "ssh -o ConnectTimeout=15" \
      --include='*/' --include='finetuning_eval.csv' --include='pretraining_*.csv' --exclude='*' \
      "$box:/root/ghm/exp/" "$MIRROR/$box/" 2>/dev/null
    timeout 30 ssh -o ConnectTimeout=15 "$box" '
      free=$(df --output=avail -BG /root 2>/dev/null | tail -1 | tr -dc "0-9")
      if [ "${free:-99}" -lt 4 ]; then find /root/ghm/exp -name "params_*.pkl" -delete 2>/dev/null;
      else find /root/ghm/exp -name "params_*.pkl" -mmin +150 -delete 2>/dev/null; fi' 2>/dev/null
  done
}

while :; do
  for slot in "${SLOTS[@]}"; do
    box=$(box_of "$slot"); gpu=$(gpu_of "$slot")
    if gpu_idle "$box" "$gpu"; then
      mapfile -t ENVS < "$ENVFILE"; ne=${#ENVS[@]}; [ "$ne" -eq 0 ] && continue
      n=$(cat "$CNT"); env="${ENVS[$(( n % ne ))]}"; seed=$(( n / ne ))
      tag=$(echo "$env" | sed 's/-navigate-singletask//; s/-play-singletask//; s/-v0//; s/-/_/g')
      name="ghm_a${n}_${tag}_s${seed}"
      pre=$(mem_prefix "$box")
      say "LAUNCH $name ($env) -> $slot"
      timeout 45 ssh -o ConnectTimeout=20 "$box" \
        "$pre bash /root/ghm/launch_infom.sh $gpu $env $seed $name --agent=agents/ghm.py ${EXTRA[$box]}" >/dev/null 2>&1
      echo $(( n + 1 )) > "$CNT"
      sleep 80
    fi
  done
  harvest
  sleep 180
done
