#!/usr/bin/env bash
# GHM keep-busy loop (EC2 control-plane). Keeps all 4 GPUs on g3 (2x4090) + g3090 (2x3090)
# pinned with horizon-conditioned GHM training on OGBench cube-single, self-refilling forever.
# GPU-idle-driven: if a slot's GPU is free (<800 MiB), launch the next job from a cycling queue.
# Harvests finetuning_eval.csv to EC2 each tick. Idempotent + restartable. Stop: remove cron + kill pid.
set -u
ROOT=/home/ubuntu/tdmpc-glass
GH=$ROOT/exp/tdmpc_glass/ghm
MIRROR=$GH/mirror
CNT=$GH/counter
LOG=$GH/loop.log
mkdir -p "$MIRROR"
echo $$ > "$GH/loop.pid"
[ -f "$CNT" ] || echo 0 > "$CNT"
SLOTS=( "g3:0" "g3:1" "g3090:0" "g3090:1" )
# Env rotation read from a FILE each tick (one OGBench env name per line) so antmaze can be
# folded into the rotation without restarting the loop. Seeded with the 5 cube-single tasks.
ENVFILE=$GH/envs.txt
if [ ! -f "$ENVFILE" ]; then
  for t in 1 2 3 4 5; do echo "cube-single-play-singletask-task${t}-v0"; done > "$ENVFILE"
fi

say(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
box_of(){ echo "${1%%:*}"; }; gpu_of(){ echo "${1##*:}"; }

gpu_idle(){ # box gpu -> 0 if <800 MiB used
  local mu
  mu=$(timeout 20 ssh -o ConnectTimeout=12 "$1" "nvidia-smi -i $2 --query-gpu=memory.used --format=csv,noheader,nounits" 2>/dev/null | tr -dc '0-9')
  [ -n "$mu" ] && [ "$mu" -lt 800 ]
}

harvest(){
  for box in g3 g3090; do
    timeout 40 rsync -az -m -e "ssh -o ConnectTimeout=15" \
      --include='*/' --include='finetuning_eval.csv' --include='pretraining_*.csv' --exclude='*' \
      "$box:/root/ghm/exp/" "$MIRROR/$box/" 2>/dev/null
    # DISK SAFETY (38h unattended): after harvesting eval CSVs, prune model checkpoints from
    # FINISHED runs (>120 min old = past one full ~2h run cycle). If disk is tight, prune all.
    timeout 30 ssh -o ConnectTimeout=15 "$box" '
      free=$(df --output=avail -BG /root | tail -1 | tr -dc "0-9")
      if [ "${free:-99}" -lt 5 ]; then find /root/ghm/exp -name "params_*.pkl" -delete 2>/dev/null;
      else find /root/ghm/exp -name "params_*.pkl" -mmin +120 -delete 2>/dev/null; fi' 2>/dev/null
  done
}

while :; do
  for slot in "${SLOTS[@]}"; do
    box=$(box_of "$slot"); gpu=$(gpu_of "$slot")
    if gpu_idle "$box" "$gpu"; then
      mapfile -t ENVS < "$ENVFILE"; ne=${#ENVS[@]}; [ "$ne" -eq 0 ] && continue
      n=$(cat "$CNT"); env="${ENVS[$(( n % ne ))]}"; seed=$(( n / ne ))
      tag=$(echo "$env" | sed 's/-play-singletask//; s/-navigate-singletask//; s/-v0//; s/-/_/g')
      name="ghm_a${n}_${tag}_s${seed}"
      say "LAUNCH $name ($env) -> $slot"
      timeout 45 ssh -o ConnectTimeout=20 "$box" \
        "bash /root/ghm/launch_infom.sh $gpu $env $seed $name --agent=agents/ghm.py" >/dev/null 2>&1
      echo $(( n + 1 )) > "$CNT"
      sleep 80   # let the new job grab GPU memory before re-scanning (avoids double-launch)
    fi
  done
  harvest
  sleep 180
done
