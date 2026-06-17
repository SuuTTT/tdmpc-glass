#!/usr/bin/env bash
# Paper-A discrimination orchestrator (EC2 control-plane side).
# Keeps g3 (2x4090) + g3090 (2 GPUs) BUSY with the 2x2 discrimination matrix
#   (clean/distractor x bisim/no-bisim), multiple seeds, multiple tasks.
# Fills idle slots from a PENDING queue, detects completion via the launcher's
# "all done status=" marker, harvests *_phase.csv (value_r2 + returns over training)
# to EC2, and SELF-REFILLS (adds a new seed round) so GPUs never go idle.
# Idempotent + restartable. State columns: name task seed dist bisim status slot
set -u
ROOT=/home/ubuntu/tdmpc-glass
STATE=$ROOT/exp/tdmpc_glass/paperA/state.tsv
MIRROR=$ROOT/exp/tdmpc_glass/paperA/mirror
LOG=$ROOT/exp/tdmpc_glass/paperA/loop.log
mkdir -p "$MIRROR"
echo $$ > "$ROOT/exp/tdmpc_glass/paperA/loop.pid"
SLOTS=( "g3:0" "g3:1" "g3090:0" "g3090:1" )   # g3090:1 protected by gpu_idle guard
STEPS=500000
MAXSEED=9   # self-refill ceiling

say(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
box_of(){ echo "${1%%:*}"; }
gpu_of(){ echo "${1##*:}"; }

if [ ! -f "$STATE" ]; then
  mkdir -p "$(dirname "$STATE")"
  cat > "$STATE" <<'EOF'
cleanNB	CheetahRun	0	0	0	RUNNING	g3:1
distNB	CheetahRun	0	32	0	RUNNING	g3:0
distB	CheetahRun	0	32	0.5	RUNNING	g3090:0
cleanB	CheetahRun	0	0	0.5	PENDING	-
cleanNB	CheetahRun	1	0	0	PENDING	-
distNB	CheetahRun	1	32	0	PENDING	-
distB	CheetahRun	1	32	0.5	PENDING	-
cleanB	CheetahRun	1	0	0.5	PENDING	-
cleanNB	CheetahRun	2	0	0	PENDING	-
distNB	CheetahRun	2	32	0	PENDING	-
distB	CheetahRun	2	32	0.5	PENDING	-
cleanB	CheetahRun	2	0	0.5	PENDING	-
wcleanNB	WalkerWalk	0	0	0	PENDING	-
wdistNB	WalkerWalk	0	32	0	PENDING	-
wdistB	WalkerWalk	0	32	0.5	PENDING	-
wcleanB	WalkerWalk	0	0	0.5	PENDING	-
wcleanNB	WalkerWalk	1	0	0	PENDING	-
wdistNB	WalkerWalk	1	32	0	PENDING	-
wdistB	WalkerWalk	1	32	0.5	PENDING	-
wcleanB	WalkerWalk	1	0	0.5	PENDING	-
EOF
  say "seeded state with discrimination queue"
fi

gpu_idle(){ # box gpu  -> 0 if free (mem.used < 800 MiB)
  local mu
  mu=$(timeout 20 ssh -o ConnectTimeout=12 "$1" "nvidia-smi -i $2 --query-gpu=memory.used --format=csv,noheader,nounits" 2>/dev/null | tr -dc '0-9')
  [ -n "$mu" ] && [ "$mu" -lt 800 ]
}

launch(){ # name task seed dist bisim box gpu
  local name=$1 task=$2 seed=$3 dist=$4 bisim=$5 box=$6 gpu=$7
  timeout 45 ssh -o ConnectTimeout=20 "$box" "cd /root/helios-rl && setsid bash -c '. .venv/bin/activate && export PYTHONPATH=/root/helios-rl/src:/root/mujoco_playground_repo MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$gpu MJPG_IMPL=jax EVAL_NEPS=5 TDMPC_GLASS_OUTPUT_TAG=paperA_${name}_s${seed} PROBE_ID=paperA_disc_${name} ALGO=tdmpc2 TASK=$task SEEDS=$seed DISTRACTOR_DIMS=$dist BISIM_COEF=$bisim TOTAL_STEPS=$STEPS CODE_SHA=paperA && bash scripts/run_dmc_baseline.sh' > logs/paperA_${name}_s${seed}.out 2>&1 < /dev/null &" >/dev/null 2>&1
}

is_done(){ # name seed box
  timeout 25 ssh -o ConnectTimeout=15 "$3" "grep -q 'all done status=' /root/helios-rl/logs/paperA_${1}_s${2}.out 2>/dev/null" 2>/dev/null
}

harvest(){ # name seed box
  local d="$MIRROR/${1}_s${2}"; mkdir -p "$d"; local f files
  files=$(timeout 25 ssh -o ConnectTimeout=15 "$3" "find /root/helios-rl/exp -path '*_${1}_*' -name 'seed_${2}_phase.csv' 2>/dev/null" 2>/dev/null)
  for f in $files; do timeout 40 rsync -az -e "ssh -o ConnectTimeout=15" "$3:$f" "$d/" 2>/dev/null; done
  timeout 40 rsync -az -e "ssh -o ConnectTimeout=15" "$3:/root/helios-rl/logs/paperA_${1}_s${2}.out" "$d/" 2>/dev/null
}

while :; do
  mapfile -t LINES < "$STATE"
  declare -A BUSY=()
  : > "$STATE.tmp"
  for ln in "${LINES[@]}"; do
    IFS=$'\t' read -r name task seed dist bisim status slot <<< "$ln"
    if [ "$status" = "RUNNING" ]; then
      box=$(box_of "$slot")
      if is_done "$name" "$seed" "$box"; then
        harvest "$name" "$seed" "$box"; status=DONE; slot="-"; say "DONE $name($task) s$seed"
      else
        BUSY[$slot]=1; harvest "$name" "$seed" "$box"
      fi
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$task" "$seed" "$dist" "$bisim" "$status" "$slot" >> "$STATE.tmp"
  done
  mv "$STATE.tmp" "$STATE"

  for slot in "${SLOTS[@]}"; do
    [ -n "${BUSY[$slot]:-}" ] && continue
    pend=$(awk -F'\t' '$6=="PENDING"{print; exit}' "$STATE")
    [ -z "$pend" ] && continue
    box=$(box_of "$slot"); gpu=$(gpu_of "$slot")
    gpu_idle "$box" "$gpu" || { say "slot $slot occupied by foreign job, skip"; continue; }
    IFS=$'\t' read -r name task seed dist bisim _ _ <<< "$pend"
    say "LAUNCH $name($task) s$seed dist=$dist bisim=$bisim -> $slot"
    launch "$name" "$task" "$seed" "$dist" "$bisim" "$box" "$gpu"; sleep 30
    awk -F'\t' -v n="$name" -v s="$seed" -v sl="$slot" 'BEGIN{OFS=FS} $1==n&&$3==s{$6="RUNNING";$7=sl}{print}' "$STATE" > "$STATE.tmp" && mv "$STATE.tmp" "$STATE"
    BUSY[$slot]=1
  done

  # self-refill: nothing pending or running -> add next CheetahRun 2x2 seed round
  if ! grep -qE $'\t'"(PENDING|RUNNING)"$'\t' "$STATE"; then
    ns=$(awk -F'\t' '$2=="CheetahRun"{if($3>m)m=$3}END{print m+1}' "$STATE")
    if [ "$ns" -le "$MAXSEED" ]; then
      say "REFILL CheetahRun 2x2 seed $ns"
      { printf 'cleanNB\tCheetahRun\t%s\t0\t0\tPENDING\t-\n' "$ns"
        printf 'distNB\tCheetahRun\t%s\t32\t0\tPENDING\t-\n' "$ns"
        printf 'distB\tCheetahRun\t%s\t32\t0.5\tPENDING\t-\n' "$ns"
        printf 'cleanB\tCheetahRun\t%s\t0\t0.5\tPENDING\t-\n' "$ns"; } >> "$STATE"
    else
      say "MAXSEED reached, stopping refill"; break
    fi
  fi
  sleep 180
done
say "LOOP EXIT"
