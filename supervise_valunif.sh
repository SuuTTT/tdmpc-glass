#!/usr/bin/env bash
# Durable supervisor for the valunif 3-arm DMControl runs.
# Owns 6 jobs over 4 GPU slots; adopts already-running matching jobs;
# relaunches crashes; never double-books a GPU; exits when all CSVs complete.
set -u
REPO=/root/helios-rl
cd "$REPO" || exit 1
LOG=/tmp/valunif_supervisor.log
echo "[sup] start $(date -u +%FT%TZ)" >> "$LOG"

# job queue (TASK:SEED) and the 4 gpu slots
QUEUE=(WalkerWalk:1 WalkerWalk:2 WalkerWalk:3 CheetahRun:1 CheetahRun:2 CheetahRun:3)
NGPU=4
declare -A SLOT     # gpu -> TASK:SEED currently assigned
qi=0

csv_path() {  # $1=TASK $2=SEED
  ls "$REPO/exp/tdmpc_glass/$1_unif_dmc_$1_L16_valunif_s$2"/seed_*_phase.csv 2>/dev/null | head -1
}
rows() { f=$(csv_path "$1" "$2"); [ -n "$f" ] && wc -l < "$f" 2>/dev/null || echo 0; }
is_done() { [ "$(rows "$1" "$2")" -ge 11 ]; }
is_live() {  # any run_benchmark proc for this exact TASK+SEED
  pgrep -f "tasks $1 --total_steps 250000 --seed $2 " >/dev/null 2>&1
}
launch() {  # $1=GPU $2=TASK $3=SEED
  echo "[sup] launch gpu=$1 $2:$3 $(date -u +%T)" >> "$LOG"
  setsid nohup bash "$REPO/exp_unif_run.sh" "$1" "$2" "$3" valunif 16 250000 25000 \
      >/tmp/vu_sup_${2}_s${3}.out 2>&1 </dev/null &
  disown
}

while true; do
  alldone=1
  for q in "${QUEUE[@]}"; do
    t=${q%:*}; s=${q#*:}
    is_done "$t" "$s" || alldone=0
  done
  [ "$alldone" -eq 1 ] && { echo "ALL_DONE $(date -u +%FT%TZ)" >> "$LOG"; break; }

  for g in $(seq 0 $((NGPU-1))); do
    job=${SLOT[$g]:-}
    if [ -n "$job" ]; then
      t=${job%:*}; s=${job#*:}
      if is_done "$t" "$s"; then SLOT[$g]=""; job=""; fi
    fi
    if [ -z "${SLOT[$g]:-}" ]; then
      # assign next not-done, not-already-assigned job
      while [ "$qi" -lt "${#QUEUE[@]}" ]; do
        cand=${QUEUE[$qi]}; ct=${cand%:*}; cs=${cand#*:}
        assigned=0
        for gg in $(seq 0 $((NGPU-1))); do [ "${SLOT[$gg]:-}" = "$cand" ] && assigned=1; done
        if is_done "$ct" "$cs" || [ "$assigned" -eq 1 ]; then qi=$((qi+1)); continue; fi
        SLOT[$g]=$cand; break
      done
      job=${SLOT[$g]:-}
    fi
    [ -z "$job" ] && continue
    t=${job%:*}; s=${job#*:}
    if ! is_done "$t" "$s" && ! is_live "$t" "$s"; then launch "$g" "$t" "$s"; fi
  done
  sleep 60
done
echo "[sup] exit $(date -u +%FT%TZ)" >> "$LOG"
