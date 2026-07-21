#!/usr/bin/env bash
# EXP#13 autonomous eval guard. Waits for the 3 training pids to finish, then runs
# eval_all_13.sh, waits for the 3 curve JSONs, then finalizes RESULTS.json.
# Defends against the campaign's recurring "training finishes but eval never runs" bug.
set -u
DIR=/root/tdmpc_glass/exp/tdmpc_glass/hl_orient
LR=$DIR/logs
cd $DIR
echo "[watch] started $(date) pids=$*" >> $DIR/watch.log
# wait for all training pids to exit
for pid in "$@"; do
  while kill -0 "$pid" 2>/dev/null; do sleep 120; done
  echo "[watch] pid $pid exited $(date)" >> $DIR/watch.log
done
sleep 30
echo "[watch] launching eval_all_13.sh $(date)" >> $DIR/watch.log
bash eval_all_13.sh >> $DIR/watch.log 2>&1
# wait for the 3 curve jsons to have a 'final' key (eval done)
need="curve_orient_a1_s1 curve_orient_a1_s2 curve_topdown_a1_s1"
for i in $(seq 1 240); do  # up to ~2h
  done_n=0
  for c in $need; do
    [ -f "$DIR/$c.json" ] && grep -q '"final"' "$DIR/$c.json" 2>/dev/null && done_n=$((done_n+1))
  done
  echo "[watch] eval progress $done_n/3 $(date)" >> $DIR/watch.log
  [ $done_n -ge 3 ] && break
  sleep 60
done
echo "[watch] finalizing RESULTS.json $(date)" >> $DIR/watch.log
source /root/tdmpc_glass/venv/bin/activate
python3 finalize_13.py >> $DIR/watch.log 2>&1
echo "[watch] DONE $(date)" >> $DIR/watch.log
