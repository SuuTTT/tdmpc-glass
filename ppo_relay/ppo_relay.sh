#!/usr/bin/env bash
# Relay PPO sweep CSVs b3060b -> (EC2 staging) -> b3060, every 10 min.
# b3060 cannot ssh b3060b directly, so EC2 mediates: pull, then push.
# Idempotent rsync; restartable; logs every cycle. Touches ONLY ppo_*.csv.
set -u
RELAY_DIR="/home/ubuntu/tdmpc-glass/ppo_relay"
STAGE="$RELAY_DIR/stage"
LOG="$RELAY_DIR/relay.log"
SRC="b3060b:/root/helios-rl/exp/benchmark/"
DST="b3060:/root/helios-rl/exp/benchmark/"
INTERVAL=600
mkdir -p "$STAGE"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
while true; do
  # pull only ppo_*.csv from b3060b into local stage
  if rsync -az --timeout=120 -e ssh --include='ppo_*.csv' --exclude='*' \
        "$SRC" "$STAGE/" >>"$LOG" 2>&1; then
    PULLED=$(ls "$STAGE"/ppo_*.csv 2>/dev/null | wc -l)
    # push staged ppo_*.csv to b3060
    if rsync -az --timeout=120 -e ssh "$STAGE"/ppo_*.csv "$DST" >>"$LOG" 2>&1; then
      echo "$(ts) OK relayed $PULLED ppo csv(s) b3060b->b3060" >>"$LOG"
    else
      echo "$(ts) ERR push to b3060 failed (rc=$?)" >>"$LOG"
    fi
  else
    echo "$(ts) ERR pull from b3060b failed (rc=$?)" >>"$LOG"
  fi
  sleep "$INTERVAL"
done
