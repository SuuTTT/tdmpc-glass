#!/bin/bash
cd /root/helios-rl/exp/tdmpc_glass/results_page
while true; do
  /root/helios-rl/.venv/bin/python gen_page.py >> update.log 2>&1
  sleep 600
done
