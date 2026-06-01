# CLAUDE.md — TD-MPC-Glass (standalone)

Guidance for agents working in this repo.

## Read first
- `AGENT_HANDOFF_CONTEXT.md` — live experiment status (more authoritative than README for what's running).
- `docs/operations/launch_dashboard.md` — control-plane runbook + queue cheat sheet.

## The one rule that cannot be broken
**This box (EC2) has NO GPU and NEVER trains.** There is no `"local"` entry in
`BOXES`. All training runs on remote vast.ai GPU workers via the queue daemon.
Do not add a local training slot, and do not run `run_benchmark.py` here.

## Two path layouts (do not confuse)
- **Control plane = THIS box**, rooted at `/home/ubuntu/tdmpc-glass`. The three
  daemons in `control/` run here against `scripts/queues/central_queue.json`.
- **Workers = vast.ai boxes**, rooted at `/root/helios-rl` with their own
  `/root/venv` (JAX/GPU) and `/root/mujoco_playground_repo`. The daemon rsyncs
  `scripts/` + `src/` to `/root/helios-rl/` on each worker before launching.
  Remote commands in the daemons use `/root/helios-rl`; local paths use the EC2
  path. When editing the daemons, keep that split — don't rewrite remote `/root/...`
  paths to the EC2 path.

## Control plane
- `control/start_center.sh` — start/restart all three daemons (idempotent; pkills first).
- `control/web_dashboard.py` — Flask, port 5055 (`DASHBOARD_PORT` to override).
- `control/task_queue_daemon.py` — claims `pending` tasks, SSH-launches on idle workers. One-master rule.
- `control/iter5_stream_remotes.sh` — rsyncs worker CSVs into `exp/tdmpc_glass/remote_mirror/`.
- Venv: `.venv` (Flask only). Worker training deps are NOT installed here.

## Queue hygiene (from docs/operations)
- Back up `scripts/queues/central_queue.json` before any bulk mutation.
- After a fleet shutdown, reset stale `running` → `pending` (append `recovery_history`)
  BEFORE restarting the queue daemon. Keep historical `done`/`failed` rows.
- `fcntl` file lock protects concurrent edits; the daemon writes via tmp+rename.

## Training code (runs on workers)
- `scripts/run_benchmark.py --algos tdmpc-glass --tasks HopperHop ...` is the only
  driver that matters. Glass knobs are `--glass_*` flags.
- Algorithm milestones live in `src/helios/algorithms/*.py`; each file's docstring
  is the spec (milestone reward, architecture, key fixes). `tdmpc_glass.py` imports
  only numpy/jax/flax/optax + sibling `tdmpc2`.
- Output tagging: set `TDMPC_GLASS_OUTPUT_TAG=<name>` so phases don't clobber each other.
- Known-falsified — do NOT repropose: `stopgrad_graph=false` (Phase 2),
  Path P/Pa intrinsic cluster entropy, Path 7 cluster-id obs, Phase 1c act_noise anneal.

## research-os alignment
This repo is the origin of the research-os queue design and is registered as the
`tdmpc_glass` research-os project. Keep conventions aligned (file-backed queues,
worker registry, inspect-don't-daemon-by-default). The live queue stays here.
