# TD-MPC-Glass

Standalone research codebase for **TD-MPC-Glass** — TD-MPC2 augmented with
prototype-clustering of latents (a structural-entropy auxiliary loss) — on
**MuJoCo Playground** continuous-control tasks (primarily `HopperHop`).

Extracted from `helios-rl` into a standalone repo on 2026-06-01. This is the
**origin of the file-backed queue design** that `research-os` generalizes
(see [research-os HANDOFF](../research-os/HANDOFF.md)); this repo keeps its own
proven control plane and aligns its conventions to that paradigm.

---

## Architecture: control plane vs. workers

```
   ┌─────────────────────────────┐         SSH (key-based)         ┌──────────────────┐
   │  THIS BOX (EC2, NO GPU)      │  ───────────────────────────▶  │ vast.ai GPU box  │
   │  = control plane only        │   rsync scripts/+src/, launch   │ /root/helios-rl  │
   │                              │                                 │ /root/venv (JAX) │
   │  control/web_dashboard.py    │  ◀─── rsync exp CSVs (mirror) ── │ run_benchmark.py │
   │  control/task_queue_daemon.py│                                 └──────────────────┘
   │  control/iter5_stream_*.sh   │              ... × N workers
   │  scripts/queues/central_queue.json
   └─────────────────────────────┘
```

**Hard rule: this box NEVER trains.** It has no GPU. There is intentionally no
`"local"` entry in `BOXES`. All training runs on remote vast.ai GPU workers.
(The original local-4070-Ti training center was destroyed; its role is gone, not
replaced on the control box.)

---

## Layout

```
tdmpc-glass/
├── README.md                 # this file
├── CLAUDE.md                 # guidance for agents
├── AGENT_HANDOFF_CONTEXT.md  # live experiment status (read first)
├── .venv/                    # control-plane venv (Flask only; NOT for training)
├── control/                  # the control plane (runs HERE, on EC2)
│   ├── start_center.sh       # launch all three daemons
│   ├── web_dashboard.py      # Flask UI + REST queue API (:5055)
│   ├── task_queue_daemon.py  # claim pending tasks → SSH-launch on idle workers
│   ├── iter5_stream_remotes.sh # rsync remote CSVs into exp/.../remote_mirror/
│   └── idea_queue.py
├── scripts/                  # code that RUNS ON WORKERS (rsync'd to /root/helios-rl)
│   ├── run_benchmark.py      # the only training driver that matters
│   ├── render_glass_rollout.py
│   ├── run_phase*.sh         # phase launchers referenced by the queue
│   └── queues/
│       ├── central_queue.json     # ← the live task queue (source of truth)
│       └── *.bak_*                # timestamped backups
├── src/helios/               # algorithm code (import path: helios.algorithms.*)
│   └── algorithms/{tdmpc_glass,tdmpc2,ppo,sac,...}.py
├── docs/                     # operations + iteration ledgers (was docs/tdmpc-glass)
│   └── operations/{launch_dashboard,experiment_ops,ec2_*,...}.md
└── exp/tdmpc_glass/          # experiment outputs + remote_mirror/ (dashboard reads these)
```

## Quick start (control plane)

```bash
cd ~/tdmpc-glass
bash control/start_center.sh        # starts dashboard + queue daemon + streamer
# dashboard: http://localhost:5055   (tunnel: ssh -L 5055:localhost:5055 <ec2>)
```

Stop everything:
```bash
pkill -f 'tdmpc-glass/control/web_dashboard.py'
pkill -f 'tdmpc-glass/control/task_queue_daemon.py'
pkill -f 'tdmpc-glass/control/iter5_stream_remotes.sh'
```

Add a training task (UI form, or curl):
```bash
curl -s -X POST http://localhost:5055/api/queue -H 'Content-Type: application/json' \
  -d '{"label":"phaseX seed 1","launcher":"scripts/run_phasei9_glass_probe.sh","env":"SEEDS=1","priority":10}'
```
Lower `priority` runs first (default 10). Full cheat-sheet:
[docs/operations/launch_dashboard.md](docs/operations/launch_dashboard.md).

## Worker fleet (2026-06-01)

| Tag | Host:port | GPU |
|---|---|---|
| ssh1_2080ti | ssh1.vast.ai:34217 | RTX 2080 Ti 22GB |
| ssh1_a4000 | ssh1.vast.ai:24456 | RTX A4000 16GB |
| ssh2_a4000 | ssh2.vast.ai:18950 | RTX A4000 16GB |
| ssh3_a4000 | ssh3.vast.ai:17426 | RTX A4000 16GB |
| ssh6_titanv | ssh6.vast.ai:31740 | Titan V 12GB |
| ssh9_a4000 | ssh9.vast.ai:16690 | RTX A4000 16GB |
| ssh9_2060_gpu0-3 | ssh9.vast.ai:17647 | 4× RTX 2060 6GB |

Registry lives in `BOXES` at the top of both `control/web_dashboard.py` and
`control/task_queue_daemon.py` — **keep them in sync**. SSH key:
`~/.ssh/vastai_id_ed25519` (override with `SSH_IDENTITY_FILE`).

> **One-master rule:** never run a second queue daemon against the same fleet —
> two daemons double-claim tasks and launch duplicate runs.

## Relationship to research-os

This is registered as the `tdmpc_glass` research-os project
(`research-os/research/tdmpc_glass/`). Its `central_queue.json` is the concrete,
battle-tested ancestor of research-os's `run_queue.json` + `ros.py` dispatch.
Conventions (file-backed queues, worker registry, no-daemon-by-default inspection)
are aligned; the live queue/daemon are kept here because they already work.

See [docs/operations/](docs/operations/) for the full operational runbook
(experiment_ops, fleet_rebuild_recovery, hardware_req, storageAWS, …).
