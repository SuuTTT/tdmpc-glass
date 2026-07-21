# Cross-benchmark comparison matrix (Part 17) — PLAN + PROGRESS
Goal: comparable table of {TD-MPC2 vanilla, TD-MPC2 jumpy(ours), InFOM, CompPlan} x {Panda-manip, OGBench cube-single, OGBench antmaze-medium}.
Honest caveat: mixes paradigms (TD-MPC2 online MBRL vs InFOM/CompPlan offline goal-conditioned) -> a capability/landscape matrix, label protocols.

ALREADY HAVE:
- TD-MPC2 vanilla+jumpy on Panda: PickCube +60%, Pick-Orient +161% (returns, D2 suite).
- InFOM on cube-single: reproduced (peak 1.0/0.94/0.90/0.96/0.90).
- CompPlan on cube/antmaze: diagnosed (occupancy fixable; action-agnostic -> needs TD-Flow).

TO RUN:
- Phase 0: tdmpc MJX env on b3060 + OGBench bridge (adapter to train our TD-MPC2 online on OGBench gym envs).
- Phase 1: TD-MPC2 vanilla+jumpy on cube-single + antmaze-medium (online), 3 seeds, 500k. [feasibility spike first]
- Phase 2: generate offline goal-conditioned Panda dataset -> run InFOM + CompPlan on Panda.
- Phase 3: assemble matrix + Part-17 blog (with paradigm caveat).

---
## 2026-06-20 — Phase 0 + Phase 1 EXECUTED on b3060 (4x RTX 3060, all 4 GPUs)

**Box:** b3060 = 4x RTX 3060 (12GB each). SEPARATE env from /root/ghm (InFOM). Worker root
/root/helios-rl with its own .venv. mujoco_playground cloned to /root/mujoco_playground_repo.

### Phase 0 — env setup (DONE)
- rsync EC2 scripts+src -> b3060:/root/helios-rl/. venv: jax[cuda12] (jax 0.10.2, 4 devices),
  flax/optax/hydra/omegaconf/numpy, `pip install -e mujoco_playground_repo`, then pinned
  `mujoco==3.8.0 mujoco-mjx==3.8.0`, plus `pip install ogbench` (1.2.1) in the SAME venv.
- CONFLICT resolved: ogbench pulled mujoco 3.9.0 (via dm_control 1.0.41); re-pinned to 3.8.0.
  Only residual: dm_control wants >=3.8.1 (cosmetic; dm_control not used by OGBench locomotion/
  manip envs). Both `from mujoco_playground import registry` AND `import ogbench` work at 3.8.0.
- SMOKE (DMC CheetahRun, MJPG_IMPL=jax, 5k steps): PASS. obs=17 act=6, JIT 21.8s, sps~105
  incl. warmup. Our TD-MPC2 trains on MJX. -> Phase 0 GREEN.

### Phase 1 — OGBench bridge (DONE) + runs (LAUNCHED, advancing)
**Bridge approach (additive, no MJX-loop rewrite):** `scripts/ogbench_adapter.py` wraps a vector
of N CPU OGBench gym envs behind a brax/mujoco_playground-style batched interface
(.reset/.step/.observation_size/.action_size; State with .obs/.reward/.done numpy arrays; brax-style
AUTO-RESET on terminal). run_benchmark.py gets an env-source switch:
`TDMPC_ENV_SRC=ogbench:<name>` builds the adapter instead of registry.load, makes batch_step /
single_env_step/reset NON-jit on this path (CPU gym can't be jit-traced), forces a small
N_ENVS (TDMPC_OGBENCH_NENVS, default 8), and logs success to `*_success.csv` (OGBench info['success']).
All changes guarded by `_is_ogbench`; DMC/MJX path unchanged.

**Env dims probed:** cube-single obs=28 act=5 ep_len=200; antmaze-medium obs=29 act=8 ep_len=1000.
Action space [-1,1]. CPU-only step rate: cube ~317 sps, antmaze ~1231 sps (env stepping is NOT the
bottleneck — GPU gradient updates are).

**SPIKE (cube-single, vanilla, 2000 online steps):** PASS at ~12 sps. Full pipeline (collect ->
buffer -> K updates -> pi/MPPI eval -> success log) works; untrained agent gives return -200,
success 0.000 (correct: dense -1/step until solved). >3 sps threshold cleared -> FEASIBLE.

**LAUNCHED (self-refilling launcher /root/ghm_launch.sh, launcher pid 553247):**
2 algos {vanilla --jumpy_k 0; jumpy --jumpy_k 8 --jumpy_plan --jumpy_n_macro 3 --mppi_horizon 16}
x 2 tasks {cube-single-play-singletask-task1-v0, antmaze-medium-navigate-singletask-task1-v0}
x 3 seeds = 12 runs, total_steps=200k, EVAL_NEPS=5, eval every 20k. Pinned 1/GPU, staggered 30s,
self-refills as slots free to keep all 4 GPUs busy. Logs: /root/helios-rl/ghm_logs/<tag>.log;
CSVs: /root/helios-rl/exp/benchmark/tdmpc2_<Task>{,_success}.csv.
First-fill confirmed advancing, all 4 GPUs at 100% util, 0 errors. Throughput: vanilla ~12 sps
(~5h/run), jumpy ~5 sps (~11h/run; extra k-step heads + jumpy-plan eval). jumpy mechanism healthy
(jumpy_err<iter1_err ratio<1 early). Returns/success at launch = baseline (-200 / 0.0), as expected
pre-training. HARVEST: pull tdmpc2_*_success.csv from b3060:/root/helios-rl/exp/benchmark when runs
progress; success>0 is the real signal for these sparse goal tasks.

## 2026-06-20 DIAGNOSIS — matrix is PARADIGM-LOCKED (not fairly fillable)
- OGBench cube-single: reward {-1,0} (sparse, 0 only at goal); random policy 0/5 success, -200 return.
- Our TD-MPC2 = ONLINE dense-reward MBRL -> 0 signal -> 0 success at 100k (confirmed). OGBench is OFFLINE by design.
- => "TD-MPC2 on OGBench (online)" cells are STRUCTURALLY ~0 (paradigm mismatch), not informative. Phase 1 STOPPED.
- InFOM/CompPlan on Panda: FEASIBLE only via a generated OFFLINE goal-conditioned Panda dataset (schema: observations(N,28) actions(N,5) terminals next_observations rewards{-1,0}). Phase 2 prep agent building it.
- CONCLUSION: cannot do a fair cross-paradigm head-to-head. Feasible = run InFOM/CompPlan on Panda (offline) -> compare to their cube/antmaze; present our TD-MPC2 on its native online Panda. Drop the online-on-offline cells.

## 2026-06-20 REAL-SUCCESS re-run — jumpy vs vanilla TD-MPC2 on PandaPickCube (HONEST scoring)

**Why.** Prior PandaPickCube "wins" were on the DENSE shaping return, which is reward-hacked:
render eval showed return ~400-500 with SUCCESS=False, max_box_target=0.000, reached_box mostly
False (0% real picks). This re-run scores on the REAL TASK SUCCESS signal and gives a longer
budget (1.5M vs prior 500k) to test if more training escapes the reward-hack.

**Success definition (from mujoco_playground pick.py + render_planned.py, identical logic):**
`box_target = (1 - tanh(5*(0.9*pos_err + 0.1*rot_err))) * reached_box` (raw, unscaled — the env's
reward `scales` apply 8.0 only to the summed reward, NOT to `state.metrics`). SUCCESS = box_target
metric >= 0.9 at ANY step in the episode. Also track reached_box (gripper grasped, state.info) and
box_target_max. Source: state.metrics["box_target"], state.info["reached_box"].

**STEP 1 code change (additive, guarded by `_is_pickcube = "PandaPickCube" in env_id`):**
- run_benchmark.py train_tdmpc2: eval_pi/eval_mppi/eval_jumpy now track per-episode
  max(box_target metric) and max(reached_box) and set diag {success=float(bt_max>=0.9),
  box_target_max, reached}. (DMC/ogbench paths untouched — same pattern as the `_is_ogbench` branch.)
- New sidecar `tdmpc2_PandaPickCube_<tag>_realsuccess.csv`:
  step,seed,pi_return,mppi_return,jumpy_return,pi/mppi/jumpy_success,_reached,_box_target_max.
  Console prints "[REAL-SUCCESS PandaPickCube] pi: succ=.. reached=.. box_target_max=.. | mppi: ..".
- Backup: scripts/run_benchmark.py.bak_realsuccess.
- 5k smoke (seed99, eval@2k) used to verify the success column populates (see below).
- NOTE: run requires PYTHONPATH=/root/helios-rl/src:/root/mujoco_playground_repo (helios pkg).

**STEP 2 launch — campaign script scripts/realsucc_campaign.sh (self-refilling, 1/GPU x4, setsid):**
- vanilla: --jumpy_k 0 ; jumpy: --jumpy_k 8 --jumpy_plan --jumpy_n_macro 3 --mppi_horizon 16
- 3 seeds each = 6 runs; TOTAL_STEPS=1,500,000; TDMPC_EVAL_INTERVAL=50000 (eval ~every 50k).
- Logs: /root/helios-rl/ghm_logs/realsucc_<name>.log ; markers in ghm_logs/realsucc_campaign_state/.
- CSVs: /root/helios-rl/exp/benchmark/tdmpc2_PandaPickCube_realsucc_<name>{,_realsuccess}.csv
  + per-seed curves under exp/tdmpc_glass/PandaPickCube_realsucc_<name>/.
- Coexists with the CPU-only scripted-pick agent (GPU-only here; do NOT touch scripted_pick.py/hl_pickcube/).

**STEP 3 headline questions to track as they train (hours):**
- Does REAL success (box_target>=0.9) ever rise above ~0 for vanilla or jumpy by 1.5M?
- Does jumpy beat vanilla on REAL success (NOT return)?
- Secondary: does reached_box (grasp) rate rise even if full place-at-target success stays 0?
