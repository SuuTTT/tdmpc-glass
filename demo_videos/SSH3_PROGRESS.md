# SSH3 Progress (SHARED box ssh3.vast.ai:11731, 4x RTX 3060)

## Date: 2026-06-21

## Connectivity / box
- ssh -o ConnectTimeout=25 -p 11731 root@ssh3.vast.ai (EC2 key). python3.12.3.
- GPUs: 4x RTX 3060 12GB. GPU0 = MAHJONG project (untouched). We use GPU1/2/3 ONLY.

## GUARDRAILS HONORED
- Did NOT touch GPU0, /root/mahjong, /root/*.py, /root/ckpt, or any mahjong process.
- Mahjong is alive (currently in eval phase: bench_vs_bot.py / verify_lever.sh / __main__.py workers,
  fanbw_gate.sh orchestrator). GPU0 idle BETWEEN their own training cycles — their lifecycle, not us.
- All our work isolated under /root/tdmpc_glass/ (separate dir + venv).

## Setup: DONE
- Code mirrored from b3060 via EC2 staging:
  /root/tdmpc_glass/helios-rl/{src,scripts} (6.7M), /root/tdmpc_glass/mujoco_playground_repo (2.4G).
- venv /root/tdmpc_glass/venv (python3.12) installed with EXACT b3060 pins via --no-deps
  (avoids dm-control 1.0.41 vs mujoco 3.8.0 resolver conflict; mirrors b3060 1:1).
  jax/jaxlib/jax-cuda12-*==0.10.2, nvidia-*-cu12 12.9.x, brax 0.14.2, mujoco/mjx 3.8.0,
  flax 0.12.7, optax 0.2.8, numpy 2.4.6, dm_control 1.0.41. playground editable installed.
- ENV setup OK: YES

## VERIFY EARLY: PASS
- jax-sees-GPU: YES. CUDA_VISIBLE_DEVICES=1 -> [CudaDevice(id=0)], 2048x2048 matmul OK.
- mujoco import: OK with MUJOCO_GL=egl (loader already present on this box).
- Smoke test (8k steps, GPU1, tdmpc2 PandaPickCube): PASS, RC=0, wrote .csv + _realsuccess.csv. Cleaned up.

## RUNS LAUNCHED (detached setsid, staggered 25s): YES
Env per run: PYTHONPATH=.../helios-rl/src:.../mujoco_playground_repo MJPG_IMPL=jax MUJOCO_GL=egl
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 (mem-cap 0.5 -> ~400-700MB/GPU, never hogs a 3060)
NO --save_full_state (disk-safe). eval_interval=50000 (PandaPickCube default). 30M steps each.
- GPU1: ssh3_vanilla_s2  tdmpc2 PandaPickCube seed2 jumpy_k0   log run_vanilla_s2.log
- GPU2: ssh3_small_s2    + TDMPC_LATENT_DIM=256 TDMPC_HIDDEN=256,256 seed2  log run_small_s2.log
- GPU3: ssh3_small_s3    + TDMPC_LATENT_DIM=256 TDMPC_HIDDEN=256,256 seed3  log run_small_s3.log

## STATUS (first sps, warmup ramp; will climb toward 3060 steady-state ~241 vanilla / ~337 small)
- vanilla_s2: training, sps~158 @ step 25.6k
- small_s2:   training, sps~190 @ step 25.6k
- small_s3:   training, sps~92  @ step 10k (still ramping)
- All 3 procs alive on GPU1/2/3. First realsuccess CSV writes at 50k eval.

## DISK: 15G free (started 21G; our footprint ~6G after pip-cache purge). Safe (>5G).
## ETA: ~25-34h for 30M at 3060 speeds. Not waiting for completion.
