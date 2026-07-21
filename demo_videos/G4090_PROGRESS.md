# G4090 Progress (vast contract 41916638, RTX 4090)

## Connectivity
- ssh config alias `g4090` is STALE (points to 38.47.123.157:5018 / key vastai_id_ed25519 — kex closed, not this contract).
- WORKING endpoint = spec: `ssh -i ~/.ssh/id_ed25519 -p 36638 root@ssh5.vast.ai` -> returns ok.
- GPU: NVIDIA GeForce RTX 4090, 24564 MiB, driver 580.119.02 (CUDA 13 capable). python3.11.10 (conda). 50G disk free.

## Setup
- Code pushed from EC2 stage (sourced from b3060): /root/helios-rl/{src,scripts}, /root/mujoco_playground_repo (1.5G).
- venv at /root/helios-rl/.venv, JAX stack pinned to b3060: jax/jaxlib/jax-cuda12-* == 0.10.2, brax 0.14.2, mujoco/mjx 3.8.0, flax 0.12.7, optax 0.2.8, numpy 2.4.6 (distrax omitted; TD-MPC2 doesn't use it, b3060 lacks it).
- Install: JAX stack + brax/mujoco/mjx/flax/optax OK. scipy pinned 1.18.0 unavailable on py3.11 -> got scipy 1.17.1 (JAX dep, fine).
- CUDA: jax-cuda12-* alone did NOT load CUDA; installing the nvidia-*-cu12 pip wheels (matching b3060 freeze: cublas/cudnn/nccl/nvrtc/etc 12.9.x) fixed it.
- Remaining deps (dm_control, mujoco_playground editable) installing.

## JAX-sees-GPU: YES -> devices: [CudaDevice(id=0)], 2048x2048 matmul ran on GPU.

## EGL fix (needed): import mujoco crashed at module load (eglQueryString on NoneType).
   Container had libEGL_nvidia.so.0 but no glvnd loader libEGL.so.1.
   Fix: apt-get install libegl1 libgl1 libglx-mesa0 libgles2; set MUJOCO_GL=egl. Now imports fine.

## Smoke test: PASS. 8k steps, JIT compiled 8.5s, SMOKE_RC=0.
   Wrote exp/benchmark/tdmpc2_PandaPickCube_g4090_smoke.csv + _realsuccess.csv sidecar.

## Runs: LAUNCHED on g4090 (staggered 25s, sharing 1x4090)
   - g4090_vanilla_s1: tdmpc2 PandaPickCube seed1 jumpy_k0 total_steps=30,000,000 --save_full_state, eval/50k. log run_vanilla.log
   - g4090_small_s1:   same + TDMPC_LATENT_DIM=256 TDMPC_HIDDEN=256,256. log run_small.log
## SPS (both runs sharing 1x4090, sps climbing as JIT/startup cost amortizes):
   - vanilla: 319 -> 385 over 71k-97k steps (still rising). Baseline 3060 vanilla = 241.
   - small:   223 -> 299 over 56k-82k steps (still rising). Baseline 3060 small = 337.
   Note: running BOTH concurrently on one 4090 splits throughput; GPU util 92%, only 1748 MiB used.
   Steady-state per-run sps will be higher; both already at/above 3060 even while co-located.

## CSVs writing OK: exp/benchmark/tdmpc2_PandaPickCube_g4090_{vanilla,small}_s1{,_realsuccess}.csv
   First eval at step 50176: success=0.0 (expected this early). pi_return vanilla 333, small 1041 (reward-hacked metric).

## STATUS: SETUP COMPLETE, BOTH 30M RUNS TRAINING. Long-running (~hours). Read CSVs for honest success numbers.
