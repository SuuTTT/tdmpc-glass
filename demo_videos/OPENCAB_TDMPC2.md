# TD-MPC2 on PandaOpenCabinet — does a world model crack the multi-step skill?

**Date:** 2026-06-22 | **Box:** ssh3 (4× RTX 3060), GPU2=vanilla / GPU3=jumpy
**Task:** PandaOpenCabinet (multi-step reach → grasp → **pull**), mujoco_playground, impl=jax
**Success metric:** `box_target >= 0.9` — handle pulled to target, where `box_target = (1 - tanh(5·‖target−box‖))` **gated by `reached_box`** (must grasp first). Per-eval success = fraction of 5 eval episodes whose per-episode max box_target ≥ 0.9.
**Steps:** 2,000,000 each, eval/50k (40 evals). **All numbers READ FROM the `*_realsuccess.csv` sidecars — not fabricated.**

> Code change: the real-success sidecar guard was PickCube-only (`_is_pickcube = "PandaPickCube" in env_id`). I extended it to also fire for `PandaOpenCabinet` (identical reward structure: `box_target` metric + `reached_box` info). Backup at `run_benchmark.py.bak_opencab`. Verified the sidecar CSV writes correctly for OpenCabinet.

## Results

| Run | Real success (box_target≥0.9) peak | @ step | Real success @2M (final) | Grasp (reached) peak | Grasp @2M | Peak return | Wallclock |
|---|---|---|---|---|---|---|---|
| **PPO (prior baseline)** | **0.98** | ~33M | — | — | — | — | — |
| **vanilla TD-MPC2** (jumpy_k=0) | **0.20** (1/5 eps, pi policy) | 1.2M & 1.3M | **0.00** | pi 0.40 @800k, mppi 0.33 | **0.00** | 3971 @800k | 16,173 s |
| **jumpy TD-MPC2** (k=8, plan, n_macro=3, H=16) | **0.00** | — | **0.00** | **0.00** (never grasped) | **0.00** | 2658 | 28,350 s |

- **Vanilla:** only **sporadic** genuine pulls — 1 of 5 eval episodes at 1.2M and 1.3M (peak 20%), then collapses back to **0% by 2M**. Grasps (reached) flicker up to 0.40 (pi) / 0.33 (MPPI plan) then vanish. MPPI planning never produced a single success-level pull.
- **Jumpy:** across **all 40 evals and all 3 controllers (pi/mppi/jumpy), success=0, reached=0, box_target=0 — it NEVER grasped the handle once** in 2M steps. The k=8 macro-dynamics head *did* win on prediction (jumpy_err ≈0.15 < iter1_err ≈0.25, ratio ≈0.6, stable throughout) — temporal abstraction works as a *world-model mechanism* but delivered **zero task progress**. ~3× slower wallclock.

## Verdict

**A learned world model (TD-MPC2), with or without temporal/skill abstraction, does NOT solve PandaOpenCabinet — and is NOT more sample-efficient than PPO.**

- **World model:** vanilla finds only transient, unstable pulls (peak 20% @1.2–1.3M, 0% at 2M). PPO's **98%@~33M is uncontested**; TD-MPC2 at 2M is nowhere near.
- **Temporal abstraction:** jumpy is **strictly worse** than vanilla (0 grasps vs vanilla's sporadic grasps/pulls). Better k-step prediction ≠ better control on this multi-step skill.
- **Reward-hacking confirmed (partial):** return climbs to ~4000 (dense `gripper_box` / `robot_target_qpos` shaping) while real success stays ~0 — the policy **hovers/touches the handle to farm shaping reward but does not execute the pull**. Same dense-reward decoupling observed on PickCube. The peak-return eval (800k) had reached=0.40 but **success=0** — touched, never pulled.

**Bottom line:** the world model's planning surfaces transient grasps but cannot reliably chain reach→grasp→**pull**; temporal abstraction does not help. Neither cracks the multi-step skill in 2M steps.

Full machine-readable results: `ssh3:/root/tdmpc_glass/opencab_tdmpc2_RESULTS.json`
CSVs: `ssh3:/root/tdmpc_glass/helios-rl/exp/benchmark/tdmpc2_PandaOpenCabinet_opencab_{vanilla,jumpy}_s1_realsuccess.csv`
