# InFOM on PandaPickCube (offline flow-occupancy RL, neutral PPO dataset)

Box b3060 GPU3. InFOM scaffold /root/ghm/infom/. Dataset:
/root/helios-rl/exp/tdmpc_glass/baselines_ppo_sac/panda_ppo_dataset/panda_pickcube.npz
(76,800 transitions, 512 eps, 75% success, obs66/act8, neutral PPO rollouts).

## Did InFOM ingest + train on Panda?  YES.
End-to-end pipeline verified: data ingest -> flow-occupancy + BC pretrain -> critic/actor
finetune -> eval in a mujoco_playground PandaPickCube env. Success metric = box_target>=0.9
at any step in an episode (SAME honest metric as PPO 66->83% and TD-MPC2 scoring).

## Env branch added (the remaining adaptation flagged by the prior agent)
- NEW: /root/ghm/infom/envs/panda_utils.py
  - load_panda_dataset(): loads the local npz DIRECTLY (ogbench.load_dataset drops rewards),
    derives next_observations within trajectories, reward = per-step `successes` flag
    (box_target>=0.9, achieved-on-arrival), masks=1-terminal, 95/5 train/val split.
  - PandaPickCubeGymEnv(gymnasium.Env): wraps mujoco_playground registry.load(
    "PandaPickCube", impl="jax"); gymnasium reset/step; info['success']=box_target>=0.9.
- PATCH: /root/ghm/infom/envs/env_utils.py make_env_and_datasets -> `elif "panda" in env_name`.
- PATCH: main.py + utils/log_utils.py wandb-optional (wandb not installed); installed distrax.

## Smoke (2k pretrain + 2k finetune, 5 eval eps) — PASS
  step 2001: episode.success=1.0 (5/5), box_target=0.965
  step 4000: episode.success=0.8 (4/5), box_target=0.779
InFOM grasps+lifts from the neutral PPO data even at tiny budget.

## Full run (500k pretrain + 250k finetune, eval every 50k, 50 eval eps) — IN FLIGHT
Results CSV: /root/ghm/infom/exp/panda_full/debug/<run>/finetuning_eval.csv
(peak/final to be filled when run completes).
