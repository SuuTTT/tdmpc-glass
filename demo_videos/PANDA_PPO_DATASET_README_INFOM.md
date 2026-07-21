# Panda PPO offline dataset for InFOM (Task 3)

NEUTRAL dataset: rolled out the BEST PPO baseline policy (ckpt 000108134400, the 83.2% real-success
peak of the 80M run) on PandaPickCube. PPO is a standard baseline (not a method under test), so these
rollouts are a defensible neutral "expert/play" dataset (vs TD-MPC2 rollouts which would be unfair).

## Contents
- panda_pickcube.npz  -- the dataset (OGBench raw layout)
- DATASET_META.json   -- machine-readable stats

## Schema (panda_pickcube.npz), OGBench raw layout
- observations : (77312, 66) float32  -- per-traj s0..sT (includes final post-terminal state)
- actions      : (77312, 8)  float32  -- a0..a_{T-1} + dummy last (masked by terminal)
- terminals    : (77312,) float32     -- 1.0 on last row of each of the 512 trajectories
- rewards      : (77312,) float32     -- STOCK env reward (documentation; InFOM goal-relabels)
- successes    : (77312,) float32     -- per-step box_target>=0.9 flag (documentation)

OGBench `ogbench.utils.load_dataset(...)` derives next_observations (or valids) by shifting within
trajectories. InFOM GCDataset then relabels goal-conditioned rewards/masks at sample time.

## Stats
- 512 episodes (256 deterministic/expert + 256 stochastic/exploratory), ep_len 150
- 76,800 usable transitions (77,312 rows - 512 final-states)
- success fraction = 0.75 (det 0.777, stochastic 0.750), obs_dim 66, act_dim 8
- actions in [-1, 1]

## InFOM ingestion status
SMOKE TEST PASSED (smoke_infom_ingest.py): the npz loads cleanly through
`ogbench.utils.load_dataset` (-> 76800 transitions, 512 traj, next_observations derived) AND through
InFOM `utils.datasets.Dataset.create` + `.sample(256)`. So the dataset is well-formed for InFOM's
core data pipeline (Dataset + GCDataset goal-relabeling).

REMAINING ADAPTATION NEEDED for a full InFOM run (NOT done here, by design -- nontrivial):
InFOM `envs/env_utils.py make_env_and_datasets()` only dispatches on env_name containing
"singletask" (OGBench, downloads <name>.npz) or DMC keywords (walker/cheetah/quadruped/jaco, ExORL).
There is NO Panda branch. To run InFOM `main.py` on this dataset, add ONE branch to
make_env_and_datasets for a name like "panda-pickcube" that:
  1. calls `ogbench.utils.load_dataset` on THIS local npz (handles raw -> next_obs), splits train/val;
  2. returns a gym-style eval env wrapping mujoco_playground PandaPickCube (for online eval rollouts).
Step 2 (a gymnasium eval env exposing success in info) is the only real work; the data side is proven.
Alternatively, bypass env_utils and load the npz directly in main.py for offline-only pretraining
(no online eval). Either is a small, well-scoped change; the dataset itself needs no further work.
