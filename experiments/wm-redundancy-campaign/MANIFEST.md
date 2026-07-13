# WM-Redundancy Campaign — experiment scripts (synced 2026-07-13)

These are the run-recipe scripts for the July TD-MPC2 world-model-redundancy campaign
(Papers A & 3). They lived only on the rented GPU boxes; synced here so the recipes
survive box destruction. Results/ledger live in the `wm-redundancy-paper` repo; write-ups
in `suuttt.github.io` (Parts 15–18).

## Boxes
- `b3060/`  = box 156.238.224.242:62366 (RTX 3060 x4), stack `/root/helios_wmablate`
  (helios ablation stack). Runs: VBN suff ablations, SAC entropy grid (P1), Hopper
  reward-margin (H3), Lean+ (target-EMA), and the V2* planner-collection dissociation.
- `b3060b/` = box 217.171.200.22:62684 (RTX 3060 x4), stack `/root/tdmpc_glass` +
  `helios-rl`. Runs: the value-sufficiency-bottleneck (VBN) grid A1 (Cheetah/Walker/Acrobot).

## Runners
- `run_arm.sh` / `run_arm_v2.sh` / `run_arm_lean.sh` (b3060): wrap `scripts/run_benchmark.py`.
  `run_arm_v2.sh` sets `--mppi_n_samples 512` and `MPPI_COLLECT=1` (planner-collection).
- `run_vbn.sh` (b3060b): wraps the VBN benchmark with `VBN_DIM` bottleneck width.

## Uncommitted source gate-edits (STILL ONLY ON THE BOXES as `.bak_*` backups — recover before destroy)
| box | file | backup | gate added |
|---|---|---|---|
| b3060 | src/helios/algorithms/tdmpc2.py | .bak_lean | `LEAN_TAU` env → target-EMA rate at both EMA sites (Lean+) |
| b3060 | scripts/run_benchmark.py | .bak_v2 | `MPPI_COLLECT=1` → planner collects (batch_mppi_targets) |
| b3060 | (helios-rl) scripts/run_benchmark.py | .bak_p1 | `SAC_TENT_SCALE` target-entropy scale |
| b3060 | (helios-rl) sac.py | .bak_p1 | `SAC_ALPHA_FLOOR` clamp after alpha update |
| b3060 | mujoco_playground_repo/.../hopper.py | .bak_h3m | `HOP_SPEED`/`HOP_MARGIN`/`HOP_REWARD_MODE` reward gates |

To recover a source edit: `diff <file> <file>.bak_*` on the box gives the exact patch.
