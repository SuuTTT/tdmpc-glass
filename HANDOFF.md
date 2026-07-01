# HANDOFF — tdmpc-glass (control plane + blog)
**2026-07-01 04:29 UTC.** Full master handoff (with box connection details, kept off public GitHub): `/home/ubuntu/HANDOFF_tdmpc-glass_2026-07-01.md` on the EC2 control box.

## What this repo is
Control plane + training code + the public blog (`docs/_posts/`, Jekyll → suuttt.github.io/tdmpc-glass). **This EC2 box has NO GPU and never trains** — training runs on two vast.ai workers (`b3060`, `b3060b`; SSH aliases in `~/.ssh/config`) rooted at `/root/helios-rl` (b3060) and `/root/tdmpc_glass` (b3060b).

## ⛔ Hard constraints
- **b3060b is SHARED with another user's Mahjong RL** (`moyuHarv` tmux + `botzone` process). NEVER touch/kill it; cap your VRAM there. Verify alive: `ssh b3060b 'tmux ls | grep moyu'`.
- Kill only your own pids. Never `--save_full_state`. Read every number from disk (this project has a history of fabricated numbers — always deterministic real eval, report n, matched-budget controls). Keep disk >7G (b3060) / >3G (b3060b).

## Live right now (do not kill)
**Beat-PPO scan** training on both boxes since ~03:00 UTC 2026-07-01: TD-MPC2 (b3060) vs matched-budget PPO (b3060b) on PandaPickCubeOrientation, LeapCubeReorient, LeapCubeRotateZAxis, PandaRobotiqPushCube. Harvest anytime:
```
ssh b3060b '/root/tdmpc_glass/venv/bin/python /root/tdmpc_glass/exp/beat_ppo_scan/harvest.py ppo'
ssh b3060  '/root/helios-rl/.venv/bin/python /root/helios-rl/exp/beat_ppo_scan/harvest.py tdmpc'
```
Early read: PandaPickCubeOrientation = big TD-MPC2 sample-eff win (≈PPO ceiling at ~100× fewer steps, asymptote ~tie); dexterous **Leap** tasks are the open contest for a clean asymptote win (both arms still exploring <1M steps). Full state + per-env table: `beat_ppo_scan/VERDICT.md` on both boxes, and §4 of the master handoff.

## Where results/verdicts live
- Verified ledger: **`wm-redundancy-paper/bet2_null_results.md`** · master verdict: `SYNTHESIS_beat_ppo.md` · queue/state: `AUTONOMOUS_BACKLOG.md`.
- Blog: latest is **Part 52** (`docs/_posts/2026-06-30-*part52*`). Parts 50–52 cover the H-JEPA Panda solve, the learned-residual result, and the anti-collapse taxonomy.
- Read first in this repo: `docs/INDEX.md`, `docs/CHANGELOG.md`, `CLAUDE.md`.

## Campaign in one paragraph
H-JEPA solves PandaPickCube (0.367) but the lever is a competent low-level primitive, not learned hierarchy. Analytic skills cap ~0.37 on contact physics; a *learned* residual breaks that (0.72/0.98) yet matched PPO wins the asymptote — so a structured prior buys sample-efficiency, not a higher ceiling (2 Panda tasks). Anti-collapse for JEPA latents is downstream-dependent (relational/uniformity helps geometric, hurts value-control). The live scan tests whether new dexterous envs give a clean asymptote win over PPO.

## Next steps
1. Finish + record the beat-PPO scan (in flight). 2. (If greenlit) pixel-JEPA-vs-Dreamer — the one untested JEPA angle. 3. (Lower priority) different Panda contact primitive.
