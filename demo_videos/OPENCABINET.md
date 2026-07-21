# PandaOpenCabinet — PPO baseline + Heuristic-Learning (HL) loop

Box b3060, GPU1. Env: mujoco_playground PandaOpenCabinet (reach -> grasp -> PULL a
vertical handle bar on a slide joint to target_pos [0.3+-0.1, 0, 0.5]).
SUCCESS = metrics box_target >= 0.9  <=>  reached_box (site within 0.012 m of handle)
AND |target - handle| <= ~0.02 m.

ALL NUMBERS READ FROM JSON / LOG FILES. Peak + final reported.

## PART A — Official PPO
Config: brax_ppo_config("PandaOpenCabinet") = 40M timesteps, 2048 envs, policy
(32,32,32,32), value (256,)x5. Launched full official recipe (no edits) on GPU1.
Training reward curve (eval/episode_reward, 1000-step protocol wrapper):
  step 0=75.7  9.8M=665  22.9M=953  32.8M=1104  40M~=1115  49M=1140 (plateau ~1130)
PPO eval (box_target>=0.9, eval_ckpts_oc.py, N=256, 150-step protocol matching
training episode_length), READ FROM ppo_eval_s150.json:
  VERDICT: **PPO SOLVES IT.** success 0% till 13M -> 84% @16M -> 92% @23M ->
  97% @26M -> plateau **98.05%** (peak @32.8M, FINAL @75M = 0.9805).
  reached_rate == success_rate (98%); mean_max_box_target 0.98; max_box_target 1.0.
  So the handle reach IS kinematically achievable; PPO learns it cleanly.

## PART B — Heuristic-Learning loop
Skill decomposition: APPROACH (pre-grasp standoff -x) -> ALIGN (site onto handle)
-> GRASP (close on bar) -> PULL (site->target along -x) -> HOLD.

HL loop ran 12 single-knob iterations, all eval'd at box_target>=0.9 over N=256
(real telemetry, READ FROM JSON; see hl_opencabinet/LOG.jsonl + KNOWLEDGE.md).

### iter 0-7: characterising the wall
v0 baseline iterative analytic IK: approach_ok 1.0, align_ok 0, grasp_ok 0,
  reached_box 0, success 0. Site lands ~10cm short in x of handle.
ROOT CAUSE (diag5/7/8): the handle [0.5,0,0.5] is at the Panda's forward-reach
  FRONTIER. Iterative per-step IK stalls -- joint 5 saturates ~0.5 rad short of
  command, achieved site_x maxes ~0.41. FK-vs-site calib fine (3mm); the pose IS
  IK-reachable (perr~0). The wall is CONTROL AUTHORITY at the workspace edge.
v1-v4 (min-travel branch / direct goal / pure DLS): still reached_box 0 -- confirms
  the wall is physical reach, not the IK method.
v5-v7 committed-reach (freeze grasp q, straight drive): per-env/zero-seeded branches
  pick non-reaching postures; reached_box 0.

### iter 8-12: cracking the reach
v8 HARDCODE the verified reaching grasp q + hold gripper OPEN: reached_box 4.7%,
  grasp_ok 27.7% -- reproduces the open-loop reach ceiling.
LEVER FOUND (over_q): command joints a small % PAST the IK solution to fight PD
  droop at the frontier. over_q=0.02, hold-open: align_ok 8%->44%, grasp_ok
  28%->95%, reached_box 4.7%->**7.4%** (the HL PEAK for touching the handle).
  over_q=0.05 overshoots and breaks it; cartesian overshoot also fails (branch flip).
v9-v12 full reach->grasp->PULL pipeline (over_q 0.015-0.03): grasp_ok 44-58%,
  pull moves the handle, BUT reached_box falls back to ~0.8% and **success 0%**.

### BINDING WALL (final, fully diagnosed)
SUCCESS needs reached_box (site within 1.2cm of handle CENTER) AND handle within
2cm of target, SIMULTANEOUSLY. Two coupled limits beat the static-IK heuristic:
 1) REACH: handle is at the workspace frontier; even with the over_q droop-fix the
    gripper settles ~1.5-2cm from the bar center, so reached_box (<1.2cm) latches on
    only ~7% of envs at best, and ONLY while holding still.
 2) GRASP-OFFSET: when the gripper grasps the 2cm bar side-on and PULLS, the site
    sits offset from the bar center, so reached_box drops to ~1% during the pull.
HL PEAK: reached_box 7.4% (hold-open, over_q=0.02); box_target>=0.9 SUCCESS = 0%.
PPO solves it (98%) because it learns closed-loop frontier reaching + centered
contact that a static analytic-IK phase machine cannot reproduce.
