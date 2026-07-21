# HL Learning Curve — RL units (1000-step eval_pi protocol)

Files: `hl_learning_curve.csv`, `hl_learning_curve.png`.
Eval script: `eval_curve.py` (mirrors `scripts/run_benchmark.py` `eval_pi`).

## Protocol (EXACT same axis as TD-MPC2)
Each controller version is run under the identical PandaPickCube eval used to score
TD-MPC2's return:
- env = `wrapper.wrap_for_brax_training(registry.load("PandaPickCube",
  impl=jax), episode_length=1000, action_repeat=1)`
- per episode: reset, run 1000 steps, `er += float(state.reward)` (dense env reward,
  reward UNMODIFIED — consumed as-is), break on `done`.
- return = mean `er` over n=64 parallel episodes; success = max `box_target >= 0.9`;
  reached = max `reached_box > 0.5`.

VERIFIED protocol fact: at `episode_length=1000` the brax `EpisodeWrapper` runs ONE
continuous 1000-step episode — the inner 150-step env does NOT auto-reset (the wrapper
counter overrides inner termination; `done` stays 0 until step 1000). So the HL return
is the dense reward accumulated over a single grasp+lift+place+HOLD rollout, exactly
how `eval_pi` scores TD-MPC2.

## The curve (return_1000step / real success)
| iter | version    | return_1000 | success | reached | note |
|------|------------|-------------|---------|---------|------|
| 1  | v1         | 4403 | 0.000 | 0.89 | baseline phase machine, no aik (hover-near-cube) |
| 3  | v3         | 4231 | 0.000 | 0.89 | split LIFT/TRANSPORT/PLACE |
| 4  | v4         | 4488 | 0.000 | 0.84 | faster IK + earlier transport |
| 7  | v7         | 4549 | 0.000 | 0.84 | DEAD-END iterative 6DOF orient (reverted) |
| 8  | v8         | 3069 | 0.016 | 0.81 | ANALYTIC level-IK from GRASP — cracked 0% wall |
| 9  | v9         | 2880 | 0.063 | 0.81 | longer grasp-hold + tighter xy |
| 10 | v9_rhold   | 3406 | 0.063 | 0.83 | **BEST** (controller.py): hold 50, grasp_max 55 |
| 11 | v11_xyservo| 3522 | 0.031 | 0.97 | Job-2: broke reached cap 0.78->0.97, success-neutral |

(Job-2 success measured at n=256 multi-seed in LOG.jsonl; the curve uses the n=64
1000-step return axis. Best honest success = 0.094, 4-seed mean at n=256.)

## HONEST comparability framing (IMPORTANT)
- TD-MPC2 vanilla plateaus at **~2500 return** — but this is a **HOVER** with
  **0 real success** (box_target=0; the policy reward-hacks the ungated `gripper_box`
  term, w=4, by hovering near the cube and never grasping).
- The HL curve shows the OPPOSITE trade: the early HL versions that never succeed score
  the HIGHEST return (~4400-4550) by the SAME hover mechanism. As HL learns to actually
  grasp+lift+place (v8+), its return DROPS to ~2880-3520 because real manipulation
  ABANDONS the cheap hover reward — yet it earns REAL task success (up to ~9.4%).
- So in this reward, **return is anti-correlated with real success**. The HL line's
  value is not a higher number than 2500 — it is that HL converts the cheap hover
  return into ACTUAL picks, which TD-MPC2 never does. Read the return ALONGSIDE the
  success column: TD-MPC2 = 2500 return / 0% success; HL best = ~3400 return / ~9.4%
  real success.
