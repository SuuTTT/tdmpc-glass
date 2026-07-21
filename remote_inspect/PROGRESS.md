# Experiment #5 — live option-controller + alpha-annealed learned residual (Markov in (s,z))

GOAL: beat PPO by keeping the abstraction IN the loop. Dual criterion:
  (A) reach >= 0.82 real success (match end-to-end PPO ceiling), AND
  (B) cross 0.66 real success in < 33M env-steps (beat PPO escape speed).

## Design
executed action (what base PandaPickCube.step receives):
    a_t = clip( a_option(s_t, z_t) + alpha(t) * pi_res(s_t, z_t) , -1, 1 )
- a_option: analytic phase controller (param_controller.ParamPick, v9 BEST knobs,
  d==0), executed LIVE; z_t = its phase carry (APPROACH..HOLD).
- pi_res: the brax-PPO policy output (8-d in [-1,1]); PPO optimizes it.
- alpha(t) = clip(ctrl_gsteps_per_env / ANNEAL, 0, 1): linear 0->1 (inherits
  controller competence early -> full residual authority late, exceeding 0.24 cap).
- Markov fix vs #4: phase z fed into obs (one-hot phase + sub-goal offset + grip
  intent) -> obs 66 -> 77; pi_res is Markov in (s,z). NO distillation.
- Milestone shaping (Markov-checkable): +0.2*grasped +0.3*lifted on top of task
  reward. SELECT/REPORT on TRUE success box_target>=0.9 only.
- z threading/autoreset: z in state.info; reset when EpisodeWrapper steps==0
  (fires on initial + autoreset). ctrl_gsteps monotonic per-env (drives alpha).

## Comparison anchors (real success, from #4/#2, files)
- from-scratch PPO        : ~0.80-0.82 peak; crosses 0.66 around 33M  (BAR)
- #2 abstraction (options): 0.24
- HL analytic controller  : 0.0625 (curve) / 0.0938 (smoke, 1000-step)
- #1 raw residual         : ~0.11

## Smoke (verified, real rollouts)
- ResidualPickCube observation_size=77; brax-wrap + PPO train + eval all work.
- alpha=0 + zero residual -> TRUE success 0.0938, reached 0.76 (== controller
  baseline, NOT 0 -> z threaded & controller live). Phase advances to PLACE/HOLD.
- alpha schedule confirmed 0->1 over ANNEAL per-env steps.
- 600k-step PPO train OK (reward 581->567, JIT 42s). Eval at schedule-matched
  alpha=0.08 -> 0.055 success / 0.36 mean_max_bt (controller-dominated, as expected).

## Runs (ssh3, 40M env-steps each, num_evals=24, one job/GPU)
- GPU0 s1_anneal10k : seed1, ANNEAL per-env=10000 (alpha->1 @ 20.48M total = 50%)
- GPU1 s2_anneal10k : seed2, same schedule
- GPU2 s1_anneal5k  : seed1, ANNEAL per-env=5000  (alpha->1 @ 10.24M = 25%) [ablation]
- GPU3 s1_fixed0p5  : seed1, constant alpha=0.5                              [ablation]

## Eval protocol
eval_residual_curve.py: per checkpoint at total-step S, alpha=clip(S/20.48M,0,1)
(training-matched), n=256, 1000-step, success=max box_target>=0.9. PEAK+FINAL,
steps_to_cross_0.66.

STATUS: launched 2026-06-23, compiling. Curves pending.
