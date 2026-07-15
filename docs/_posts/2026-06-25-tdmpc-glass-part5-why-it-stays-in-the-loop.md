---
layout: post
title: "Why the Abstraction Has to Stay in the Loop — Negative Results, a Physical Ceiling, and What an Abstraction Is Actually For"
date: 2026-06-25
description: "Follow-up to the beat-PPO update: the in-loop residual on a hand-coded controller ties PPO (0.79 vs 0.81) and beats it 1.7x on sample-efficiency. This post details the loop and the analytic controller, shows that 0.83 on PandaPickCube is a physical kinematic ceiling (proven both directions, so both methods tie there), and lays out the plan to beat PPO on success on tasks with headroom by combining the abstraction's prior with selective model-based planning — and, longer term, with TAMP, which is itself both a planner and an abstraction. A brief necessity check (distillation, authority-annealing, and a learned gate all do worse than keeping the controller in the loop) confirms the in-loop coupling."
---

> **⚠ Reconciliation note (2026-07-02):** the 0.79-vs-0.81 "near-tie" below came from the milestone-shaped
> variant; the later canonical matched-harness study (base reward, n=3, CI-separated) reads **0.716 &lt; 0.810 —
> vanilla PPO wins the asymptote**, residual ~1.6× faster. The "planning +8% on HopperHop" quoted in §6 was later
> measured at **+1.7 return points** (n=31) in the controlled plan-vs-π test. The speed-not-ceiling conclusion
> stands; those two numbers don't.

> Continuing from the [last update](https://suuttt.github.io/projects/2026-06-24-tdmpc-glass-part4-jumpy-to-beat-ppo/):
> the headline was a heuristic-in-the-loop residual that **ties PPO (0.79 ± 0.01 vs 0.81) and reaches competence
> ~1.7× faster**. This post spells out the method (the loop and the analytic controller), shows *why* the
> ~0.83 success on PandaPickCube is a shared physical ceiling rather than a method gap, and lays out the concrete
> **plan to beat PPO** on tasks with headroom — by adding selective planning, and eventually TAMP, on top of the
> abstraction. Beating the strong model-free baseline is the goal; this is the path.

## 1. The loop and the analytic controller (what "in the loop" means)

The method has two parts that run **together at every timestep**.

**(a) The analytic controller** — a hand-coded, closed-loop phase machine for PandaPickCube. It reads the box
and target poses directly and steps through phases, solving a small analytic level-IK each step to produce a
Cartesian action:

```
phase ∈ {REACH, DESCEND, GRASP, LIFT, PLACE}
loop each step:
    if   REACH:   goal ← above(box);          if EE over box      → DESCEND
    elif DESCEND: goal ← box;                 if EE at box        → GRASP
    elif GRASP:   close gripper;              if grasped          → LIFT
    elif LIFT:    goal ← box + z_clear;       if lifted           → PLACE
    elif PLACE:   goal ← target; open near target
    a_ctrl ← levelIK(goal, current_pose)      # closed-loop, per step
```

On its own this controller solves the task only ~9% of the time (open-loop grasp dynamics on a small cube are
the limit) — it is a *competent prior*, not a solution.

**(b) The in-loop residual.** We keep the controller **live** and learn a small policy that corrects it. The
key fix is making the problem Markov: the controller's hidden phase index \(z\) is appended to the observation,
so the learned residual sees \((s, z)\). The executed action is

$$ a = \mathrm{clip}\big(a_{\text{ctrl}}(s) + \alpha\,\pi_{\text{res}}(s, z)\big),\qquad \alpha = 1.0\ \text{(persistent)}. $$

\(\pi_{\text{res}}\) is trained with ordinary PPO inside an environment wrapper that adds \(a_{\text{ctrl}}\) and
exposes \(z\). Crucially \(\alpha\) is a **constant**, not annealed (Section 3 explains why). This persistent
in-loop residual reaches **0.79 ± 0.01** real success (box_target ≥ 0.9, n=256), a near-tie with end-to-end
PPO's **0.81**, and crosses the 0.66 competence bar at **19.7M** steps vs PPO's **32.8M** — the ~1.7× sample
-efficiency win.

## 2. A brief necessity check — the controller can't be removed

This is a small supporting result, not the headline: it just closes the obvious question, *"is the controller
only a training-time scaffold you can drop afterward?"* No. **Three independent attempts to take the controller
out of the runtime loop are all worse than keeping it in** — the standard way to show a component is necessary
(remove it, watch it break):

| route to *remove* the controller | what it does | result |
|---|---|---|
| **Distillation** | BC/DAPG the controller+residual into a standalone raw-action net | **0.0** (BC); DAPG only matches plain PPO 0.82 with worse sample-efficiency |
| **Authority-annealing** | anneal \(\alpha: 0\to 1\) so the learned policy takes over | **collapses to 0.008** |
| **Learned gate** | learn a per-state gate that switches the controller off when "not needed" | **0.613** — strictly below persistent \(\alpha\)=1.0's **0.781** |

Three independent eliminations all degrade or collapse ⇒ **the analytic controller must stay in the loop at
runtime.** The mechanism is identical in all three: the abstraction is **non-Markov** — its action depends on
the hidden phase state — so the controller is a *runtime provider* of the phase structure the residual corrects,
**not a discardable training scaffold**. Distill it and the reactive net can't reconstruct the phase; anneal it
and a half-trained residual inherits a non-stationary target; gate it off and the phase context is lost
mid-episode. The residual has nothing to stand on without the controller beneath it.

That is the real content of "the abstraction has to stay in the loop": not a slogan, but a claim falsified three
ways and surviving.

## 3. Beating PPO is the goal — but on PandaPickCube ~0.83 is a shared *physical* ceiling

Beating a strong model-free baseline is the bar, and it stays the goal. So we threw seven levers at
PandaPickCube to exceed PPO and approach 100%: bigger nets, a capacity sweep, closed-loop retry, two curricula,
deploy-time best-of-k, an orientation-aware controller, longer budget, and end-to-end learned grasping. **None
beats ~0.83 — and crucially, neither does PPO.**

The reason is not a learning shortfall — it is **physics**, and we proved it both directions:

- **Forward:** at far horizontal reach (>0.84 m) the Panda **cannot hold a grasped cube within ~8° of upright**,
  which the `box_target ≥ 0.9` metric requires — an upright-grasp IK feasibility test finds **99.9% of the
  far-reach configs infeasible** (while 100% are *position*-reachable). Position-reachable ≠ upright-graspable.
- **Backward (the clean falsification):** simply **remove** the far-reach configs from the spawn distribution
  (17.6% of spawns exceed 0.84 m, matching the ~17% failure rate) and PPO success goes to **1.000**.

So on PandaPickCube, ~0.83 is a *shared* ceiling — PPO sits there too, and no method can pass it because the
last ~17% is kinematically infeasible. You cannot "beat PPO" on success on a task where both are pinned to the
physics. The right response is not to give up on beating PPO — it is to (a) bank the win the abstraction
*already* has here (sample-efficiency, ~1.7×), and (b) chase the success win on **tasks with headroom**, with a
sharper tool: planning.

## 4. Where the abstraction stands today — and what's already won

To be precise about the *current* standing (the success win is still open and pursued in §6): **the in-loop
residual matches PPO on success where the controller's prior fits and trails where it doesn't** — while already
winning on the efficiency/robustness axes:

| task | in-loop residual | PPO |
|---|---|---|
| PandaPickCube (prior fits) | 0.79 (tie; 1.7× faster) | 0.81 |
| PickCubeOrientation, **fixed** prior | 0.33 (loses) | 0.82 |
| PickCubeOrientation, **parametrized** prior | 0.67 (recovers ~70%) | 0.82 |
| PickCubeOrientation, *fuller* parametrization | 0.68 (class ceiling) | 0.82 |

The orientation task is the clean test of *reusability*: the fixed top-down/fixed-yaw grasp is the wrong prior
for a random-yaw target, so the abstraction **loses** (0.33, worse than learning from scratch). Making the
controller **orientation-aware** — grasp/place at the target yaw, with the target pose fed to the residual —
recovers it to **0.67** (~70% of the gap), but fuller parametrization plateaus at **0.68**: a residual
correcting an analytic grasp has a ceiling below end-to-end RL on hard tasks. The design lesson: **abstractions
are reusable only when their priors are *parametrized* to the task's degrees of freedom, not hard-coded.**

So the banked result so far is:

> **A small residual on a task-matched, parametrized abstraction matches PPO's success with ~1.7× better
> sample-efficiency, more stability, and interpretability — and transfers across tasks when the prior is
> parametrized.**

That is the foundation, not the finish line. The remaining goal — *beating* PPO on success — is what the next
section is for, and the tool is planning.

(Stability shows up concretely: the matched/parametrized residual barely moves from peak to final, whereas an
ill-matched residual exhibits a fragile optimum that collapses under continued training. Interpretability is
structural — a legible phase machine with phase-attributable failures, which is *how* we localized the ceiling
to the grasp phase and then proved it kinematic.)

## 5. The original mission, closed honestly (InFOM)

The week began as a reproduction of CompPlan + InFOM. Result: **InFOM-base reproduces on its actual
(manipulation) benchmark** — cube-single ~0.96/0.98 on 2 of 3 seeds (with one seed collapsing to 0.16, real
seed-variance). AntMaze, which an initial draft scored as a "failure," turned out to be **outside InFOM's
benchmark entirely** (InFOM's finetuning datasets are generated locally for manipulation envs only; antmaze
appears nowhere in the repo or paper). So the antmaze number is an out-of-benchmark extension, not a
reproduction — a correction we made rather than leave a misleading comparison standing.

## 6. The plan to beat PPO — selective planning, then TAMP

Beating PPO on *success* is the open goal. On PandaPickCube it's physically impossible (shared 0.83 ceiling),
so the plan targets **tasks with headroom** and brings a sharper instrument — planning — on top of the
in-loop abstraction:

1. **Use planning where it actually pays.** Model-based planning (MPPI through the learned model) helps
   *precisely when the learned model is accurate*: +8% over the reactive policy on HopperHop, but it *hurts*
   PandaPickCube, where it plans over a reward-hacked value. We're running a controlled correlation across
   several DMC tasks — *planning advantage vs learned-model accuracy* — to turn this into a **"plan-when-the
   -model-is-accurate" rule** and a **confidence-gated planner** that plans only in trustworthy regions and
   defers to the policy/controller elsewhere.
2. **Abstraction prior + selective planning on harder tasks.** The abstraction supplies a competent prior and
   sample-efficiency; selective planning adds lookahead exactly where the model supports it. On longer-horizon /
   sparse tasks — where PPO's exploration struggles (recall TD-MPC2's own failure on PandaPickCube was a
   *learning-time exploration* failure) — this combination is where beating PPO on success is realistic.
3. **TAMP as the principled controller.** Longer term, **task-and-motion planning (TAMP)** is a natural fit: it
   is *simultaneously a planner and an abstraction* — a symbolic/geometric layer that is essentially a
   principled, general version of our hand-coded controller. Swapping the analytic controller for a TAMP layer
   (keeping the learned residual + selective planning on top) would generalize the prior across tasks while
   preserving the in-loop coupling that Section 2 shows is necessary. This unifies the two threads of the
   project — *planning* and *abstraction* — under one framework.

(Also open: whether the parametrize-the-prior recipe generalizes cleanly to a third task — the fixed-vs
-parametrized contrast is harder to construct than it looks, and we're choosing the right task.)

Full per-experiment write-ups (Parts 25–37), with learning curves, the benchmark table, rollout videos, the
upright-IK feasibility proof, and the workspace-constraint falsification, are in the
[TD-MPC-Glass series](https://suuttt.github.io/tdmpc-glass/).
