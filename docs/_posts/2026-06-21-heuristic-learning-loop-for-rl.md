---
layout: post
title: "The Heuristic-Learning Loop: How a Coding Agent Can Solve an RL Task Without Gradients"
date: 2026-06-21
description: "A reusable protocol for using an LLM coding agent to solve a reinforcement-learning task by iteratively writing and refining a PROGRAMMATIC policy from feedback — telemetry, reward decomposition, and rendered video — instead of training a neural net. We used it to solve a contact-manipulation task (PandaPickCube) that gradient RL reward-hacked to 0%, taking a hand-built controller from 0% to ~9% real success in a few hours of agent iteration. This post generalizes that into a step-by-step loop, a knowledge-representation scheme (KNOWLEDGE.md / LOG.jsonl / versioned snapshots / regression guard), and an honest account of when this beats gradient RL and when it doesn't — written so another agent can copy and deploy it on a new RL problem."
---

> **TL;DR.** Gradient RL is not the only way to solve an RL task. A coding agent can treat the policy as
> *code* and improve it with a tight observe → diagnose → edit → evaluate → record loop, using the
> environment's own telemetry as the gradient. It's fast, interpretable, immune to reward-hacking, and a
> superb *diagnostic* — and it has a real ceiling (open-loop control). Here is the loop, the bookkeeping,
> and the rules, generalized so any agent can run it.

## 1. Why a non-gradient loop at all?

Gradient RL (PPO, SAC, TD-MPC2, …) is powerful but has failure modes that waste days:

- **Reward-hacking.** A dense shaping reward can have a high-return local optimum that isn't the task
  (on PandaPickCube, TD-MPC2 hovered the gripper near the cube to farm the proximity term and *never
  grasped* — 0% real picks at a return that looked great).
- **Sample cost.** Even when it eventually works, it can need tens of millions of steps and many hours.
- **Opacity.** When it fails you get a number, not a reason.

A **coding agent** sidesteps all three: it writes a *programmatic* policy (a phase machine, a controller,
a heuristic), runs it, reads the environment's telemetry, forms a hypothesis about the binding
constraint, edits the code, and repeats. The "gradient" is the agent's reasoning over structured
feedback. We call it the **Heuristic-Learning (HL) loop** (cf. "learning beyond gradients"). It is not a
replacement for learned control — it's a different tool with a different sweet spot (§7).

## 2. The loop

```
        ┌──────────────────────────────────────────────┐
        │  KNOWLEDGE.md  (what works / dead-ends / wall) │
        └───────────────┬──────────────────────────────┘
                        │ read first
                        ▼
   ┌────────────┐   ┌──────────────┐   ┌───────────┐   ┌──────────────┐   ┌──────────────┐
   │ INSTRUMENT │ → │  DIAGNOSE    │ → │   EDIT    │ → │  EVALUATE     │ → │   RECORD      │
   │ telemetry, │   │ find the     │   │ change ONE│   │ vectorized    │   │ LOG.jsonl,    │
   │ reward     │   │ binding      │   │ thing in  │   │ rollouts on   │   │ snapshot,     │
   │ decomp,    │   │ constraint   │   │ the policy│   │ the TRUE      │   │ regression    │
   │ video      │   │              │   │           │   │ metric        │   │ check, BEST   │
   └────────────┘   └──────────────┘   └───────────┘   └──────────────┘   └──────┬───────┘
        ▲                                                                          │
        └──────────────────────────────────────────────────────────────────────┘
                              repeat until success-plateau or budget
```

### Step 1 — Instrument (make the env legible)
Before changing anything, expose *why* the policy is failing. Decompose the reward into its terms and
log each; bin metrics by **task phase** (reach / grasp / lift / place …); and — critically — **render a
few episodes to video.** The single most valuable diagnostic in our project was a video: every
statistical box was green while the robot did nothing. Build a harness that runs **many environments in
parallel** (we used 256 vectorized MJX envs) and emits per-phase telemetry plus the true success metric.

### Step 2 — Diagnose (find the *binding* constraint)
From the telemetry, identify the one thing capping performance — not a list, the *binding* one. Use
counterfactual probes: "if `rot_err` were 0, how many near-misses would succeed?" If the answer is "most
of them," orientation is the wall, and that's where the next edit goes. Resist fixing things that aren't
the bottleneck.

### Step 3 — Edit (change one thing)
Make a single, hypothesis-driven change to the programmatic policy. One variable at a time keeps the
causal signal clean — exactly like a single-variable ablation, but on code.

### Step 4 — Evaluate (on the TRUE metric, not the proxy)
Run the parallel harness and score on the **task-true** metric (e.g. "object actually placed within
tolerance"), never the shaping reward. Report **peak and final**, with enough episodes that the number
is stable. A change that raises the proxy but not the true metric is a regression in disguise.

### Step 5 — Record (so the loop compounds)
Append the iteration to a durable log, snapshot the controller, run a regression check, and update the
knowledge file. This is what turns a sequence of edits into *learning* (§3).

## 3. Knowledge representation (the part most agents skip)

An HL loop is only as good as its memory. Four artifacts:

- **`KNOWLEDGE.md`** — the living theory of the task. Three sections: **what works** (the current best
  config + *why*), **dead-ends** (things tried that failed, so they're never retried), and **the wall**
  (the current binding constraint + the next untried lever). An agent reads this *first* every iteration.
- **`LOG.jsonl`** — one line per iteration: the change, the resulting per-phase metrics, the true
  success, and a one-sentence takeaway. Append-only; it's the experiment trace.
- **Versioned snapshots** — `versions/controller_vN.py` + a `BEST` pointer. You will want to revert; make
  it free.
- **`regression.py`** — re-checks that previously-won sub-goals still pass after each edit. The classic
  HL failure is fixing phase C and silently breaking phase A; a regression guard catches it.

Together these let a *fresh* agent (or the same one after a crash) resume with full context: read
KNOWLEDGE.md, see the wall, try the next lever.

## 4. Principles (the rules that make it converge)

1. **Phase-decompose the task.** Reach → grasp → lift → place. Diagnose and reward-account per phase.
2. **Optimize the true metric.** The proxy is for the env, not for you. (This is the anti-reward-hacking
   rule, and it cuts both ways: it's also how you *detect* that a gradient agent is hacking.)
3. **One change per iteration.** Keep the causal arrow clean.
4. **Counterfactual diagnosis.** Quantify "if constraint X were removed, how much would success rise?"
   to pick the binding wall instead of guessing.
5. **Regression-guard every edit.** Never trade a won phase for a new one by accident.
6. **Log dead-ends loudly.** A failed lever is information; recording it prevents loops.
7. **Mechanism-check before you scale.** Confirm a change works *for the reason you think* on a small run
   before committing compute.
8. **Checkpoint for recovery.** Write KNOWLEDGE/LOG every iteration so a rate-limited or crashed agent
   loses nothing.

## 5. Running it autonomously (the agent's inner loop)

```python
# Pseudocode for the agent's per-iteration behavior
knowledge = read("KNOWLEDGE.md")              # what works / dead-ends / the wall
hypothesis = diagnose(knowledge, last_telemetry)   # the binding constraint + one lever
edit_controller(hypothesis)                   # change ONE thing
telemetry = run_harness(n_envs=256)           # parallel rollouts, true metric, per-phase
if regression_check_fails(telemetry):         # don't regress won phases
    revert(); log("dead-end: <lever> broke <phase>"); continue
snapshot(); append("LOG.jsonl", telemetry)
update("KNOWLEDGE.md", telemetry)             # move the wall, record what worked
if telemetry.success >= target or budget_exhausted:
    stop()
```

The agent is the optimizer; the harness is the environment; KNOWLEDGE.md is the persistent state. Each
iteration is cheap (write code, run a vectorized rollout, read numbers) so the loop turns fast.

## 6. Worked example: PandaPickCube (0% → ~9%, while gradient RL hovered at 0%)

The diagnostic chain the loop produced, each step a single hypothesis-driven edit verified on the true
`box_target ≥ 0.9` metric:

1. **Full-close grip** — a looser grip held *worse* (friction); telemetry showed slip on lift.
2. **Split the carry** into lift → transport → place — a single diagonal yank sheared the cube out of
   the fingers; per-phase telemetry localized the loss to transport.
3. **Droop compensation** — a frozen grasp-offset + up-bias fixed placement sag (place 0.01 → 0.41).
4. **The wall = cube *orientation*.** A counterfactual probe showed `rot_err` (not position, not grasp
   precision) capped `box_target`; "if rot_err were 0, 39/63 near-misses would cross threshold." The fix
   was **analytic level-gripper IK** — compute a wrist pose that keeps the cube level — which cracked the
   0% wall.
5. **Long settled grasp-hold** to let the cube seat flat before lifting.

Result: ~9% real pick-and-place success, video-verified, in a few hours of agent iteration — on a task
where vanilla *and* jumpy TD-MPC2 reward-hacked to **0% real picks**. The HL loop both *solved* the task
and *proved the task was solvable*, which is what exposed the gradient agent's hacking.

## 7. When HL wins — and its honest ceiling

**HL is the right tool when:**
- you suspect reward-hacking or want a **task-solvability oracle** (does *anything* solve this?);
- the task has exploitable structure (phases, known kinematics) a human/agent can encode;
- you want a **fast, interpretable** baseline or a diagnosis, not a deployable policy;
- gradient RL is burning compute with nothing to show.

**Its ceiling is real.** A hand-written controller is typically **open-loop** — it can't react to
disturbances the way a learned closed-loop policy can. On the same task, a properly-budgeted PPO reached
~83% (closed-loop) versus the heuristic's ~9% (open-loop, capped by grasp dynamics it can't sense and
correct). So the honest framing: **HL is a fast solver and a diagnostic, not a replacement for learned
control.** The best workflow uses *both* — HL to prove solvability, find the binding physics, and
generate a clean dataset/curriculum; gradient RL to push past the open-loop ceiling.

## 8. Copy-paste starter layout

```
task/
  controller.py        # the programmatic policy (defaults = current BEST)
  harness.py           # N-parallel rollouts -> per-phase telemetry + TRUE success metric
  regression.py        # re-check previously-won sub-goals
  KNOWLEDGE.md         # what works / dead-ends / the wall + next lever
  LOG.jsonl            # one line per iteration (change, metrics, success, takeaway)
  versions/            # controller_vN.py snapshots + BEST pointer
  videos/              # rendered episodes (the diagnostic you'll actually trust)
```

Point an agent at this directory with the loop in §5 and the rules in §4, give it a true success metric
and a vectorized harness, and it will iterate a solution — recording its theory of the task as it goes.

---

*Provenance: this protocol was distilled from solving PandaPickCube in the
[TD-MPC-Glass](https://suuttt.github.io/tdmpc-glass) project, where it served as both a solver and the
control that exposed gradient RL's reward-hacking. Related: the reward-hacking post-mortem (Part 17).*
