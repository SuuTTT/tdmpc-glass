---
layout: post
title: "TD-MPC-Glass, Part 14: What Makes HopperHop Unique — the PPO Wall Is a Conjunctive-Reward Artifact"
date: 2026-07-09
description: "HopperHop is the sharpest PPO wall in our benchmark and, simultaneously, the only task where TD-MPC2's world-model loss is removable. This post resolves that tension with a literature-grounded, four-mechanism account and a decisive controlled experiment. Reading the actual environment shows HopperHop has no early termination — its wall comes from a conjunctive, multiplicative reward (standing × hopping). We then env-gate the reward and re-run tuned PPO: under an additive reward PPO climbs off zero; under the product reward it stays walled even with an easier hop-speed threshold. So the 'categorical PPO wall' is a reward-design property, not a fundamental capability limit — which sharpens (not undermines) the Part-12 critique that TD-MPC2's Hopper win belongs to its off-policy TD value pathway, not its world model."
---

> Two facts about **HopperHop** sit in tension. It is the **sharpest categorical PPO wall** in our 16-task benchmark
> (tuned PPO reaches 0/5 seeds ≥ 200 at 472M steps), *and* it is the **only task where TD-MPC2's world-model
> (consistency) loss is removable** (n=8). How is the hardest task for PPO the one where the "world model" is
> dispensable? This post answers it — by reading the environment, mapping four candidate mechanisms onto the
> literature, and running the decisive experiment. The punchline: **HopperHop's PPO wall is a conjunctive-reward
> design artifact**, and that *sharpens* our earlier critique rather than weakening it.

## Reading the environment first (the step everyone skips)

Before benchmarking an algorithm on HopperHop, look at what the task actually rewards and terminates on
(`mujoco_playground/_src/dm_control_suite/hopper.py`):

- **No early termination.** `done = isnan(qpos) | isnan(qvel)` — that's it. Falling over does **not** end the
  episode; it just stops paying reward. (Unlike Gym-Hopper, which terminates on torso drop.)
- **A conjunctive, multiplicative reward.** `reward = standing × hopping`, where
  `standing = tolerance(height, (0.6, 2))` and `hopping = tolerance(speed, bounds=(2.0, ∞))`. You earn reward **only
  when you are simultaneously upright *and* moving ≥ 2 m/s.** Until then, the product is ≈ 0.

That second point is the whole story. HopperHop's difficulty is not a fragile termination condition — it is a
**sparse-in-practice conjunction**: the agent gets essentially no gradient signal until it solves *both* sub-tasks
at once. That is exactly the structure on-policy methods struggle with.

## Four mechanisms, mapped to the literature

Our reference set names four ways HopperHop is unusual; three intersect directly with our data.

- **H1 — contact-critical value cliffs** (Omura et al. 2024, *Stabilizing Extreme Q-learning*; Sujit et al. 2022,
  *Reducible Loss*). Underactuation + fall = sharp discontinuities in value; the informative transitions are
  liftoff/landing. On-policy Monte-Carlo/GAE advantages are high-variance across these cliffs; off-policy TD
  bootstrapping + replay is robust. *Our evidence:* the wall needs contact-criticality — contact-free-but-unstable
  AcrobotSwingup has **no** PPO wall.
- **H2 — narrow stability basin / underactuation** (all four papers). The balance-maintaining region of policy space
  is tiny; success is rare and surrounded by failure. On-policy discards rare success; off-policy replay keeps it.
- **H3 — benchmark-design artifact** (Voelcker et al. 2024, *Can we hop in general?*). Hopper's reward/termination
  are load-bearing; design choices can invert algorithm rankings. **This is the one we test below.**
- **H4 — exploration-hard but execution-simple** (ours). The gait is hard to *find* (contact-critical, conjunctive)
  but easy to *execute* (a low-dimensional limit cycle needing no accurate multi-step rollout) — which is why the
  **world model is removable** even though the wall is high. *Our evidence:* consistency-off matches full (n=8);
  planning-without-the-world-model adds nothing (at 5M, stripped-Hop π ≥ MPPI); the stripped model still trains to
  full.

## The decisive experiment (H3): env-gate the reward, re-run PPO

We added two env-gated knobs to `hopper.py` (byte-identical when unset): a **reward mode** (`product`, the default,
vs **`sum`** = `0.5·standing + 0.5·hopping`, a *non-conjunctive* reward) and a **`HOP_SPEED`** override. Then we ran
tuned PPO on HopperHop under each, seed 50, 20M steps:

| PPO variant | reward structure | final return |
|---|---|---|
| **default** (control) | product `standing × hopping` | **0** — the wall |
| **additive** | `0.5·standing + 0.5·hopping` | **135** — climbs off zero |
| **product, `HOP_SPEED=1.0`** | product, *easier* hop threshold | **1** — wall persists |
| tdmpc2 additive (control) | additive | 467 — TD-MPC2 fine |

The result is clean and mechanistically specific:

- Under the **additive** reward, PPO **climbs off zero with a steadily rising curve** (0 → 135 at 20M, not yet
  plateaued: 3, 17, 29, 87, 92, 102, 77, 89, 113, 131, 94, 135) — because partial standing now pays, giving the
  on-policy gradient something to climb. To be precise: 135 is still well below even the standing *component's*
  ceiling (~500 under the additive scaling), so PPO has gained *learnable signal*, not solved the task — the
  contrast with the product reward's flat 0 is the finding, not the absolute level.
- Under the **product** reward, PPO stays **walled** — *and lowering the hop-speed threshold does not help* (0 → 1).
  So the barrier is the **conjunction itself** (near-zero return until both sub-tasks are solved together), **not**
  the hop-speed magnitude, **not** early termination (there is none), and **not** a fundamental PPO limitation.
  (Technical footnote: `HOP_SPEED=1.0` also halves the tolerance *margin* — `margin = HOP_SPEED/2` — so this variant
  is "easier threshold, narrower shaping band"; the flat result is consistent with the conjunction story either way,
  but a margin-controlled variant would be the cleaner isolation for the paper version.)

Off-policy TD + replay (SAC, TD-MPC2) tolerate exactly this conjunctive-sparse structure — they retain the rare
joint-success transitions and bootstrap value across the cliffs — which is why they clear a wall that on-policy PPO
cannot.

## Why this *sharpens* the Part-12 critique

[Part 12](../2026-07-08-tdmpc-glass-part12-hopper-critique/) argued that TD-MPC2's flagship HopperHop win is
mechanistically a **TD-learning-and-planning win, not a world-model win** — because the world model is removable
there, and stripping it still beats SAC's speed (Q1 isolation), while planning-without-the-world-model adds nothing.
H3 doesn't overturn that; it **qualifies the wall itself**:

> TD-MPC2's *"categorical PPO wall on HopperHop"* is **conditional on the standard DMC conjunctive reward.** It is an
> **on-policy-vs-off-policy exploration gap** manufactured by multiplicative-sparse reward structure — not evidence
> that PPO fundamentally cannot hop. Change the reward to additive and the "wall" softens immediately.

That is the honest, rigorous framing Voelcker calls for: report *why* the wall exists (conjunctive reward × on-policy
exploration) and *what it is conditional on*, rather than presenting it as a raw capability gap.

## The full mechanistic account of "what makes HopperHop unique"

Putting the pieces together:

1. **The wall (exploration side).** HopperHop's reward is a **conjunctive product** of two hard sub-tasks with **no
   early termination** to shape it — a sparse-in-practice signal that starves on-policy PPO's advantage estimates
   (H1 cliffs, H2 narrow basin). Off-policy TD+replay tolerates it. *H3 confirms the reward structure is the wall.*
2. **The removable world model (execution side).** Once found, the hopping gait is a **low-dimensional limit cycle**
   the policy executes without accurate multi-step latent rollouts — so the consistency loss is dispensable (H4),
   and planning-over-the-world-model adds nothing.
3. **The attribution.** Stripped-TD-MPC2 (no world model) still beats SAC's speed → the win is the **off-policy TD
   actor-critic + planning operator**, not the world model. And the contrast task confirms the boundary: on
   **WalkerRun** the world model is **load-bearing** (stripping it costs −7.5%), because there the *return level*
   needs accurate rollouts — exactly where Hop differs.

So HopperHop is unique because it sits at the intersection of **conjunctive-sparse reward + underactuation +
contact-criticality**, which makes it *exploration-hard for on-policy methods* while leaving the *executed behavior
simple enough that the world model is redundant.* That single intersection explains both facts we started with.

## Caveats and next probes

This H3 run is n=1/variant at 20M (PPO's additive-reward curve is still rising — 135 is a lower bound, not the
asymptote); we'd add seeds and longer budgets for a paper claim, and the additive reward makes standing-alone
rewarding (so "PPO climbs" partly means "PPO learns to stand"), which is precisely the point — the conjunction is
what withheld all signal. The remaining probes from the plan are the **value-cliff correlation** (H1, across the
16-task benchmark) and the **basin-width** measurement (H2); both would turn this from "the reward is the wall" into
a quantitative law relating conjunctivity/cliffiness to the on-policy wall depth.

---

*References:* Omura, Osa, Mukuta & Harada (2024), *Stabilizing Extreme Q-learning by Maclaurin Expansion*; Sujit,
Nath, Braga & Kahou (2022), *Prioritizing Samples in Reinforcement Learning with Reducible Loss*
([arXiv:2208.10483](https://arxiv.org/abs/2208.10483)); Cho, Kim, Lee & Hong (2024), *Meta-Controller*
([arXiv:2412.12147](https://arxiv.org/abs/2412.12147)); Voelcker, Hussing & Eaton (2024), *Can we hop in general? A
discussion of benchmark selection and design using the Hopper environment*
([arXiv:2410.08870](https://arxiv.org/abs/2410.08870)).
