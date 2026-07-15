---
layout: post
title: "TD-MPC-Glass, Part 17: The Handbook — Our Stack From Scratch, So You Can Design Algorithms With Me"
date: 2026-07-10
description: "A from-scratch tutorial book for this research program: what TD-MPC2 actually is (the five losses, the latent, the planner), exactly how our JAX reimplementation works and where it deviates from canonical, how the experiment harness runs (env-gates, tags, markers, the ledger), the complete map of what we have established so far with real numbers, and — the point of the whole document — the recipe for proposing a new algorithm or experiment and having it running on the fleet within hours."
---

> **Why this exists.** You asked to know what I am doing at a level where we can *design algorithms together*.
> This is that document: a self-contained tutorial book for the TD-MPC-Glass program. Chapter 1–2 teach the agent
> from scratch. Chapter 3 is our actual implementation, deviations included. Chapter 4 is the experiment harness
> and its discipline. Chapter 5 is the results map with real numbers. Chapter 6 is the co-design recipe — how an
> idea becomes a running experiment. Skim 1–2 if you know TD-MPC2; never skip 3.

## Chapter 1 — The agent in five losses

TD-MPC2 is a model-based RL agent that lives entirely in a learned latent space. Everything follows from five
components trained end-to-end off a replay buffer:

**The latent.** An encoder maps observation to \\(z_t = \text{enc}(o_t) \in \mathbb{R}^{512}\\), normalized by
**SimNorm**: the 512 dims are split into groups of 8 and each group is softmaxed. This bounds the latent (helps
stability) and makes it *distributed* — no single dimension means anything alone. Remember this when we get to the
bottleneck experiments: SimNorm is why "read only the first D dims" is a real constraint.

**Loss 1 — consistency (this is the "world model").** A dynamics MLP predicts the next latent:
$$\mathcal{L}_{\text{cons}} = \sum_{t} \rho^t \,\big\lVert \text{dyn}(z_t, a_t) - \text{sg}[\text{enc}(o_{t+1})]\big\rVert^2$$
This is JEPA-style latent self-prediction (no pixel reconstruction). It is the ONLY loss that makes the latent
*predictive* — the thing you can roll forward. When we say "stripped world model" we mean exactly: this loss off.

**Loss 2 — reward.** A head predicts \\(r_t\\) from \\((z_t, a_t)\\), as a two-hot distributional target (the scalar
is encoded as weight on two adjacent bins of a fixed grid; the loss is cross-entropy). The planner uses this head to
score imagined rollouts.

**Loss 3 — value (the engine, part 1).** A \\(Q\\)-ensemble (2 heads in our config) trained by TD:
target \\(= r + \gamma\, Q_{\text{target}}(z', \pi(z'))\\), two-hot cross-entropy again, target network via EMA.
Ablating this is fatal on every task we ever tested — the agent cannot even stand.

**Loss 4 — policy (the engine, part 2).** An actor trained to maximize the ensemble-min Q at its own action.
In our implementation this is fully deterministic:
$$\mathcal{L}_\pi = -\,\mathbb{E}\big[\min_2 Q(\text{sg}(z), \tanh(\mu_\pi(z)))\,/\,\text{RunningScale}\big]$$
RunningScale is an EMA of the Q-value inter-quantile range (p95−p5) that keeps the loss magnitude stable across
tasks. Canonical TD-MPC2 samples the action and adds a tiny entropy bonus (~1e-4); ours doesn't (see Ch. 3).
Also fatal to ablate — the planner alone cannot compensate.

**Loss 5 — there is no loss 5.** Exploration is NOT a loss in this design: it is Gaussian noise added to actions at
collection time, annealed from 0.3, plus a random-action warmup (first 25k steps). Contrast SAC, which puts entropy
*inside* the actor objective. This distinction turned out to be a main finding (Ch. 5, P1).

## Chapter 2 — The planner (MPPI)

At decision time the agent can do better than \\(\pi\\): plan. **MPPI** (Model Predictive Path Integral) is
derivative-free trajectory optimization over the learned model:

1. Sample \\(N\\) action sequences of horizon \\(H\\)=3 from a Gaussian \\((\mu_{1:H}, \sigma_{1:H})\\); a few are
   seeded by rolling \\(\pi\\) (the "policy prior").
2. Roll each through \\(\text{dyn}\\), scoring \\(\sum_t \gamma^t \hat r_t + \gamma^H Q(z_H, \pi(z_H))\\) — reward
   head along the way, value head as terminal bootstrap.
3. Reweight elites (softmax over scores), update \\(\mu, \sigma\\), iterate a few times; execute \\(\mu_1\\), replan
   next step (receding horizon).

Three places planning could appear: **eval** (we do this — `eval_mppi` vs `eval_pi` are logged separately at every
checkpoint, which is what let us measure "does planning help" per task), **data collection** (canonical does this;
ours does not — the V2 experiment adds it via a gate), and **TD targets** (neither does this in the configs we run).

## Chapter 3 — Our implementation, exactly (read this one)

The stack is a from-scratch JAX/Flax implementation (`helios`), running mujoco_playground/MJX physics, on two
Vast.ai 4×3060 boxes (`b3060` = helios_wmablate, the Hopper-capable box; `b3060b` = tdmpc_glass, cannot build
HopperHop). Entry point: `scripts/run_benchmark.py` (task/algo/seed/steps CLI; ~2600 lines; there are TWO copies —
`/root/helios_wmablate/scripts/` used by the ablation harness and `/root/helios-rl/scripts/` used by SAC/PPO runs;
they share the collection-loop structure but are patched independently — always check which one a runner calls).
Algorithms live in `src/helios/algorithms/tdmpc2.py` (+ sac.py, tdmpc_glass.py).

**The training loop** (the part that matters for design work):
```
while env_steps < total_steps:
    # COLLECT (vectorized N_ENVS): a = tanh(π_mean(enc(o))) + Gaussian(σ annealed from 0.3)
    #   [random uniform actions for the first 25k steps]
    #   [V2 gate: MPPI_COLLECT=1 → a = MPPI(z) instead of π-mean]
    buffer.add(...)
    # UPDATE (k_update=128 gradient steps per collect chunk):
    #   sample sequences, compute the 4 losses, masked by ABLATE, one optimizer step each
    # EVAL every ~50k steps: eval_pi (raw policy) AND eval_mppi (planner) → CSV row
```

**Verified deviations from canonical TD-MPC2** (Part 16 has the audit):
1. collection is π+noise, never the planner (canonical: planner); 2. actor loss is deterministic-mean, no entropy
term (canonical: sampled + ~1e-4 entropy); 3. MJX backend, not dm_control; 4. MPPI 2048 samples (canonical 512).
Parity status: hopper-hop final at parity with official (449 vs ~420±113); SAC-failure reproduced exactly;
Walker/Acrobot −17/−23% below official — consistent with missing planner-collection where the WM is load-bearing.

**The env-gate pattern** — our single most-used engineering idiom. Every experimental modification is a small code
patch gated by an environment variable, byte-identical when unset:
```python
_VBN = int(__import__('os').environ.get('VBN_DIM', '0'))
...
z = z[..., :_VBN] if _VBN > 0 else z     # in QEnsemble and Pi
```
Active gates today: `ABLATE` (turn off one loss: none/consistency/value/policy/reward), `VBN_DIM` (value/policy
heads read only the first D of 512 latent dims), `VAC_LAM`/`URC_LAM` (loss reweighting — closed, null),
`HOP_REWARD_MODE`/`HOP_SPEED`/`HOP_MARGIN` (HopperHop reward-structure knobs in hopper.py), `SAC_ALPHA_FLOOR`/
`SAC_TENT_SCALE` (SAC entropy surgery), `MPPI_COLLECT` (planner-collection, V2). Each was applied with a `.bak_*`
backup, an AST parse check, and a 25–35k smoke run before any real launch.

## Chapter 4 — The experiment harness and its discipline

**Anatomy of one run.** A driver script (written via `printf`, never heredoc-over-ssh — those fail silently) sets
env vars and calls `run_arm.sh` / `run_vbn.sh`, which sets a **task-qualified tag** like
`wmabl_WalkerRun_consistency_s70` (a tag missing the task name once caused cross-task data contamination — the F1
incident). Output: a benchmark CSV (`tdmpc2_<task>_<tag>.csv`, one row per eval: step, π-return, MPPI-return), a log
with `es=` step counters (space-padded!), and — when a whole batch finishes — a **marker file** (e.g.
`exp/P4_WALKER_SUFF_DONE`) written by the driver's `wait`.

**The rules, each paid for by a past mistake:**
- **Harvest only at markers.** Near-final evals lie (a "final" harvested pre-completion was wrong by 10%; amended).
- **Verify every launch** by `pgrep` + log existence ~45s after; compound-ssh backgrounds fail silently.
- **One GPU roller only.** Two schedulers double-booked GPUs and (likely) OOM-killed a whole box's jobs.
- **Never trust an in-context ETA**; recompute from `es=` deltas. Packing 2 jobs/GPU halves per-job speed.
- **Every result → the ledger** (`wm-redundancy-paper/bet2_null_results.md`, append-only, git-pushed) with n, seeds,
  step counts, and honest caveats — nulls, truncations, and mistakes included. The ledger is the source of truth;
  blogs and papers cite it.
- **Report n. Never fabricate. Nulls are results.**

**The fleet:** 8×RTX3060 across the two boxes. Budget arithmetic that governs design: a 5M-step tdmpc2 run ≈ 10h
unpacked (~138 sps), ~2× that packed; SAC ≈ 35 min; PPO(brax) 20M ≈ 3h; planner-collection tdmpc2 ≈ 2.4× slower
(~58 sps). A 4-arm × 2-seed ablation is an overnight; a 3-task × 4-width × 5-seed grid is a week.

## Chapter 5 — The results map (what is established, with numbers)

**A. The redundancy result (Paper A).** Four independent ways of imposing structure on the latent, all null-to-harmful
on dense control: added SE/graph structure (glass — long-run null), loss reweighting (VAC −4/−9%, URC −8.2%/−3.8%),
value-conditioned metric (bisimulation −46/−55%: strong no-go), architectural bottleneck (VBN: graded null).

**B. The positive instrument (Paper A).** The **value-sufficient-bottleneck curve**: force Q/π to read only the first
D latent dims → return(D) is smooth, monotone, saturating, and no width recovers vanilla. Cheetah n=2(→3):
516/572/628/738 vs 855. Walker n=2(→3): 604/652/670/708 vs 727 (97% at D=128 — most compressible). Acrobot n=2:
280/351/291/491 vs 511 (least compressible). This is the valid replacement for the decode-R² probe the paper shows
is broken.

**C. The world model is task-conditional (Papers A+3).** Ablating the consistency loss: HopperHop **removable**
(n=8, stripped ≈ full ≈ 420); WalkerRun **−7.5%** (n=4, tight; the historical −23% was a seed outlier); CheetahRun
**≈−19%** (n=2 truncated, wide spread, full-5M seeds running); Acrobot −44%. Official-vs-ours deficits track this
same ordering (Part 16) — external cross-validation.

**D. The HopperHop mechanism (Paper 3).** The PPO wall (0/5 seeds ≥200 at 472M) is caused by the **conjunctive
reward** `standing × hopping`: additive reward → PPO climbs off zero (0→135@20M); product with easier threshold →
still walled (1); product with easier threshold AND default margin (the de-confounded variant) → **still walled
(2.8/3.6 @20M)**. No early termination exists in this env — the conjunction is the whole wall.

**E. The attribution (Paper 3, "P1").** What beats PPO *and* SAC on Hop is the **TD core itself, planner-free**:
our stack never uses the planner in training, yet matches official TD-MPC2 on Hop. SAC fails at every entropy
setting tested — auto-α collapses to 0.003 and stands forever (76/23/101); α floored at 0.05 never stands
(0.26/0.002/0.006); 0.01 splits (1/0/51). The mechanism: SAC puts stochasticity in the *objective* (lose-lose on a
conjunctive-sparse contact task); TD-MPC2 keeps the objective deterministic and puts exploration in the *data*.
Scoped honestly: a ≥4–8× sample-efficiency gap at 5M, not a capability wall (official SAC's best seed crosses at 1M).

**F. Open right now:** V2 (does Hop-removability survive planner-collection?) — the scoping verdict for the whole
critique; Cheetah sufficiency full-5M; VBN → n=5; the Lean+ agent (below).

## Chapter 6 — Designing algorithms with me: the recipe

**The pipeline an idea goes through** (typical wall-clock: idea → running arms in 2–4 hours):
1. **State the claim as an arm-pair.** "X helps" must become: arm A (gate on) vs arm B (gate off), same seeds, same
   task, same budget, and a *pre-declared* success criterion. If the claim can't be phrased this way, it isn't ready.
2. **Pick tasks by what they probe.** HopperHop = conjunctive-sparse exploration, WM-removable. WalkerRun/CheetahRun
   = dense, WM-load-bearing (mild/moderate). Acrobot = threshold-shaped, least-compressible latent. A method claim
   should predict *different* effects across these — uniform effects are usually bugs or artifacts.
3. **Implement as an env-gate** (default byte-identical), `.bak` backup, AST check, 25k smoke on one GPU.
4. **Launch n≥2 seeds per arm** with task-qualified tags + a completion marker; verify by pgrep+logs; the loop
   babysits, harvests at the marker, ledgers, and pushes.
5. **Interpret against the map** (Ch. 5): every new number lands in an existing table; contradictions get flagged,
   not smoothed. Then: extend seeds, add the control you now realize you need, or kill the idea and ledger the null.

**Live design targets where your ideas plug in directly:**
- **Lean+ TD-MPC2** — the minimal agent: value+policy+reward+MPPI+SimNorm, consistency gated per task (on for dense,
  off for Hop-like). Open improvement bets: value stabilization (XQL/Maclaurin-style robust losses on the Q-target),
  better data-side exploration (the P1 result says this is THE lever on conjunctive tasks), adaptive UTD.
- **The exploration lever** — SAC dies on Hop by entropy dosage; we win by annealed data-noise. Is there a principled
  scheduler (e.g., noise scaled by TD-error or by standing-component saturation)? Cheap to gate, high paper value.
- **The reward-conjunctivity knob** — `HOP_REWARD_MODE` generalizes: any product-of-tolerances env can be
  additive-ized. A conjunctivity→wall-depth *law* across envs is a benchmark-design contribution (Paper 3 stretch).
- **The VSB diagnostic as a tool** — return(D) as a per-task compressibility fingerprint; predicts where abstraction
  can/cannot help before you spend a week training one.

**How to hand me an idea:** one sentence of mechanism ("X should help because Y"), which arm-pair tests it, which
task ordering it predicts, and what number kills it. I'll take it from there — gate, smoke, launch, ledger — and
you'll see it in the live log with your criterion applied.

*Companion docs: Part 15 (living revision plan + log), Part 16 (validity audit), the append-only ledger in the
paper repo. Everything in this book is as of 2026-07-10; the live log supersedes it where they disagree.*
