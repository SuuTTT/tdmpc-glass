# GHM / CompPlan reproduction plan — "Compositional Planning with Jumpy World Models"

Status: 2026-06-17. Scaffolds cloned to `ghm_repro/`. Training NOT yet started (GPUs are on
the Paper-A discrimination matrix). This is a multi-DAY reproduction, not an 8h one — see honesty note.

## What the paper is
- Farebrother, Pirotta, Tirinzoni, Munos, Lazaric, Touati. arXiv **2602.19634**, ICLR-2026
  **Workshop on World Models** (table name: **CompPlan**). Builds on **Temporal Difference Flows
  (TD-Flow)**, ICML 2025, arXiv **2503.09817** (same group).
- Method: a **Geometric Horizon Model (GHM)** = a flow-matching generative model of the discounted
  future-state (successor/occupancy) measure, **policy- and horizon-conditioned**, used to plan over
  sequences of pre-trained goal-conditioned policies. Eval on **OGBench** (antmaze navigate + cube manip).
- Headline: ~+200% relative over primitive-action planning on long-horizon tasks.

## Code availability (verified, 2026-06-17)
- **No official code** for CompPlan, TD-Flow, or the paper's GHM (checked arXiv, OpenReview
  j6H7c3aQyb / 6WTGIu4NVN, facebookresearch, JesseFarebro, ahmed-touati). Reimplementation required.
- Best adaptable scaffolds (all JAX, all already on OGBench):
  - `ghm_repro/infom`  — InFOM (ICLR'26): **flow-matching over the discounted occupancy/successor
    measure** on OGBench cube tasks. ~80% of the GHM generative core + OGBench train/eval loop.
  - `ghm_repro/value-flows` — Value Flows: flow-matching satisfying a **Bellman/TD** equation; the
    pattern for wiring a TD bootstrap into a flow-matching loss (the TD-Flow piece).
  - `ghm_repro/ogbench` — benchmark envs + reference base policies (GCBC, CRL) in `impls/`.

## Two cheapest target tasks (to match the paper's Table 1)
1. `antmaze-medium-navigate-v0`  (paper "antmaze-medium"; e.g. GC-BC 0.49 -> 0.85 with CompPlan)
2. `cube-single-play-v0`         (paper "cube-1";        e.g. CRL 0.28 -> 0.86 with CompPlan)

## Recipe (from paper)
- GHM flow-matching model: 3M grad steps, Adam, batch 256, U-Net core, discount-consistency on
  25% (antmaze)/12.5% (cube) of each minibatch. Plan: sample 256 (antmaze)/1024 (cube) subgoal seqs.
- Base policies: pre-trained GC-BC / GC-TD3 / CRL etc. (OGBench impls). CompPlan plans *over* them.
- LR + GPU-hours: NOT reported. State-based tasks ≈ 2–3 h/agent reference (OGBench); 3M-step flow
  model likely longer.

## Execution plan (when a GPU is allocated)
1. Fresh venv on a worker: `pip install ogbench`, jax[cuda12], plus InFOM's requirements (keep it
   ISOLATED from the tdmpc `.venv`). Smoke: `ogbench.make_env_and_datasets('cube-single-play-v0')`.
2. Reproduce a BASE policy first (GC-BC on cube-single via OGBench impls) — confirms env+data+eval.
3. Stand up the GHM = adapt InFOM's flow-occupancy model; add (a) horizon-conditioning, (b) policy-
   conditioning, (c) the TD-Flow bootstrap target from value-flows. Train on cube-single (cheapest).
4. Plan-over-policies eval; compare success to paper Table 1 (cube-1 row). Then antmaze-medium.
5. Gate: if a single base+GHM run reproduces the cube-1 lift (e.g. ~0.3 -> ~0.8) → scale to antmaze.

## Honesty note
Full reproduction of CompPlan numbers is a multi-day effort (no code, 3M-step flow model, base-policy
pretraining, two benchmarks). In an 8h unattended window the realistic deliverable is: scaffolds
cloned (done), env+data pipeline stood up, one base policy training, GHM scaffold drafted — NOT
reproduced numbers. Do not report reproduction until a CSV shows the cube-1 success lift.
