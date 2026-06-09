# Iteration 21 — Abstraction-grounded exploration on sparse tasks ("explore")

*2026-06-08. Grounded in the deep-research synthesis (docs/research/dr-synthesis-iter21.md):
abstraction/skills beat flat model-based RL only in the SPARSE/long-horizon/exploration regime.
User-approved direction for the 8h window. Reframed goal: not "+X% on dense" but
"TD-MPC2 FAILS on sparse exploration (≈0); an abstraction-grounded mechanism ENABLES it (0→solved)".*

## Arms (intrinsic exploration reward, training-only, normalized; eval untouched)
- **vanilla** TD-MPC2 — control (CartpoleSparse H3 ≈ 0; BallInCup/Acrobot bimodal).
- **RND** (Random Network Distillation) — simple proven baseline. Answers: does ANY exploration
  rescue sparse TD-MPC2? The lower-risk control the research flagged.
- **Laplacian** (DCEO-style eigenpurpose) — the abstraction bet. Learn graph-Laplacian rep
  phi(s) (smooth-over-transitions + decorrelated); reward ||phi(s')-phi(s)|| = movement along
  slow manifold directions / crossing bottlenecks. The RIGHT use of transition-graph abstraction
  (eigen-directions as exploration), unlike iter-19 reach-centroid. NO continuous-control or
  world-model precedent → genuine gap.

Implementation: src/helios/algorithms/intrinsic.py (flax+optax, online update on collection
transitions, raw-obs, running-norm). Flags: --intrinsic {none,rnd,laplacian} --intrinsic_coef.

## Tasks
Sparse MJX suite where vanilla ≈ 0: CartpoleSwingupSparse, BallInCup, AcrobotSwingupSparse.

## Pre-registered gates (solve = last-2-eval mean ≥ 800 @1M; or task-appropriate threshold)
- **G1 (rescue):** RND and/or Laplacian solve-rate ≫ vanilla (which ≈ 0/3) — does exploration
  rescue sparse TD-MPC2 at all?
- **G2 (the abstraction claim):** Laplacian solve-rate > RND solve-rate (with ≥5 seeds, CI/
  Fisher) — does abstraction-grounded exploration beat generic novelty? This is the publishable
  claim ("spectral-abstraction exploration enables sparse control beyond generic intrinsic reward").
- If RND rescues but Laplacian ≈ RND → honest "exploration helps, abstraction-flavor doesn't add".
- If neither rescues → sparse exploration on these tasks needs more than an intrinsic bonus
  (consistent with iter-17 novelty getting 745 sub-800); reconsider.

Honest prior: RND-rescues ~50%; Laplacian>RND ~25% (DCEO unproven in continuous control,
Laplacian objective finicky). Downside = clean publishable negative + the methodology.

## Status
- intrinsic.py BUILT (RND + Laplacian), RunningNorm verified, run_benchmark wired
  (--intrinsic/--intrinsic_coef), compiles. Smokes queued (rnd+laplacian, CartpoleSparse 60k).
- Coef default 1.0 (normalized reward). If no signal, sweep coef {0.5,2.0}.

## Results
*(from CSVs only — verification discipline.)*

## VERDICT (2026-06-09, n=2, partial) — RND > Laplacian; abstraction-exploration loses
- CartpoleSparse: RND FOUND reward (max 561, last2 372) where vanilla H3=0; Laplacian did NOT (0-5).
- BallInCup: both found (~837) but vanilla already could (not novel). AcrobotSparse: neither rescued.
- G1 (rescue): RND partially yes (Cartpole), seed-dependent/bimodal. G2 (Laplacian>RND): FAILS —
  Laplacian <= RND everywhere. The abstraction-grounded eigenpurpose exploration does NOT beat
  generic novelty in continuous control (consistent with deep-research: DCEO unproven/finicky here).
- HONEST: a simple intrinsic reward (RND) can rescue some sparse TD-MPC2 tasks, but the
  ABSTRACTION flavor (Laplacian) adds nothing. Not the abstraction win sought. Don't expand
  Laplacian. (RND-rescue could be firmed with more seeds, but it's a generic-exploration result.)
