# External deep-research (Claude / GPT / Gemini), 2026-06-08 — verbatim archive

Three external deep-research reports on "beat TD-MPC2 with abstraction/skills in continuous
control", provided by the user. Saved verbatim for the record. Synthesis + decision in
`dr-synthesis-iter21-FINAL.md`.

================================================================================
## CLAUDE DR
================================================================================
TL;DR: Lowest-risk architectural bet = a "jumpy" temporally-abstract latent prediction head
(learned k-step latent dynamics d^(k)(z_t,·)->z_{t+k}) that MPPI plans over at macro scale —
NOT eigenoptions, NOT skills-over-MPPI. Directly attacks the compounding 1-step error (H=9
helps sparse, collapses Panda). Survives all 3 prior failures (it's temporal prediction, not
latent clustering/bisimulation/reach-centroid). Evidence: SPlaTES (RLC 2025), THICK (ICLR
2024) — skill/abstract-outcome models beat flat MBRL on long-horizon/sparse; distilling a
hierarchical agent back into flat TD-MPC2 DEGRADES to myopic behavior.
- Eigenoptions/DCEO eigen-direction exploration = HIGH-risk/high-novelty, NOT low-risk: the
  entire graph-Laplacian/eigenoption line has NEVER been demonstrated on continuous-ACTION
  control (only discrete gridworlds/MiniWorld/Atari). "Continuous" there = continuous STATE,
  not actions. Porting to MJX is itself an unproven contribution.
- "Beat TD-MPC2 with abstraction" is wrong framing for dense proprio (TD-MPC2 already a
  sufficient self-predictive abstraction, Ni et al. ICLR 2024 — explains the proprio null).
  Reframe to exploration / long-horizon credit assignment on sparse tasks.
- Skill discovery (METRA/LSD/CSD/DADS/DIAYN/CIC): improve coverage/zero-shot/adaptation, NOT
  task return on dense DMC. Skill+planner beats flat ONLY on exploration-hard/long-horizon
  sparse. Puppeteer (hierarchical TD-MPC2, ICLR 2025) does NOT beat flat TD-MPC2 on return
  (wins on naturalness). DADS+MPC = cleanest "plan-in-skill-space beats flat MBRL" but on Ant
  navigation.
RANK: 1=jumpy k-step head (lowest-risk, reuses SimNorm latent/consistency/MPPI/value, add one
head+loss). 2=eigen-direction Laplacian intrinsic reward (high-novelty, high-null-risk, first
continuous-action eigenoption). 3=METRA/LSD skills+MPPI (transient skills, weak fit). 4=ensemble
-disagreement (Plan2Explore) intrinsic (cheap complement, not standalone).
KILL-GATES for jumpy: sparse — non-overlapping IQM CI above vanilla on >=2/3 sparse tasks;
guardrail — PandaPickCube IQM not below ~90% vanilla; mechanism — k-step latent error < iterated
1-step error. Expected: modest defensible sparse/long-horizon win plausible; clean dense-Panda
win unlikely (tie at best). Jumpy = most likely publishable, least likely a 3rd null.
Cites: TD-MPC2; Ni et al. ICLR2024; SPlaTES RLC2025; THICK ICLR2024; Director NeurIPS2022;
Puppeteer ICLR2025; DADS ICLR2020; METRA/LSD/CSD/HILP/CIC; eigenoptions/DCEO/ALLO; Choreographer;
Plan2Explore.

================================================================================
## GPT DR
================================================================================
- METRA/LSD/CSD/DADS learn useful skills in high-dim continuous control and can lift downstream
  via a low-level controller; DIAYN/CIC mainly improve exploration diversity (limited direct
  return). Spectral/Laplacian methods: barely applied to continuous control; gains are in
  exploration coverage, NO strong evidence of improving final return; online auto-discovery
  often "deviates and hinders learning" (Kotamreddy & Machado 2025); bottleneck options can
  HURT in locomotion (bottlenecks not meaningful there).
- Skills+model-based: SkiMo (CoRL 2022 / cited 2025) — learn skill set + skill-dynamics model,
  plan in skill space; on long-horizon sparse (maze, kitchen) SkiMo >> flat MBRL (Dreamer/
  TD-MPC): only SkiMo reliably finds sparse maze goal; ~5x fewer samples on kitchen. DADS+MPC
  beats flat MBRL on Ant nav (dense+sparse). No public report of METRA+WorldModel / LSD+Dreamer
  beating single-level TD-MPC2/Director on the same benchmark.
RANK: 1=METRA/LSD skills + MPPI (recommended — directly uses proven skills as planning
primitives; like DADS+MPC but better skills). 2=Jumpy latent model (predict z_{t+k}, reduce
compounding error — moderate change/uncertain payoff; SkiMo validates the family). 3=DCEO/
spectral intrinsic reward (novel, high risk). 4=multi-resolution/other (future).
Reframe: TD-MPC2 already strong; target sparse-exploration / multi-task where skills shine.
Define contribution by exploration efficiency / solution feasibility (success rate, convergence)
not single-task return. Protocol: CartpoleSwingupSparse/BallInCup/PandaPickCube, IQM+bootstrap
CI >=5 seeds, pre-registered fail thresholds.

================================================================================
## GEMINI DR
================================================================================
Strong recommendation: **Horizon-Consistent Jumpy TD-MPC2 (HC-TDMPC).** Keep TD-MPC2's
self-predictive SimNorm latent; replace the 1-step dynamics with a MULTI-STEP jumpy model
z_{t+k}=d_θ(z_t, a_{t:t+k-1}, k), trained with a HORIZON-CONSISTENCY loss aligning predictions
across timescales (predict z_{t+2k} directly == sequence two k-step preds):
  L_HC = || d(z_t,a,2k) - d(d(z_t,a,k),a,k) ||^2
plus multi-step reward/value CE targets + an InfoNCE contrastive alignment (anti-collapse).
Precedent: Jumpy World Models (Farebrother et al. 2026) — 200% over 1-step on long-horizon
nav/manipulation; Temporal Difference Flows; THICK; Director; variable-length MBRL.
- Method comparison table: METRA (high MJX-compat, biased to locomotion, coverage util), DADS
  (low JAX compat — dynamics model in plan loop slow), CIC (kNN overhead), LSD/CSD (scale
  sensitive), DCEO/Laplacian (high compat but buffer-coverage dependent; redundant local
  options). Director: discretization hurts proprio. TAAC: dynamic action-repeat beats flat on
  14 continuous tasks (model-free). JWM: 200% over 1-step — the rigorous validated combo.
- Path analysis: A (DCEO eigen-exploration) = attacks sparse, avoids Louvain-centroid failure,
  but high complexity + must suppress on dense/manipulation. B (Jumpy z_{t+k}) = directly solves
  the H-vs-compounding-error collapse, keeps SimNorm (dodges proprio null), JAX-friendly. C
  (METRA/LSD skills+MPPI) = compounding error if 1-step model; high-variance skill-outcome model
  on precision manipulation.
- JAX impl: static horizon bucketing k∈{1,2,4,8} (avoid recompile); jax.lax.associative_scan
  for O(log H) jumpy rollout (associativity d(z,a,4)=d(d(z,a,2),a,2)).
- Pre-registered KILL GATES: (1) compile >3x vanilla OR SPS <50% -> kill; (2) PandaPickCube
  @100k H_eff=8 IQM < 1500 -> kill (manipulation collapse); (3) CartpoleSparse/AcrobotSparse
  @100k IQM fails to exceed 500 -> kill (jumpy prevents fine swingup search).
- Reframe: contribution = long-horizon planning WITHOUT compounding error (precision on
  manipulation + exploration on sparse), a unified world model. High probability of a
  publication-grade win; circumvents proprio null by altering only temporal composition.
