# TD-MPC-Glass research ledger — what worked, what didn't, what to probe next

*2026-06-09. Single source of truth for the campaign's verdicts and the forward queue. Every entry is
backed by run CSVs / mechanism-checks (no notebook numbers). Goal: beat vanilla TD-MPC2 at the
architecture/algorithm level under a fair protocol (single-variable, compute-matched, pre-registered
peak+final CI gates, mechanism-check before fan-out, no procedure tricks).*

## ✅ WHAT WORKED
| thing | evidence | honest status |
|---|---|---|
| **Jumpy (k-step) world model beats vanilla on manipulation** | PandaPickCube n=5: peak +966/+44% CI[714,1248], final +1266/+88% CI[877,1642], both CI-separated; mechanism pre-confirmed (jumpy_err/iter1_err 0.99→0.82 as k grows) | REAL win, but jumpy is **known prior art** (Farebrother 2602.19634, 2026). Fair-protocol reproduction+evaluation, not our invention. |
| **Methodology: peak+final + pre-registered CI gates + mechanism-check-before-fanout** | caught all 8 mirages; killed SE-k/F in an afternoon instead of a multi-week campaign | The most transferable output of the project. |
| **Ensemble-free uncertainty signal** (jumpy-pred vs iterated-1-step-pred) | tracks true k-step error, Spearman +0.72 | Validated, reusable (exploration/abstention) — just not for horizon-gating (see below). |

## ❌ WHAT DID NOT WORK (nulls, most→least thoroughly killed)
| # | lever | iter | why it died |
|---|---|---|---|
| 1 | Geometric prototype clustering (structural-entropy Glass) | 14 | redundant with SimNorm's soft-categorical latent (IQM 0.748 vs 0.738, overlap) |
| 2 | Behavioral / reward-grounded clustering | 14 | null at n=34; gain crossed CI-separation 3× then settled in overlap |
| 3 | Bisimulation auxiliary (BS-MPC style) | 14 | actively hurts (0.549); brittle to coef; **failed twice** |
| 4 | Distractor robustness from abstraction | 14 | falsified (1.23× < 1.5× gate); both encoders crushed equally |
| 5 | Sparse-task rescue via grouping | 14 | it's an exploration problem, not latent geometry (0/3 vs 1/3) |
| 6 | "Floor effect" / weak-seed tail | 14 | inverted — behavioral arm produced the study's worst seed (0.127) |
| 7 | Laplacian / eigenpurpose exploration | 21 | generic RND ≥ it everywhere; **unsupervised-abstraction fail #2** |
| 8 | Community-detection *skills* | 19 | communities = motion phases, not reachable subgoals |
| 9 | `rho` consistency-horizon schedule | 20 | task-dependent tuning knob, not architecture (helps Panda, hurts sparse) |
| 10 | **SE-k adaptive jump-length** (structural entropy → k) | 23 | SE pre-check PASSED (53% gap) but mechanism-check FAILED: boundary score does NOT track k-step error (Spearman +0.09 Panda / −0.18 Cart) |
| 11 | **F: uncertainty-gated horizon** | 23 | signal valid (disc↔err Spearman 0.72) but **no headroom**: k-step error uniform in-dist AND under MPPI-perturbed actions (inflation 1.06×) — nothing to gate |

| 12 | **SI2E / VCSE SE-exploration** (value-conditional kNN entropy + cluster term) | 24 | NULL: no rescue of sparse Cart/Acro beyond vanilla; doesn't beat RND (all 0/n); at coef 1.0 mildly HURTS (best seeds < vanilla) |
| 13 | **wmsi2e — SE-exploration over the WORLD-MODEL latent (the novel bet)** | 24 | NULL: ties si2e at 0/n; WM-latent + critic-value conditioning adds nothing over random-encoder SI2E or RND. 3rd exploration null (after community-skills, Laplacian) |

**Root cause for #10–11 (and the whole adaptive-k family):** the trained jumpy model is **uniformly
accurate** over the states/actions a near-optimal policy visits — which is exactly why fixed-k jumpy
already works, and why adaptive jump-length has nothing to adapt to.

**Cross-cutting lesson:** a strong self-predictive world model (TD-MPC2 + SimNorm) is a high bar; most
"abstraction" is redundant with what it already learns ([Ni et al. 2024] sufficient-abstraction theory).

## 🔭 WHAT'S PROMISING (untested; NOT killed by the uniform-error finding)
- **Hermite-spline action bottleneck** (Gemini DR): macro-action = target (q,v), cubic-Hermite interp +
  PD tracker. Shrinks MPPI search dim \(k\cdot d \to 2d\); **no learned codec → no online repr-shift**
  (its edge over TAP/PLAS). Independent of error-variance, so this campaign's verdict doesn't touch it.
- **Value-equivalent macro head**: train \(d_k\) return-equivalent over k steps (predict same macro-Q)
  not state-faithful — abstraction keeps only control-relevant info. Single-variable loss change.
- **SI2E-style SE-driven EXPLORATION** (the real home for our validated SE structure). SI2E (NeurIPS
  2024, 2410.06621) minimizes SE over a VALUE-difference state-action graph -> encoding tree -> a
  *value-conditional structural-entropy* intrinsic reward for coverage; beats SOTA exploration baselines
  on MiniGrid/MetaWorld/DMControl. KEY: this points SE at coverage (where motion-phase/community
  structure IS the relevant axis), NOT at adaptive-k (where it died). Sparse-task regime = where
  abstraction has evidence and SimNorm's "sufficient representation" does not help (bottleneck is
  exploration, not encoding). We already have the harness (iter-21 intrinsic.py RND+Laplacian, --intrinsic
  flag, sparse tasks) + the RND baseline. CAVEAT: iter-21 Laplacian (geometric SE-flavored exploration)
  LOST to RND; SI2E is value-conditional (richer) so a fair re-attempt, but prior is cautious.
- **Reusing the disagreement signal** (jumpy vs iterated-1-step) for safe/abstain planning — cheap,
  orthogonal, the one positive by-product of the F null.

WHY "SE to enhance return" splits: DIRECT SE-clustering-for-return on DENSE tasks = ALREADY NULL
(geoglass/behavglass mirages #1-2; redundant with SimNorm — a better clusterer can't fix redundancy).
INDIRECT (SE->exploration->finds reward->return) on SPARSE tasks = the live path (= SI2E). Pre-check-1
passing means the SE structure is REAL, NOT that clustering will raise dense-task return.
- **High-DoF done right**: our Humanoid probe FLOORED at 500k (needs millions of steps); the literature
  locates abstraction headroom on Dog/Humanoid, but only a real-budget run can test it honestly.

## 📋 NEXT PROBES — in priority order
1. **Hermite-spline action bottleneck.** Mechanism-check first: does spline-restricting the action
   sequence preserve the achievable-return envelope (i.e. can splines still express the winning Panda
   trajectories)? If yes → pre-registered beat-jumpy gate (5 seeds, peak+final, CI, sparse+Panda).
   *Highest novelty-per-risk; doesn't need error-variance.*
2. **Value-equivalent macro head.** Add a macro-Q-equivalence loss to \(d_k\); single-variable vs
   state-faithful jumpy; test under distractors (where state-faithful wastes capacity).
3. **SI2E-style SE-driven exploration** (the SE lever's correct home). Wire a value-conditional-SE
   intrinsic reward into the existing iter-21 intrinsic harness; sparse tasks (CartpoleSparse, BallInCup,
   AcrobotSparse). PRE-CHECK: value-difference SE encoding tree is non-trivial on our latents. KILL-TEST
   / GATE: **SI2E must BEAT RND** (not just rescue) — the iter-21 G2 bar Laplacian failed — IQM over 5
   seeds, peak+final, CI. If SI2E <= RND -> abstraction-flavored exploration adds nothing (consistent
   with Laplacian), record and stop. (Sub-option: jumpy/iter-1-step disagreement as an alternate bonus.)
4. **Humanoid/Dog at real budget** (millions of steps) — only with the compute to do it honestly;
   settles whether jumpy's manipulation win extends to high-DoF.

## Gate discipline (applies to every probe above)
Single-variable vs the right baseline (jumpy, not vanilla); compute-matched; rliable IQM, ≥5 seeds,
**peak AND final**, bootstrap CI; **mechanism-check before any multi-seed fan-out**; read ≥400k, distrust
any single snapshot; no restarts/PBT/per-task tuning. A win = ≥10% with non-overlapping CI on ≥3/4 tasks.
