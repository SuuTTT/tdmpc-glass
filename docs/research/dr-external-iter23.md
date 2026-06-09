# External deep-research reports — iter-23 (novel abstraction lever on jumpy TD-MPC2 substrate)

*2026-06-09. Three external DRs (Claude, Gemini, GPT), commissioned by the user, who asked them to
ground the search in our previous Structural-Entropy (Glass) paper. Captured faithfully (rankings,
mechanisms, citations, gates, failure modes, verdicts preserved). Synthesis → iteration_23_ideation.md
round-2. Internal round-1 agent report: dr-iter23-agent-claude.md.*

---

## Report 1 — Claude (Structural-Entropy / SIDM-anchored)

**TL;DR / verdict:** the single most novel-yet-feasible lever is **Lever I: build a structural-entropy
encoding tree over the jumpy TD-MPC2 LATENT TRANSITION GRAPH, and use directed-structural-entropy
minimization to (a) derive macro-actions/skills and (b) choose the jump length k that MPPI plans over.**
A dedicated sweep found NO prior work combining structural entropy with a learned latent world-model
space, and NONE using SE minimization to set temporal abstraction for an MPPI/CEM planner → a real
mechanism-level gap, not a re-skin. Hedge with Lever F (uncertainty-gated horizon) as the safe fallback.

**Anchor paper precision — SIDM** ("Hierarchical Decision Making Based on Structural Information
Principles," Zeng, Peng, Su, Li; JMLR 26 (2025); arXiv 2404.09760): model-free HRL grounded in Li & Pan
structural information theory. Builds state & action similarity (kNN cosine) graphs over encoder-
compressed primitives, constructs an encoding tree minimizing K-dimensional structural entropy → state/
action communities. Novel theory contribution = **directed structural entropy** for asymmetric abstract-
state transitions; high-frequency inter-community transitions = skills/options; hierarchy scale set by
entropy minimization, not fixed k. Base RL = SAC/PPO. **Model-free throughout — no learned dynamics
model, no planner.** Benchmarks: gridworld nav, HSD-3 bipedal locomotion, 7-DoF Fetch, DMControl
(state-abstraction track), SC2. Headline: bipedal Hurdles +18.75% vs HSD-3; Fetch Slippery Push +32.70%
vs Reskill; DMControl hopper-hop +13.55%, hopper-stand sample-eff +64.86% vs model-free baselines.
**No model-based/world-model/planning baseline (TD-MPC/Dreamer/MuZero/PETS) anywhere.**

**SE-in-ML lineage (all model-free/diffusion):** Li & Pan 2016 (founding, IEEE Trans IT); SEP (ICML
2022, graph pooling); SISA (IJCAI 2023, arXiv 2304.12000 — SE state abstraction, SAC, +up to 18.98
reward / 44.44% sample-eff vs CURL/DBC/SAC-AE/DrQv2/SimSR); SIRD (AAAI 2023, MARL roles); SI2E (NeurIPS
2024, structural-mutual-info exploration); **SIHD** (NeurIPS 2025, arXiv 2509.21942 — SE encoding tree
over kNN state graph → adaptive multi-scale **diffusion** hierarchy; D4RL; beats Diffuser/HDMI/HD by up
to 12.6%; **NO latent world model, NO MPPI** — MPPI only a weak baseline). 2025 IJCAI SE survey confirms
RL apps are all model-free/discrete/graph.

**Gap thesis (the novelty):** (1) SE/encoding-tree over a learned world-model latent / latent transition
graph: NONE FOUND. (2) SE minimization to choose jump length k / define macro-actions an MPPI/CEM planner
plans over: NONE FOUND. → **SE × jumpy-latent-WM × MPPI is genuinely unoccupied**; close to novel-
mechanism because directed-SE over a *learned latent transition graph* has never been formulated.

**Lever-by-lever deltas:** A ≈ TAP/PLAS re-skin (LOW-MED novelty; offline, not shown to beat online
TD-MPC2). B = adaptive jump length (MED; THICK ICLR2024, Time-Aware WM 2506.08441; planner-side k-per-
state in MPPI underexplored). C = bisimulation (LOW; **failed twice for us → drop**). D = hierarchical
(LOW; Puppeteer ICLR2025 = two TD-MPC2 models, SPlaTES RLC2025 — lane occupied; Puppeteer's edge is
*naturalness* via n=51 user study, TD-MPC2 ties on reward). E = empowerment/DIAYN/METRA (LOW; **failed
twice → drop**; excels at coverage not dense return). F = uncertainty-gated horizon (MED; MBPO/PETS/
Plan2Explore when-to-trust; gating jumpy-k by latent-dynamics ensemble disagreement inside MPPI is clean
+ underexplored; TD-MPC2 already ensembles Q). G = compositional operators (MED, HIGH RISK). H = cross-
task (LOW; multi-task TD-MPC2 + HILP occupy it).

**Ranked top-4:** 1=**I** (SE k-selection first, skills phase-2), 2=**F** (uncertainty-gated horizon,
safe), 3=**B** (adaptive k — natural non-SE ablation vs SE-k), 4=A (only as a component of I, not standalone).
De-prioritize C, E, D, H, G.

**Lever I pre-registered gate:** CartpoleSwingupSparse, BallInCup, AcrobotSwingupSparse + PandaPickCube;
compute-matched, single-variable vs plain jumpy; rliable IQM, 5 seeds, bootstrap CI; **GATE: SE-k beats
plain-jumpy IQM ≥10% (peak AND final) on ≥3/4 tasks with non-overlapping 95% CIs; kill if it fails CI
separation on ≥2 tasks.** Mirage risks: (i) SimNorm latents are dense → SE may yield trivial partitions —
**check 2-D vs 1-D SE gap is materially >0 BEFORE trusting**; (ii) graph-rebuild cadence can leak compute
(must compute-match); (iii) communities may not align with reward boundaries — small-n seed luck.
**Threshold that flips ranking:** if jumpy latent graph has negligible community structure (2D-SE ≈ 1D-SE),
demote I to rank 3 and promote F to rank 1.

**Caveats:** no published case of SE beating a strong MB planner in continuous control (the opportunity +
the risk). Two close 2026 works use different machinery (not SE): Compositional Jumpy WM (TD-flow
timescales) and HWM/Hierarchical Planning w/ Latent World Models (arXiv 2604.03208 — top-down latent
planning, 70% Franka pick-&-place from goal image vs 0% VJEPA2-AC). SimNorm density is the principal risk.

---

## Report 2 — Gemini (architectural; scored table)

**Verdict / rank-1: Lever F — Epistemic-Uncertainty-Gated *Elastic* Jumpy Horizon** (~80 LOC; uses
existing ensemble; integrable + validatable in 3 weeks). Mechanism: ensemble of E k-step dynamics heads
{d_k^(e)} + V distributional critics; per macro-step compute normalized dynamics-prediction variance
U_dyn; soft gate λ_j = σ((β − U_dyn)/τ); cumulative return G = Σ (Π_{l≤j} λ_l) γ^{jk} R_k(...); same
gating sequence decays λ-returns in imagined actor-critic updates. Closest: Neubay (offline rollout
truncation), ELVIS (uncertainty-aware λ-return on 1-step RSSM, SOTA on 14 DMC visual + real-robot under
occlusion). Novelty delta = applying it to a *jumpy* latent model (deep rollout, zero single-step
compounding, auto-collapse to near-step in unstable/contact states). Rationale: kills MPPI "hallucination
exploitation." **Failure mode flagged: SimNorm "epistemic saturation"** — ensembles on heavily-normalized
latents can be identically confident yet wrong OOD → U_dyn stays low → no truncation.

**Scored table (novelty / feasibility / overall rank):** F 8/9 **rank 1**; Lever I (Gemini's = **Hermite
Spline Action Bottleneck**) 9/8 **rank 2**; A 6/8 rank 3; Lever J (**Bilinear Spectral Jumpy Dynamics**)
9/4 rank 4; C 7/5 rank 5; G 8/3 rank 6; B 8/2 rank 7 (**impractical: variable horizons break GPU-
parallel MPPI batching — serial/padding destroys speedup**); D 5/3 rank 8; E 4/4 rank 9; H 5/2 rank 10.

**New Lever — Hermite Spline Action Bottleneck (rank 2):** replace raw a_{t:t+k} with a cubic-Hermite
spline; macro-action m_j = (q_{t+k}, v_{t+k}) ∈ R^{2d} (target joint pos+vel); reconstruct trajectory via
Hermite basis + a PD tracking controller; condition d_k on m_j; MPPI samples end-states (search dim
k·d → 2d). **No learned encoder/decoder → no representational shift online** (its key edge over TAP/PLAS).
Closest: Alvarez-Padilla et al. (cubic-Hermite-spline MPPI in raw joint space on MuJoCo, reference-free
locomotion, Go2 gaits on CPU with 20–30 samples), TAP. Novelty = splines as a structural action
bottleneck *inside a learned latent WM*. Failure mode: C¹ smoothness can deprive exploration of
discontinuous actions (e.g. breaking static friction) → local minima.

**New Lever — Bilinear Spectral Jumpy Dynamics (rank 4):** factor the k-step operator low-rank-bilinear:
z_{t+k} = Ψ(z_t) diag(Λ(a_M)) Φ(z_t)^T 1, rank r≪d_z. Multi-step rollout becomes linear matrix scaling
→ fast + regularized + stable long-horizon. Closest: Spectral-Representation RL (linear MDP value/dynamics;
validated on 20+ DMC incl. visual). Novelty = spectral decomposition fused into a decision-time MPPI loop
on TD-MPC2 latents. Failure mode: low-rank linear smooths over contact discontinuities → mispredicts collisions.

**Also (lower): Macro-Temporal Revised Bisimulation** (RevBis at k-step scale; adaptive α(t); filters
high-freq micro-contact noise). Failure: representational collapse if loss over-scaled (matches our 2 fails).

**Pre-registered gates (Gemini):** F → PandaPickCube success-rate, base 68.4% → **≥82%** @N=64 MPPI, 5
seeds CI. Hermite → DMC Humanoid Run, base 480 → **≥650** @2M, N=32. Spectral → Cheetah Run latency 12.4ms
→ **≤3.5ms** with ≤5% reward decay. Macro-bisim → Walker-Walk+distractors, base 320 → **≥580**.

---

## Report 3 — GPT (consensus-leaning; conservative)

**Verdict: Lever F (uncertainty-gated jumpy horizon) is the single most promising lever** — least new
structure (reuses model uncertainty), offline-RL precedent (Neubay adaptive long-rollouts), compute-
matched robust improvement; modest novelty but most feasible/likely to beat the jumpy baseline.

**Top-4:** 1=uncertainty-gated jumpy horizon (mechanism: stop extending k / truncate rollout when k-step
ensemble variance exceeds threshold; only trust in-distribution model regions; single-variable
threshold). 2=state-dependent adaptive jump length (small k at contacts/complex dynamics, large k on
smooth segments; little prior precedent → higher novelty; risk: harder-to-train variable-k model,
possible instability). 3=reward-predictive temporal abstraction (compress z_{t+k} to return-relevant
info, bisimulation-flavored at macro scale; cites Lehnert et al. 2020 reward-predictive representations;
risk: over-compression drops planning-relevant info; no continuous-control precedent). 4=two-timescale
hierarchical jumpy planning (high-level latent subgoal + low-level short-horizon MPPI; closest = **HWM /
Hierarchical World Models, Zhang et al. 2026** — Franka Pick-&-Place 0%→70%, Push-T 17%→61%; so the
mechanism already works but lane occupied + violates single-variable cleanliness).

**Falsification (all):** vs plain jumpy on PandaPickCube + MuJoCo locomotion, score mean+95% CI; invalid
if CI covers 0 / improvement < ~3–5%. **Failure modes:** F — threshold tuning / explore-exploit (too
strict = short horizon = no gain; too wide = no gating; mis-estimated uncertainty truncates good
rollouts). B — training complexity, may amplify model bias / search space. Reward-predictive — over-
abstraction discards useful info, unstable objective, no priors. Hierarchical — implementation
complexity, multi-variable, miscalibrated subgoals early, compute cost.

---

## Cross-report synthesis (one paragraph)
**Strong 3/3 consensus that F (uncertainty-gated jumpy horizon) is the safe, feasible, likely-to-work
pick** (Gemini #1, GPT #1, Claude #2) — but all three concede its novelty is modest. **Claude uniquely
champions the SE lever (I)** as the genuinely-novel, SE-paper-anchored pick (highest upside, higher risk).
**Gemini contributes two genuinely-novel architectural levers** (Hermite-spline action bottleneck — no
learned codec so no online representational shift; bilinear-spectral dynamics — fast/stable rollouts).
**Convergent kills:** bisimulation (C) and empowerment (E) — failed twice for us, all agree drop; D and H
lanes occupied (Puppeteer/HWM, multi-task TD-MPC2/HILP). **Two SimNorm-specific risks gate the two front-
runners:** SE may yield trivial partitions on dense SimNorm latents (check 2D-SE > 1D-SE first); ensembles
may saturate (be confidently wrong) on normalized latents. **Practical constraint (Gemini):** variable-k
breaks GPU-parallel MPPI batching → B/SE-k need a batch-friendly formulation (fixed macro-grid with
per-trajectory soft masks, not ragged horizons).
