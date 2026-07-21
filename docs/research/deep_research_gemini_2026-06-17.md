# Deep Research (Gemini) — Temporal Abstraction, Value-Sufficiency, Training Stability in MBRL
Saved 2026-06-17. External deep-research-agent output (Gemini), kept verbatim as the literature backbone
for paper B. Our own findings + positioning are in docs/CHANGELOG.md and the homepage Part-3 post.

## Key deltas vs our own research pass (read first)
- **Farebrother = Geometric Horizon Models (GHM) built on Temporal Difference Flows (TD-Flow)**, policy- &
  horizon-conditioned, evaluated on **OGBench** (navigation+manipulation), not plain AntMaze. Even more
  distinct from our primitive-action k-step macro than we thought.
- **Our RunningScale-saturation finding may be NOVEL.** TD-M(PC)² owns the *policy-mismatch* account of the
  late collapse (+ KL policy constraint), but Gemini reports it does **not** document the **RunningScale
  saturation** (cap=4 << true IQR≈16) numerical-scaling mechanism. => late collapse is a **DUAL failure**
  (control-theoretic policy mismatch + numerical scale saturation); the scaling half is uncharted. This is
  more favorable than our earlier "pre-empted" read — the scale-aware fix could be a real contribution.
- **Planning-vs-representation disentanglement is significant + under-published** (SPlaTES & HWM only
  implicitly support it). Our "macro-model uniformly accurate; locomotion failure is planning-centric not
  representation-centric" is a clean, citable result.
- **Redundancy criterion gap confirmed:** no one has formulated an *operational, predictive* "value-
  sufficiency sieve" (R²≥0.99 ⇒ auxiliary representation losses are null) validated by a negative campaign.

## Gemini's ranked directions (adopt #1+#3 as the paper, #2 as the method extension)
1. **Value-Sufficiency Sieve** (redundancy criterion as an online R² monitor + meta-analysis of ~16 null
   levers). Closest: Grimm VE, Ni et al. ZP. Rank 1 (novelty×feasibility high).
2. **Decision-time adaptive-horizon via local dynamics-forgiveness** (scale k/H at decision-time from
   planning-time divergence; single non-hierarchical model). Closest: HWM, ASI, THICK, SPlaTES. Rank 2.
   Duplication risks: THICK (representation-level gating), SPlaTES (low-level skill stabilization), HWM
   (two-timescale action encoder). Differentiator must be: planning-time switching, single model, no skills.
3. **Scale-Aware Advantage Normalization** (RunningScale saturation fix: dynamic IQR-expanding cap).
   Closest: TD-M(PC)², TD-MPC2. Rank 3, feasibility high. Frame policy-mismatch as reproduction; claim the
   RunningScale-saturation discovery + dynamic fix as novel.

## Verdict (Gemini)
Don't ship "jumpy macro-model + selective positive results" (reviewers see TD-MPC2+HWM+SPlaTES). Ship a
**critical deconstruction**: (1) prove the Redundancy Criterion via the value-sufficiency metric; (2)
disentangle locomotion-vs-manipulation as a *planning* (not representation) failure via the uniformly-accurate
macro-model; (3) expose the RunningScale-saturation bug + an elegant scale-aware fix. Honest + grounded →
NeurIPS/ICLR oral candidate.

## Performance reference targets (from Gemini's compiled table — VERIFY before citing)
| Benchmark | Flat TD-MPC2 | Jumpy/GHM | SPlaTES | TD-M(PC)² |
|---|---|---|---|---|
| Antmaze-Large | 0.21 | 0.61–0.90 | 0.85 | 0.92 |
| Franka Pick&Place | 0.00 | 0.70 (HWM) | 0.65 | 0.88 |
| Humanoid 61-DoF | collapse @2M | n/a | n/a | 900+ return |
| Cube-4 manip | 0.01 | 0.67–0.76 | 0.72 | 0.85 |

## Taxonomy + 17 paper cards (verbatim, Gemini)

(a) Jumpy/skip-step latent: TD-VAE, Buesing SSMs, Temporal Difference Flows, Compositional Planning w/ Jumpy WMs
(b) Temporally-abstract/slow latents: Clockwork VAE, ResDreamer
(c) Hierarchical WMs: Director, THICK (C-RSSM), Hierarchical Planning w/ Latent WMs (HWM)
(d) Options/skills-as-models: SPlaTES, ALPS (Laplacian subgoals)
(e) Adaptive/variable temporal resolution: Adaptive Skip Intervals, Neural-ODE SMDPs
(f) Macro-action MPC: TD-MPC, TD-MPC2, HWM action encoder

Cards (title / venue / link / method / planning / benchmarks / result / helps-vs-hurts):
1. Buesing Jumpy SSMs — NeurIPS 2018 — arxiv 1802.03006 — abstract latent SSM, MC tree over primitives —
   Atari — fast+accurate vs model-free — helps visually-rich/local; no unstable-continuous results.
2. TD-VAE — ICLR 2019 — openreview S1x4ghC9tQ — jumpy latent transition, no MPC — 3D mazes — consistent jumpy
   long-horizon — helps long-horizon POMDP; hurts high-freq reactive control.
3. Adaptive Skip Intervals — NeurIPS 2018 — neurips 8188 — model picks step-size τ dynamically — bouncing/
   synthetic control — better long-pred + train efficiency — helps sparse event-driven; hurts chaotic.
4. Clockwork VAE — NeurIPS 2021 — arxiv 2102.09593 — latent tiers at fixed clock rates — long video/control —
   100s-step consistent — helps long-horizon memory; hurts high-freq reactive control.
5. DreamerV2/RSSM latent overshooting — ICLR 2021 — arxiv 2010.21934 — discrete RSSM + overshooting —
   Atari/DMC — human-level Atari — strong locomotion; open-loop compounding error at long horizon.
6. Director — ICLR 2023 — arxiv 2206.02042 — 2-level RSSM, manager subgoals every k — Minecraft/DMC/Crafter —
   solves long-horizon sparse from pixels — helps sparse manip/nav; hurts dense reactive locomotion.
7. THICK (C-RSSM) — ICLR 2024 Spotlight — openreview TjCDNssXKU — GateL0RD sparse context updates, event-
   driven high level — MiniHack/VisualPinPad/Multiworld — interpretable categorical TA — helps discrete-event
   tasks; neutral/hurts continuous-uniform.
8. Limits of Hierarchical WMs — Sci Reports 2024 — arxiv 2406.00483 — RSSM stack, static resolutions —
   DMC — hierarchy did NOT beat flat (abstract-level model exploitation) — hurts standard continuous control.
9. Compositional Planning w/ Jumpy WMs (Farebrother) — ICLR 2026 ws/arxiv 2602.19634 — GHM on TD-Flow,
   policy+horizon-conditioned, plan over POLICY sequences — OGBench nav+manip — +200% zero-shot long-horizon —
   helps long-horizon compositional; hurts if base policies unstable.
10. SPlaTES (Long-Horizon Planning w/ Predictable Skills) — RLC 2025 — rlj 136 — predictable temporally-
    extended skills + abstract WM, iCEM — pointmass/quadruped/nav under perturbation — beats MB+skill
    baselines — helps long-horizon under drift; states primitive-action MBRL fails on unstable/perturbed.
11. TD-MPC — ICML 2022 — arxiv 2203.04955 — short latent rollout + TD terminal value, MPPI/CEM over primitives
    — DMC — high sample-eff — strong locomotion; sparse visual manip hard.
12. TD-MPC2 — ICLR 2024 — arxiv 2310.16828 — SimNorm discrete latent, two-hot, MPPI over primitives —
    DMC/ManiSkill/HumanoidBench — robust multi-task — late-training value overestimation/collapse high-DoF.
13. TD-M(PC)² — arxiv 2502.03550 — KL policy constraint vs planner/prior mismatch — 21 high-DoF DMC/
    HumanoidBench — eliminates overestimation/collapse, big gains 61-DoF — redundant on low-DoF.
14. HWM (Hierarchical Planning w/ Latent WMs) — arxiv 2604.03208 — multi-resolution shared latent + action
    encoder for macro-actions, subgoal + low-level MPC — Franka/Push-T/PLDM maze — 0%→70% zero-shot pick&place,
    4x cheaper — helps long-horizon; flat enough for reactive.
15. Value-Equivalence Principle (Grimm) — NeurIPS 2020 — arxiv 2011.03506 — VE models match Bellman updates on
    (V,Π) — abstract MDPs — model need not reconstruct states — helps noisy/visual; less in task-agnostic.
16. Proper Value Equivalence — NeurIPS 2021 — arxiv 2106.10316 — PVE via policy value fns; MuZero ~ PVE upper
    bound — Atari — efficient from value loss.
17. Self-Predictive (Ni et al.) — ICLR 2024 — arxiv 2401.08898 — unify SPR/DBC/TD-MPC/MuZero under ZP; ZP+RP
    prevents collapse — MiniGrid/POMDP — minimalist single-aux-loss — helps distractors; redundant on clean control.
