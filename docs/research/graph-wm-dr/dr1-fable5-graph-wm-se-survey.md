# DR1 (Fable5) — Graph/Structured World Models for Control, and Structural Entropy as an Abstraction Objective: A Critical Survey

*Saved 2026-06-11. External deep-research report #1 (Fable5). Verbatim archive for the graph-WM+SE direction.*

## TL;DR
- The redundancy thesis is largely correct & well-supported: a strong self-predictive vector latent (TD-MPC2+SimNorm, DreamerV3) already encodes value-aligned, implicitly-factored structure; published object-centric/graph WMs repeatedly **fail to beat monolithic baselines on control** (DLPWM < DreamerV3; DreamerV3 > DreamerV3:slate; OCRL benefits task-conditional). Few wins (SOLD) are on bespoke multi-object relational tasks, not replicated.
- The specific direction — a **differentiable SE loss on a learned dynamics graph inside a world model** — is a genuine EMPTY GAP. SE-in-RL (SISA, SIDM, SI2E) uses discrete greedy encoding-tree on a precomputed replay/value graph, not differentiably inside a Dreamer/TD-MPC latent; differentiable SE exists only for static graph clustering (LSEnet, DeSE). But it's plausibly a **null-result-shaped gap** for the same reason the 15 levers failed.
- **Go/No-Go: conditional GO on a cheap mechanism-check, NO-GO on building the full system first.** Build a plain entity-factored transformer/GNN WM on multi-object manipulation; probe (a) linear value-decodability from the flat entity concatenation and (b) whether learned attention/adjacency already shows object-block structure. If value is linearly decodable and structure already present (likely), the SE loss is redundant — kill before building.

## Key findings
1. Object-centric WMs = crowded field, weak control track record: structured/slot latents help prediction & interpretability but usually do NOT beat monolithic DreamerV3/TD-MPC2 on control, often underperform. SOLD (2410.08822) is the most credible counterexample but only on bespoke relational manipulation.
2. SE-in-RL = small single-group niche (Zeng/Peng/Li, Beihang): SISA(2304.12000), SIDM(2404.09760), SIRD, SI2E. All use SE for state/action/skill abstraction on a precomputed transition graph via discrete greedy encoding-tree minimization, never as a differentiable loss inside a WM latent.
3. SimNorm plausibly provides implicit clustering (softmax-simplex per group ≈ soft cluster assignment) — supports "clustering for free."
4. Self-predictive theory (Ni et al. 2024) supports linear value-decodability as meaningful (not complete) sufficiency: R²≈1 = latent already contains value-relevant structure an abstraction objective would re-extract.
5. Genuine headroom is in compositional/combinatorial generalization (held-out object counts/combos) and sparse-interaction causal factorization — NOT in-distribution sample efficiency; even there evidence is thin/contested.

## Comparison: graph/structured WMs (control vs prediction; beats monolithic?)
- GNS (2002.09405, 2020): particles/proximity, GNN MP, fluids/rigids — prediction only, not compared.
- C-SWM (1911.12247): object slots/relational GNN contrastive — prediction/representation, no policy.
- RoboCraft (2205.02909): particles+GNN+MPC, real deformable — control, different task class.
- FOCUS (2307.02427): object latents on DreamerV2 — ≈DreamerV2 dense; gains in sparse-reward exploration.
- ROCA (2310.17178): SLATE slots+GNN — DreamerV3 BEATS DreamerV3:slate.
- IRIS (2209.00588): image tokens+transformer — Atari100k SOTA but monolithic-token.
- STORM (2310.09615): image tokens+stochastic transformer — competitive, monolithic-token.
- SOLD (2410.08822): SAVi slots+OCVP — beats DreamerV3/TD-MPC2 on relational tasks (3 seeds, 12M-matched); narrow on standard.
- Dyn-O (NeurIPS'25): object slots — prediction (rollout) only.
- DLPWM (2511.06136): disentangled object latents — UNDERPERFORMS DreamerV3 (representation drift).
- ObjectZero (2601.06604): slots+GNN+MCTS — ≈EZ-V2; slightly > DreamerV3.

## Comparison: SE / graph-abstraction methods (inside a WM?)
- Li&Pan SE (IEEE TIT 2016): foundational, greedy encoding-tree.
- SISA(2304.12000)/SIDM(2404.09760)/SIRD/SI2E(NeurIPS'24): discrete greedy on precomputed graphs — NOT inside a WM.
- SI Hier. Diffusion (2509.21942): SE = exploration regularizer on diffusion policy — not RSSM/TD-MPC latent.
- SEP pooling (2206.13510): static graph classification.
- LSEnet (2405.11801) / DeSE: DIFFERENTIABLE soft-assignment SE — static clustering, not RL.
- DiffPool/MinCutPool/DMoN: differentiable pooling, not RL.
- **Target: differentiable SE on learned WM dynamics graph = EMPTY GAP.**

## Q5 — honest prior / go-no-go
Single most likely null reason: the graph WM's per-entity latents + attention adjacency are already individually value-sufficient and already exhibit object-block structure → SE re-derives structure message-passing/attention learned for free (the exact mechanism behind the 15 nulls, one level up). Secondary: standard manipulation benchmarks (few objects) don't require relational abstraction at tested scale.

Cheap mechanism-check (kill first): train plain entity-factored transformer/GNN WM on multi-object task, probe (1) value linearly decodable (R²) from flat concatenation of entity latents? (2) learned attention/adjacency shows block structure matching objects (modularity/NMI vs ground-truth)? (3) post-hoc SE/encoding-tree of learned dynamics graph reveals non-trivial communities correlating with task structure? If yes to (1) and (2 or 3) → SE redundant → kill. Only proceed if value NOT linearly decodable AND attention structureless.

## Stages
- Stage 0 (1-2 wk, decisive): mechanism-check on 3-6 object ManiSkill/Robosuite stacking, ground-truth entity states + flat transformer-over-entities. Keep going only if value-decode R²<~0.9 AND attention modularity/NMI near chance.
- Stage 1: add differentiable soft-assignment SE aux loss (LSEnet/DeSE-style, NOT greedy); must beat identical no-SE backbone on held-out-object-count compositional split (≥5 seeds).
- Stage 2: add param-matched TD-MPC2+SimNorm & DreamerV3 + full compositional suite; beat both on OOD object counts.
- Kill threshold: if SE only matches (not beats) no-SE backbone, or entity-factored backbone loses to monolithic → abandon.

**Strategic note:** highest-value contribution may not be "SE makes control better" (likely null) but a rigorous negative/diagnostic result: "linear value-decodability + attention-block-structure jointly predict abstraction-objective redundancy across monolithic AND graph latents" — converting 16 nulls into a falsifiable principle. Publishable, de-risks the program.

## Caveats
- SOLD's claim is authors' on self-selected relational tasks (3 seeds), large only on "Distinct" variants, not replicated.
- "Empty gap" = absence-of-evidence (SE-RL dominated by one fast-moving group).
- Linear value-decodability is a training-distribution property; doesn't certify compositional/OOD.
- CDL/causal-factorization gains on low-dim ground-truth state, not pixels.

Key refs: Li&Pan IEEE TIT 2016; TD-MPC2 2310.16828; DreamerV3 2301.04104/Nature2025; C-SWM 1911.12247; GNS 2002.09405; FOCUS 2307.02427; SOLD 2410.08822; ROCA 2310.17178; DLPWM 2511.06136; ObjectZero 2601.06604; IRIS 2209.00588; STORM 2310.09615; Ni et al. 2401.08898; SISA 2304.12000; SIDM 2404.09760; SI2E (NeurIPS'24); SEP 2206.13510; LSEnet 2405.11801; CDL 2206.13452; eigenoptions 1703.00956; L³P 2011.12491; IJCAI'25 SE survey.
