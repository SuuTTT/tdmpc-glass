# DR3 (Gemini) — Information-Theoretic Abstractions in Relational World Models: Graph Latents, Structural Entropy, Visual Control

*Saved 2026-06-11. External deep-research report #3 (Gemini). Faithful archive (condensed math).*

## Taxonomy of Graph/Relational/Object-Centric WMs (2020–2026)
Monolithic latents (DreamerV3, TD-MPC2) waste capacity on backgrounds & struggle with multi-object relational physics. Four GWM branches:
1. **Object-Centric Slot Transformers** — SOLD (2410.08822): SAVi encoder + OCVP (decoupled temporal + relational attention) + Slot Aggregation Transformer for reward. Slot-MPC: action-conditioned cOCVP for goal-conditioned gradient MPC.
2. **GNN Dynamics** — ObjectZero (2601.06604): frozen SLATE/DINOSAUR slots + online GNN message passing + MCTS. CWMI: GNN + latent confounder estimators + causal constraints.
3. **Relational Token Transformers** — STICA (2511.14262): object-centric visual tokens + action/reward tokens, causality-aware policy/value; avoids rigid slot-count.
4. **Object-Addressable Hybrids** — OA-WAM (2605.06481), OC-STORM (2501.16443): semi-supervised/tracking priors (SAM/DINOv3 masks) to fix **slot-identity drift** (swap/merge/permute during long interactions). OA-WAM splits slot = persistent addr + time-varying content; cross-slot attention routes through static identity keys → zero-shot robustness to swaps.

## Control vs Prediction dichotomy (the key empirical insight)
Object-centric models give superior long-term structurally-consistent **prediction** but often **underperform/match** monolithic on **control** (Atari100k, DMControl). Cause: dynamic envs cause slot-identity drift → non-stationary noise in latent transitions → destabilizes policy/value. Monolithic bypasses this: categorical latents (DreamerV3) / SimNorm (TD-MPC2) perform a soft, task-sufficient clustering WITHOUT object-tracking overhead. **Explicit relational architectures only show clear headroom when the task demands OOD compositional scaling, multi-object coordinate alignment, or combinatorial-relation reasoning.**

## Structural Entropy (Li & Pan 2016)
SE = min bits to encode a one-step random walk under a hierarchical partition (encoding tree T). For community α: H^T(G;α) = -(g_α/V_G) log2(V_α/V_{α⁻}), where g_α=cut, V_α=volume, V_{α⁻}=parent volume. Minimizing H^T(G) decodes the natural hierarchy.

### SE in RL / latent dynamics
- **SIDM (2404.09760, JMLR 2025)**: directed weighted transition graph from trajectories; optimizes DIRECTED SE → encoding tree clusters states/actions into abstract communities → skills/options from high-freq inter-community transitions. No reward engineering.
- **SIHD (AAAI 2026)**: offline long-horizon sparse-reward; multi-scale diffusion hierarchy + SE regularizer (penalizes OOD transitions, encourages underrepresented communities).
- **SEPC (2412.08841)**: SE minimization on latent-similarity graph for representation learning (prevents collapse).
- **SEMA (2603.23875)**: SE-based observation pruning for LLM RTS agents (−50% latency).
- **EntroAD**: ViT self-attention as relational graph; row-normalized Shannon entropy of attention = patch-level SE descriptor for anomaly routing.
- Adjacent pooling: DiffPool (fixed cluster count), MinCutPool (spectral), Co-Pooling (node+edge views), **SEP (2206.13510)** (SE-guided, no fixed ratio).

### Novelty: differentiable SE on a learned dynamics graph
Historically blocked by non-differentiable encoding trees. **HypCSE (AAAI 2026, 2512.00524)**: maps node embeddings into Poincaré hyperbolic space, models hierarchical trees continuously via LCA volumes + softmax-relaxed partition indicators → SE optimizable by backprop. But HypCSE = offline continuous clustering; SIDM = offline transition graphs. **Applying a differentiable SE loss ONLINE to a learned latent dynamics graph in an active MBRL control loop is entirely novel — no framework uses differentiable SE to enforce hierarchical bottlenecks on the slot-slot interaction graph or state-transition graph of a WM during online training for visual continuous control.**

## Theoretical headroom (4 regimes where graph structure should help)
1. **High-object density / visual distractors**: monolithic global vector → representation collapse as object count scales (exponential joint config); GWM allocates O(N) linear capacity per slot, ignores distractors.
2. **Compositional extrapolation / zero-shot**: monolithic generalizes by interpolation → OOD object counts fail; GWM permutation-equivariance + shared interaction rules → add nodes at test time → zero-shot to novel counts/layouts.
3. **Long-horizon planning via graph temporal abstraction**: monolithic compounding error; SE-coarsening (SIDM) → plan over macro-actions/skills → shorter effective horizon.
4. **Multi-agent credit assignment / role discovery**: monolithic critic struggles (exponential joint action); SE over inter-agent graph → role discovery, localized value over partitions.

## Recommended architecture (semi-supervised object-addressable + HypCSE)
1. **Nodes**: frozen DINOv3 features → SAM3 masks → mask-pool → slot s_{i,t} = [MaskPool(F_t ⊙ M_{i,t}) ; addr_i] (persistent learnable identity addr_i → avoids slot drift).
2. **Dynamics**: action-conditioned object-centric Slot Transformer (cOCVP): decoupled temporal attention (per-slot causal) + relational attention (cross-slot at t), relational key-query constrained to addr_i subvectors.
3. **Differentiable SE objective**: slot similarity graph W_{ij,t}=softmax(τ·cos(s_i,s_j)); map to Poincaré disk; HypCSE continuous SE loss L_SE = CSE(W_t, {s_i}) → clusters functionally-redundant slots (gripper+grasped-object → "tool" community), maximizes separation of independent entities.
4. **Optimization**: L_total = L_pred + β1 L_reward + β2 L_SE (SE auxiliary, never primary).
5. **Control**: gradient-based MPC (differentiable WM) — minimize Σ γ^τ d_H(s_target,t+τ, s_goal) over action sequence by backprop.
6. **Benchmarks**: Robosuite multi-block, ManiSkill2 dexterous+distractors, CausalWorld; **compositional-generalization splits with held-out object counts**.
7. **Baselines that MUST be beaten**: param-matched TD-MPC2+SimNorm; DreamerV3; SOLD; **the critical no-SE entity-factored backbone ablation**.
8. **Failure modes**: semantic collapse (all slots → one community; mitigate w/ strong L_pred + dynamic β2); hyperbolic gradient explosion (clip ≤0.5, Riemannian Adam); slot collapse on realistic visuals; quadratic compute in slot count.

## Prior assessment + go/no-go
**Single most likely null: "Task-Decoupled Topological Compression"** — SE minimizes partition by topological/visual similarity, but visual/geometric similarity ≠ MDP value function. SE might cluster two red blocks (visual similarity) even though task needs the blue block → misaligned bottleneck → restricts policy → worse than monolithic (which extracts task-sufficient value-aligned bottleneck via value-gradient backprop).

**Cheap mechanism-check (do FIRST, before HypCSE pipeline):** train a standard non-regularized slot WM (SOLD); on eval episodes extract slot trajectories → build slot similarity graph W_t post-hoc → compute analytical discrete SE H^T(G) over time (Li&Pan greedy) → **cross-correlate drops in SE (community merging) with task landmarks (gripper contact, grasp, reward events)**. If analytical SE does NOT show clean step-like decreases aligned with task-critical transitions → forcing differentiable SE will introduce a task-irrelevant bottleneck and fail.

**GO (conditional):** build the object-centric WM, but condition the differentiable HypCSE loss on the post-hoc analytical-SE check. If SE drops align with contacts/value-landmarks → proceed to end-to-end differentiable SE. If decoupled → abort SE, pivot to **value-equivalence / task-centric bottlenecks on the relational attention layers** directly.

Refs: SOLD 2410.08822; STICA 2511.14262; ObjectZero 2601.06604; OA-WAM 2605.06481; OC-STORM 2501.16443; Slot-MPC 2605.14937; SIDM 2404.09760; SIHD AAAI2026; SEPC 2412.08841; SEP 2206.13510; HypCSE 2512.00524; SEMA 2603.23875; Li&Pan IEEE TIT 2016.
