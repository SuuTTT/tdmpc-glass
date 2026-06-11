# Proposal: Value-Gated Structural Entropy on a Relational World Model (VG-SE-WM)

*2026-06-11. My synthesis of the 3 deep-research reports (Fable5/GPT/Gemini, in graph-wm-dr/) + this project's
hard-won lessons. Brainstorm — the original direction that I think actually has a shot, and the de-risked
fallback that's publishable either way.*

## What the 3 DRs agree on (the convergent verdict)
1. **Object-centric/graph WMs rarely beat monolithic on CONTROL** (DLPWM<DreamerV3, ROCA shows DreamerV3>slate, ObjectZero≈EZ-V2; only SOLD wins, narrow, bespoke, 3 seeds). They win on *prediction*, not control.
2. **The gap is real but null-shaped:** a differentiable SE loss on a *learned* WM dynamics graph is genuinely unexplored — but it most likely fails for the same reason our 15 levers did.
3. **The #1 predicted failure (all 3 name it):** *task-decoupled topological compression* — SE/clustering groups by visual/topological **similarity**, which ≠ the MDP value function. A similarity-SE bottleneck is misaligned and *hurts*. This is EXACTLY our campaign's recurring finding: SE finds control-irrelevant structure (SimNorm's 53% motion-phase gap) or no structure (our transformer-attention-graph NO-GO, 2026-06-11).
4. **Headroom exists ONLY in compositional/OOD generalization** (held-out object counts/combos), not in-distribution efficiency where monolithic is already value-sufficient.
5. **All 3 prescribe: mechanism-check FIRST, build LATER**; and Fable5's strategic note — the highest-value output may be a **rigorous redundancy PRINCIPLE**, not a win.

This maps perfectly onto our methodology. We already have **2 data points** for the principle (monolithic SimNorm: value-decodable R²=1.0 → SE redundant; token-attention-graph: structureless → SE NO-GO). An entity-graph would be the 3rd.

## My proposal — two things, in order

### A) The de-risked, guaranteed output: the redundancy-criterion paper (do regardless)
Turn 16 nulls into a falsifiable **predictive criterion**:
> *"Explicit abstraction (clustering/SE/value-equivalence) is redundant for a world model exactly when its latent is **linearly value-decodable** and its interaction graph **already carries task-aligned structure** — and this holds across monolithic (TD-MPC2/SimNorm, R²=1.0), token-transformer (attention-graph NO-GO), and entity-graph latents."*
We have the tools (`value_probe`, `se_attention_graph`) and 2/3 data points already. One entity-graph probe completes it. This is the publishable spine and it de-risks everything.

### B) The original positive bet (only if the mechanism-check below passes): **Value-Gated Structural Entropy**
The novelty that attacks the #1 failure mode head-on. Three moves, none of which the DRs' proposals make:

1. **Build the abstraction graph from VALUE-COUPLING, not similarity.** The killer of every prior attempt (and ours) is clustering by similarity (visual/latent), which is task-decoupled. Instead define edge weight
   `w_ij = sensitivity of return to the i–j interaction` (e.g. |∂Q/∂(message_ij)| or a reward-gated dynamics-coupling), then run differentiable SE (`se_jax`/HypCSE-style) on **this value-coupling graph**. SE now finds **value-relevant communities** (which entities' interactions matter for the task), not red-blocks-look-alike communities. This directly converts "task-decoupled topological compression" into "task-coupled compression."
   - *Why this could be non-redundant where everything else was redundant:* a monolithic latent is value-sufficient for the *state*, but it does **not expose the sparse relational structure of which interactions drive return** — that's a property of the *dynamics graph*, not the state vector. Monolithic doesn't factor it; SimNorm's softmax doesn't hand it to you. So the redundancy argument may not apply here.

2. **Reframe the target: from "cluster the latent" (redundant) to "discover the sparse value-relevant interaction graph" (relational).** This is closer to value-aware causal factorization (CDL) + an SE-hierarchy prior than to slot-SE. The SE term's job is to make the discovered value-coupling graph *hierarchically sparse* (few macro-interactions), which is what transfers across object counts.

3. **Evaluate ONLY in the compositional-OOD regime.** Train on N objects, test on held-out N′≠N counts/combinations. The falsifiable bet: a value-gated sparse interaction graph transfers (shared interaction rules, permutation-equivariant) where a monolithic latent shifts OOD and collapses. In-distribution we *expect* a tie (monolithic is sufficient) — don't even claim it.

## The cheap mechanism-check that gates B (do FIRST — our discipline + all 3 DRs)
Train a plain **entity-factored** WM (ground-truth entity states first, to dodge slot-collapse confounds — per DLPWM's drift finding) on a 3–6-object manipulation task (ManiSkill/Robosuite). Then probe, with tools we already have:
- **(i) OOD value-decodability** (`value_probe`, sharpened): is value linearly decodable from the flat entity concat **in-distribution AND at held-out object counts**? Monolithic is sufficient in-dist — the question is whether it **collapses OOD**. *Headroom requires R² to drop OOD.*
- **(ii) value-coupling-graph structure** (`se_attention_graph`, but on the value-coupling graph, not the attention/similarity graph): does the value-coupling graph show task-aligned communities — SE drops at contact/grasp/reward events (Gemini's check, on the *value* graph)?
- **(iii) transfer**: does the value-coupling structure persist across object counts?

**GO** only if **R² collapses OOD** (structure genuinely needed) **AND** the value-coupling graph is task-aligned + transferable. **NO-GO / write the principle paper** if value stays decodable OOD or the value-graph is structureless (the likely outcome, honestly — prior ~25%).

## Why this is better than the DRs' default proposal
The DRs' default ("differentiable SE on the slot-similarity graph + HypCSE") walks straight into the failure mode they themselves name. My version (a) swaps similarity→value-coupling to defeat task-decoupling, (b) reframes the target to the one relational property monolithic doesn't expose, (c) tests only where headroom exists, and (d) keeps the principle paper as the guaranteed output. It's the difference between "re-run the 16th null with more machinery" and "test the one hypothesis our own nulls haven't excluded."

## Honest prior + first step
Prior on B being a real win: ~25–30% (better than the DRs' implicit ~15% because of the value-gating + regime targeting, but still a bet). Prior on A (the principle paper) being publishable: high. 
**First concrete step (cheap, ~1 wk):** build the entity-factored WM on a multi-object ManiSkill task + run probes (i)/(ii)/(iii). That single experiment either opens B or completes A — and we already have `value_probe`, `se_jax`, `se_attention_graph`, and the WM infra. Do the mechanism-check before any HypCSE/differentiable-SE build.
