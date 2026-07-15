---
layout: post
title: "TD-MPC-Glass, Part 13: JEPA From Scratch — Representation, Collapse, Hierarchy, Energy, and Structural Information"
date: 2026-07-08
description: "A from-scratch explainer of the JEPA line and where our structural-entropy work fits. What a Joint-Embedding Predictive Architecture is and what it was designed for (image/video self-supervised representation learning), what representation collapse is and how it's measured and prevented, how JEPA is turned into a controller (frozen encoder + latent dynamics + planning), what Hierarchical JEPA and hierarchical planning are, how all of this differs from the Dreamer line of generative world models, LeCun's energy-based-model framing (planning as energy minimization), and the information-theory thread — from JEPA's implicit information-maximization to Li-Pan structural entropy as an explicit structural-information objective for the latent."
---

> A from-scratch tour of the JEPA family, written to fix the vocabulary our program keeps using loosely — collapse,
> hierarchy, energy, structural information — and to place our structural-entropy (SE) work inside it. If you only
> take one thing: **JEPA is a way to learn a representation by predicting in *latent* space instead of pixel space,
> and everything interesting about it — collapse, anti-collapse, energy, hierarchy — is a consequence of that one
> choice.**

## 1. What is a JEPA?

**JEPA = Joint-Embedding Predictive Architecture** (Yann LeCun). Take an input, split it into a context part
\\(x\\) and a target part \\(y\\). Encode both, and train a predictor to predict the *embedding* of \\(y\\) from
the embedding of \\(x\\):

$$\text{minimize}\quad \big\lVert \text{pred}(\text{enc}(x)) - \text{sg}[\text{enc}_{\text{tgt}}(y)]\big\rVert^2 .$$

The key word is *joint-embedding*: the loss lives in **representation space**, not input space. Contrast the three
families of self-supervised learning:

- **Generative / reconstructive** (autoencoders, MAE, Dreamer's RSSM): predict the *raw* \\(y\\) (pixels), so the
  model must spend capacity on every predictable *and unpredictable* detail (texture, noise).
- **Contrastive** (SimCLR): pull embeddings of matching pairs together, push non-matching apart — needs negative
  samples and careful batch construction.
- **JEPA (joint-embedding predictive)**: predict the *embedding* of \\(y\\). Because the target is itself a learned
  representation, the model **can discard unpredictable detail** and keep only what is predictable/structural — the
  property LeCun emphasizes ("choose to ignore details that are not easily predictable").

## 2. What was JEPA designed for?

Representation learning for **perception**, not RL. The flagship instances:

- **I-JEPA** (image): mask patches, predict their embeddings from visible patches. Evaluated on ImageNet-1k
  linear-probe / low-shot, CIFAR, Places, iNaturalist.
- **V-JEPA / V-JEPA 2** (video): predict masked space-time embeddings. Evaluated on Kinetics-400 (~82%),
  Something-Something-v2 (~71%), ImageNet.

So a JEPA's native "task" is: *produce a frozen encoder whose features a simple probe can read off for downstream
recognition.* Control came later, and always as an add-on (§5).

## 3. What is collapse, and how is it measured and prevented?

**Collapse** is the trivial solution to "predict your own embedding": map *every* input to the *same* (or a very
low-dimensional) latent. Then prediction is perfect — you predicted a constant — and the representation carries no
information. It is the central failure mode of any non-contrastive joint-embedding method.

**How we measure it** (two metrics that can disagree, and the disagreement matters):
- **Effective rank** of the latent covariance — how many dimensions the representation actually uses. Healthy =
  high; collapsed → ~0–1 (in our nav runs it hit ~1e-7).
- **Frozen-encoder readout \\(R^2\\)** — how well a ridge probe decodes a target (geometry, or value/return-to-go)
  from the frozen latent. Collapsed = low \\(R^2\\).

These diverge: an anti-collapse term like *uniformity* can **maximize eff-rank yet lower readout-\\(R^2\\)** — it
spreads the latent out but destroys usable information. So eff-rank alone is a misleading collapse metric; readout
is what matters. (This is one of our sharper empirical findings.)

**How it's prevented — two routes:**
1. **Architectural asymmetry (BYOL-style):** a predictor on the online branch + an EMA/stop-grad target branch. The
   asymmetry makes the constant solution unstable, so the network avoids collapse *without any explicit term*. Our
   experiments showed a pure JEPA on broad DMControl data does **not** collapse for exactly this reason.
2. **Explicit anti-collapse regularizers (VICReg-style):** a **variance** term (keep each latent dim's std above a
   floor) + a **covariance** term (decorrelate dims). Uniformity and structural-entropy penalties are alternatives
   in the same slot.

**When does anti-collapse actually help?** Only when the latent *genuinely collapses* — which in our program
happened in exactly one regime: closed-loop, online, goal-conditioned navigation. There, anti-collapse restored
rank and lifted goal-decodability (point-maze 0.53→0.95). On broad data or value-anchored control latents (which the
BYOL asymmetry or a dense TD gradient already keep healthy), extra anti-collapse is neutral-to-harmful. *A fair test
of any structuring objective, including SE, therefore has to be run on tasks where collapse actually occurs.*

## 4. How is a JEPA turned into something that acts?

A JEPA is just an encoder. To *do* a task you add a head:
- **Perception:** freeze the encoder, train a linear probe (the standard eval).
- **Control:** add an **action-conditioned latent dynamics** model and **plan** in the latent — this is where JEPA
  meets world models.

## 5. What is H-JEPA, and how does hierarchical planning relate?

**H-JEPA = Hierarchical JEPA**: a stack of JEPAs at increasing time/abstraction scale. A low-level JEPA-1 predicts
short-horizon, detail-rich embeddings; a higher-level JEPA-2 predicts longer-horizon, coarser embeddings. The point
is **multiscale planning**: the high level proposes coarse subgoals over long horizons; the low level fills in the
fine actions to reach them. This is LeCun's blueprint for planning-based agents that don't use RL — they *infer*
actions by prediction, not maximize reward by trial and error.

**How we used it** (#51–#57): we built a faithful decoder-free H-JEPA (encoder + latent predictor + EMA target +
VICReg, 2-level with a high-level latent subgoal) and drove the high level two ways — a reactive selector and a
**latent-MPPI planner** — on PandaPickCube and nav. Result: a multi-seed **null** on Panda, because the bottleneck
was the low-level *motor primitive* (it never learned to reach a sub-1.2 cm target), not the abstraction; and the
nav *collapse* studies. The lesson matched the field's: JEPA-for-control needs a **competent low-level primitive**;
the hierarchy doesn't manufacture one.

## 6. How does this differ from Dreamer (and the generative world-model line)?

The **Dreamer** line — **PlaNet → DreamerV1/V2/V3** (Hafner et al.) — is the *generative* counterpart:
- **Reconstruction:** Dreamer's **RSSM** (Recurrent State-Space Model) is trained variationally to **reconstruct
  observations** (and rewards) — it predicts *pixels*, unlike JEPA's latent-only prediction.
- **Learning:** Dreamer learns an **actor-critic in imagination** — it rolls the latent dynamics forward and trains
  a policy + value entirely on *imagined* trajectories (RL in the model).
- **Latents:** DreamerV3 uses 32 categorical latents; strong across 150+ tasks with fixed hyperparameters.

So the axis is: **reconstruction (Dreamer) vs latent-only prediction (JEPA)**, and **RL-in-imagination (Dreamer) vs
planning/energy-inference (LeCun's JEPA vision)**. There is a middle ground for control — decoder-free latent
prediction + a value + planning — and **that is exactly what TD-MPC2 is**: its *consistency* loss is JEPA-like
(latent self-prediction, no decoder), but it adds a **TD value + policy** (the RL engine) and **MPPI planning**.
"Latent planning" methods (TD-MPC, PlaNet's CEM planner) sit on this spectrum too — plan in a learned latent, with
or without a value. Our whole program is, in effect, an audit of which piece of that spectrum is doing the work:
Dreamer's reconstruction, JEPA's latent prediction, or the TD value.

## 7. Energy: what LeCun means, and why planning = minimization

LeCun frames all of this as **energy-based models (EBMs)**. Define an **energy** \\(E(x,y)\\) that is *low* when
\\(y\\) is a compatible continuation of \\(x\\) and *high* otherwise. For a JEPA, the energy is precisely the
**latent prediction incompatibility**:

$$E(x,y) = \big\lVert \text{pred}(\text{enc}(x)) - \text{enc}(y)\big\rVert^2 .$$

Two consequences:
- **Training** shapes \\(E\\) so real \\((x,y)\\) pairs have low energy — and the collapse problem is exactly the
  degenerate way to make energy low everywhere (constant embedding), which is why EBMs need a
  contrastive-or-regularizer term to keep energy *high* elsewhere.
- **Planning = inference = energy minimization.** To act, search over an action/latent sequence that **minimizes
  the accumulated energy** to a goal (goal = a target embedding). Hierarchical planning minimizes energy at multiple
  scales. This is why LeCun's agents *plan* rather than *reward-maximize*: MPC/CEM/gradient-based inference over the
  learned energy replaces the policy-gradient of RL. (TD-MPC2's MPPI is a sampling-based energy/return minimizer of
  the same flavour, but scored by a learned *value*, not a pure prediction-energy.)

## 8. Information theory: from JEPA's implicit maximization to structural entropy

Anti-collapse is, at bottom, an **information** constraint: don't let the representation lose information (don't
collapse its entropy/rank). VICReg's variance-covariance terms are a hand-designed way to keep the latent's
information content high. This is the thread our SE work pulls on, but with a *different, graph-structural* notion of
information.

**Structural information / structural entropy** (Li & Pan). Ordinary Shannon entropy measures the information in a
distribution; **structural entropy** measures the information encoded in the **structure** of a graph — specifically,
the minimum number of bits to encode the destination of a one-step random walk, minimized over all hierarchical
partitions ("**encoding trees**"). Grouping tightly-connected vertices into communities *lowers* this cost; the
optimal encoding tree reveals the graph's natural community hierarchy. It has been used for community detection,
graph neural networks, and — recently — hierarchical RL.

**How this connects to JEPA.** Build a k-NN graph on the batch of latent vectors; its **structural entropy** is an
information measure of the latent's *community structure*. Minimizing 2-D structural entropy (our "glass"/SE
objective, via `selib`) organizes the latent into a compact community hierarchy — a *structural* information prior,
as opposed to VICReg's per-dimension variance prior. That is the genuinely novel angle we have not yet properly
tested: **use SE not as an anti-collapse penalty that competes with the predictor, but as the objective that
*defines* the latent's (and, in H-JEPA, the hierarchy's) structure** — an information-theoretic, graph-structural
route to the abstraction JEPA is reaching for. Whether it beats VICReg/uniformity is an open question, and — per our
redundancy results — one that must be tested on the goal-conditioned / collapse-prone / long-horizon tasks JEPA is
actually for, not on dense value-based control where every structure is redundant.

## Where our program sits, in one line

TD-MPC2 is a **decoder-free latent-prediction (JEPA-like) world model welded to a TD value and MPPI planning**; the
Dreamer line is its **generative, RL-in-imagination** cousin; LeCun's H-JEPA is the **planning-by-energy-minimization,
no-RL** ideal. Our findings say the *value* is the engine on dense control, so the honest place to look for a JEPA
or SE win is *outside* that regime — perception, goal-conditioned planning, long-horizon hierarchy — which is
exactly where the JEPA literature already lives.

---

*Sources:* [DINO-WM (arXiv 2411.04983)](https://arxiv.org/html/2411.04983v2) ·
[V-JEPA 2 (Meta AI)](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/) ·
[LeCun, Latent-Variable EBMs / path to autonomous intelligence (arXiv 2306.02572)](https://arxiv.org/abs/2306.02572) ·
[DreamerV3 (Hafner et al.)](https://arxiv.org/abs/2301.04104) ·
[Survey of Structural Entropy (IJCAI 2025)](https://www.ijcai.org/proceedings/2025/1183.pdf) ·
[Li & Pan, Structural Information & Dynamical Complexity of Networks (IEEE T-IT 2016)](https://dl.acm.org/doi/10.1109/TIT.2016.2555904).
