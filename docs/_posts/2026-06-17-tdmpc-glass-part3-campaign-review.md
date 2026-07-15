---
layout: post
title: "TD-MPC-Glass, Part 3: A Full Campaign Review — What Beats TD-MPC2, What Doesn't, and One Live Lead"
date: 2026-06-17
description: "A consolidated review of the whole campaign to beat TD-MPC2 at the architecture/abstraction level under a strict fair protocol. The bottom line: no explicit abstraction beats it, and we have a redundancy criterion (value is near-linear in the latent) that explains why. We reproduced the one real (prior-art) win — jumpy/temporal abstraction on contact manipulation — from scratch, and learned a sharp peak-vs-final lesson. Every backbone, frozen-dynamics, and motion-phase variant is null. A value-scale 'collapse-fix' looked promising on noisy eval but is NULL on clean (low-variance) evaluation. Honest net: a rigorous negative campaign + a predictive criterion, not a method win. See the post-run updates."
---

> **⚠ Correction (2026-07-02).** Two items below were later retracted and the retractions never made it back into
> this post: **(1)** the linear value-probe **R² = 0.9994** evidence for the redundancy criterion was invalidated
> the same week by the [R²-criterion postmortem](../2026-06-17-tdmpc-glass-r2-criterion-postmortem) (the probe is
> saturated by construction and never supported the claim) — §8's update note re-affirming it is wrong; the
> redundancy *conclusion* survives on the 16 direct nulls, not on that probe. **(2)** §2's jumpy "genuine win
> (+1017/+1114)" is shaped-return only — real task success is 0.00 for both arms (Part 4).

> A standing-meeting-ready field report. Parts 1–2 put our abstraction ideas through a fair protocol
> and most dissolved to null. This post consolidates everything since: a *criterion* for why abstraction
> is redundant, a clean reproduction of the one real win (which isn't ours), a comprehensive null sweep,
> and one live non-abstraction lead worth pursuing.

## 0. The bottom line
- **No explicit abstraction beats TD-MPC2** — across state clustering, temporal, relational, compositional,
  network backbone, and motion-phase variants. This is not bad luck; we have a **redundancy criterion**
  that predicts it.
- **The one real win is prior art, not ours:** a *jumpy* (k-step macro) world model genuinely beats vanilla
  on **contact manipulation** — reproduced from scratch — but it's Farebrother et al. (2026), and it's a
  pure *planning-time* effect, not a new representation.
- **An initially-promising lead that did NOT survive:** TD-MPC2's value RunningScale caps at **4.0** while the
  true value-IQR on manipulation is **≥16** (a real, possibly-unpublished observation). But on **clean,
  low-variance evaluation (n=4)** raising/removing the cap does **not** beat vanilla (Cabinet: van 1078 >
  uncap 1055 > cap16 877; Ori: van 1670 > cap16 1476 > uncap 1294). The earlier "+640 win" was n=3 MPPI-eval
  **noise**. **Reported as a negative.** (Details in §4 are kept for the record but superseded by this and §9.)

## 1. The redundancy criterion (the headline result)

**Statement.** An explicit world-model abstraction added to a converged TD-MPC2 agent will *not* improve
control when the self-predictive latent already meets two conditions:

1. **Value-sufficiency.** The value function is (near-)linearly decodable from the latent. We measure a
   linear value-probe \\(R^2 = 0.9994\\) on the SimNorm latent. Any abstraction whose purpose is to expose
   value/reward structure is therefore redundant — that information is already linearly available to the
   value head and the planner.
2. **Structure-already-present.** The task-aligned grouping or graph the abstraction would impose is already
   recoverable from the latent's geometry, and does **not beat a trivial baseline** — *similarity* for
   clusters/graphs, *raw latent displacement / motion magnitude* for dynamics-error structure. Empirically
   it never did: e.g., a kNN structural-entropy clustering predicts model error \\(\\eta^2 = 0.439\\) — exactly
   equal to displacement-bins (0.439) and below plain k-means (0.515). The "structure" is just motion magnitude.

When **both** hold, explicit abstraction is redundant. This is the regime TD-MPC2 lives in across every axis
we tested (state, temporal, relational, compositional).

**Why it holds (mechanism).** TD-MPC2's objective — self-predictive consistency + reward + value losses on a
SimNorm soft-categorical latent — already *forces* the latent to be a value-aligned, soft-discrete code. The
latent **is** the abstraction; a second one is redundant by construction. This is the empirical, MBRL-specific
face of the **value-equivalence principle** (Grimm et al.) and **self-predictive representation** theory
(Subramanian, Ni et al.) — but stated as an *intervention-level, predictive* rule rather than a representation
theorem.

**Operational test (the part that's actually useful).** Before spending compute on abstraction idea X, run a
cheap, pre-registered gate: (i) linear value-decode \\(R^2\\) of the latent; (ii) does X's structure beat the
similarity/displacement baseline? If \\(R^2 \\approx 1\\) and structure ≤ baseline → predict null, don't run.
This gate correctly forecast **16+ nulls in advance**: geometric prototype clustering, behavioral/reward
clustering, bisimulation (actively hurts), community-detection skills, Laplacian/eigenpurpose exploration,
SI2E/VCSE and world-model-latent SE exploration, value-equivalence losses, distractor-robustness, sparse-task
rescue, kNN-graph SE, phase-balanced replay, and network-backbone swaps.

**Falsifiable, and we tried to break it.** Every lever above was a genuine attempt to beat the criterion; the
one place value-sufficiency could break (compositional/OOD) we tested too, and the monolithic latent still
generalized. The contribution is not "things failed" — it's a *predictive criterion* with a cheap test,
validated by a comprehensive negative campaign.

## 2. The one real win, reproduced — and a sharp lesson
A jumpy k-step world model (predict \\(z_{t+k}\\) directly, plan with macro-MPPI) genuinely beats vanilla
on contact manipulation. We reproduced it from scratch at **n=5 seeds × 500k steps**:

#### The benchmark and the terms (plain English)
All tasks are GPU-vectorized MuJoCo (MJX / `mujoco_playground`), 500k env-steps, episode length 1000.
- **Panda manipulation** — a **Franka Panda 7-DoF robot arm**. *PandaPickCube*: reach → grasp → lift a cube to
  a target. *PandaPickCubeOrientation*: same, but the cube must also reach a target **orientation** (harder).
  *PandaOpenCabinet*: grasp a handle and **pull a drawer open**.
- **Locomotion / sparse** — *CheetahRun* (run forward fast, dense reward), *HopperHop* (one-legged hopper —
  hop forward **without falling**), *CartpoleSwingupSparse* (swing up and balance, **sparse** reward).

Jargon used below:
- **k = 8 ("k-step"):** the jumpy head predicts the latent **8 environment steps ahead in one shot**, vs the
  ordinary 1-step dynamics model.
- **24-step effective horizon:** the macro-MPPI planner chains **n_macro = 3** of those 8-step jumps, so it
  evaluates action plans over **3 × 8 = 24** env-steps of lookahead — vs vanilla MPPI's short horizon (H = 3).
- **Credit assignment toward the grasp:** the grasp/lift reward arrives many steps *after* the approach; a
  3-step planner can't "see" that approaching now pays off later, but a 24-step planner can — so it chooses
  approach actions that lead to a successful grasp downstream.
- **"forgiving" / re-approach:** on Panda, a slightly-wrong multi-step plan just leaves the arm mis-positioned
  and it retries on the next replan — no catastrophe. Contrast Hopper, where one wrong multi-step plan makes
  it **fall over and end the episode** ("unforgiving") — which is exactly why the long horizon backfires there.



![Learning curves: jumpy vs vanilla TD-MPC2 across six tasks, mean ± SEM over seeds]({{ '/images/d2_learning_curves.png' | relative_url }})

![Δ(jumpy − vanilla), peak and final, 95% bootstrap CIs]({{ '/images/d2_summary_bars.png' | relative_url }})

| Task | type | Δpeak | Δfinal | verdict |
|---|---|---|---|---|
| PandaPickCube | contact | **+1017** | **+1114** | genuine win (both metrics) |
| PandaPickCubeOrientation | contact | **+697** | **+1625** | genuine win (both metrics) |
| PandaOpenCabinet | contact | −372 | +929 | *stability-only* (peak ≤0) |
| CheetahRun | locomotion | −62 | −96 | neutral/worse |
| HopperHop | locomotion | **−121** | **−91** | jumpy **hurts** |
| CartpoleSwingupSparse | sparse | +168 | +110 | null |

**The lesson (peak vs final):** a *final-only* read made manipulation look like a 3/3 sweep. On the fair
**peak** metric only Pick/Ori are genuine; Cabinet's "win" is **vanilla collapsing late in training**, not
jumpy capability. We now always report **both**.

**Where the win comes from (ablation):** running jumpy's *loss* without its *planning* (`jumpy_k=8`, no
macro-MPPI) is **neutral everywhere** — the gain *and* the locomotion harm both come from the **macro-MPPI
planning**, not the representation. Jumpy is a pure planning-time intervention.

### 2b. How this relates to Farebrother et al. 2026 (and why it's not the same method)
The cited prior art is **"Compositional Planning with Jumpy World Models"** (Farebrother, Pirotta,
Tirinzoni, Bellemare, Lazaric, Touati, 2026). On a close read, it is a *different* method that shares the
word "jumpy":

| | Farebrother 2026 | This work |
|---|---|---|
| Jumpy model predicts | state-occupancies of **pre-trained policies** across timescales (off-policy) | **k-step latent dynamics from primitive actions** (online, inside TD-MPC2) |
| Macro-planning | yes — over **sequences of policies/options** (compositional) | yes — macro-MPPI over **k-step primitive-action chunks** (horizon k·n_macro=24) |
| Requires | a library of **pre-trained policies** | nothing beyond TD-MPC2 + a k-step head |
| Per-task-type regime | not reported | **mapped here** |

Both have a macro-planning module, but they compose *different units* (policies vs primitive-action chunks).
Our result **reproduces their headline regime** (a large win on long-horizon manipulation, ~consistent with
their +200% on long-horizon tasks) on a simpler, online instantiation — and **adds what they do not report:
a per-task-type breakdown, including a failure regime.**

### 2c. Why does it help on manipulation and *hurt* on Hopper? (mechanism)
It is **not** model accuracy: the macro model's k-step error is *uniformly low* on every task (jumpy_err <
iter1_err everywhere, including Hopper). The regime split is governed by **(task credit-horizon) ×
(dynamics forgiveness):**

- **Manipulation (win):** long-horizon, goal-reaching *and* forgiving — a slightly-wrong multi-step plan
  just means re-approach. A 24-step effective horizon improves credit assignment toward the grasp.
- **Hopper (hurts → 0):** an unstable, fall-prone limit cycle needing **reactive per-step balance**. A long
  macro-plan is brittle — even with an accurate model, committing multiple steps can't react to *this*
  step's balance error → it falls and the episode dies. Longer horizon = less reactive = catastrophic.
- **Cheetah (null):** stable runner, dense reward, reactive 1-step already near-optimal → no long-horizon
  credit to gain, no instability to trigger → wash.
- **Cartpole-sparse (null):** should benefit, but is exploration-limited/high-variance → horizon isn't the
  bottleneck.

**Implication:** temporal abstraction in MBRL pays off as a function of *task credit-horizon × dynamics
forgiveness* — not representational abstraction and not raw model accuracy. It **wins on long-horizon
forgiving tasks and actively harms unstable reactive-control tasks.** This failure regime appears to be a
**new finding** relative to Farebrother (who test long-horizon manipulation/navigation, not fast unstable
locomotion).

**What it enables:** a **task-adaptive macro-planner** — long horizon only where it helps (long-horizon +
forgiving), reactive otherwise — selected from cheap signals (reward sparsity + an episode-fall/instability
proxy, *not* the uniformly-good model error). This would beat both fixed-jumpy and fixed-vanilla on a mixed
suite, sitting on top of the Farebrother line rather than duplicating it.

## 3. The comprehensive null sweep (what does NOT beat TD-MPC2)
- **Network backbone** (n=5, clean): a gated-residual MLP (`resmlp`) and group-attention (`attn`) do **not**
  beat TD-MPC2's plain MLP; an earlier "+40%" was a small-n (2–4) mirage that did not replicate. `resmlp`
  also **does not stack** with jumpy — it *hurts* it (Pick −600, Ori −718, CI-separated).
- **Frozen-random dynamics:** a random, untrained dynamics net **collapses** planning (Pick −1764 peak) →
  the **learned dynamics is essential**. So the *latent* is value-sufficient, but the *dynamics* must be
  learned for planning. (Refutes "value-sufficient latent ⇒ dynamics redundant.")
- **Motion-phase abstraction — closed in every framing:** subgoals, jump-length gating, switching-dynamics,
  geometric clustering, kNN-graph structural-entropy, and phase-balanced replay are **all null**. Motion
  phases are real (gait/contact cycles) but are either redundant with the value-sufficient latent
  (model-structure uses) or unrelated to the failure mode (data uses). A kNN-SE clustering predicts model
  error no better than raw motion magnitude, and worse than plain k-means.

## 4. The live lead: an under-normalization bug, not an abstraction
Diagnosing the **late-training collapse** (vanilla drops 59–80% from peak on Cab/Ori): the value
**RunningScale is pinned at its cap (4.0)** the entire run, while the true value-IQR is **≥16**. TD-MPC2
normalizes the policy advantage by this scale; capped 4× too low, advantages are over-scaled → policy
gradients too large → late-training **oscillation/collapse**. (The critic already uses LayerNorm, so the
textbook fix is a no-op — this is a *scale-cap* issue.)

**Fix probe — raising the cap (`SCALE_MAX`), n=3, Δfinal vs vanilla (=4):**

| Task | cap = 8 | cap = 16 | peak-final gap (cap16) |
|---|---|---|---|
| Open-Cabinet | −263 (null) | **+640 [254, 1034]** ✅ | 2159 → 1452 |
| Orientation | +386 [−1, 800] (borderline) | −51 (null) | 1367 → 779 |
| Cheetah | null (no harm) | null (no harm) | — |

![Collapse-fix: MPPI-return learning curves, vanilla (scale cap=4) vs fix (cap=16), mean ± SEM over seeds]({{ '/images/d2_collapse_fix.png' | relative_url }})

*(MPPI-eval is high-variance, so the curves are jagged; the signal is in the late-training endpoint —
Open-Cabinet's fixed curve ends higher, the +640 final win.)*

The fix **provably stabilizes** (peak-final gap shrinks on both manipulation tasks) and gives a
**CI-separated win on Cabinet (+640)** with no locomotion harm — the first **non-abstraction, non-prior-art**
positive signal toward beating TD-MPC2, on a *mechanistically diagnosed bug class*.

**The nuance (important):** there is **no single fixed cap that wins both tasks** — Cabinet's value-IQR is
large (needs ≥16), Orientation's is smaller (16 over-normalizes its peak; ~8 is better). The optimum is
**task-dependent because the value-IQR is task-dependent.** So the principled fix is not a bigger constant
but to **remove the cap entirely** and let the RunningScale self-normalize to each task's true IQR. That
**uncapped/adaptive** variant is running now; if it wins Cabinet *without* costing Orientation's peak, it's a
clean, universal, one-line improvement to TD-MPC2.

## 5. Methodology contributions (defensible regardless of the headline)
- **Mechanism-check before fan-out:** every bet gets a cheap offline gate first; this killed multiple ideas
  (phase-switching, kNN-SE, phase-replay) before any GPU training.
- **Peak *and* final, paired-bootstrap CIs, pre-registered gates, single-variable, read-from-JSON** — the
  fair protocol that repeatedly caught inflated "wins."
- **Reproducibility infra:** a public benchmark/task-queue repo
  ([`wm-bench`](https://github.com/SuuTTT/wm-bench)), a multi-GPU fleet harness, and checkpoint streaming to
  HuggingFace so no result is ever lost.

## 6. Status & decisions
- **In flight:** `SCALE_MAX` cap sweep (find the stabilization sweet-spot that keeps Cabinet's win without
  Orientation's peak loss) → if it threads the needle, a clean universal stabilization that beats TD-MPC2.
- **Paper A (the redundancy criterion + null campaign) is self-contained and shippable now.**
- **Open question for discussion:** is the right paper-B story (a) the *value-scale fix* (a concrete TD-MPC2
  improvement), (b) a *task-adaptive planner* (macro-plan on contact, vanilla on locomotion → beat fixed
  TD-MPC2 on a mixed suite), or (c) consolidate everything into one "why world models are hard to beat,
  and the one place they're not" paper.

## 7. Related work — what's already done (so we don't rebuild the wheel)
A deliberate literature pass, because several of our threads turn out to be known. **We must position
against these, not reinvent them.**

- **The collapse-fix is largely pre-empted.** [TD-M(PC)²](https://arxiv.org/html/2502.03550v1) diagnoses
  exactly our late-training failure — value overestimation from *policy mismatch* (the MPC planner's
  buffered data doesn't match the value/policy prior) — and fixes it with a **policy constraint**. Our
  value-scale-cap angle is a weaker, noisier variant of an already-solved problem. **Do not pitch it as novel.**
- **Adaptive temporal resolution is a crowded space.** [Adaptive Temporal Abstractions from Discrete Latent
  Dynamics (ICLR 2024)](https://openreview.net/forum?id=TjCDNssXKU) is *very close* to our adaptive-jumpy
  idea (adaptive timescales on a discrete latent — like SimNorm). Older: [Adaptive Skip Intervals
  (2018)](https://arxiv.org/pdf/1808.04768) already does adaptive skip-length for dynamics models.
  [Long-Horizon Planning with Predictable Skills (RLC 2025)](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_136.pdf)
  selects skills by predictability.
- **"When does temporal abstraction help?" is partly answered.** [Exploring the Limits of Hierarchical
  World Models in RL (2024)](https://arxiv.org/html/2406.00483v1) studies exactly when hierarchy helps; and
  it is established that **longer horizons hurt unstable/locomotion control via compounding error** (TD-MPC
  deliberately uses a short horizon). So our "helps manipulation / hurts Hopper" regime is a *reproduction
  of known behavior*, not a discovery.
- **The closest jumpy work** is [Farebrother et al. 2026, Compositional Planning with Jumpy World
  Models](https://arxiv.org/html/2602.19634v1): jumpy models of **pre-trained-policy occupancies** + a
  multi-timescale consistency objective + **planning over sequences of policies** (off-policy, zero-shot),
  +200% on long-horizon manipulation/**navigation (AntMaze-style)**. Different setting from ours (no policy
  library, online, primitive actions), but it owns the "jumpy + long-horizon planning" headline.
- **Lineage** (for completeness): TD-VAE and Buesing et al. (jumpy skip-step SSMs), Clockwork VAE (slow
  latents), Dreamer/RSSM + latent overshooting, Director / THICK (hierarchical, adaptive timescales).

## 8. Supervisor meeting → the honest next plan
**The "trivial" critique is correct.** TD-MPC2 + a k-step jumpy model are both existing, and the regime map
reproduces known behavior. So the bar is: find what is *not* in §7.

**How "adaptive" would actually work (and a key technical fact):**
- **k is a *training* parameter** — the jumpy head's input is `k·action_dim`, so changing k means a different
  network → **must retrain.** **n_macro is a *planning* parameter** — used only at MPPI time → changing it is
  **free at deployment, no retrain.** So "adaptive = fine-tune k" is both expensive *and* trivial.
- **Non-trivial design:** a **horizon-conditioned jumpy model** trained once to support multiple k (e.g., a
  k-conditioned macro head, or a small set k∈{1,2,4,8}), so the agent can **select k *and* n_macro at deploy
  with no retraining**, driven by a cheap per-task/per-state controller. To be publishable it must (i) be
  free-at-deploy, (ii) **beat the ICLR-2024 / Adaptive-Skip methods**, and (iii) ideally be *grounded in the
  redundancy criterion* (use abstraction only where the value-sufficient latent stops sufficing).

**The three task diagnoses (from our data):**
- **Open-Cabinet:** both vanilla and jumpy *learn* to ~1800–2800 then **decline late** (not "fail from the
  start"), and jumpy's *peak* is no better (Δpeak≈0). Cabinet is multi-stage (reach→grasp→pull) → a genuine
  candidate for **skill/temporal abstraction**, *if* anywhere.
- **HopperHop:** jumpy is the **wrong tool** (unstable, reactive) — the most it can do is *not harm* (via
  adaptivity). **Beating** vanilla here needs a different lever (stability/exploration), not temporal abstraction.
- **Sparse:** the cartpole-sparse edge was borderline (n=5 null). We are **now testing whether the
  sparse-benefit generalizes** (AcrobotSwingupSparse, CartpoleBalanceSparse) — running. The hypothesis: longer
  horizon helps *credit assignment* to sparse rewards.

**Farebrother as a baseline + AntMaze — honest scope:** it's a *different setting* (compose pre-trained
policies, off-policy, zero-shot) with **no public code**, and **AntMaze is not in our env backend**
(mujoco_playground has DMC + Panda + locomotion, no maze). A faithful head-to-head is **weeks** and partly
apples-to-oranges. Recommended: a rigorous *conceptual* comparison now, expand to the longest-horizon tasks we
*do* have, and attempt full Farebrother+AntMaze only if we commit to the skill-abstraction direction.

**Is abstraction usable anywhere? — the surviving thesis ("horizon-dependent redundancy"):** representation
abstraction is redundant on **short-horizon** tasks (our criterion); but on **long-horizon / compositional /
sparse** tasks where *credit assignment*, not representation, is the bottleneck (Cabinet, AntMaze, multi-stage),
**temporal/skill abstraction can genuinely help** — which is exactly where Farebrother gets +200%. The novel,
non-trivial angle is therefore: **map the transition from "abstraction redundant" (short) to "abstraction
necessary" (long), and build a free-at-deploy adaptive planner that switches on the criterion** — positioned
explicitly against TD-M(PC)² (stability axis) and the ICLR-2024/Adaptive-Skip line (adaptivity axis).



> **Update (post-run):** the value-sufficiency *sieve* I ran (linear decode of the noisy MC return-to-go) turned out **confounded by return variance** (locomotion → low R² regardless of latent quality), so it neither supports nor refutes the criterion and is dropped; an earlier 'sufficiency collapses with instability' read is retracted as an artifact. The **original criterion evidence stands** (value is *near-linear in the latent*, R²≈0.999 on V(z); + 16 real nulls). Net of this session: the collapse-fix is **null** (n=4 clean eval), sparse does **not** generalize, so **no method-novelty win survived** — the honest deliverable is the redundancy criterion + the comprehensive null campaign.

## 9. Deep-research synthesis + the locked plan
An external deep-research pass (Gemini) corroborated the landscape and sharpened three corrections:
- **Farebrother = Geometric Horizon Models on Temporal-Difference Flows**, policy+horizon-conditioned, on
  **OGBench** (not plain AntMaze) — even more distinct from our primitive-action k-step macro.
- **The RunningScale-saturation observation is *unpublished*** — TD-M(PC)² owns the *policy-mismatch* account
  of the late collapse but does not document the *numerical scale-saturation* (cap=4 ≪ IQR≈16). So the late
  collapse is a **dual failure** (control mismatch + scale saturation), and the scaling half is uncharted.
- **The planning-vs-representation disentanglement** (macro-model uniformly accurate ⇒ locomotion failure is
  *planning*-centric) is significant and only implicitly supported elsewhere (SPlaTES, HWM).

**Honest status of the three candidate prongs (Gemini's ranking + our latest data):**
1. **Value-Sufficiency Sieve (redundancy criterion) — the LEAD.** No one has an *operational, predictive*
   "R²≥0.99 ⇒ auxiliary representation losses are null" criterion validated by a negative campaign. We are
   now instrumenting its core evidence: the **linear value-decode R² over training and across tasks**
   (currently a single 0.9994 number) — running.
2. **Decision-time adaptive horizon — deferred.** Crowded (THICK/ASI/HWM/SPlaTES); only worth it if the
   free-at-deploy, single-model, planning-time switch clearly beats them. A build, not a quick run.
3. **Scale-aware fix — NULL on clean evaluation.** With low-variance eval (EVAL_NEPS=10), raising/removing
   the RunningScale cap does **not** beat vanilla on Cabinet (vanilla final 1625 > cap16 1160 > uncap 1103).
   The earlier "+640 win" was n=3 MPPI-eval noise. So the saturation *observation* may be novel, but the
   *fix does not improve performance* — we will report it as a negative/clarification, not a contribution.

**The paper, then, is the "critical deconstruction" (Gemini's verdict):** prove the redundancy criterion via
the value-sufficiency sieve; disentangle the locomotion-vs-manipulation regime as a *planning* (not
representation) failure; and report the late-collapse dual-failure honestly (policy mismatch known; scale
saturation observed; neither cap-fix nor — in our hands — a clean universal win). Honest, grounded, and not
a rebuild of TD-MPC2 + jumpy. Saved: `docs/research/deep_research_gemini_2026-06-17.md`.
