---
layout: post
title: "TD-MPC-Glass, Part 14: Jumpy World Models from Scratch — γ-Models, TD-Flow, and Compositional Planning (vs Dreamer & TD-MPC)"
date: 2026-06-18
description: "A from-scratch primer for newcomers on the line of work we're now reproducing: world models that don't predict one step at a time but JUMP straight to far-future states. We build up every term and equation — what a world model is, the one-step view (Dreamer, TD-MPC2) and its compounding-error problem, the discounted future-state (occupancy/successor) measure and the 'geometric horizon', γ-models / Geometric Horizon Models (Janner 2020), flow matching, Temporal-Difference Flows (TD-Flow, ICML 2025), and Compositional Planning with Jumpy World Models (CompPlan, ICLR-WS 2026). Then a side-by-side comparison with Dreamer and TD-MPC2: who predicts what, who plans how, and where each one wins."
---

> We just closed the redundancy chapter of this project (Part 13: the R² criterion couldn't be built),
> and the one technique that genuinely *won* was temporal abstraction — predicting the future in big
> jumps instead of small steps. So we're reproducing the 2025–2026 "jumpy world model" line. This post
> is the primer I wish I'd had: every term and equation from scratch, and how it differs from the world
> models you may already know (Dreamer, TD-MPC2). No prior background assumed beyond "an agent picks
> actions to maximize reward."

## 1. What is a world model, and why plan with one?

An agent lives in a Markov decision process: states $s$, actions $a$, transition $P(s' \mid s, a)$,
reward $r(s,a)$, discount $\gamma \in [0,1)$. The goal is a policy $\pi(a \mid s)$ maximizing the
expected discounted return $\mathbb{E}\big[\sum_{t} \gamma^t r(s_t,a_t)\big]$.

A **world model** is a learned simulator of the environment. Instead of only learning *what to do* (a
policy) or *how good a state is* (a value function), you learn *what happens next*. With a world model
you can **plan**: imagine candidate futures and pick actions whose imagined consequences look best.
This is the model-based promise — better sample efficiency and the ability to "think before acting."

The central design question — the one this whole post is about — is: **what exactly does the model
predict?**

## 2. Family 1: step-by-step world models (Dreamer, TD-MPC2)

The classic answer: predict the **next** state, one step at a time, then roll forward.

**Dreamer** learns a latent recurrent state-space model (RSSM). Encode observations into a latent
$z_t$, and learn a transition that predicts the next latent:

$$ z_{t+1} \sim p_\theta(z_{t+1} \mid z_t, a_t), \qquad \hat r_t = r_\theta(z_t),\qquad \hat o_t = \text{dec}_\theta(z_t). $$

To plan/learn, Dreamer **rolls the model out in imagination** — $z_t \to z_{t+1} \to z_{t+2} \to
\dots$ — and trains an actor-critic on these imagined trajectories.

**TD-MPC2** (the baseline this whole project chases) is similar in spirit but planning-first. It learns
a latent $z = \text{enc}(s)$, a latent dynamics $z' = d_\theta(z, a)$, a reward head, and a value head,
trained with a self-predictive **consistency** loss so the rolled-out latent matches the encoded future
latent. At decision time it does **MPPI** — sample action sequences, roll them through $d_\theta$ for a
short horizon $H$, score each by predicted reward plus a terminal value, and act:

$$ a_{0:H} \;\approx\; \arg\max_{a_{0:H}}\; \mathbb{E}\Big[\textstyle\sum_{h=0}^{H-1}\gamma^h \hat r(z_h,a_h) + \gamma^H \hat V(z_H)\Big]. $$

**The catch — compounding error.** Both predict one step at a time, so to reason about a state $k$
steps away you compose the model with itself $k$ times: $\hat P^k = \hat P \circ \hat P \circ \cdots$.
Each application adds error, and the errors **compound geometrically**. That's fine for short, reactive
control (where these methods shine) but it makes *long-horizon* reasoning — "how do I get across the
maze?" — brittle. You either take tiny confident steps or accept a blurry, drifting rollout.

## 3. The other idea: don't step — jump

What if, instead of "where am I after **one** step," the model answered "where am I likely to be
**eventually**"? Concretely, fix a policy $\pi$ and ask for the distribution over **future** states you
will visit, with nearer futures weighted more. That object is the **discounted state-occupancy measure**
(a.k.a. the **successor measure**):

$$ d^\pi_\gamma(s' \mid s) \;=\; (1-\gamma)\sum_{t=0}^{\infty} \gamma^{t}\, P\big(s_t = s' \,\big|\, s_0 = s,\ \pi\big). $$

Read it slowly: start at $s$, follow $\pi$, and $d^\pi_\gamma(\cdot\mid s)$ is the (normalized,
geometrically-discounted) distribution of *all* the states you'll touch. The $(1-\gamma)$ makes it a
proper probability distribution.

There's a beautiful equivalent view. Draw a random horizon from a **geometric distribution**,
$T \sim \mathrm{Geometric}(1-\gamma)$, so $P(T = t) = (1-\gamma)\gamma^{t}$, and then look at the state
$s_T$. Sampling $s' \sim d^\pi_\gamma(\cdot \mid s)$ is *exactly* "roll out $\pi$ and stop at a random
geometric time." Hence the name **Geometric Horizon Model (GHM)**: a generative model that, given $s$,
**samples a plausible discounted-future state in one shot** — no stepping. (Origin: Janner et al.,
"γ-models / Generative Temporal-Difference Learning," NeurIPS 2020.)

This is the "jumpy" part. A GHM jumps over time. To reason about the far future you **sample once**
instead of composing a one-step model $k$ times — so there's no geometric error compounding.

## 4. The self-consistency that makes it learnable: a Bellman equation for the future

How do you train a model of "all future states" without rolling out forever? The occupancy measure
satisfies its own **Bellman recursion** — the next state is either *this* step (weight $1-\gamma$) or
drawn from the occupancy of *where you land next* (weight $\gamma$):

$$ d^\pi_\gamma(\cdot \mid s) \;=\; (1-\gamma)\, P(\cdot \mid s, \pi) \;+\; \gamma\, \mathbb{E}_{s'\sim P(\cdot\mid s,\pi)}\big[\, d^\pi_\gamma(\cdot \mid s')\,\big]. $$

This is the exact analog of the value-function Bellman equation $V = r + \gamma P V$, but for
*distributions of future states* instead of *scalar returns*. It means we can learn the GHM with a
**temporal-difference (TD) bootstrap**: match a mixture of (a) the real next state and (b) the model's
own prediction at the next state (via a slowly-updated *target network*). That's the seed of TD-Flow.
But first we need a way to learn to *sample* from a distribution. Enter flow matching.

## 5. Flow matching in one section (from scratch)

Suppose you want to sample from a complicated distribution $q(x)$ (here: future states) but you only
have samples from it. **Flow matching** learns a *velocity field* that continuously transports simple
noise into data.

Pick noise $x_0 \sim \mathcal N(0, I)$ and a data point $x_1 \sim q$. Connect them with a straight line
indexed by $\tau \in [0,1]$:

$$ x_\tau = (1-\tau)\,x_0 + \tau\, x_1, \qquad \text{so the velocity along this path is } \frac{dx_\tau}{d\tau} = x_1 - x_0. $$

Train a network $v_\theta(x, \tau)$ to predict that velocity:

$$ \mathcal L_{\text{FM}} = \mathbb{E}_{\tau,\,x_0,\,x_1}\;\big\| v_\theta(x_\tau, \tau) - (x_1 - x_0)\big\|^2. $$

To **sample**, start from noise $x_0$ and integrate the learned ODE from $\tau=0$ to $\tau=1$:

$$ \frac{dx}{d\tau} = v_\theta(x, \tau), \qquad x(0) = x_0 \;\Rightarrow\; x(1)\ \text{is a sample from } q. $$

That's it: flow matching = "learn the arrows that push noise onto the data manifold, then follow the
arrows." (It's the continuous, simulation-free cousin of diffusion models, and it's what InFOM — the
open-source scaffold we build on — uses.)

## 6. TD-Flow (ICML 2025): flow matching meets the Bellman bootstrap

Now combine §4 and §5. We want a flow model that samples from the *occupancy measure* $d^\pi_\gamma$,
not from a fixed dataset. **Temporal-Difference Flows (TD-Flow)** trains the flow's target using the
occupancy Bellman recursion:

- with probability $1-\gamma$, the flow's "data point" $x_1$ is the **actual next state** $s'$;
- with probability $\gamma$, it is a **sample from the (target) flow at the next state** $s'$ —
  i.e. bootstrap: $x_1 \sim d^{\pi}_{\bar\theta}(\cdot \mid s')$ using a target network $\bar\theta$.

Plugging that bootstrapped target into the flow-matching loss gives a model of the full discounted
future from only one-step data — and the TD construction keeps the **variance low over long horizons**
(the headline TD-Flow result: stable learning of far-future distributions where naive Monte-Carlo
sampling of $T \sim \mathrm{Geom}$ would be hopelessly noisy). Authors: Farebrother, Pirotta,
Tirinzoni, Munos, Lazaric, Touati (arXiv 2503.09817).

So: **TD-Flow = how to train a good GHM.**

## 7. CompPlan (ICLR-WS 2026): planning by jumping over policy sequences

A GHM tells you *where you'll end up* under a policy. **Compositional Planning with Jumpy World Models**
(CompPlan, arXiv 2602.19634, same group) uses that to **plan**. Two extra ingredients turn a single
GHM into a planner:

1. **Horizon-conditioning.** Instead of one fixed $\gamma$, condition the flow on a chosen horizon so
   you can ask "where will I be in *roughly* $h$ steps," $G_h^\pi(\cdot \mid s)$ — short jumps for
   local moves, long jumps for big ones.
2. **Policy-conditioning.** Condition the GHM on *which* base policy $\pi_i$ is running, so one model
   predicts the outcomes of a whole *library* of pre-trained policies $\{\pi_1,\dots,\pi_m\}$.

Then planning is a search over **sequences of policies**. Given start $s$ and goal $g$:

$$ \text{find } (\pi_{i_1}, \pi_{i_2}, \dots)\ \text{such that } s \xrightarrow{\,G^{\pi_{i_1}}\,} \hat s_1 \xrightarrow{\,G^{\pi_{i_2}}\,} \hat s_2 \cdots \to \hat s_K \approx g, $$

by sampling candidate subgoals $\hat s_k$ from the GHM and scoring them by proximity to the goal. Each
arrow is a **jump** — one GHM sample skipping many primitive steps — so a $K$-policy plan reasons over
a horizon of *hundreds* of environment steps without ever composing a one-step model. On OGBench
(antmaze navigation, cube manipulation) this delivers the paper's headline ~**+200%** on long-horizon
tasks versus primitive-action planning.

So: **CompPlan = how to *plan* with a GHM** — the "jumpy world model."

## 8. Side-by-side: Dreamer vs TD-MPC2 vs CompPlan/GHM

| | **Dreamer** | **TD-MPC2** | **CompPlan / GHM** |
|---|---|---|---|
| What the model predicts | next latent $z_{t+1}$ | next latent $z_{t+1}$ | a **discounted-future state** $s' \sim d^\pi_\gamma$ |
| Time granularity | one step | one step | **jumpy** (random geometric horizon) |
| How it's trained | recon + KL, imagination | self-predictive consistency + reward + value | **flow matching + TD bootstrap** (TD-Flow) |
| Planning / acting | actor-critic in imagination | **MPPI** over short rollouts | **search over sequences of base policies** |
| Long-horizon behavior | compounding error over rollout | compounding error; short $H$ | **no step-composition** → strong long-horizon |
| Best at | pixels, general control | reactive continuous control, sample-eff. | long-horizon navigation / compositional goals |
| Weakness | blurry long rollouts | drifts past short $H$; reactive only | needs good base policies; less about fine reactive control |

The one-sentence contrast: **Dreamer and TD-MPC2 imagine the future one step at a time and pay for it
over long horizons; a GHM samples the far future directly, and CompPlan stitches such jumps across a
library of policies to plan over very long horizons.** They're complementary — jumpy models are not a
replacement for reactive control, they're a different tool for the long-horizon/compositional regime.
(This matches what *we* found independently in Part 12: our own jumpy k-step world model helped exactly
on the contact-structured, longer-horizon manipulation tasks, and was neutral-to-harmful on reactive
locomotion.)

## 9. Where our reproduction stands

There is **no public code** for TD-Flow or CompPlan, so we build on **InFOM** (Intention-conditioned
Flow Occupancy Models, ICLR 2026) — an open-source JAX implementation of flow-matching over the
occupancy measure on OGBench. It is training now on our boxes and reproduces sensible cube-single
success. The work in flight is the **GHM extension**: adding §7's horizon-conditioning,
policy-conditioning, and the plan-over-policies loop on top of InFOM to reach a CompPlan-style jumpy
planner. That build is the next post.

## 10. The benchmark: OGBench tasks, and how to read the numbers

All of this is evaluated on **OGBench**, a standard offline-RL / goal-reaching benchmark. Two task
families matter for the jumpy-model story:

- **cube-single** — *robot-arm manipulation.* A Franka-style arm must move **one cube** into a target
  configuration. Observation 28-D (arm + cube state), action 5-D (arm control), **short-to-medium
  horizon** (~100–200 steps). `cube-double/triple/quadruple` add more cubes (harder, longer). `play` =
  the offline dataset was collected by a scripted "play" policy; `singletask-taskN` = a fixed goal.
- **antmaze-medium** — *quadruped navigation.* A 4-legged "ant" walks through a **maze** to a goal
  location. Observation 29-D, action 8-D (leg torques), **long horizon** (hundreds of steps, many
  turns). This is the regime jumpy/compositional planning is built for — and where CompPlan's headline
  ~+200% gains mostly come from.

**How to read an eval** (each eval runs ~50 episodes):

- **success rate** — `episode.success` $\in [0,1]$, the **fraction of episodes that reached the goal**.
  This is *the* headline metric (what OGBench and CompPlan's Table 1 report). `0.60` = solved 60% of
  the test episodes.
- **return** — cumulative reward per episode. These tasks are sparse/negative: reward $\approx -1$ each
  step until the goal, then the episode ends. So $\text{return}\approx -(\text{steps taken})$ — closer
  to $0$ means *faster*, and e.g. $\text{return}=-200,\ \text{length}=200$ means it hit the 200-step
  cap and **never reached the goal** (a failure).
- **length** — average steps until success or the 200-step timeout.

A practical caveat we keep hitting: report **peak** success, not final. The online-finetuning phase is
unstable at the very end, so final-step success collapses for the InFOM baseline and our GHM alike;
the peak over training is the meaningful number.

**Reproduction status (honest):** we have run **cube-single only** — enough to *validate the pipeline*
(the horizon-conditioned GHM trains end-to-end and lands cube-single peak success inside the InFOM
range). We have **not run antmaze yet**: InFOM's training script needs a locally-generated `-ft-`
fine-tuning dataset, and its generator covers cube/scene/puzzle but **not antmaze**. Since antmaze is
the *long-horizon* test that actually probes the jumpy hypothesis, unblocking that dataset path is the
priority before reading too much into short-horizon cube numbers.

### Pointers (verify before citing)
- γ-models / GHM: Janner et al., NeurIPS 2020, arXiv 2010.14496.
- TD-Flow: Farebrother et al., ICML 2025, arXiv 2503.09817.
- CompPlan: "Compositional Planning with Jumpy World Models," ICLR-WS 2026, arXiv 2602.19634.
- InFOM (our scaffold): arXiv / github.com/chongyi-zheng/infom. OGBench: github.com/seohongpark/ogbench.
- Dreamer (Hafner et al.) and TD-MPC2 (Hansen et al., ICLR 2024, arXiv 2310.16828) for the step-by-step family.
