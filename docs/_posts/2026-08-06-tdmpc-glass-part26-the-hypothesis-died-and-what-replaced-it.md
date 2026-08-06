---
layout: post
title: "TD-MPC-Glass, Part 26: The Hypothesis Died, the Ceiling Objection Was Right, and What to Build Next"
date: 2026-08-06
description: "The reward-conjunctivity hypothesis from Part 25 is dead — killed by the difficulty control written into its own pre-registration. This post explains exactly what the control changed and why it is fatal, then takes seriously the objection that any improvement must shrink as its baseline rises: a headroom analysis shows planning captures a near-constant ~41% of whatever the policy leaves, which makes our headline finding closer to arithmetic than to science. Ends with the design that escapes the confound (toggle search on a frozen checkpoint) and a concrete proposal for deliberate, budget-aware planning — an agent that decides when to think."
---

> Part 25 ended by nominating one hypothesis as the ICLR bet: *planning's value is set by reward
> conjunctivity.* Within 24 hours it was dead, killed by a control we had written into its own
> pre-registration before any data existed. This post is the autopsy, an honest reckoning with the
> objection that our replacement finding might be arithmetic, and the design that gets us out.

---

## 1. Background: what "the planner" is, and what we are measuring

TD-MPC2 has a world model, and at every timestep it can choose an action in one of two ways:

- **plan** — sample 512 candidate action sequences, roll each one forward through the learned model,
  score them, refine, and take the first action of the winner (MPPI, horizon 3, 6 iterations)
- **prior** — just ask the policy network `π(z)` for an action

One config flag, `mpc`, selects between them. Everything else — the model, the loss, the replay
buffer, the network sizes — is identical. So `mpc=true` vs `mpc=false` isolates **decision-time
search**, and the ratio between them is what we have been calling the **planning gain**.

Measured across six tasks, planning is load-bearing on exactly one:

| task | plan | no plan | gain |
|---|---|---|---|
| **hopper-hop** | 364.2 (n=5) | 103.3 (n=5) | **3.53×**, p=0.008 |
| walker-run | 807.4 | 705.4 | 1.14× |
| acrobot-swingup | 417.2 | 305.0 | 1.37× |
| cheetah / cup / finger | — | — | ~1.00× |

The planner also costs about **35% of wall-clock**. So "when is it worth paying for?" is a real
question with a real price attached.

---

## 2. The hypothesis: conjunctive rewards

The idea came from our supervisor's observation that hopper-hop's 4× is the surprising number, and
from asking *why that task*. The answer looked like it was in the reward function:

```python
standing = rewards.tolerance(physics.height(), (_STAND_HEIGHT, 2))   # NO margin
hopping  = rewards.tolerance(physics.speed(), bounds=(_HOP_SPEED, inf),
                             margin=_HOP_SPEED/2, value_at_margin=0.5, sigmoid='linear')
return standing * hopping
```

`rewards.tolerance` with no `margin` argument is a **binary gate**: it returns exactly 1 inside the
band and exactly 0 outside. Multiply that into a product and you get a reward that is **identically
zero whenever the hopper is not upright**, no matter how fast it is moving.

For a gradient-based learner this is poison. Over a large region of state space the reward is 0 and
so is its gradient — there is nothing to climb. The agent has to *stumble* into satisfying both
conditions at once before it gets any signal at all. A planner does not stumble: it rolls candidates
through the model and can **search** for a sequence that holds height in band while building speed.

We scanned the whole dm_control suite. hopper-hop is the **only** multiplicative reward containing a
hard binary factor. It is also the only task with a big planning gain. That is a satisfying story.

It is also exactly the shape of claim that has died six times in this campaign: one positive case,
noticed *after* seeing which task had the effect.

---

## 3. Making it causal: conjunctivity as a knob

The fix for a post-hoc pattern is not more tasks — there are no other hard-gated ones — it is to
**manipulate the cause**. We patched dm_control so conjunctivity became a dial, additive and inert
unless an environment variable is set:

- **Direction A** — soften hopper's gate: give `standing` a margin so reward decays smoothly instead
  of snapping to zero. *Predicted:* gain falls.
- **Direction B** — harden walker's gate: remove `standing`'s margin and remove the `1/6` floor on
  the move term, making it a true conjunction. *Predicted:* gain rises.

Both moved as predicted:

| condition | plan | never | gain |
|---|---|---|---|
| hopper stock (hard gate) | 364.2 (n=5) | 103.3 (n=5) | 3.53× |
| hopper **softened** | 450.5 (n=3) | 315.6 (n=3) | **1.43×** ↓ |
| walker stock (smooth) | 807.4 (n=4) | 705.4 (n=4) | 1.14× |
| walker **hardened** | 804.9 (n=3) | 457.7 (n=3) | **1.76×** ↑ |

Two directions is much better than one, because "you just made the task harder" cannot explain a
gain that moves *both* up and down. At this point the hypothesis looked strong.

---

## 4. The control that killed it

Here is the thing the two directions still could not rule out. **Softening a gate also hands out
partial credit.** A soft gate gives you something for being nearly upright; a hard one gives you
nothing. So Direction A changed two things at once: how *conjunctive* the reward is, and how *hard*
it is to get any reward at all.

The control separates them. Instead of changing the gate's **shape**, change the **bar**:

```python
standing = rewards.tolerance(physics.height(), (0.35, 2))   # still NO margin
```

Read that carefully. It is still a binary gate — still 1-or-0, still multiplied into a product,
still exactly zero reward whenever the hopper is down. **Conjunctivity is untouched.** The only
change is that the height you must clear dropped from 0.60 to 0.35.

If conjunctivity is what produces the planning advantage, this variant must keep it. The
pre-registration said so explicitly, months before the data: *the control must not move the gain.*

| hopper variant | plan | never | gain | difference | perm p |
|---|---|---|---|---|---|
| stock (hard gate @0.60) | 364.2 (n=5) | 103.3 (n=5) | **3.53×** | +261.0 | **0.008** |
| softened (margin) | 450.5 (n=3) | 315.6 (n=3) | 1.43× | +134.9 | 0.200 |
| **control (hard gate @0.35)** | 522.2 (n=4) | 432.1 (n=4) | **1.21×** | +90.2 | 0.657 |

The gain collapsed anyway, from 3.53× to 1.21×.

**That is fatal, and it is fatal in the cleanest possible way.** We removed nearly the whole effect
*without changing conjunctivity at all*. Whatever produced the 3.53×, it was not the conjunctive
structure of the reward — because the conjunctive structure is still fully present in the control.
And it retro-explains Direction A: softening the gate shrank the gain because it made reward easier
to obtain, not because it made it less conjunctive.

One honesty note. At n=2 this cell rested entirely on a single seed: the no-planning arm read 147.7
and 848.4, and deleting the 848 gave 3.31×, leaving the hypothesis untouched. We refused to score it
and ran more seeds. At n=4, deleting that seed still gives 1.78× — well below 3.53×. The verdict no
longer depends on which seeds you keep.

---

## 5. The objection that matters: is the replacement finding just arithmetic?

What survived the control looked like this:

> Planning's benefit is a decreasing function of how competent the search-free policy already is.

Our supervisor's immediate reaction was the right one, and it is the sharpest critique this campaign
has received:

> *Any improvement shrinks as the thing it improves gets better — that's just the improvement space,
> `ceiling − baseline`, getting smaller. How do you make this scientific?*

This is exactly right, and it has to be taken literally rather than deflected. If planning always
captured some fixed share of the remaining headroom, then "gain decreases with competence" would be
a restatement of subtraction, not a finding about planning.

So we tested it. Assume a ceiling `C` and compute what fraction of the remaining headroom
`(C − baseline)` planning actually captures, for the three hopper variants:

| assumed ceiling | stock | softened | easier | spread |
|---|---|---|---|---|
| 550 | 50.8% | 57.5% | 76.4% | 25.6 pp |
| 600 | 45.7% | 47.4% | 53.7% | 7.9 pp |
| **650** | **41.5%** | **40.3%** | **41.3%** | **1.2 pp** |
| 700 | 38.1% | 35.1% | 33.6% | 4.4 pp |
| 1000 | 25.3% | 19.7% | 15.9% | 9.5 pp |

At a ceiling of 650, all three variants capture **~41% of remaining headroom**, agreeing to within
1.2 percentage points. And the cross-task check at matched baseline lands nearby: with baselines
around 430–470, hopper captures ~41% and walker ~47%.

**So the objection is substantially correct.** Most of the variation in "planning gain" across our
conditions is headroom arithmetic. Planning takes a roughly constant slice of whatever the policy
leaves on the table.

Two caveats, in both directions. Against us: that fit has **one free parameter (the ceiling) over
three points**, and we chose the ceiling to minimise spread — which is circular, and we are not
going to pretend otherwise. In our favour: "constant fraction of headroom" is a *sharper* and more
falsifiable claim than the vague version it replaces. It predicts a number, not a direction.

We tried to falsify it on the saturated tasks and could not, because they cannot discriminate:
cup-catch (981.5 → 982.8) and finger-spin (984.7 → 985.0) have ~15 points of headroom, where 41%
predicts about +6 and we observe about +1. Consistent, uninformative.

---

## 6. How to make it actually scientific

Three moves, in increasing order of decisiveness.

**(a) Estimate the ceiling independently instead of fitting it.** Train a reference agent far longer
on each variant and use its asymptote. Removes the free parameter.

**(b) Predict out of sample, pre-registered.** Take α ≈ 0.41 from hopper, measure a new variant's
baseline, and *publish the predicted planning gain before running the planning arm.* A model with
one fitted constant that predicts a held-out cell is doing real work. One such test is already in
flight (walker with a raised speed target).

**(c) Hunt the dissociation — the design that settles it.** Constant-fraction says the *only* thing
that matters is how much headroom exists. So it forbids two cells:

- a **weak** policy with **large** headroom where planning buys ~nothing
- a **strong** policy where planning still buys a lot

If either exists, headroom is not the whole story and the residual is the actual mechanism. We have
a principled place to look: a policy that is weak because of **aleatoric noise** should get nothing
from search — you cannot plan your way around dice — while a policy that is weak because of
**credit assignment** should get a great deal. Same headroom, opposite predictions.

**And the confound-free measurement, which is now running.** Every comparison above changes the
*training* run, so the policy, the state distribution and the headroom all move together. Instead:
train once with `mpc=false`, **save a checkpoint at every evaluation**, then toggle search on the
frozen weights. Same policy, same states, same headroom, one bit different. That gives planning's
value as a function of policy competence with everything else nailed down — and it is the honest
form of "PPO + planner", with no on-policy importance-weight breakage to debug.

We patched the trainer to checkpoint every 50k steps and relaunched (four runs, hopper and walker,
~25% of progress discarded to get it). Each run becomes a curve rather than a point.

---

## 7. The next idea: an agent that decides when to think

The third question our supervisor raised is the most interesting, and our data argues for it:

> *Could the agent decide when to plan and when not to — the way a person plans if there is time,
> and otherwise just acts?*

Yes, and it is the natural successor for three reasons. It is a **method**, not another
measurement — which is what this campaign has been missing. It has a **real price tag**: the planner
is 35% of wall-clock, so any saving is measurable rather than notional. And it is exactly what our
results imply: if planning's value is concentrated where the policy is weak, then paying for it
uniformly at every timestep is obviously wasteful.

This is classical **rational metareasoning** — decide whether to compute by estimating the value of
computation against its cost — pointed at a modern model-based agent.

**The cheap experiment that comes first, and it needs no new algorithm.** Take a frozen checkpoint.
At every visited state compute Δ = Q(MPPI action) − Q(π action): how much did search actually buy
*here*? Then plot the distribution.

- If Δ is roughly uniform across states, adaptive planning cannot help — you must pay everywhere.
- If Δ is heavy-tailed (most states ≈ 0, a few states large), then an oracle that plans only on the
  top-k% of states captures most of the benefit at a fraction of the cost.

Sorting states by Δ and sweeping k traces the **achievable Pareto frontier before building any
gating mechanism at all.** If that frontier is flat, the idea is dead for one day's compute. If it
is steep, we have the target curve, and the research question becomes concrete and modest: *find a
signal available at decision time that approximates the oracle ranking.* Candidates: ensemble Q
disagreement, policy entropy, one-step model error, TD error, or the planner's own improvement on a
truncated rollout.

The human intuition in the question then becomes the actual objective — maximise return subject to
an average planning budget, e.g. "plan on at most 20% of steps". That is a constrained optimisation
with a knob a practitioner would genuinely turn.

One caution from our own results. We already found that gating planning **mid-training** is
**bimodal** — acrobot's four gate seeds came out 154, 407, 572, 656, a 4× spread on one
configuration. So "recovers X% of the gap" was never an estimable quantity. Deployment-time gating
on a *frozen* policy is a cleaner problem — no feedback loop into what data gets collected — but the
mid-training version should be treated as hard until shown otherwise.

---

## 8. Where this leaves the story

**Dead:** conjunctivity sets the value of search. Killed by its own pre-registered control.

**Weakened to near-arithmetic:** planning's benefit falls as the policy improves. True, but mostly
headroom, and we say so.

**The sharper replacement, on probation:** planning captures a roughly constant fraction (~41%) of
the headroom the policy leaves. One fitted constant, three points, needs out-of-sample confirmation.

**Still standing, and the most robust thing we have:** whatever the mechanism, a world model earns
its keep only where the policy is failing — and this holds *across agents that exploit the model in
completely different ways*. DreamerV3 has **no search anywhere**; it uses its model purely to train
the policy on imagined rollouts. It shows the same task pattern: hopper 2.40× (p=0.009, n=6), walker
0.95×, a clean null (n=13/11). A hypothesis about search cannot explain a planner-free agent
behaving identically.

**The practical rule, honestly downgraded.** We wanted "read the reward function, then decide" — free,
no training run. That is gone with conjunctivity. What replaces it is a probe: train briefly without
search and pay for it only if the policy arm stalls. Less elegant, still useful, and now the thing
being measured is the thing that matters.

*Every number above is in git with per-seed values. The corrections in this post — including one
where we used random-policy return as a difficulty proxy when it actually measures the reward
floor — are recorded where they happened rather than quietly fixed.*
