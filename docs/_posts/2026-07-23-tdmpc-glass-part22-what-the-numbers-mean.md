---
layout: post
title: "TD-MPC-Glass, Part 22: What the Numbers Actually Mean — And Whether We Ever Removed the World Model"
date: 2026-07-23
description: "A from-scratch explanation of the vocabulary this campaign has been using — cells, per-seed values, unimodal vs bimodal, inversions, permutation p-values, margins, design floors — followed by the hardest question anyone has asked about this work: if setting consistency_coef=0 leaves the latent dynamics still trained by the reward and value heads, did we ever actually remove the world model? And is that why DreamerV3 degrades under its ablation while TD-MPC2 barely notices?"
---

> Part 21 was an audit. This one is a glossary and a reckoning. Every term the
> last two days of results have leaned on, explained from scratch with the
> actual numbers — and then the question that matters most: **have we been
> ablating the thing we said we were ablating?**

## Part 1 — The setup, in plain language

We train the same agent twice on the same task. One copy is normal. The other
has one loss term switched off. Then we ask whether the second copy is worse.

That's it. Everything else is bookkeeping about *how much* worse, *how
reliably*, and *what kind of worse*.

Two details make the bookkeeping harder than it sounds:

**Runs are random.** Each training run starts from a different random seed —
different network initialisation, different exploration noise, different data.
Two identical configurations can land far apart. So we don't run each
configuration once; we run it many times and look at the spread.

**The number we compare is not the last number.** An agent's score bounces
between evaluations. On pendulum a single run read 360 at one checkpoint and 0
at the next. So we take the **last-6-evaluation median**: the six most recent
evaluations of a finished run, sorted, middle value. One number per run, robust
to a single bad checkpoint.

---

## Part 2 — The vocabulary

### "A cell"

Our main table is a grid: rows are tasks (cheetah, walker, ball-in-cup, …),
columns are the four combinations of *full vs stripped agent* × *policy π vs
planner Π*. Each box in that grid is a **cell**.

"Which cell moved" is shorthand for: when we switched off the loss term, which
of those four boxes changed? If the full-vs-stripped difference shows up under
the planner but not under the policy, that tells you the effect lives in
*planning*, not in the representation. Acrobot does exactly that, and that's
how we know.

### "Per-seed values"

Instead of reporting one number per configuration, we report the whole list.
Cheetah's stripped arm at n=12:

```
498.3  543.5  566.8  572.6  586.4  587.6  603.4  612.1  615.7  617.5  627.8  648.1
```

Twelve runs, twelve numbers. Reporting the list rather than the average is the
single most useful thing this campaign started doing, because averages hide the
two shapes below.

### "Unimodal" vs "bimodal"

**Unimodal** means the numbers cluster around one value. Cheetah's stripped arm
above is unimodal: it runs smoothly from 498 to 648 with no gaps. Every run
learned the task; they just learned it to slightly different degrees.

**Bimodal** means the numbers fall into two separate clumps with a gap between.
Finger-spin's stripped arm at n=9:

```
641.1  701.3  702.6  │   926.4  930.5  948.9  953.9  957.5  966.7
└──── three runs ────┘   └──────────── six runs ─────────────────┘
                    224-point gap
```

That is *not* "the average is 838." No run scored 838. Three runs failed to
learn properly and six learned fine. The mean describes nothing that happened.

**Why it matters so much:** the right summary statistic depends on the shape.
For a unimodal arm, an average or a gap is meaningful. For a bimodal arm, the
only honest summary is a **success rate** — how many runs landed in the good
clump. Using an average on a bimodal arm is how we ended up retracting two
claims.

### "Graded" vs "all-or-nothing"

This follows directly.

- **Graded**: every run learns, and the ablation changes *how well*. Cheetah's
  full arm averages 753, stripped averages 574. Everyone succeeds; the stripped
  ones succeed less. The effect is a continuous quantity.
- **All-or-nothing**: runs either succeed completely or fail completely, and the
  ablation changes *how often*. Ball-in-cup's full arm solves 8 times out of 14
  — and when it solves, it scores 972–976, essentially a perfect score. When it
  doesn't, it scores 0. There is no middle.

These are different phenomena and they need different statistics. That
distinction is the backbone of the current hypothesis.

### "Inversions", and "0/144"

Take every full-arm run and pair it with every stripped-arm run. With 12 runs
each, that's 12 × 12 = **144 pairs**. An **inversion** is a pair where the
stripped run beat the full run — the "wrong" way round.

- Cheetah at n=12: **0/144**. Every single full run scored higher than every
  single stripped run. The two lists don't overlap at all.
- Walker at n=12: **16/144**. Mostly ordered, but 16 pairs go the wrong way —
  the lists overlap.

"0/144 inversions" is the cleanest possible result for a comparison of this
kind. It's what "complete separation" means, stated as a count.

### "p = 3.7 × 10⁻⁷" — where that number comes from

This is an **exact permutation test**, and it's simpler than it looks.

Ask: *suppose the labels are meaningless.* Suppose those 24 cheetah numbers are
just 24 numbers, and calling twelve of them "full" and twelve "stripped" is
arbitrary. How many ways are there to split 24 numbers into two groups of 12?

$$\binom{24}{12} = 2{,}704{,}156$$

Of those 2.7 million possible splits, **how many put all twelve highest numbers
in one group?** Exactly one. So if the labels meant nothing, the chance of
seeing a split this extreme is 1 in 2,704,156 —

$$p = \frac{1}{2{,}704{,}156} = 3.7 \times 10^{-7}$$

No distributional assumptions, no normality, no t-test. Just counting
arrangements. That's why it's called *exact*.

### "The design floor"

Notice what that implies: with 12 runs per arm, **the smallest p you can ever
get is 1/2,704,156**, because perfect separation is the most extreme outcome
available. That's the **design floor** — the best possible p-value for that
sample size.

| runs per arm | possible splits | best achievable p |
|---|---|---|
| 2 | 6 | 0.167 |
| 3 | 20 | 0.050 |
| 5 | 252 | 0.004 |
| 8 | 12,870 | 7.8 × 10⁻⁵ |
| 9 | 48,620 | 2.1 × 10⁻⁵ |
| 12 | 2,704,156 | 3.7 × 10⁻⁷ |

This table cost us a day. The DreamerV3 hopper comparison was run at **2 seeds
per arm**, where the best possible p is 0.167 — *it could never have reached
significance no matter how large the effect was.* We'd have run it, seen a
gap, and reported something unsupportable. Now the first question about any
design is: what's the best p this could produce?

### "The margin"

The margin is the *gap* between the worst full run and the best stripped run.
Cheetah at n=12: worst full is 681.25, best stripped is 648.10, so the margin is
**33.15 points**.

An inversion count and a margin measure different things. Zero inversions says
the lists don't overlap. The margin says *by how much*. You can have zero
inversions with a huge gap (robust) or with a hair's-breadth gap (fragile).

And here's the finding that surprised us: **margins shrink as you add runs.**
Cheetah's margin was 92.85 at n=3, n=5, and n=8 — rock stable across ten added
runs — and then fell to 33.15 at n=12, when an unusually weak full run (681) and
an unusually strong stripped run (648) both showed up.

That's not bad luck. It's arithmetic. The margin depends on the *extremes* of
both lists, and the more runs you do, the more extreme your extremes get. A
margin is an **extreme-order statistic**, and it drifts downward with n by
construction. So a stable margin is evidence only *relative to the n at which
you measured it*. We had asserted cheetah's margin was stable; at n=12 that
turned out to be wrong, and correcting it produced a better methodological point
than the original claim.

---

## Part 3 — The hard question: did we ever remove the world model?

**Short answer: no, and this is the most important thing in the paper.**

### What TD-MPC2's latent dynamics is trained by

TD-MPC2 has a learned dynamics network `d(z, a)` — given a latent state and an
action, predict the next latent state. Three separate signals train it:

1. **Self-prediction (the "consistency" loss).** Roll `d` forward one step, and
   pull the result toward a target encoding of the *actual next observation*.
   This is what grounds the latent in the world.
2. **The reward head.** Predict reward from latents that were produced by
   rolling `d` forward. Errors flow backward *through* `d`.
3. **The value head.** Same structure: Q-values are computed on rolled-out
   latents, so its gradients also flow back through `d`.

Our ablation sets `consistency_coef = 0`. That deletes **only signal 1.**

Signals 2 and 3 remain untouched, and the planner still rolls trajectories
through `d` when choosing actions. So the "stripped" agent:

- **still has a dynamics model**
- **still trains it**, using reward and value prediction
- **still plans through it**

It has lost exactly one thing: the requirement that its imagined next state
resemble the *actual* next state. It no longer has to model the world — only
the parts of the world that affect reward and value.

### Why this isn't a flaw, and what it makes the paper about

That object has a name. It's a **value-equivalent model** — Grimm et al. (2020),
and the principle behind MuZero. The idea: a model doesn't need to predict
observations, only quantities that affect decisions.

So our experiment is not "world model vs no world model." It's:

> **Does observation-grounded self-prediction earn its keep on top of value
> equivalence?**

This is a narrower question and a better one. It's the question the
value-equivalence literature actually poses, and it's *answerable* — where
"does a world model help?" is under-specified, because it depends entirely on
what you mean by removing it.

It also **explains our results** in a way the loose framing couldn't. The
stripped agent stays competitive on dense-reward tasks like cheetah and walker,
because reward and value gradients alone are enough to shape a usable latent
there. It falls apart on acrobot's *planner*, because planning three steps
through a latent that was never grounded in observations produces garbage. The
old framing had to call those unrelated. The value-equivalence framing predicts
both.

**What we genuinely cannot test:** a true model-free TD-MPC2 — no dynamics
network at all. That's not an ablation, it's a different algorithm. The closest
reference point is the tuned SAC baseline from the official TD-MPC2 release, and
we use it as exactly that.

---

## Part 4 — So is that why Dreamer behaves differently?

Partly — but let me correct a framing first, because the difference is real and
the word "crash" is doing damage.

**What actually happened.** In DreamerV3 on hopper, the ablated agent scored 188
against the full agent's 322. That's a substantial degradation. It is *not* a
crash to zero. The thing that genuinely collapsed in this campaign was the JEPA
arm — and that was a configuration pathology (the anti-collapse regulariser
saturating against a one-hot softmax), not a consequence of the ablation.

**Why the same nominal ablation has different leverage.** DreamerV3's ablation
is `loss_scales.dyn = 0`, which switches off the KL term pulling the RSSM's
*prior* toward its *posterior*. And here's the structural difference:

- **DreamerV3 trains its actor and critic entirely inside imagination.** They
  never see real states. Every gradient they receive comes from trajectories the
  model dreamed. Break the prior and you have poisoned the only source of
  training signal the controller ever gets.
- **TD-MPC2 trains its policy on real latents**, and its planner only rolls
  **3 steps** before falling back on a learned value estimate. Break the
  self-prediction and you've degraded a 3-step lookahead that was already
  backstopped.

So: **same switch, very different blast radius.** In Dreamer the model *is* the
training environment. In TD-MPC2 the model is a short-horizon lookahead
attached to a controller that has other sources of signal.

This is a genuine confound for cross-family comparison, and the paper says so.
If we reported "Dreamer degrades more than TD-MPC2, therefore generative world
models matter more," we'd be measuring the ablations' leverage, not the
architectures' reliance.

---

## Part 5 — Do we need to fix it?

Three options were on the table.

**Option 1 — actually delete the model from TD-MPC2.** No dynamics network at
all. Rejected: that's not an ablation of the model, it's a swap to a different
algorithm, and the comparison it produces (model-based vs model-free) is the one
the SAC baseline already provides.

**Option 2 — match the ablations across families.** Find a Dreamer intervention
with the same leverage as TD-MPC2's. Rejected as intractable inside a week: the
architectures differ in exactly the way that makes "the same leverage" hard to
define. Dreamer has no equivalent of "the reward head still trains the dynamics,"
because in Dreamer the dynamics *is* the substrate everything else runs on.

**Option 3 — state precisely what is removed and make the claim about that.**
This is what we did, and I want to be clear it isn't a consolation prize. Naming
the ablation correctly turned a vague question into a sharp one, made the
results explicable rather than merely tabulated, and connected the work to an
existing theoretical framework instead of leaving it as a pile of ablation
numbers.

The one thing we owe the reader is the caveat stated plainly, in the paper, in
the place where the cross-family comparison appears — not buried in an appendix.
That's now written.

---

## Part 6 — What it all adds up to

Three channels, each with its own signature:

- **Representation** — graded and unimodal. Cheetah (0/144 inversions,
  p = 3.7 × 10⁻⁷) > walker (16/144, p = 2 × 10⁻⁴) > finger (redundant) > hopper
  (nothing). The benefit shrinks smoothly to zero and never goes negative.
- **Planning** — visible only under the planner. Acrobot separates under Π
  (p = 2.1 × 10⁻⁵) but not under π, and the planner actively *harms* the
  stripped agent on **9 seeds out of 9**.
- **Exploration** — all-or-nothing, measured as a success rate. Ball-in-cup:
  full solves 8/14, stripped solves **0/14 and never records a single non-zero
  evaluation in any run**.

And the methodological finding nobody was looking for: **five of five effects in
this campaign drifted as seeds accumulated.** Hopper's reversed and was
retracted. Pendulum's went to zero. Walker's margin inverted. Finger's
significance turned out to rest on one outlier. Even cheetah — the one result
whose *conclusion* never budged — saw its margin fall by two thirds.

That's the base rate, in our own hands, on our own results. It is the reason
this paper reports per-seed values, inversion counts, margins *and* their
trajectories, rather than a table of means with stars next to them.

---

*Every number here is read from run CSVs on the training boxes, gated on
completion at 1.5M steps. The permutation tests are exact and enumerate the full
set of arrangements; no distributional assumptions are used anywhere.*
