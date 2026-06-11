---
layout: post
title: "TD-MPC-Glass, Part 3: The Latent Was Already the Abstraction"
date: 2026-06-10
description: "Thirteen abstraction levers, all null; the one method that beat vanilla TD-MPC2 (a jumpy k-step world model) is prior art, not ours. This is the post where we stop guessing and measure why — a cheap probe shows the trained latent is already a sufficient, value-aligned abstraction (value linearly decodable at R²≈1.0; criticality near-uniform). Ni et al.'s sufficient-self-predictive-abstraction theory, showing up as a number on our own checkpoints. The transferable lesson is the mechanism-check-before-fanout methodology that called every null correctly."
---

> Part 2 ended with a scoreboard nobody wants to publish: eight abstraction mirages dissolved to null,
> one real win (a jumpy k-step world model) that turned out to be someone else's idea, and a methodology
> that kept catching us being wrong. This post finishes the story by answering the question the whole
> campaign was really asking — **not "did abstraction help?" (it didn't) but "why didn't it?"** We stopped
> bolting things onto TD-MPC2 and instead ran a cheap probe on a trained checkpoint. The answer is clean
> and a little humbling: **a trained TD-MPC2 latent is already a sufficient, value-aligned abstraction.**
> Value is linearly decodable from it at $R^2 \approx 1.0$; the structure an explicit abstraction would
> impose is already there. The negative result and the methodology are the contribution — and we think
> they're worth writing down honestly.

---

## 0. Where Part 2 left us

If you read [Part 2]({{ '/2026/06/09/tdmpc-glass-part2-mechanism-check-saved-a-campaign/' | relative_url }}), you know the verdict and you know the
caveats. The one-line recap, for everyone else:

We set out to **beat TD-MPC2 at the architecture/algorithm level** with an "abstraction" idea — our own
structural-entropy *Glass* augmentation, and then a parade of successors — under a strict fair protocol:
single-variable, compute-matched, pre-registered peak-AND-final CI gates, mechanism-check before fan-out,
no procedure tricks. The protocol was the product. And under it:

- **Thirteen abstraction levers came back null** — geometric clustering, behavioral clustering,
  bisimulation auxiliaries, distractor-robustness, sparse-task rescue, Laplacian/eigenpurpose exploration,
  community-detection skills, structural-entropy adaptive jump-length, uncertainty-gated horizons, two
  flavors of SE-driven exploration, and more.
- **One thing genuinely beat vanilla** — a **jumpy (k-step) world model** on PandaPickCube manipulation:
  peak **+966 (+44%)**, 95% CI **[714, 1248]**; final **+1266 (+88%)**, CI **[877, 1642]**, both
  CI-separated, mechanism pre-confirmed (the k-step head's error vs iterating the 1-step model drops
  $0.99 \to 0.82$ as k grows). But the jumpy world model with cross-timescale consistency is **published
  prior art** ([Farebrother et al., 2026](https://arxiv.org/abs/2602.19634)). It's a fair-protocol
  reproduction-and-evaluation win, **not our invention**.

So the honest situation at the end of Part 2 was: every idea that was *ours* is a null, and the one win
isn't. That's not a tragedy — it's a question. **Why does a strong latent world model keep being
un-improvable by abstraction?** Part 2 gestured at an answer (SimNorm is already a soft-clustering; Ni et
al. 2024 say the self-predictive objective is a *sufficient* abstraction). This post is where we stop
gesturing and **measure** it.

---

## 1. The shift: stop bolting things on, start probing the substrate

Every lever in Part 2 followed the same shape: hypothesize that TD-MPC2 is *missing* some abstraction,
build it, run a gate. After thirteen of those, the pattern was loud enough to reframe the question. Maybe
nothing is missing. Maybe the trained latent already *is* the abstraction every lever was trying to add —
in which case the levers aren't wrong so much as **redundant**, and you'd predict exactly the null parade
we got.

That's a testable claim, and crucially it's testable **without training anything**. You take a trained
PandaPickCube checkpoint — a latent the self-predictive objective has already shaped — and you ask two
direct questions of it:

1. **Is the latent already value-sufficient?** If an explicit value-equivalence objective is supposed to
   "keep only what matters for control," then there has to be control-relevant information the current
   latent is *losing*. Is there?
2. **Does value-criticality vary across states?** Every adaptive-horizon / value-critical idea assumes some
   states are decision-critical and others aren't, so you can spend planning budget unevenly. Is that
   variation actually there?

Both are cheap kill-tests on a frozen checkpoint (`scripts/value_probe.py`, standalone, no hot-path edits).
This is the iteration-28 mechanism-check, and it's the spine of the whole post.

## 2. Probe 1 — the latent is already value-sufficient

The lever this gates is the **value-equivalent macro head**: train the k-step dynamics $d_k$ to be
*return*-equivalent (predict the same macro-$Q$) rather than state-faithful, on the theory that an
abstraction organized around value keeps only what control needs and discards the rest.

For that to help, value has to be *hard* to recover from the current latent — there has to be headroom. So
we fit the simplest possible probe: a **linear** decode of the value function from the frozen latent, over
12k states from 12 episodes on a self-predictive PandaPickCube checkpoint.

$$ \texttt{linear\_V\_decode\_r2} = 0.9994 $$

Value is **already linearly decodable from the latent at $R^2 \approx 1.0$**. Not after a deep nonlinear
head — a *linear* readout recovers it almost perfectly. There is essentially nothing for a value-equivalence
objective to *add*, because the self-predictive latent has already arranged itself so that value is a
trivial linear function of it.

The supporting numbers tell the same story from the other side:

| quantity | value | reading |
|---|---|---|
| `linear_V_decode_r2` | **0.9994** | value already trivially decodable |
| `effective_dim_latent` | ≈ 6.96 | the latent uses ~7 effective dimensions |
| `effective_dim_value_subspace` | ≈ 7.08 | the value-relevant subspace is ~7 dims |
| `value_irrelevant_variance_frac` | ≈ 0.978 | ~98% of latent variance is value-irrelevant |

That last row is the interesting twist, and it's worth not over-reading. ~98% of the latent's *variance* is
value-irrelevant — which sounds like enormous waste an abstraction could trim. But the value-relevant
subspace (~7 dims) and the latent's effective dimensionality (~7 dims) **match**: the value information
isn't buried, it's cleanly carried in a low-dimensional subspace that a linear map reads off perfectly. The
value-irrelevant variance is along directions the value head already ignores. So an explicit
value-equivalence loss isn't *finding* hidden value structure — the structure is already exposed; it's
mostly redundant pressure toward something the self-predictive objective achieved for free.

And when we actually ran it (iter-26, value-equivalence at coef 0.5 vs the matched baseline, MPPI), it
didn't just fail to help — it **hurt**: PandaPickCube `ve` 1616/916 vs `vebase` 2692/2243 (Δ **−1076 peak /
−1327 final**); CheetahRun −129/−83. The mechanism-check predicted "neutral-at-best, probably redundant,"
and the campaign delivered "redundant and slightly harmful." A cheap probe and a multi-seed run agreeing is
exactly the corroboration the methodology is built to produce.

## 3. Probe 2 — value-criticality is near-uniform

The second lever is the **value-critical adaptive horizon**: plan farther where decisions are critical,
shorter where they aren't. This is the value-flavored cousin of the error-gated adaptive-k we killed in
Part 2 — and it dies the same way, for the same reason.

We measured how much the value-criticality of a state varies across the states a good policy visits. The
coefficient of variation:

$$ \texttt{crit\_cv} = 0.36 \quad (< 0.5 \text{ bar}), \qquad \texttt{flat\_state\_frac} = 0.029 $$

A CV of 0.36 means criticality is **near-uniform** — there's no spread of "critical vs trivial" states for
an adaptive horizon to exploit. Only ~3% of states are flat (genuinely don't-care). This is the exact same
finding that killed the error-gated adaptive-k family in Part 2 (the jumpy model is *uniformly accurate*, so
there's nothing to gate), now showing up in the value geometry: the states a near-optimal policy visits are
**uniformly decision-relevant**. There is nothing to adapt *to*.

So both value-organized levers fail their mechanism-check before a single multi-seed campaign. Same root
cause as all thirteen prior nulls, now stated as a measurement rather than a theory:

> A strong self-predictive world model (TD-MPC2 + SimNorm) already encodes a **value-sufficient
> abstraction**. The latent isn't *missing* the structure these levers add — it already has it.

## 4. The aha, stated plainly

Here is the thing it took us a campaign of nulls to see, and an afternoon of probing to confirm.

When we started, "abstraction" felt like an *addition* — a representation TD-MPC2 lacked, that we could
build and bolt on. The probes say the opposite. A trained TD-MPC2 latent is **already**:

- a soft-clustering — SimNorm partitions the latent into groups and softmaxes each one, so the encoder
  output is V parallel soft codebooks *by construction*;
- temporally coherent — the self-predictive consistency loss shapes transitions to be smooth and
  predictable;
- and **value-aligned** — value is linearly decodable at $R^2 \approx 1.0$, criticality is near-uniform,
  the value subspace is low-dimensional and cleanly exposed.

This is precisely what [Ni et al. (2024)](https://arxiv.org/abs/2401.08898) predict in theory: a
self-predictive objective learns a **sufficient** abstraction. We didn't set out to confirm their theorem;
we set out to beat it, failed thirteen times, and then *measured it on our own checkpoints*. The
$R^2 = 0.9994$ is what "sufficient abstraction" looks like when you put a ruler on it.

Which reframes every null in the ledger. They aren't thirteen unrelated failures — they're **one finding,
thirteen times**: explicit abstraction is **redundant** with what a strong self-predictive world model
already learns, and where it isn't redundant it's **misaligned** (Glass's transition-graph entropy captured
real *motion-phase* structure — swing/stance/contact arcs — that is genuinely beyond SimNorm but
**irrelevant to control**). Redundant or misaligned; either way, null.

## 5. The structure is real — it's just not useful for control

It would be tidy to conclude "there's no structure in the latent, so abstraction has nothing to grab." That
would be wrong, and the most interesting part of the project is *why* it's wrong.

The structure is unambiguously there. When we built the latent transition graph correctly — sparsifying the
dense SimNorm "blob" first — and measured its structural entropy, we found a **53% two-dimensional vs
one-dimensional structural-entropy gap** on the k-step transition graph (47% via kNN geometry). That's a
strong, real community structure. The raw graph is a ~0%-gap blob (40–76% edge density); sparsify it and the
communities pop out crisply.

So the latent *does* carry rich structure. The punchline of the entire project is the next sentence: **the
structure is real but not useful for control.** Across three independent probes:

- The communities are **motion phases**, not reachable subgoals → useless as skills (Part 2 §8).
- Their boundaries **don't coincide** with model-error regions → useless for adaptive-k (Spearman +0.09 /
  −0.18).
- SE-driven coverage **didn't beat** generic novelty → useless for exploration.

Why does rich structure refuse to help? Because SimNorm already hands the model a soft-categorical code, and
the self-predictive + value objectives already extract the *control-relevant* slice of it (the ~7-dimensional
value subspace from §2). The 53%-gap communities describe *what the dynamics do* — they are an
**interpretability** result about what a world-model latent represents — but control needs *what to do*, and
that's already encoded, value-sufficiently, in a different and lower-dimensional structure. Re-clustering by
motion phase, or steering jump-length / exploration by it, is re-deriving structure the policy and value
heads already exploit. (Full retrospective: `docs/analysis/why-glass-failed-simnorm-redundancy.md`.)

## 6. The real contribution: mechanism-check before fan-out

If the result is "abstraction is redundant with a sufficient world model," the *method* is "how we found
that out without burning a year of GPU time on it." That's the genuinely transferable piece, and it's worth
being explicit about, because it called every null correctly.

The discipline is one rule: **before you fan out a lever into a multi-seed, multi-week campaign, run the
cheapest possible test of the mechanism it depends on.** Not "does it work?" — that needs the campaign.
"Does the thing it *assumes* even exist?"

It works because every abstraction lever has a load-bearing assumption you can interrogate on a frozen
checkpoint in an afternoon:

| lever | the assumption it needs | cheap kill-test | result |
|---|---|---|---|
| value-equivalent macro head | value is hard to recover from the latent | linear V-decode $R^2$ | **0.9994** → no headroom |
| value-critical horizon | criticality varies across states | criticality CV | **0.36** → near-uniform, nothing to gate |
| SE adaptive jump-length | boundaries mark where the model errs | boundary-vs-error Spearman | **+0.09 / −0.18** → uncorrelated |
| uncertainty-gated horizon | error varies under planning perturbations | error inflation under MPPI noise | **1.06×** → uniform, nothing to gate |

Each of these is hours, not weeks. Each one predicted the campaign verdict that followed (when we spent the
compute anyway). The contrast with the *original* Glass effort is the whole argument: iterations 8–9 spent
**months** tuning a "turn-Glass-off-at-1M" schedule — when an iter-23-style mechanism-check ("does Glass's
structure track anything control needs?") would have returned *no* in an afternoon. We paid the multi-week
price once so we could stop paying it. The lesson generalizes to anyone tempted to bolt abstraction onto
TD-MPC2 or Dreamer: **measure that the headroom exists before you build the thing to exploit it.**

There's a second methodological thread from Part 2 worth restating here, because it's a precondition for the
probes meaning anything: **peak-AND-final reporting**. Several mirages looked publishable at an interim
snapshot — a "growing lead" that reversed by 450k, a "+104% on final" that shrank to +44% peak once you
accounted for vanilla's late collapse. We report both metrics for every arm and gate on CI separation. The
jumpy win survives on both; the abstraction effects survive on neither.

## 7. Honest limitations

This is a negative result with a measured mechanism, not a closed book. The places it could be wrong:

- **Low-DoF regime.** Every probe here is on PandaPickCube and DMC-scale tasks (~7 effective latent
  dimensions). The theory permits abstraction headroom exactly where there is **more value-irrelevant
  capacity to discard** — i.e. **high-DoF** control (Humanoid, Dog, dexterous manipulation). Our one
  high-DoF attempt (Humanoid) floored at 500k steps because it needs millions; we couldn't test it honestly.
  The right move is the same one this post argues for: run a high-DoF `value_probe` *first* — if value is
  *not* linearly decodable at $R^2 \approx 1.0$ there, the headroom is real and worth a budget; if it is,
  the verdict extends and we've saved the budget.
- **TD-MPC2-specific.** "Sufficient self-predictive abstraction" is a property of TD-MPC2's objective.
  DreamerV3 learns a *reconstruction*-based recurrent latent and trains its actor-critic purely in
  imagination — a reconstruction objective is **not** value-sufficient by construction, so there may be
  genuine abstraction headroom there that TD-MPC2 doesn't have. We haven't tested it. The redundancy claim
  is about strong *self-predictive* world models specifically.
- **$R^2$ is necessary, not sufficient, for "no headroom."** Value being linearly decodable on the
  *visited* state distribution doesn't prove it's decodable off-distribution (under distractors, transfer).
  The honest residual open question is whether an explicit value-equivalence objective helps where the
  *implicit* one is capacity-limited — distractors and transfer — even though it's redundant on-distribution.
  That's a smaller, sharper claim than the one we set out to make, but it's the one the data leaves open.

## 8. What's next

The honest read is that this closes a *family*, not just a lever. The thing that kept failing — clustering,
structural entropy, value-equivalence, criticality-gated horizons — all tried to **re-organize a monolithic,
already-sufficient vector latent**. That's the dead end. The finding tells us exactly where to look instead:
substrates where the abstraction is *not* already baked in, and regimes where "sufficient" stops holding.

- **Graph world models + structural entropy.** The redundancy that killed our Glass arm was specific: SE on a
  SimNorm latent is redundant because SimNorm's softmax is *already* a soft clustering. A graph-structured
  world model — nodes as entities/objects, edges as relations, GNN dynamics — has **no such built-in
  clustering**, so structural entropy (the information-theoretic graph-hierarchy objective) is non-redundant
  there *by construction*. This is the direction we're most excited about: SE-guided hierarchical abstraction
  in graph world models, on genuinely relational domains (multi-object manipulation, multi-agent), with the
  same mechanism-check up front — *does the trained graph latent already carry the SE hierarchy, or not?*
- **High-DoF, done right.** The one regime the sufficiency claim may not survive — more state detail means more
  potentially value-irrelevant capacity. Our one Humanoid attempt floored at 500k steps (needs millions); the
  right move is a high-DoF `value_probe` *first* — if value isn't already decodable there, headroom is real.
- **Generality across world models.** Everything here is measured on TD-MPC2. DreamerV3 learns a
  reconstruction-based recurrent latent and trains its actor in imagination — does the same value-sufficiency
  hold? We're wiring up the comparison; the claim only generalizes if the probe says so.

And one clean byproduct worth reusing: the **ensemble-free disagreement signal** (jumpy prediction vs
iterated one-step) tracks true k-step error at Spearman **+0.72**. Useless for horizon-gating (the error is
uniform), but a real, validated uncertainty signal for exploration bonuses or safe/abstained planning.

## 9. The bottom line

We tried to beat TD-MPC2 with abstraction. We didn't. The one method that beats it isn't ours. And the
reason — measured, not asserted — is that a trained self-predictive latent is **already** the abstraction we
kept trying to add: value linearly decodable at $R^2 = 0.9994$, criticality near-uniform at CV 0.36, the
control-relevant structure cleanly exposed in ~7 dimensions while 98% of the variance the model carries is
value-irrelevant slack the value head already ignores. Ni et al.'s sufficient-abstraction theorem, as a
number on our own checkpoints.

That's a negative result, and we think it's a useful one: it tells anyone building on a strong
self-predictive world model that explicit abstraction is the wrong place to look for wins, and it ships the
**mechanism-check-before-fanout** discipline that will tell you the same thing about your lever in an
afternoon instead of a year. The wins, when they come, will be small, specific, and only believable behind a
pre-registered, peak-and-final, mechanism-checked gate. We'd rather report that honestly than fake the
headline.

---

*Reproducibility: the value probe is `scripts/value_probe.py` (standalone, no hot-path edits); per-run CSVs
under `exp/tdmpc_glass/`; iteration records in `docs/iterations/` (iter-26 value-equiv, iter-28
value-organized mechanism-check); campaign verdicts in `docs/iterations/RESEARCH_LEDGER.md`; the SimNorm
structural-entropy retrospective in `docs/analysis/why-glass-failed-simnorm-redundancy.md`. All numbers are
read from run CSVs / probe JSON, not notebooks — verification discipline, the hard way.*


{% include mathjax.html %}
