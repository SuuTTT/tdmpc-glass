---
layout: post
title: "TD-MPC-Glass, Part 7: Five Bets, Resolved — What Survived Contact With the Data"
date: 2026-07-03
description: "The five-bet program from Part 6, run in parallel and resolved. The headline is a negative: the flagship 'planning-as-exploration' thesis does not survive a controlled plan-vs-policy-only test — three ways. What survives is quieter and true: planning is a sample-efficiency and exploitation lever (speed, not ceiling), and pure-JEPA anti-collapse is a non-effect on DMControl (state, pixels, and narrow data alike). A capstone with the honest scorecard, the two hypotheses the controls killed, and where the real paper is."
---

> The capstone. [Part 6](../2026-07-02-tdmpc-glass-part6-five-bets-next-phase) laid out five concrete bets after a
> supervisor review; we ran them in parallel on two GPU boxes. This post is the scorecard — including two
> appealing hypotheses that controlled ablations killed. Each thread has its own living log (linked below) with
> the full numbers; here is the synthesis.

## Scorecard

| bet | one-line verdict | state |
|---|---|---|
| **A — planning as a directed-exploration operator** (flagship) | **NULL** — planning is sample-efficiency + exploitation, *not* exploration | [Thread A](../tdmpc-glass-thread-a-planning-exploration) |
| **B — "when does a behavioral prior help RL?" taxonomy** | the surviving positive contribution (speed-not-ceiling) | [Thread B](../tdmpc-glass-thread-b-behavioral-prior-taxonomy) |
| **C — abstraction as variance-reduction** | **NULL** — the 6/16 "wins" are a wash | [Thread C](../tdmpc-glass-thread-c-abstraction-variance-reduction) |
| **D — JEPA anti-collapse, done right** | **reversal** — pure JEPA doesn't collapse (state/pixels/narrow) | [Thread D](../tdmpc-glass-thread-d-jepa-anticollapse-done-right) |
| **E — SE for structure discovery** | folded into A3; deprioritized once A went null | [Thread E](../tdmpc-glass-thread-e-se-structure-discovery) |

Two of the five were genuine, testable, *appealing* hypotheses — and the controls said no. That is the point of
running controls.

## The flagship went null — and it's the most useful result here

Part 6's flagship was the strongest *new* story: that the real reason model-based planning beats PPO on
exploration-hard tasks is **exploration** — "planning is a directed-exploration operator." The sharpest datapoint
was TD-MPC2 beating PPO on `HopperHop` **367 vs 33**. The claim: planning explores its way to states (and hence
policies) that on-policy PPO never reaches.

The honest test isn't TD-MPC2-vs-PPO (that confounds the *model/algorithm* with planning). It's **planning vs a
policy-only ablation of the same agent** — MPPI on versus off, everything else held fixed — with state-space
**coverage** logged across training. We ran that three ways:

- **A1-core (`HopperHop`, from scratch).** No coverage gain from planning. But nothing learned in the budget, so
  inconclusive.
- **A1-full (`WalkerWalk`, learns).** Planning *narrows* coverage post-competence (directed exploitation onto the
  gait manifold) — the opposite of the hypothesis — yet reaches competence **~1.5–2× faster** (training return
  leads the policy-only arm by +130–160 through the whole competence phase, converging by ~120k). Coverage does
  **not** mediate the benefit: planning covers *less* and learns *faster*.
- **A1-decisive (`CartpoleSwingupSparse`, sparse *and* exploration-hard — the quadrant the thesis needs).** The
  clean test of "does planning discover reward where policy-only stalls?" Answer: **no.** Both arms discover the
  sparse reward **3/3**; the policy-only arm discovers it *slightly earlier* (140k vs 157k). Planning's advantage
  is entirely *post*-discovery exploitation (final return 540 vs 331).

Three controlled tests, one conclusion: **planning is not a directed-exploration operator.** Its real, repeatable
value is **sample-efficiency and exploitation** — faster to competence, higher and steadier returns once the
reward is found. That is the campaign's "speed, not ceiling" law, restated at the level of the planner itself.
The HopperHop 367-vs-33 remains real, but it is a *TD-MPC2-vs-PPO* fact (a better model + value learning), not a
*planning-vs-no-planning* one — the planning-vs-π-only return edge there was only **+1.7**.

Because the mechanism was exploration, A2 (novelty-seeking MPPI) and A3 (SE-discovered exploration subgoals) —
and with them Thread E — were deprioritized: there is no exploration effect to amplify or direct.

### A correction we made on ourselves
The first A1-full read reported "policy-only slightly ahead, a flat null," from a *noisy single-episode eval* at
140k. Reading the clean \(n{=}3\) training curve overturned it — planning has a large, consistent mid-training
lead. The verdict is *not* "planning does nothing"; it's "planning buys speed/exploitation, not exploration." The
lesson: read the training curve, not one eval snapshot.

## The JEPA thread reversed — cleanly, four ways

Part 5 reported a "downstream-dependent anti-collapse taxonomy." Part 6 flagged that we had tested it on the wrong
object — TD-MPC2 is value-anchored, not a pure JEPA. Testing it *right* reversed the story:

- **D2 / D2-ext (pure JEPA on DMControl state).** A pure self-predictive latent (encoder + jumpy predictor + EMA,
  **no** reward/value loss) **does not collapse.** The zero-anti-collapse arm has the *best* readouts; the
  collapse is prevented by the **predictor+EMA (BYOL) asymmetry**, not by any explicit repulsion. Adding
  anti-collapse ranges from neutral to harmful, and the earlier geometric-vs-value split was **nav-specific**, not
  a law.
- **D3 (pixels).** Same picture in the high-dim regime LeCun's argument actually targets — no collapse, uniformity
  *maximizes* effective rank but gives the *worst* readouts.
- **D1 (SE arm on TD-MPC2).** On a value-anchored latent, structural entropy hurts value-control just like
  uniformity, and SE+uniformity combined is worse than either — a redundancy result, correctly scoped.
- **Open cell (narrow / on-policy data).** Does a narrow, reward-seeking data distribution collapse a pure JEPA?
  **No** — effective rank stays 10–22 as the buffer narrows; the value-R² that falls just tracks the narrow
  buffer's low value variance, not encoder collapse.

So the collapse regime is *narrower* than any data-distribution story: pure JEPA is robust on broad DMControl
(state and pixels) **and** on narrow/on-policy data. The one setting that did collapse — the closed-loop online
nav H-JEPA — is not explained by data breadth, which points at the value/goal closed loop itself as the trigger.
That is the honest open question, and it is a much sharper one than we started with.

## What survives: the taxonomy (Thread B)

The quiet result is the real one. Across Panda and DMControl, injecting a behavioral prior (an analytic skill, a
hand-controller, a structured latent) is governed by two axes — **prior-fit × exploration-difficulty** — with the
escape-difficulty frontier operationalizing the second. A fitting prior is a **speed lever** (~1.6–7× faster to
competence, same ceiling); a mismatched one is **dead weight**; on exploration-hard tasks only the prior-equipped
agent survives at a practical budget. The Part-7 planning result slots straight in: planning is another *speed*
lever, not a ceiling or exploration lever. This is the paper — an empirical taxonomy, honest about what
abstraction does and does not buy. Finalizing it is the next step.

## The meta-lesson

Two appealing, publishable-sounding hypotheses — "planning is the exploration breakthrough" and "abstraction
reduces variance" — did not survive controlled tests. A third — "JEPA needs anti-collapse" — reversed once tested
on the right object. None of that is failure; it's the controls doing their job. The through-line of this whole
program holds: **abstraction and planning buy sample-efficiency — speed, not ceiling** — and the fastest way to a
true result is a matched-budget control and the discipline to read the \(n\)-seed curve rather than the number you
were hoping for.

*Every figure and number above is pulled from the disk-backed results ledger; the nulls are reported as
deliberately as the positives. Per-thread detail lives in the five linked living logs.*
