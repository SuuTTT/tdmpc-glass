---
layout: post
title: "The R² Criterion That Couldn't: Why You Can't Probe Your Way to 'Redundant'"
date: 2026-06-17
description: "For months I explained a campaign of null abstraction results with one tidy story: TD-MPC2's latent is already 'value-sufficient' — value decodes linearly from it at R²≈1 — so any abstraction that re-organizes toward value is redundant. A reviewer-style challenge forced the real question: does that R² actually prove redundancy? I built the discrimination experiment, and the answer is a clean no. Two ways of measuring 'value-sufficiency by linear decode' both fail — one is confounded by return variance (it anti-correlates with performance), the other is saturated by construction (it reads ~0.98 whether the policy scores 653 or 1). The famous '0.9994' was an artifact. Here is the post-mortem."
---

> The redundancy criterion was the most-cited idea in my TD-MPC2 campaign — the one-line answer to
> "why does nothing beat it at the abstraction level?" It turns out the *evidence* for it (a
> linear-decode R²) never measured what I claimed. The negative result stands; the explanation I bolted
> on does not. This is how it came apart, with receipts.

## Where the criterion came from

The campaign result is consistent and brutal: **every explicit *representation* abstraction I tried —
state, temporal, relational, compositional, 16+ levers — is null** under a fair single-variable
protocol versus vanilla TD-MPC2.

You need a *why*. The story was the **redundancy criterion**: TD-MPC2's SimNorm latent is trained with
a value loss, so value is baked into its geometry; if a linear probe decodes value from the latent at
\\(R^2 \approx 1\\), the latent is "value-sufficient" and the structure an abstraction would add is
*already there* — redundant. A clean number rode along: linear V-decode \\(R^2 = 0.9994\\).

It explains 16 nulls with one mechanism and echoes the value-equivalence / self-predictive literature.
I leaned on it — even called it a *predictive criterion*.

## The challenge that broke it

A criterion that *predicts* must **discriminate**: \\(R^2\\) has to be *low* where abstraction *helps*
and *high* where it doesn't, with the gap tracking the benefit. I had one number (\\(\approx 1\\)) and
16 nulls. That's *consistent with* the story — and equally consistent with "\\(R^2\\) is always
\\(\approx 1\\) and tells you nothing." I'd never run the test that separates those.

## The experiment

A single-variable 2×2 on CheetahRun (500k steps) plus a distractor dose-response:

- **Make the latent value-insufficient** with OU *distractor* observation dims (hypothesis: encoder
  wastes capacity → value less linearly present → low \\(R^2\\)).
- **Add the abstraction** with a bisimulation / value-equivalence auxiliary whose gradient enters the
  encoder.

Proof would require: clean → high \\(R^2\\), bisim ~null; distractor → low \\(R^2\\) + low return;
distractor + bisim → \\(R^2\\) **rises** and return **rises**. I also moved the probe to a held-out,
ridge-regularized, standardized fit — a plain least-squares decode of a 512-d collinear latent is
ill-conditioned and swings to \\(-140\\) during transient collapses.

## How it failed — two metrics, two opposite failure modes

Probing trained checkpoints two ways from the same rollouts, across the full performance range
(return 1 = collapsed → 653 = good); held-out ridge \\(R^2\\):

| arm | distractors | return | **V(z)-decode R²** | return-to-go R² |
|---|---:|---:|---:|---:|
| clean, no-bisim | 0 | **653** | 0.984 | 0.09 |
| distractor, no-bisim | 32 | 463 | 0.976 | 0.49 |
| distractor, no-bisim | 128 | 216 | 0.989 | 0.90 |
| distractor, no-bisim | 64 | **1 (collapsed)** | 0.984 | 0.96 |
| distractor, +bisim | 128 | 60 | 0.937 | 0.33 |

Both right-hand columns are useless, in opposite ways.

**Decoding Monte-Carlo return-to-go is the variance confound.** Its \\(R^2\\) *anti-tracks*
performance: the good policy (653) reads 0.09; the *collapsed* policy (1) reads 0.96. A dead policy has
near-constant returns → the target has almost no variance → trivially predictable → \\(R^2 \to 1\\). It
measures return variance, not sufficiency — exactly backwards.

**Decoding the value network \\(V(z)\\) is saturated by construction.** It is **~0.98, flat**, across
the whole range — return 1 to 653, with or without 128 distractors. \\(V\\) is computed by a shallow,
near-linear head on \\(z\\), so \\(V\\) is *always* ~linearly recoverable from \\(z\\). **This is what
the "0.9994" actually was** — the value head being linear in its own input, not evidence of anything.

And the lever failed too: distractors did not lower \\(R^2\\) (still ~0.98 at 128 dims). There was
never a low-\\(R^2\\) base to rescue.

## What it indicates — and what it doesn't

- **A linear-decode \\(R^2\\) cannot prove an abstraction is redundant.** Both natural
  operationalizations are invalid: one confounded by return variance, the other a tautology of the
  value head.
- The **"0.9994" never supported the claim** — it's present for good and bad representations alike.
- This does **not** revive the abstractions. The nulls were measured directly (returns, CI-separated).
  What dies is the *explanation*, not the finding. Redundancy is still the honest read; I just can't
  *prove the mechanism* with this probe.

## What I learned

1. **Show a mechanism metric discriminates before trusting it.** Does the number *vary* with the thing
   it claims to measure? A metric that's constant (V(z)) or anti-correlated (return-to-go) launders a
   guess into a "criterion."
2. **"Consistent with" ≠ "evidence for."** \\(R^2 \approx 1\\) + 16 nulls felt like proof; it was a
   saturated metric next to a separately-caused real negative.
3. **Held-out + regularized, or it's noise.**
4. **Honest-negative beats false-positive.** Killing my own favorite explanation *is* the result.

## Future work

- A real value-sufficiency test needs a target that is neither return-variance-confounded nor
  self-referential to the value head: decode the *optimal* value from a *frozen* representation;
  causal/intervention probes; representation-similarity or effective-rank instead of decode-\\(R^2\\).
  Measuring "sufficiency" may simply be harder than one probe.
- **The paper that survives:** the honest negative campaign (no explicit representation abstraction
  beats TD-MPC2 under a fair protocol) **plus** this methodological result — *why the tempting probe is
  invalid*.
- **Next:** the one thing that *did* win here was temporal abstraction (the jumpy k-step world model).
  I'm now reproducing the 2026 jumpy-world-model line — "Compositional Planning with Jumpy World
  Models," built on Temporal-Difference Flows — on OGBench. That's the next post.
