---
layout: post
title: "TD-MPC-Glass, Part 13: The R² Criterion That Couldn't — Why You Can't Probe Your Way to 'Redundant'"
date: 2026-06-17
description: "For months we explained a campaign of null abstraction results with a tidy story: the latent is already 'value-sufficient' — value decodes linearly from it at R²≈1 — so adding an abstraction that re-organizes toward value is redundant. A reviewer-style challenge forced the question: does that R² actually PROVE redundancy? We built the discrimination experiment to find out, and the answer is a clean no. Two different ways of measuring 'value-sufficiency by linear decode' both fail — one is confounded by return variance (it anti-correlates with performance), the other is saturated by construction (it reads ~0.98 whether the policy scores 653 or 1). The famous '0.9994' was an artifact. This is the post-mortem: how the criterion arose, exactly how it fails, what it does and doesn't indicate, and what an honest value-sufficiency test would have to look like."
---

> The redundancy criterion was the most-cited idea in this project — the one-line answer to "why does
> nothing beat TD-MPC2 at the abstraction level?" It turns out the *evidence* for it (a linear-decode
> R²) never actually measured what we said it measured. The negative result still stands; the
> explanation we attached to it does not. This is how that came apart, with receipts.

## 1. Where the criterion came from

The campaign's running result is brutal and consistent: **every explicit *representation* abstraction
we tried — across state, temporal, relational, and compositional axes, 16+ levers — is null** under a
fair single-variable protocol. Bisimulation, value-equivalence auxiliaries, self-predictive extras,
entity factoring, graph world models: all neutral-to-harmful versus vanilla TD-MPC2.

You need a *why*. The story we told was the **redundancy criterion**:

> TD-MPC2's SimNorm latent is trained with a value loss, so value is already *baked into* the latent's
> geometry. If a linear probe can decode the value from the latent at R²≈1, the latent is
> "value-sufficient" — the value structure an abstraction would add is *already there*. Hence
> redundant. A clean number was attached: linear V-decode **R² = 0.9994**.

It's a seductive story. It "explains" 16 nulls with one mechanism, and it sounds like the
value-equivalence / self-predictive literature (Grimm et al.; Ni et al.). We leaned on it hard — even
called it a *predictive criterion*.

## 2. The challenge that broke it

The right question, asked bluntly: **does a high R² actually prove an abstraction is redundant?**

A criterion that *predicts* must **discriminate**. To earn the word, R² has to be *low* somewhere that
abstraction *helps*, and *high* where it doesn't — and the gap has to track the benefit. We had never
shown that. We had one number (≈1) and 16 nulls. That's *consistent* with the story, but it's also
consistent with "R² is always ≈1 and tells you nothing." We never ran the test that distinguishes
those two worlds.

So we ran it.

## 3. The discrimination experiment

Single-variable 2×2 on CheetahRun (500k steps, multiple seeds), plus a distractor dose-response:

- **Make the latent value-*insufficient*** with `--distractor_dims D` — append `D` temporally-correlated
  (OU) nuisance dimensions to the observation. The hypothesis: the encoder wastes capacity on
  distractors → value becomes *less* linearly present → **low R²**.
- **Add the abstraction** with `--bisim_coef` (a bisimulation / value-equivalence auxiliary whose
  gradient flows into the encoder; it should reorganize the latent around behavioral/value distance).

The criterion would be **proven** iff: clean → high R², bisim ~null; distractor → low R² + low return;
distractor + bisim → **R² rises and return rises** (the causal chain). We also fixed the probe to a
held-out (80/20), ridge-regularized, feature-standardized linear regression — because a plain
least-squares fit on a 512-d, highly collinear SimNorm latent is ill-conditioned and produces
nonsense (we watched it swing to **−140** during transient training collapses).

## 4. How it failed — two metrics, two distinct failure modes

To diagnose cleanly we probed trained checkpoints two ways from the *same* rollouts, spanning the full
performance range (return 1 = collapsed → 653 = good). Held-out ridge R²:

| arm | distractors | return | **V(z)-decode R²** | return-to-go decode R² |
|---|---:|---:|---:|---:|
| clean, no-bisim   | 0   | **653** | 0.984 | 0.09 |
| distractor, no-bisim | 32  | 463 | 0.976 | 0.49 |
| distractor, no-bisim | 128 | 216 | 0.989 | 0.90 |
| distractor, no-bisim | 64  | **1 (collapsed)** | 0.984 | 0.96 |
| distractor, +bisim   | 128 | 60  | 0.937 | 0.33 |

Read those two right-hand columns carefully. They are *both* useless, in *opposite* ways.

**(a) Decoding the Monte-Carlo return-to-go is the variance confound.** Its R² *anti-tracks*
performance: the good policy (653) reads 0.09, the *collapsed* policy (1) reads 0.96. Why? A dead
policy produces near-constant returns → the return-to-go target has almost no variance → it's
trivially "predictable" → R² → 1. This metric measures *return variance*, not value-sufficiency. It is
exactly backwards as a quality signal. (This is the same confound we'd flagged and "dropped" before —
it quietly poisoned the entire discrimination matrix until we isolated it here.)

**(b) Decoding the value network V(z) is saturated by construction.** It is **~0.98, flat**, across the
entire range — return 1 collapsed to return 653, with or without 128 distractors. It does **not move**.
The reason is structural: V is computed by a shallow, near-linear head *on top of* z, so V is *always*
~linearly recoverable from z, no matter how good or bad the policy or how many distractors you add.
**This is what the "R² = 0.9994" actually was** — not evidence of value-sufficiency, just the value
head being linear in its own input. It carries zero information about whether an abstraction would
help.

And the **lever itself failed too**: distractors did not lower V(z)-decode R² (still ~0.98 with 128
nuisance dims) — TD-MPC2's representation learning just absorbs them. There was never a low-R² base to
rescue, so the causal chain could not even be set up. (Separately, the bisim auxiliary at coef 0.5 was
training-unstable; several runs died around 50k.)

## 5. What it indicates (and what it doesn't)

- **A linear-decode R² cannot prove an abstraction is redundant.** Period. Both natural
  operationalizations are invalid for the job: one is confounded by the policy's return variance, the
  other is a tautology of the value head's linearity.
- The **"0.9994" never supported the redundancy claim.** It was an artifact of architecture, present
  for good and bad representations alike.
- This does **not** resurrect the abstractions. The nulls are real and were measured directly (returns,
  CI-separated, fair protocol). What dies is the *mechanistic explanation we bolted on*, not the
  empirical finding. Redundancy is still the honest read of the data — we just can't *prove the
  mechanism* with this probe.

## 6. What we learned

1. **A mechanism metric must be shown to discriminate before you trust it.** The single most important
   check we skipped: does the number *vary* with the thing it claims to measure? A metric that's
   constant (V(z): always ~0.98) or anti-correlated (return-to-go) is worse than no metric — it
   *launders* a guess into a "criterion."
2. **"Consistent with" is not "evidence for."** R²≈1 + 16 nulls felt like proof. It was a coincidence
   of a saturated metric and a real (separately-caused) negative result.
3. **Held-out + regularized or it's noise.** In-sample R² on a 512-d collinear latent overfits to ~1;
   our held-out ridge probe is what exposed the −140 swings and the true picture.
4. **Honest-negative beats false-positive.** Killing our own favorite explanation is the result. The
   project's recurring discipline — read from JSON, report peak *and* final, mechanism-check before
   fan-out — is exactly what caught this.

## 7. Future work

- **What a real value-sufficiency test would need:** a target that is neither return-variance-confounded
  nor self-referential to the value head. Candidates worth exploring: decoding the *optimal* value
  (not the learned head's output) from a *frozen* representation; intervention/causal probes (does
  perturbing value-irrelevant latent directions change control?); representation-similarity (CKA) or
  effective-rank arguments rather than decode-R². None of these is obviously clean — measuring
  "sufficiency" may simply be harder than a single probe.
- **The paper that survives:** the honest negative campaign — *no explicit representation abstraction
  beats TD-MPC2 under a fair protocol* — plus this methodological result: *why the tempting
  value-sufficiency probe is invalid*. The "why the obvious metric fails" is itself a contribution.
- **Where the energy goes next:** the one technique that *did* deliver real gains in this project was
  **temporal abstraction** (the jumpy k-step world model, Part 12). So we're now reproducing the 2026
  jumpy-world-model line directly — "Compositional Planning with Jumpy World Models" (built on
  Temporal-Difference Flows) — on OGBench. That's the next post.

> Receipts: probe data in `exp/tdmpc_glass/paperA/vzprobe/*.json`; discrimination matrix in
> `exp/tdmpc_glass/paperA/`; verdict in `docs/iterations/paperA_criterion_VERDICT_2026-06-17.md`.
