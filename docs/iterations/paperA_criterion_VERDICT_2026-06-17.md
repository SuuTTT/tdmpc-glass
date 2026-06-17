# Paper-A "R² redundancy criterion" — VERDICT: not operationalizable (2026-06-17)

## Question
Can a linear-decode R² prove a representation is "value-sufficient" (and thus that an added
abstraction is redundant)? The user demanded the existence proof: R² low → add abstraction →
R² high AND return high. We ran the discrimination matrix to test it.

## Two candidate R² metrics — BOTH fail, for distinct, demonstrated reasons

Held-out ridge linear probe from latent z, on CheetahRun checkpoints spanning the full
performance range (return 1 = collapsed → 653 = good). Data: `exp/tdmpc_glass/paperA/vzprobe/*.json`.

| arm | distractors | return | **V(z)-decode R²** | return-to-go R² |
|---|---|---|---|---|
| cleanNB | 0 | 653 | 0.984 | 0.09 |
| distNB | 32 | 463 | 0.976 | 0.49 |
| distNB128 | 128 | 216 | 0.989 | 0.90 |
| distNB64 | 64 | **1 (collapsed)** | 0.984 | 0.96 |
| distB128 | 128 | 60 | 0.937 | 0.33 |

1. **return-to-go decode is the variance confound:** R² ANTI-tracks performance (ret 653→0.09,
   ret 1→0.96). A collapsed policy has ~constant returns → trivially "decodable" → high R². It
   measures return variance, not value-sufficiency. (This is the confound flagged & "dropped"
   sessions ago — it fully poisons the discrimination matrix.)
2. **V(z) decode is uniformly ~0.98, flat across return 1→653:** it does NOT discriminate at all.
   The value head is near-linear in z by construction, so V is ALWAYS linearly present regardless
   of policy quality or distractors. The old "linear V-decode R²=0.9994" numbers are exactly this
   by-construction artifact — they never proved value-sufficiency.

## Also: the value-insufficiency LEVER failed
Distractors do not make the latent value-insufficient — V(z)-decode stayed ~0.98 with up to 128
OU distractor dims. TD-MPC2's representation just absorbs them. So there was no low-R² base to
rescue, and (separately) bisim-coef=0.5 runs were unstable (several died ~50k).

## Verdict
**The R² criterion cannot be operationalized.** Neither metric measures "value-sufficiency" in a
way that discriminates regimes. This definitively answers the user's session-opening challenge:
**a linear-decode R² does NOT prove abstraction is redundant.** The real evidence for redundancy
remains the empirical null campaign (16+ abstraction levers fail); R² was never valid support.

## Implication for Paper A
Drop the "predictive R² criterion" framing. The defensible paper is the **honest negative**: the
null campaign + the methodological finding *why the obvious value-sufficiency probe is invalid*
(variance-confounded one way, by-construction-saturated the other). That "why the tempting metric
fails" result is a genuine contribution.

## Fleet
Self-refill loop + watchdog cron STOPPED (experiment concluded). In-flight seed-3 arms left to
finish naturally. GPUs then free for GHM repro (`docs/ghm_repro/PLAN.md`) or release.
