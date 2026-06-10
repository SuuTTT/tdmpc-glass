# FINAL synthesis (6 sources) + decision — 2026-06-08
*My 3 agents (Laplacian / MI-metric / model-based×abstraction) + Claude DR + GPT DR + Gemini DR.*

## The convergent verdict
**The top recommendation flipped from what I was building.** All six sources agree on the
landscape; on the *method to build next*, the weight of evidence is:

1. **JUMPY / k-step "horizon-consistent" world model — the lead bet (Claude #1, Gemini #1, GPT
   #2, and my model-based agent's jumpy thread).** Learn z_{t+k}=d(z_t, a_{t:t+k-1}, k) with a
   **horizon-consistency loss** (k-step prediction == iterated shorter preds) so MPPI plans long
   *effective* horizons with FEW model applications → no compounding error. This **directly fixes
   the exact failure we found** (H9 helps sparse / collapses Panda) and **keeps SimNorm latent →
   dodges the proprio null**. Strong recent precedent: **Jumpy World Models (Farebrother 2026):
   +200% over 1-step on long-horizon; SPlaTES (RLC 2025): distilling hierarchy into flat TD-MPC2
   → myopic; THICK (ICLR 2024)**. Lowest null-risk, most publishable. **This vindicates the
   iter-20 jumpy idea I had parked for the rho lever — it was the right idea, wrong (rho) shortcut.**

2. **Skills+MPPI (METRA/LSD) — GPT #1 but de-weighted.** Claude & Gemini rank it below jumpy:
   skills improve coverage/zero-shot/transfer, NOT task return; and they're **transient in
   continuous control — the same "motion phases not subgoals" wall we already hit in iter-19**.
   SkiMo is the one positive (skills+model beats flat on sparse maze/kitchen) but needs a
   skill-dynamics model = high-variance on precision manipulation. Hold as a later option.

3. **Laplacian/DCEO eigen-exploration (my current iter-21 Laplacian arm) — repriced to
   HIGH-RISK by ALL sources.** Zero continuous-ACTION precedent; online auto-discovery "deviates
   and hinders learning" (Kotamreddy & Machado 2025); bottleneck options can hurt locomotion. It
   is the high-novelty *fallback*, not the lead. RND (the baseline arm) is fine to keep as the
   exploration control.

## Unanimous reframe (all 6)
"Beat TD-MPC2 with abstraction" is the wrong goal for DENSE proprio (TD-MPC2 already a sufficient
self-predictive abstraction — Ni et al. — explains our null). The credible, reviewer-grade
contribution = **a unified world model that plans long horizons WITHOUT compounding error**
(precision on manipulation + reach on sparse), i.e. TEMPORAL abstraction for credit-assignment/
exploration, NOT representational re-abstraction.

## DECISION — steer + next
**STEER current run (iter-21):** keep it, scoped down. RND + Laplacian on the sparse suite still
answers "does exploration rescue sparse TD-MPC2?" (RND = clean control; Laplacian = the
now-confirmed high-risk novelty). Let the queued runs finish; do NOT expand Laplacian. This is
the *exploration* half and a useful baseline for whatever wins.

**NEXT = iter-22: Horizon-Consistent Jumpy TD-MPC2 (the lead bet).** Build per Gemini's blueprint
+ Claude's gates:
- k-step jumpy latent head d(z, a_{1:k}, k) over the EXISTING SimNorm latent; horizon-consistency
  loss (align k vs iterated); multi-step reward/value targets; InfoNCE anti-collapse.
- MPPI plans macro-steps (k∈{1,2,4,8} static buckets; associative-scan rollout); effective
  horizon Nk≈9-15 with N applies.
- Tasks: sparse {CartpoleSwingupSparse, BallInCup, AcrobotSwingupSparse} + **PandaPickCube as the
  guardrail** (the H9-collapse fix is the headline).
- PRE-REGISTERED KILL GATES (adopt Gemini's): (1) compile >3x or SPS <50% vanilla → kill;
  (2) Panda IQM collapses (<~90% vanilla / Gemini's <1500 proxy) → kill (jumpy inaccurate);
  (3) sparse IQM fails to exceed vanilla CI on ≥2/3 → kill. Mechanism check: k-step latent error
  < iterated 1-step error (else gain is confounded). ≥5 seeds, IQM+bootstrap CI, compute-matched
  (equalize gradient steps AND wall-clock, not just env steps).
- Honest prior (consensus): modest-but-real win plausible on sparse/long-horizon; Panda the
  binding constraint; jumpy is the bet least likely to be a 3rd null and most likely publishable.

**Demote:** rho+H = tuning footnote (done). Skills+MPPI and DCEO-eigen = ranked fallbacks if
jumpy's go/no-go fails.

Key new cites to pull: Jumpy World Models (Farebrother et al. 2026); SPlaTES (Gürtler & Martius,
RLC 2025); THICK (Gumbsch et al., ICLR 2024); SkiMo (CoRL 2022); Temporal Difference Flows;
Puppeteer (ICLR 2025, the "hierarchy doesn't beat flat TD-MPC2 on return" datapoint).
