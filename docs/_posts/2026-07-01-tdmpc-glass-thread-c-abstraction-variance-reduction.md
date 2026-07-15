---
layout: post
title: "Thread C — Abstraction as Variance-Reduction? (living log — NULL)"
date: 2026-07-01
description: "The glass (SE) latent beats TD-MPC2 on 6/16 DMControl tasks — a wash on the mean, but with lower seed-variance on several. Hypothesis: SE structure reduces collapse-seed variance even if it doesn't move the mean. Verdict: NULL. This log records the honest negative."
---

> **Living log.** Context: [Part 5 method map](../2026-07-01-tdmpc-glass-part5-beat-ppo-reality-check),
> [Part 6 five bets](../2026-07-02-tdmpc-glass-part6-five-bets-next-phase).

## The bet

On the full 16-task DMControl benchmark, the SE-structured "glass" latent beats plain TD-MPC2 on 6/16 tasks. The
means are small and both-ways — a wash. But on several tasks glass had visibly **lower seed-variance**
(e.g. `ReacherHard` 976±3 vs 883±151). The hypothesis worth a cheap check: maybe SE structure doesn't raise the
mean, it **reduces collapse-seed variance / improves worst-case** — "abstraction as variance-reduction." That
would be a defensible *section* (not a headline).

## Status board

| item | state | verdict |
|---|---|---|
| Per-seed analysis of the 6 win-tasks + out-of-sample across all 16 | ✅ done | **NULL** |

## Progress log

### 2026-07-02 — VERDICT: NULL (analysis-only, n=4–5)
Tested whether glass's 6/16 wins are a real worst-case/variance-reduction effect. They are not.

1. **Within the 6 win-tasks**, glass has lower seed-sd and higher worst-seed 6/6 — but this is **circular**: the
   tasks were selected *because* glass's mean is higher, which (with equal ceilings) is definitionally "TD-MPC2
   owned the unlucky seed there." Dropping the worst seed erases the win on 4/6 (only `FingerSpin`, `ReacherHard`
   survive).
2. **Out-of-sample across all 16 tasks**, the effect reverses: mean seed-sd is glass **123.2 > TD-MPC2 114.5**
   (glass has *higher* variance overall); glass is lower-sd on only 8/16 and better-worst-seed on 9/16
   (coin-flips). Glass has its *own* collapses that TD-MPC2 avoids: `HopperHop` 0 vs 179, `HopperStand` seed=15
   vs 858, `ReacherEasy` ±89 vs ±3, `WalkerWalk` ±81 vs ±12.
3. Cross-algo eff_rank correlation is **unverifiable** — no TD-MPC2 latent diagnostics were saved to disk.

**Conclusion:** glass ≈ TD-MPC2 is a genuine **wash**; the 6 wins are small-mean, both-ways scatter from unlucky
seeds. **No publishable variance-reduction section.** This reinforces the broader redundancy result — an extra
relational bias on an already-value-sufficient latent buys nothing on continuous control. Recorded honestly and
closed. `b3060:exp/proposal_C_variance/{VERDICT.md,per_seed_data.json}`.
