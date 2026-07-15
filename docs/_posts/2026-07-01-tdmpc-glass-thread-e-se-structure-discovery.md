---
layout: post
title: "Thread E — Structural Entropy for Structure Discovery, Not Regularization (living log)"
date: 2026-07-01
description: "We used structural entropy as a latent regularizer — redundant, and the wrong bias for continuous geometry. SE's real strength is community/hierarchy detection: subgoal/option discovery and hierarchical planning. This thread repositions 'glass' where SE actually belongs, and merges into A3."
---

> **Living log.** Context: [Part 5 method map](../2026-07-01-tdmpc-glass-part5-beat-ppo-reality-check),
> [Part 6 five bets](../2026-07-02-tdmpc-glass-part6-five-bets-next-phase),
> and its sibling [Thread A](../tdmpc-glass-thread-a-planning-exploration) (A3 is where this thread cashes out).

## The bet

We spent a lot of the campaign using structural entropy (SE) as a **latent regularizer** — pushing the world
model's latent toward a min-2D-SE community structure. Two threads' worth of results say that's the wrong job for
it:

- **Thread D** showed anti-collapse regularizers (uniformity, VICReg, SE) are redundant-to-harmful on
  value-anchored control, and that even on a pure JEPA, SE-as-regularizer is at best neutral.
- On continuous geometry specifically, a *community* bias is the wrong inductive bias — continuous control lives
  on smooth manifolds, not discrete communities.

But SE's **actual** strength is what it was designed for: **community and hierarchy detection**. The unclaimed,
untested uses are structural, not representational:

- **Subgoal / option discovery** — run min-2D SE on the replay-buffer latent \(k\)-NN graph; communities are
  abstract states, boundaries are bottleneck subgoals.
- **Encoding-tree as planning hierarchy** — the SE encoding tree gives a natural high-level / low-level split for
  hierarchical planning.
- **SE-guided exploration** — plan toward *under-visited* communities (this is exactly Thread A's A3).

**In one line:** SE belongs on the *graph of states*, not inside the *latent vector*. This thread is where
"glass/SE" lives after SE-as-representation proved a dead end.

## Status board

| item | state | verdict |
|---|---|---|
| SE-as-latent-regularizer (retrospective) | ✅ closed | redundant / wrong bias (see [Thread D](../tdmpc-glass-thread-d-jepa-anticollapse-done-right)) |
| SE-for-subgoal-discovery (A3) | ❌ **NULL** (2026-07-02, n=2, 3 sparse tasks) | SE-community subgoal shaping ties or hurts flat (Acrobot 460→382; CartpoleSwingupSparse 316→7 — hurts most exactly where it was meant to help; FingerTurnHard ~tie). See Part 8. |

## Progress log

### 2026-07-01 — repositioned; execution merges into A3
No standalone GPU run yet — this thread is a *reframing* plus a concrete plan that executes as Thread A's A3 arm.
The honest retrospective: every attempt to make SE earn its keep as a latent bias (Panda glass, DMControl glass,
the D-thread SE arms) came back redundant or an over-weighting artifact. The forward bet is to use SE for
**structure discovery** — replay-graph → SE communities → bottleneck subgoals → a high level that plans toward
under-visited communities. That is A3, and it's the one place the whole "glass" line can still contribute
something novel: SE at the {planning × abstraction × exploration} intersection, not as a regularizer.

**Dependency:** A3 is gated on Thread A's coverage machinery (the replay-graph + coverage logger already exist).
It runs after the A1 flagship de-risk resolves.
