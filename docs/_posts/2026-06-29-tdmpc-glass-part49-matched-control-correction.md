---
layout: post
title: "TD-MPC-Glass, Part 49: Matched-Control Correction — the \"beats PPO\" arc (Parts 42–48) was vs an UNDER-BUDGETED PPO"
date: 2026-06-29
description: "Integrity addendum. Parts 42–48 reported that a behavioral abstraction (analytic controller + learned residual, and the bootstrap-then-release curriculum) 'beats PPO' on several tasks. A later matched-control campaign showed those comparisons used an UNDER-BUDGETED benchmark PPO. Against a budget-matched vanilla PPO the picture changes: the abstraction NEVER raises the asymptotic ceiling — on FingerTurnHard and all five locomotion tasks budget-matched vanilla PPO WINS; on ReacherHard and CartpoleSwingupSparse it TIES; the only surviving win is SAMPLE-EFFICIENCY where the prior fits (ReacherHard ~3×, OpenCabinet ~7×). The honest reframe: abstraction is a sample-efficiency lever, not a better ceiling. This post does not delete or silently edit the earlier parts; it corrects the public record, and a one-line pointer to here was added to the top of each affected part. Numbers from SYNTHESIS_beat_ppo.md / the verdict JSONs only."
---

> **Correction / addendum.** Parts 42–48 framed a behavioral abstraction as
> *beating PPO* on several tasks. A subsequent **matched-control** campaign showed those
> wins were measured against an **under-budgeted benchmark PPO**. Against a **budget-matched
> vanilla PPO**, the abstraction wins only on **sample-efficiency**, never on the **asymptotic
> ceiling**. Per this project's verification discipline we do not delete or silently edit the
> earlier posts — we leave them standing and correct the record here, with a one-line pointer
> added to the top of each affected part.

## What the earlier parts claimed

- **[Part 42]({{ site.baseurl }}/2026/06/25/tdmpc-glass-part42-curriculum-beats-ppo-opencabinet.html)** — abstraction-as-curriculum (lean bootstrap then release) "beats PPO" on PandaOpenCabinet (~2.5× sample-efficiency at matched success).
- **[Part 43]({{ site.baseurl }}/2026/06/26/tdmpc-glass-part43-residual-on-tamp-beats-ppo.html)** — residual-on-TAMP "beats PPO" on PandaOpenCabinet (~9–18× sample-efficiency, more stable than the curriculum).
- **[Part 44]({{ site.baseurl }}/2026/06/25/tdmpc-glass-part44-fair-dual-protocol-table.html)** — the dual-protocol bookkeeping table for the beat-PPO cells.
- **[Part 47]({{ site.baseurl }}/2026/06/25/tdmpc-glass-part47-acrobot-prior-backfires.html)** — the honest negative: the prior backfires on AcrobotSwingup (already a negative; restated here for completeness).
- **[Part 48]({{ site.baseurl }}/2026/06/26/tdmpc-glass-part48-dmc-benchmark-glass-vs-tdmpc2.html)** — the unified DMC table, which carried the "abstraction beats" cells (Pendulum) into one place.

The common thread: the PPO bar these were compared against was **not budget-matched**. On
OpenCabinet the cited PPO reached its success at 29.5M env-steps; on the DMC tasks the PPO
column was "PPO's best at a far larger budget" (50–285M steps) rather than a PPO trained under
the same conditions/budget as the abstraction. That makes the *sample-efficiency* comparison
informative but makes any *asymptotic / ceiling* "beat" claim unsupported.

## The correction: budget-matched results

Source: `wm-redundancy-paper/SYNTHESIS_beat_ppo.md` and the per-task verdict JSONs it cites
(deterministic, multi-seed, disk-backed). The control that changed the picture is a
**budget-matched vanilla PPO** — the same paradigm (controller + residual trained by PPO) vs a
plain PPO given the same budget.

| regime | asymptotic return (ceiling) | sample-efficiency |
|---|---|---|
| Reaching, fit member (ReacherHard) | **tie** (residual 981 vs vanilla 977) | **residual wins** (~3×: solved ~1M vs ~9.8M) |
| Reaching, unfit member (FingerTurnHard) | **vanilla PPO wins** (952 vs 923) | no advantage |
| Locomotion (all 5 tasks, n=5) | **vanilla PPO wins all 5** (e.g. Cheetah 914 vs 767; 3/5 CI-separated) | no advantage |
| Sparse swing-up (CartpoleSwingupSparse) | **tie** (both solve) | residual ~2× faster |
| OpenCabinet (hard manipulation) | **tie** (0.987 vs 0.988) | **residual wins ~7×** |

**Pareto experiment: NULL.** No TD-MPC2-plus-abstraction variant Pareto-dominates PPO across
both return/sample-efficiency *and* wall-clock; adding temporal abstraction (jumpy) was
null-to-negative. The sample-efficiency ↔ wall-clock tradeoff is architectural, not a tuning gap.

## What survives, and what does not

- **Does NOT survive:** any claim that the behavioral abstraction *beats PPO on the ceiling /
  asymptotic return.* Against a budget-matched baseline it never raises the ceiling — it ties
  where the prior fits and **loses** where it does not (FingerTurnHard; all five locomotion tasks).
- **Survives (qualified):** the abstraction is a **sample-efficiency** lever, and only where the
  prior captures the task's control-relevant substructure (actuated-DOF ≈ goal-DOF). The two
  concrete survivors are **ReacherHard (~3×)** and **OpenCabinet (~7×)** sample-efficiency at
  matched final performance. CartpoleSwingupSparse is a softer ~2× at a tie.
- **Already honest, unchanged:** Part 47's AcrobotSwingup result — a weak/ill-fitting prior is an
  *anchor*, slower than scratch. The matched-control campaign is consistent with it.

## The honest conclusion (revised)

**Abstraction is a sample-efficiency lever, not a better ceiling.** A behavioral prior can lower
the *statistical / optimization* cost of reaching a competent policy when the baseline is
optimization-limited *and the prior fits the task's control structure* — that is a real and
useful win on env-step-scarce settings (real robots, costly sims). But it does **not** raise the
representational ceiling: a budget-matched strong baseline already finds a value-sufficient
solution, and where the prior does not fit, wiring it into the loop is dead weight or an anchor.

This is a narrower claim than Parts 42–48 made, and we are not over-correcting in the other
direction either: the sample-efficiency wins on ReacherHard and OpenCabinet are real, multi-seed,
and disk-backed. What changed is only the axis on which "beats PPO" is true.

### Provenance

- Corrected numbers: `wm-redundancy-paper/SYNTHESIS_beat_ppo.md` (synthesis of the 2026-06
  matched-control campaign) and the per-task verdict JSONs it cites.
- Affected posts (pointer added at top, original bodies unchanged): Parts 42, 43, 44, 47, 48.
- This post adds to — does not replace — the public record. The earlier parts stay up for
  provenance; read this correction alongside them.
