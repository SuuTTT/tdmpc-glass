---
layout: post
title: "TD-MPC-Glass, Part 25: Can We Beat PPO by Warm-Starting From the Abstraction? No — End-to-End PPO Wins (0.82), and Here's Why the Abstraction Won't Transfer"
date: 2026-06-23
description: "Part 24 gave the campaign's first positive abstraction result — a value-aware skill-options policy at 0.24 real success on PandaPickCube — but it plateaued well below PPO, and we argued the gap was a STRUCTURAL ceiling (the analytic option-controller), not a budget shortfall. Part 25 tests the obvious fix: relax the cap by warm-starting unconstrained RL from the abstraction's competence. Three arms, all evaluated on real success (box_target>=0.9, n=256): from-scratch PPO reaches 0.82 (crossing 0.66 at 32.8M); DAPG (demo-augmented PPO using the abstraction's 604k demos) reaches the SAME 0.82 but is LESS sample-efficient (0.66 at 39.3M); BC-warmstart fails outright (0.0). The structural cap is confirmed — unconstrained raw-action PPO blows past #1's 0.13 and #2's 0.24 to 0.82 — but warm-starting from the abstraction gives no win, because the option-controller's competence is NOT distillable into a raw-action policy (closed-loop-IK actions aren't BC-imitable). The honest answer to 'can we beat PPO with the abstraction?': no — end-to-end PPO is simply stronger here, and the abstraction's value is low-budget escape-from-zero, not peak."
---

> The goal was explicit: **beat PPO.** Part 24's value-aware skill-options abstraction (0.24 real success)
> was the campaign's first positive, but it sat far below PPO, and we claimed the gap was a *structural*
> ceiling — the hand-coded option-controller — not a lack of training. Part 25 tests the fix: use the
> abstraction as a launchpad for *unconstrained* end-to-end RL. Verdict: the cap is real and removable, but
> the abstraction doesn't help you remove it.

## The three arms (PandaPickCube, real success = box_target≥0.9, n=256, 1000-step)

| arm | what it is | peak success | crosses 0.66 at | final |
|---|---|---|---|---|
| **from-scratch PPO** | unconstrained raw-action RL | **0.820** | **32.8M** | 0.816 |
| **DAPG** (demo-aug PPO) | PPO + BC-aux loss on the abstraction's 604k demos | 0.816 | 39.3M | 0.805 |
| **BC-warmstart PPO** | init PPO from a BC clone of the abstraction | **0.000** | never | 0.000 |
| *(ladder for context)* | pure-TDMPC2 0 · HL 0.063 · #1 residual 0.13 · #2 skill-options 0.24 | | | |

## What it says

**1. The cap on the HL-hybrids WAS structural — and unconstrained RL removes it.** Raw-action PPO (both
scratch and DAPG) blows past #1's 0.13 and #2's 0.24 all the way to **0.82**. That confirms Part 24's claim:
the HL-hybrids were capped by the analytic option-controller's expressiveness (especially the place phase),
not by training budget. Take away the structural constraint — let the policy emit raw actions end-to-end —
and the ceiling jumps to 0.82. (Note this is *higher* than the 0.66 we'd been quoting; that figure was an
earlier shorter PPO run. Full PPO plateaus ~0.80–0.82.)

**2. But warm-starting from the abstraction gives no win.** This is the result we set out to get, and it's a
clean negative:
- **DAPG** (feeding the abstraction's 604k demos as a BC-auxiliary loss during PPO) reaches the *same* ~0.82
  ceiling as plain PPO, and is **slightly *less* sample-efficient** to the 0.66 threshold (39.3M vs 32.8M).
  The demo signal didn't speed anything up — if anything it mildly slowed convergence.
- **BC-warmstart** (clone the abstraction into a policy, then PPO-finetune) **fails outright (0.0)** — and
  earlier a naive version even NaN'd. The clone never learns; PPO from a broken init goes nowhere.

**3. Why the abstraction won't transfer (the mechanism).** The skill-options policy gets its competence from
the *analytic, closed-loop-IK option-controller* — its actions at each step depend on the controller's
internal phase state and on live feedback. That is **not imitable by a memoryless raw-action policy**: BC on
604k transitions hit low action error (MAE 0.12) yet produced **0.0 real success** (reached ≈ 0.004) — the
small per-step errors compound across the long horizon and the closed-loop corrections are gone. So the
abstraction's 0.24 is *locked to the option-controller*; it cannot be poured into the flat policy that
end-to-end RL needs as a launchpad.

## So — can we beat PPO?

**No, not with the abstraction.** On this task, **unconstrained end-to-end PPO is simply the strongest
method (0.82)**, and neither warm-starting nor demo-augmenting it from the abstraction beats it on peak *or*
on sample-efficiency. The abstraction's genuine value (Parts 22/24) is elsewhere: **fast escape from 0% at
low budget** (0.24 by ~20M ES steps, where pure TD-MPC2 is still flat 0), interpretability, and the
mechanistic lesson that *value-aware* abstraction works where predictive abstraction (jumpy/Glass) didn't.
For peak performance on a fast simulator, end-to-end RL wins — consistent with the whole campaign's
through-line (Part 18): world-models/abstractions buy sample-efficiency and structure, model-free RL buys
the ceiling and the wall-clock.

**The one lever we did not exhaust:** gradient fine-tuning of the *option policy itself* (differentiable MJX
rollout, or ES with progressively relaxed structure so it keeps the abstraction while gaining authority past
the place-phase cap). That keeps the abstraction in the loop instead of distilling it away, so it's the only
remaining route by which the abstraction could *contribute* to beating PPO. Everything tried here — clone,
demo-aug, warm-start — distills the abstraction *out*, and that's exactly why it doesn't help.

### Caveats (kept honest)
Single task (PandaPickCube), single-to-few seeds per arm, real success read from
`exp/tdmpc_glass/hl_beatppo/RESULTS.json` (n=256, 1000-step, the PPO-baseline protocol). from-scratch PPO
reproduced the baseline and is the reference. "PPO 0.66" in earlier parts was a shorter run; full PPO ~0.82.
The gradient-option-finetune lever is a hypothesis, not a result. Code/data:
`hl_pickcube/{make_options_demos,bc_warmstart}.py`, `baselines_ppo_sac/{dapg_patch,run_ppo_dapg}.py`,
`hl_beatppo/RESULTS.json`. Prior: Parts 22 (design rules), 24 (the abstraction result).
