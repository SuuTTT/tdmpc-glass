---
layout: post
title: "TD-MPC-Glass, Part 10: The First GO — Graph Latents Generalize Across Object Counts"
date: 2026-06-14
description: "After ~18 nulls across both abstraction axes, the campaign's first positive mechanism-check — and the honest fine print. On a contact-rich multi-object world, a graph (entity-factored) latent's value-decodability generalizes to held-out object counts where a monolithic latent collapses (OOD R² 0.57 vs 0.21). A fair zero-padded monolithic baseline halves the gap (0.35→0.17) — so half the win was a pooling artifact — but the residual permutation-equivariance advantage still clears the pre-registered 0.15 bar across seeds. Caveats: it's representation-level (a linear value-probe, R²~0.57 modest), it is NOT about contacts (the contact-prediction criterion failed), and control-benefit is untested. That last one is the real gate."
---

> Eighteen nulls in, the redundancy criterion has held against everything — clustering, SE,
> value-equivalence, calibration, pyramids, high-DoF. This post reports the first mechanism-check that
> came back GO. It is real, it survived its fairness control, and it is narrower and more honest than
> the headline number first suggested.

## The result

Contact-rich multi-disk world (elastic collisions, known contact graph, configurable object count).
Graph WM (entity-factored transformer) vs a param-matched monolithic WM. Pre-registered GO iff
monolithic OOD value-R² drops ≥0.15 below the graph's, OR graph predicts contacts better.

| baseline | graph OOD R² | monolithic OOD R² | gap |
|---|---|---|---|
| pooled monolithic (n=4 seeds) | 0.57 | 0.21 | **0.35** |
| **fair zero-padded monolithic (n≥2)** | 0.57 | 0.40 | **0.18** |

**GO** — graph value-decodability generalizes to held-out object counts; monolithic collapses
(N=5→9: 0.29→0.18 pooled). Crucially it **survives the fairness control**: a zero-padded monolithic
that keeps a slot per entity (lacking only permutation-equivariance) recovers a lot of ground — the
pooling *was* inflating the gap — but the residual graph advantage still clears 0.15 every seed
*(source: exp/tdmpc_glass/mechcheck/gwm_verdict_robust.json, gwm_simulator_*.json)*.

## The fine print (this is the honest part)

1. **Half the original win was a baseline artifact.** Pooling crushes N entities into 2 summary
   vectors; fixing that (pad) closed the gap from 0.35 to 0.18. We caught it because we ran the control
   before celebrating.
2. **It is NOT about contacts.** The contact-prediction criterion *failed* — the graph predicts
   contact-step dynamics no better than monolithic (ratio 1.01×; contacts are ~10× harder for both).
   So this is "graph generalizes across object **counts**," not "graph simulates **contacts** better."
   That reframes the win away from the survey's graph-as-*simulator* slice toward graph-as-*compositional-
   generalizer*.
3. **It is representation-level, not control.** R²≈0.57 from a linear value-probe is modest, and the
   campaign's entire lesson is that representation properties don't automatically become control wins.

## The real gate (next)

The pre-registered escalation question is now sharp: **does this OOD value-decodability advantage show
up in control return at held-out object counts?** Train a planner on both WMs at N_train, evaluate
return at held-out N. If yes — graph WMs earn a genuine compositional-control claim and the
ManiSkill/Robosuite escalation is justified. If no — it joins the redundancy story as "another
representation property that doesn't pay at the controller." Either way, the first GO has earned that
test.
