---
layout: post
title: "TD-MPC-Glass, Part 11: The Graph Win Doesn't Reach the Controller"
date: 2026-06-14
description: "The first GO, gated honestly. The graph latent's compositional-OOD value-decodability advantage (Part 10) was put to the real test — does it convert to control return at held-out object counts? Random-shooting MPC on the learned world models: graph, fair monolithic, and random actions all land in a dead heat at the random floor (graph−random within ±13, std ~110, every N). The OOD representation advantage does NOT reach the controller. The redundancy criterion now spans all three axes — state, temporal, and relational/graph — with one honest positive nuance that stays at the representation level."
---

> Part 10 reported the campaign's first GO: a graph latent whose value-decodability generalizes across
> object counts. We said the real gate was whether that converts to control. It doesn't.

## The control-benefit test

Random-shooting MPC (512 samples × horizon 8, receding) planned with each learned world model;
true-env return over 20 episodes × 4 seeds, at the training object count and held-out counts. The
**random-action floor** is the can't-plan baseline.

| N | regime | graph | monolithic (fair) | random | graph − random |
|---|---|---|---|---|---|
| 5 | in-dist | −374.0 | −368.3 | −360.7 | −13.3 |
| 7 | **OOD** | −367.9 | −377.3 | −364.9 | −3.0 |
| 9 | **OOD** | −386.0 | −394.9 | −384.1 | −1.9 |

*(source: exp/tdmpc_glass/mechcheck/gwm_control_benefit_s{0,1,3}.json; std ~110.)*

**Graph ≈ monolithic ≈ random, at every object count.** MPC on either learned WM is no better than
acting randomly. The graph latent's OOD value-decodability advantage (real, robust, Part 10) buys
**zero** control benefit. (The per-seed `CONTROL-GO` flags the script emitted were a floor-noise
artifact — a floor-anchored relative gain explodes when the monolithic baseline itself sits at the
random floor; the honest aggregate is the table above.)

## Honest caveat

Random-shooting MPC plans on the model's reward head, and these WMs were trained on a small budget —
so "both at the floor" partly means "neither WM is accurate enough to plan with here," not purely
"graph doesn't help control." A heavier planner or much longer training could differ. But after a
fortnight in which *every* representation-level advantage failed to reach the controller, the prior is
overwhelming and chasing it further is low-EV.

## Where the program lands

- **Representation axis:** graph latents give a genuine, fair-control-surviving compositional-OOD
  value-decodability advantage (Part 10). Worth one paragraph in the paper.
- **Control axis:** it does not convert — consistent with the redundancy criterion, now demonstrated
  across **state** (16 nulls), **temporal** (5 levers + high-DoF), and **relational/graph** (this) axes.

The criterion is the result. The graph direction closes the same way explicit abstraction did:
*a converged self-predictive world model is already a sufficient, value-aligned abstraction; explicit
structure — clustered, temporal, or graph — is redundant at the controller.* No ManiSkill escalation
is warranted on this evidence. The paper is the deliverable.
