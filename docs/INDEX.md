# Documentation index — TD-MPC-Glass

*Master map of the docs. Maintained 2026-06-10. If a doc disagrees with this index, this index + the
three canonical docs below win.*

## Read first (authoritative, current)
1. **[HANDOFF_NEXT_SESSION.md](HANDOFF_NEXT_SESSION.md)** — resume entry point: what's running now, what to do first (re-arm the autonomous loop), live blockers.
2. **[iterations/RESEARCH_LEDGER.md](iterations/RESEARCH_LEDGER.md)** — single source of truth for campaign verdicts (what worked / 13 nulls / next probes). Every number backed by run CSVs.
3. **[iterations/iteration_27_plan.md](iterations/iteration_27_plan.md)** — the live experiment spec (Panda re-benchmark + arch A/B), pre-registered gates.

## Setup & operations
- [../README.md](../README.md) — repo architecture, fleet, quick-start
- [../CLAUDE.md](../CLAUDE.md) — permanent guidance for agents (the one rule: EC2 never trains)
- [operations/launch_dashboard.md](operations/launch_dashboard.md) — control-plane runbook + queue cheat sheet (canonical ops doc)
- [operations/experiment_ops.md](operations/experiment_ops.md) — general experiment runbook
- [operations/env_setup.md](operations/env_setup.md) · [operations/hardware_req.md](operations/hardware_req.md) · [operations/storageAWS.md](operations/storageAWS.md) · [operations/data_corruption_fix.md](operations/data_corruption_fix.md) · [operations/fleet_rebuild_recovery_codex.md](operations/fleet_rebuild_recovery_codex.md)

## Scientific write-ups
- [writeup/capstone.md](writeup/capstone.md) — campaign overview ("8 mirages, 1 borrowed win")
- [writeup/draft.md](writeup/draft.md) — "Six Mirages" paper draft
- [blog/blog_phase1.md](blog/blog_phase1.md) — public Part 1 (HopperHop era)
- [blog/blog_phase2.md](blog/blog_phase2.md) — public Part 2 (mechanism-check saved a campaign; §8 = SimNorm structural-entropy result). **Canonical source for the published post.**

## Analysis (kept, still-relevant)
- [analysis/why-glass-failed-simnorm-redundancy.md](analysis/why-glass-failed-simnorm-redundancy.md) — why the Glass arm was redundant with SimNorm (the campaign's core retrospective)
- [analysis/SE_maths.md](analysis/SE_maths.md) · [analysis/cluster_to_gait_mapping.md](analysis/cluster_to_gait_mapping.md) · [analysis/mppi_vs_pi_analysis.md](analysis/mppi_vs_pi_analysis.md) · [analysis/experiment_runs_report.md](analysis/experiment_runs_report.md)

## Iteration ledger (one-line verdicts; full ledger = RESEARCH_LEDGER.md)
| iter | topic | verdict |
|---|---|---|
| 14 | Behavior-aware abstraction (geo/behav/bisim/distractor Glass) | **NULL** — redundant with SimNorm (mirages #1–6) |
| 15 | Plan in prototype space ("proto-plan") | **NULL** |
| 16 | FSQ codebook replaces SimNorm ("fsqmpc") | **NULL** (mirage #7) |
| 17 | Prototype-novelty exploration, sparse ("xnov") | **NULL** |
| 18 | Horizon-sweep gate ("hsweep") | superseded by jumpy |
| 19 | Community-detection skill discovery | **NULL** — communities = motion phases, not subgoals |
| 20 | Jumpy multi-step latent model | **groundwork for the win** |
| 21 | Abstraction-grounded / Laplacian exploration | **NULL** — RND ≥ it (mirage #8) |
| 22 | Horizon-consistent jumpy TD-MPC2 (HC-TDMPC) | ✅ **REAL WIN** on PandaPickCube (prior art: Farebrother 2026) |
| 23 | SE-k adaptive jump-length | **NULL** — boundary score doesn't track k-step error |
| 24 | SI2E / wmsi2e SE-driven exploration | **NULL** — ties RND (3rd exploration null) |
| 25 | Hermite-spline action bottleneck | lean-negative (mechanism-check); deferred |
| 26 | Value-equivalent macro head (probe #2) | **folded into iter-27** (the `ve` arm) |
| 27 | **Manipulation re-benchmark + arch A/B** | 🔴 **ACTIVE** — does jumpy generalize across Franka? does value-equiv / attn / resmlp help? |

## Archive (historical — not live; kept for provenance)
- `archive/hopperhop-iterations/` — iters 1–13 (HopperHop basin-entry era; apparent wins later shown to be basin-lottery, not Glass)
- `archive/research-syntheses/` — early deep-research syntheses (iter-14/21) and prompts
- `archive/operations-superseded/` — old dashboard/launch guides (merged into launch_dashboard.md)
- `archive/design-hopperhop/` — HopperHop ideation (idea_list, roadmap, todo_plan)
- `archive/WRAP_UP_2026-06-08.md`, `archive/why_not_beating_official.md` — superseded by RESEARCH_LEDGER
- `../AGENT_HANDOFF_CONTEXT.md` — **OBSOLETE (2026-05-13)**, superseded by HANDOFF_NEXT_SESSION.md
