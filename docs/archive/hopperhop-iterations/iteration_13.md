# TD-MPC-Glass Iteration 13 — Population-Based Training (PBT) for basin-entry robustness

Created: 2026-06-03 · CODE_SHA `4d3b935` (pinned, clean) · Claude Opus 4.8 on the EC2 control plane.

## Why PBT (carry-over from iteration 12)

Iteration 12 established that **restart-on-plateau is the best basin-entry lever found**
(phasei12b: mean 422, ~40% G1 — beats the K256 baseline 362.1/20% on both mean and G1
rate) but it **does not clear the strict ≥3/5 (60%) bar**. The visible failure mode:
restart rescues *some* stuck seeds (12.2→586, 277→550) but **~half the re-rolls re-plateau**
because each laggard re-initialises its *own* (often unlucky) actor and has to re-find the
hopping basin from scratch.

PBT attacks exactly that failure mode: instead of re-rolling its own actor, a stuck member
**inherits a basin-finder's full trained model** (exploit) and continues with a fresh RNG
seed (explore). Since nearly every run produces a basin-finder, PBT should *propagate* it to
the laggards and lift the population G1 rate toward ≥3/5.

## Mechanism: orchestrator-level truncation PBT (no `src/` changes)

`control/pbt_orchestrator.py` runs on the EC2 control plane and OWNS a pool of homogeneous
A4000 boxes (removed from `task_queue_daemon.BOXES` so the queue daemon doesn't fight it).
It exploits `run_benchmark.py`'s existing `--resume_checkpoint`, which restores
`params + target_params + opt_state + scale + glass_step + env_steps + replay_buffer` — a
full trained model. So a laggard resuming a basin-finder's `best_any.pkl` literally inherits
its policy. No algorithm/code change → clean provenance (CODE_SHA 4d3b935).

- **Members**: 5 (one per pool box), each `tdmpc-glass` off@1M (N32/K8, the iteration-11/12
  best base), seeds 100–104, 10M steps, output tag `phasei13pbt_<box>_g<gen>_4d3b935`.
- **PBT step** (every `INTERVAL_S=3600`s): read each member's `best_any` from its box CSV; rank.
- **Exploit+explore**: each bottom-30% member whose `best < top_best − MARGIN(80)` and is past
  `MIN_STEP_EXPLOIT(1.5M)` copies a top member's `best_any.pkl` (donor box → EC2 relay →
  laggard box as `pbt_inherit.pkl`), is killed, and relaunched with `--resume_checkpoint`
  + `seed += 1000` (explore). Donor must itself be ≥ G1−50. Top/mid members run uninterrupted.
- **Clean-claim discipline**: exploit uses only **in-population** top members — NO injection of
  the external iteration-12 basin-finder checkpoints. Those (550, 501) are preserved at
  `exp/tdmpc_glass/basin_checkpoints/` as artifacts/fallback only.
- **Crash-resurrection**: any member not alive and `< TOTAL_STEPS` for 2 consecutive hourly
  checks is relaunched (resume own `latest_eval.pkl`), `pkill` first to prevent the
  duplicate-launch CSV-corruption mode. Keeps GPUs from idling unattended.
- **Stop**: all members reach 10M, or `MAX_WALL_S=30h`.

## Control arm (apples-to-apples)

The queue daemon's boxes run fresh **phasei12b Glass+restart CONTROL seeds** (independent
restart, NO PBT exchange) — the baseline PBT must beat. Plus the two iteration-12 G1
basin-finders (seed3=501, seed4=550) are **left running** to completion (productive G1 runs;
not killed). seed10 (442, climbing at 1.75M) was spared from the kill and continues.

## Launch record (2026-06-03 ~17:45Z)

- PBT pool (orchestrator-owned): ssh1_a4000, ssh2_a4000, ssh3_a4000, ssh4_a4000, ssh4_a4000b.
- 5 non-G1 iteration-12 laggards stopped (user-authorized) to free the pool: seeds 9,1,11,5,6.
- Orchestrator launched gen-0 (all 5 healthy: JAX/MuJoCo on cuda:0, JIT ~40s, training).
- First PBT exploit step fires ~1h after launch; first exploits possible once a member
  passes 1.5M and another reaches the basin.

## Result (2026-06-04, ~13 hourly PBT steps in) — VERIFIED, with an honest causation caveat

Independently re-read from each box's eval CSV (MPPI metric, not the orchestrator's cache):

| member | seed | best_any (mppi) | step | status |
|---|---|---:|---:|---|
| pbt_ssh4_a4000  | 103 | **571.2** | 2.75M | G1 ✓ |
| pbt_ssh4_a4000b | 104 | **545.2** | 5.75M | G1 ✓ |
| pbt_ssh2_a4000  | 101 | **529.1** | 5.25M | G1 ✓ |
| pbt_ssh1_a4000  | 100 | 366.1 | 6.0M | laggard |
| pbt_ssh3_a4000  | 102 | (lost) | — | instance recycled ~step 3 |

**G1 = 3/5 — clears the ≥3/5 robustness bar, verified.** BUT this batch does **NOT** demonstrate
the PBT *mechanism*: the three G1 members reached the basin by **independent** from-scratch
basin-finding of the Glass off@1M base. The only exploit that fired (step 3) targeted ssh3,
which then died (vast.ai recycled the instance). Worse, a **bug** let the dead ssh3
(best=−1) sit at the bottom of every ranking and monopolise the single exploit slot, so the
real live laggard (seed100=366) was **never** exploited. So this is the *base config* hitting
3/5 on a 5-seed draw (consistent with iter-12's ~40–60% band), not PBT propagation.

To honestly attribute a win to PBT, the mechanism must be shown converting a live laggard.
Two fixes applied (2026-06-04):
- **Exploit ranks LIVE members only** (best≥0 AND running); dead members go to the resurrect
  pass, not the exploit slot. Now the next step exploits seed100 (366) from a basin donor (571)
  — if seed100 climbs to G1, that is PBT demonstrably rescuing a stuck seed (→ 4/4 live).
- **Dropped lost ssh3** from the pool (publickey denied = recycled, unrecoverable).

Caveat to carry: 3/5 on one 5-seed draw is the bar but not yet *robust evidence*; the strong
claim needs either PBT visibly converting a laggard, or ≥3/5 replicated across more draws
(the phasei12b CONTROL seeds on the daemon are the independent-restart comparison).

## What to check next session

- `control/pbt_state.json` + `exp/tdmpc_glass/logs/daemons/pbt.log` — per-step G1 count and
  EXPLOIT/RESURRECT events. Target: `G1 ≥ 3/5` in the population (the robustness win).
- Compare PBT population G1 rate vs the phasei12b CONTROL seeds (same base, no exchange).
