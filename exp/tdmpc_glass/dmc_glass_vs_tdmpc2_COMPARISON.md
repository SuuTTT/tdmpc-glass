# DMC benchmark: tdmpc-glass vs tdmpc2 (in-progress harvest)

Source: full_sweep (b3060, seed-1, glass-first) + full_sweep_b (b3060b, seed-2, tdmpc2-first).
1M-step return, read from exp/benchmark/*.csv. NOTE: glass cells = seed-1, tdmpc2 cells = seed-2
(different seeds until each box finishes its 2nd algo round → then same-seed + 2-seed available).
Status as of ~20:50 2026-06-25. NOT yet published — anomalies pending cross-seed verification.

## Both-converged @1M (clean)
| task | tdmpc-glass (s1) | tdmpc2 (s2) | note |
|---|---|---|---|
| CheetahRun     | 652.4 | 646.4 | ~tie |
| FingerSpin     | 962.6 | 707.4 | glass > |
| FingerTurnEasy | 977.4 | 977.6 | tie |
| FingerTurnHard | 585.8 | 785.0 | tdmpc2 > |
| HopperHop      | 0.17  | 255.4 | ⚠ glass collapse (ANOMALY, verify) |
| HopperStand    | 355.1 | 900.8 | tdmpc2 ≫ |
| PendulumSwingup| 742.8 | 0.0   | ⚠ tdmpc2 collapse (ANOMALY, likely divergence, verify) |
| ReacherEasy    | 779.2 | 972.4 | tdmpc2 > |
| ReacherHard    | 973.4 | 972.0 | tie |
| WalkerRun      | 664.6 | 651.2 | ~tie |
| WalkerStand    | 975.5 | 972.2 | tie |

## Pending (one side not yet @1M)
AcrobotSwingup (glass 312@1M / tdmpc2 260@250k), BallInCup (glass 973@1M / tdmpc2 0.0@400k early),
CartpoleBalance (982 / 952@500k), CartpoleSwingup (857 / 865@850k), WalkerWalk (400@900k climbing / 975).

## Emerging story (UNVERIFIED until anomalies cross-checked)
tdmpc2 (established baseline) ≥ tdmpc-glass on most DMC tasks; ties on saturated tasks; glass wins FingerSpin.
Two single-seed catastrophic failures (glass→HopperHop, tdmpc2→PendulumSwingup) MUST be cross-seed verified
(b3060 tdmpc2 round will give tdmpc2-Pendulum-s1 & tdmpc2-HopperHop-s1; b3060b glass round → glass-HopperHop-s2)
before any are stated as findings — a 0.0 is most likely a divergence/NaN, not a real capability gap.
