# TD-MPC2 Shrink Study — Progress

_Budget: 1M env-steps/run (action_repeat=1), seed 1. Headline return = MPPI-plan eval (DMC max 1000)._


## Paper baseline (TD-MPC2 5M default, DMControl, Hansen et al. ICLR 2024)

Approx values read from paper learning-curve figures (no numeric table in paper). Paper budget=4M steps@action_repeat2. Compare OUR 1M-step return to paper's at_1M column.


| task | paper @1M | paper ~4M (asymptote) | note |
|---|---|---|---|
| CheetahRun | ~575 | ~875 | steady climb |
| WalkerRun | ~425 | ~800 | accelerates 1M-4M |
| AcrobotSwingup | ~375 | ~420 | converges early, ~400-450 |
| FingerSpin | ~900 | ~975 | fast-converging, near-ceiling |
| HopperHop | ~175 | ~300 | hard, slow |
| PendulumSwingup | ~750 | ~800 | converges |
| FingerTurnHard | ~450 | ~850 | hard exploration |

## Our shrink matrix (1M steps)

| config | params | sps | speedup | peak ret | final ret | (pi pk/fn) | paper@1M | matches? | step |
|---|---|---|---|---|---|---|---|---|---|
| CheetahRun/default | 3.61M | 272 | 1.0x | 645.8 | 645.8 | 614.1/611.6 | ~575 | YES (112%) | 1,000,192 |
| CheetahRun/small | 0.95M | 409 | 1.5x | 519.3 | 465.8 | 532.7/488.8 | ~575 | YES (90%) | 1,000,192 |
| CheetahRun/tiny | 0.72M | 437 | 1.61x | 512.1 | 509.5 | 511.2/497.4 | ~575 | YES (89%) | 1,000,192 |
| CheetahRun/cheapplan | 3.61M | 293 | 1.08x | 681.8 | 681.8 | 654.3/654.3 | ~575 | YES (119%) | 1,000,192 |
| WalkerRun/default | 3.61M | 165 | 1.0x | 683.7 | 683.7 | 669.1/668.8 | ~425 | YES (161%) | 1,000,192 |
| WalkerRun/small | 0.95M | 205 | 1.24x | 655.7 | 634.6 | 644.2/631.6 | ~425 | YES (154%) | 1,000,192 |
| WalkerRun/tiny | 0.72M | 155 | 0.94x | 645.4 | 641.4 | 606.9/590.2 | ~425 | YES (152%) | 1,000,192 |
| AcrobotSwingup/default | 3.59M | 224 | 1.0x | 407.8 | 406.2 | 420.6/294.8 | ~375 | YES (109%) | 1,000,192 |
| AcrobotSwingup/small | 0.94M | 382 | 1.71x | 403.4 | 389.0 | 395.1/337.3 | ~375 | YES (108%) | 1,000,192 |
| AcrobotSwingup/tiny | 0.71M | 359 | 1.6x | 457.8 | 413.7 | 418.3/371.5 | ~375 | YES (122%) | 1,000,192 |
