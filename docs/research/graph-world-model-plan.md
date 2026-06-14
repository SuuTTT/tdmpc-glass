# Graph World Models — look-back + the one live slice to test

*2026-06-14. Trigger: the GWM survey (arXiv:2604.27895, Liu et al., 30 Apr 2026) naming GWM as a
paradigm with the taxonomy connector / simulator / reasoner; method paper arXiv:2507.10539. This doc
maps that onto what THIS project already proved, and defines the cheap go/no-go before any build.*

## What we already know (don't re-run these)
The campaign already tested most of the GWM space, one level up, and it was redundant:
- **Graph-as-reasoner / connector on proprio control** — the redundancy criterion: a converged
  self-predictive latent (TD-MPC2/SimNorm) is value-decodable (R²=0.9994) and already carries its own
  interaction structure, so explicit graph/SE/clustering objectives are redundant (16 nulls).
- **Entity-graph value-coupling** (`entity_wm.py` + synthetic world + `value_coupling_probe.py`):
  even with a known sparse ground-truth coupling, the value-coupling graph recovered it at CHANCE and
  was beaten by a similarity graph — the instrument couldn't expose relational structure even when it
  existed by construction (NO-GO).
- **Compositional-OOD on the synthetic spring world**: monolithic value-decodability did NOT collapse
  OOD (R² 0.96→0.92/0.94), i.e. no headroom signal — graph latent redundant there too.

The survey's own honest read agrees: for DMControl/proprio, the graph prior is weak; the useful slice
is narrow.

## The one slice we have NOT fairly tested: **Graph-as-Simulator on contact-rich, compositional tasks**
All three deep-research reports and the survey converge: headroom for graph WMs exists (if anywhere)
in **relational manipulation with multiple movable objects, contact events, and compositional
(held-out object-count) generalization** — the SOLD / ObjectZero / Slot-MPC regime. Our prior
synthetic test used SMOOTH springs (no contact events, no OOD-collapse), so it could not surface that
headroom. **Contacts are the relational structure springs lacked.**

## The decisive go/no-go (running now, ssh7) — mechanism-check BEFORE any real-env build
On a controlled **contact-rich multi-disk world** (`contact_entities.py`: elastic collisions, walls,
known ground-truth contact graph, configurable object count), train the **graph WM** (`entity_wm`)
vs a **monolithic WM** (`monolithic_wm`, param-matched) and measure the two things the survey says
decide it — not raw prediction loss:
1. **Compositional-OOD value-decodability** — does monolithic R² *collapse* at held-out object counts
   where the graph latent holds? (the headroom regime)
2. **Contact-conditioned prediction** — does the graph WM have lower next-state error *specifically at
   contact timesteps*? (the relational-structure payoff)

**Pre-registered GO** iff monolithic OOD value-R² drops ≥0.15 below graph's, OR graph contact-step
error ≤0.8× monolithic's. **NO-GO** otherwise → graph latent is redundant even on contacts, and GWM
collapses the same way explicit abstraction did — fold into the paper as the relational-axis closure.

## If GO → escalate (the real cost decision, user's call)
Only then is a real benchmark worth the setup: **ManiSkill2 / Robosuite multi-block** (SOLD &
Slot-MPC's turf, PyTorch — a genuine integration cost against our JAX/MJX stack), graph/object-centric
dynamics with object-slot or known-object nodes + learned contact/interaction edges, baselines
TD-MPC2/DreamerV3 (monolithic) and SOLD (pixels) / Slot-MPC (offline MPC). Stage gates on
compositional-OOD control, ≥5 seeds, peak+final CI — same protocol as the whole campaign.

## Honest prior
~25–30%. The contact regime is the one untested place the survey/DRs flag, and contacts genuinely add
relational structure a monolithic latent may not factor — but every adjacent test we ran was null, and
published object-centric WMs mostly tie monolithic on control. The mechanism-check is one ssh7 run; it
either opens a real-env program or gives the paper its relational-axis closure. Either is a result.
