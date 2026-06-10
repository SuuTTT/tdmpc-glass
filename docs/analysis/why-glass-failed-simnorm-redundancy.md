# Why the initial TD-MPC-Glass (iters 8–9) failed — and whether it was all replicating SimNorm

*2026-06-10 retrospective. Companion to the iter-8/9 blog post. Verdict: yes, largely — the part of Glass
that helped was redundant with SimNorm + the self-predictive loss; the part that wasn't redundant captured
real-but-control-irrelevant structure; and the apparent win was procedure-confounded (basin lottery).*

## 1. What iters 8–9 actually were
Glass = a structural-entropy auxiliary on TD-MPC2's latent: soft-assign each latent to K prototypes
(`proto_soft_assign`), build a KxK prototype **transition graph** (`glass_transition_graph`), and minimize
its **2-D structural entropy** (+ balance + temporal-stability terms) so the latent dynamics form a few
"coherent behavioural regions." iters 8–9 added stability losses (temporal-stability, MPPI-gated distill)
— all missed — and landed on a **handoff recipe**: run Glass early, then decay λ_Glass→0 around 1M steps.
Best family `phasei9r` (Glass off@1M) ≈ 524 vs internal TD-MPC2 ≈ 389 on HopperHop — but with a wide
baseline CI and a clean rerun (`phasei10c` ≈ 316) NOT confirming it.

## 2. What SimNorm already is (the crux)
`simnorm(x, V)` = **partition the latent into V groups and softmax each group**. So a TD-MPC2 latent is,
by construction, **V soft-categorical assignments** — i.e. the encoder output is *already a soft clustering*
(V parallel soft codebooks). And TD-MPC2's **self-predictive consistency loss** already shapes that latent
so transitions are smooth/predictable. Together: SimNorm gives per-state soft-clusters; consistency gives
temporally-coherent dynamics. Theory (Ni et al. 2024) calls this objective a *sufficient* self-predictive
abstraction.

## 3. Decompose Glass into "what it adds" — both pieces fail
Glass's pressure splits into two parts:

**(a) Geometric clustering of the latent (prototypes) — REDUNDANT with SimNorm.**
Re-clustering a latent that is already a soft-categorical simplex code is re-deriving structure the encoder
already has. Direct evidence (the iter-23 pre-check, measured on a *trained* latent): SimNorm latents show
a **53% structural-entropy gap** (k-step transition graph) / **47%** (kNN) once you sparsify — the
community structure Glass tries to *impose* is **already present**. Mirage #1/#2 (geometric & behavioral
Glass) were exactly null at adequate n: nothing to add.

**(b) Transition-graph structural entropy (temporal/relational) — NOT redundant, but control-IRRELEVANT.**
This part is genuinely beyond per-state SimNorm: it shapes the KxK *transition* graph. But what structure
does it find? The iter-19/23 follow-ups answered it: the communities are **motion phases** (swing/stance/
contact arcs of a limit cycle), not reachable subgoals or error-prone regions. So the one non-redundant
thing Glass adds is real structure that **doesn't align with anything control needs** (Q-value precision,
contact handling). Useful for *describing* the latent, useless for *improving* it.

## 4. Four pieces of evidence this is redundancy/misalignment, not a tuning miss
1. **The structure was already there** (pre-check 53% SE gap) → Glass re-imposes existing structure.
2. **The best recipe was to TURN GLASS OFF** (handoff @1M). A method whose optimal schedule is "use it
   then remove it" is signalling **asymptotic value ≤ 0** — the tell of a redundant-or-harmful auxiliary,
   not a beneficial one. (We read this in iters 8–9 as a clever schedule; in hindsight it's a diagnosis.)
3. **The authors' own §9** noted Glass "biases the representation toward cluster coherence instead of
   Q-value precision" and "fights fine control around contact dynamics" — i.e. it trades control-relevant
   capacity for cluster structure the model didn't need. That's misalignment, exactly (b).
4. **The fair protocol erased the win.** The `phasei9r` edge came with restarts / basin-lottery (HopperHop
   is bimodal). Under the clean single-variable protocol, *neither* Glass nor vanilla entered the basin
   (best 323 vs 286) — the "win" was procedure, not representation (Part 2 §1).

## 5. Direct answer to "is that arm all replicating SimNorm?"
**Largely yes.** The Glass arm's effective contribution decomposes into:
- the geometric-clustering part = **re-deriving SimNorm's built-in soft-categorical structure** (redundant),
- plus a transition-SE part that is **not** SimNorm but captures **motion-phase structure irrelevant to
  control** (real but useless),
- wrapped in a schedule (handoff) that only "works" by *removing* the auxiliary, and an apparent win that
  was **basin-lottery / procedure**, not the representation.

So: not a literal re-implementation of SimNorm, but functionally the helpful part of Glass was redundant
with SimNorm + the consistency loss, and the non-redundant part didn't matter. That is *the same root
cause* as every later abstraction null — a strong self-predictive world model already encodes a sufficient
abstraction; bolting an explicit clustering/SE objective on top is redundant or misaligned.

## 6. What would have caught it sooner (the methodology lesson)
- A **mechanism-check before the multi-week schedule sweep**: "does Glass's structure track anything
  control needs (Q-value / model-error / reachability)?" — the iter-23-style check would have shown *no*
  in an afternoon, instead of months of handoff-schedule tuning.
- **Peak-AND-final + clean protocol from the start**: would have flagged the `phasei9r` edge as
  basin-lottery, not a representation win.
- **Read "turn it off to win" as a verdict, not a recipe.**

## 7. The salvageable thread
The one durable, non-null thing from the Glass arm is the *measurement*: SimNorm latents really do carry
rich (motion-phase) structure (53% SE gap). That's an **interpretability** result about what world-model
latents represent — likely more publishable as *understanding* than as a performance lever. Future SE work
should either (a) *shape* SimNorm for planning-amenability (not re-read it), or (b) use SE to *analyze*
learned latents — see the Part-2 blog §8 future-directions.
