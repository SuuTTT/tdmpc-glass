# SE pre-check: write-up, the two caveats, and forward exploration

*2026-06-09. Companion to `scripts/se_precheck.py` and `iteration_23_ideation.md`. Records what the
structural-entropy pre-check found, elaborates the two caveats, explains how motion-phase communities
(the iter-19 failure) become an ASSET for temporal abstraction, and scopes the "improve SimNorm with
an SE objective" idea as a follow-on experiment.*

## 1. What the pre-check measured and found

Question: **does TD-MPC2's SimNorm latent space have community structure an SE method can exploit?**
Metric: Li-Pan structural entropy gap = (H¹ − H²)/H¹, where H¹ is the 1-D structural entropy (no
partition; Shannon entropy of the random-walk stationary distribution) and H² is the 2-D structural
entropy under the SE-optimal community partition. Gap ≈ 0 ⇒ partitioning buys nothing ("blob");
gap ≳ 15% ⇒ real, compressible community structure.

Result (tier-1, geoglass/behavglass CartpoleSparse prototype graph + 512-d SimNorm prototypes):
- **Raw transition graph: ~0%** — a blob (40–76% edge density; matches the iter-19 finding).
- **Sparsified (top-30% edges) + SE-optimal partition: 31.2%.**
- **kNN graph on the raw 512-d SimNorm latents (k=3): 47.5%.**

Conclusion: SimNorm latents **do** cluster strongly — but only when the graph is built right
(sparsify / kNN) and partitioned SE-optimally. The deep-research-flagged "SimNorm is too dense" risk
is **real for the naive graph but mitigable**. PASS, pending tier-2 on real jumpy latents.

## 2. The two caveats, elaborated

### Caveat 1 — it's a 32-node prototype PROXY, possibly optimistic
The tier-1 graph came from a **geoglass/behavglass** run, which trained the encoder with an *auxiliary
structural-entropy clustering loss*. That loss actively pushed latents toward crisp clusters — so the
proxy's 31–47% gap may be **inflated relative to a jumpy/vanilla encoder trained with TD-MPC2 losses
only** (no clustering pressure). Two more reasons the proxy is imperfect:
- **Resolution.** 32 prototype nodes cap H¹ at log₂32 = 5 bits and coarsen structure. The real lever
  builds a graph over ≥128 clustered rollout latents — finer, but high-dim kNN can also become *more*
  uniform (curse of dimensionality), so the gap could move either way.
- **Transition semantics.** The proxy used 1-step prototype transitions; the SE-k lever wants the
  *k-step* transition graph, which aggregates differently.
→ **This is exactly why tier-2 (real jumpy-rollout latents, ≥128 nodes, k-step pairs) is the go/no-go.**
If a jumpy encoder with no SE pressure still shows ≳15% gap, the lever is genuinely viable; if it
collapses to ~0%, the proxy was optimistic and we fall back to F (uncertainty-gated horizon).

### Caveat 2 — "structure exists" ≠ "structure is useful" (the iter-19 motion-phase failure)
iter-19 found communities **and** tried to use them as SKILLS ("reach community c" = an option with an
initiation set + goal). That failed because the communities turned out to be **motion phases** — e.g.
for a cyclic gait, the swing phase, the contact phase, the stance phase — not *reachable subgoals*.

Why motion-phase communities are bad *skills*:
- An option needs a **goal the agent should deliberately reach**. But in a limit cycle the agent passes
  through every phase regardless of what it "wants" — phase membership is a **consequence** of the
  dynamics, not a **choice**. "Reach the swing phase" is not a decision; the agent will be there next
  cycle anyway. So goal-conditioning a low-level policy on community id gives it a target with no
  decision content → no useful abstraction (this is what iter-19 measured).
- Useful subgoals are **bottlenecks** (doorways, grasp points) — states you must deliberately route
  through. Cyclic locomotion has no such bottlenecks; its "communities" are just arcs of an oscillation.

## 3. How motion-phase communities become an ASSET (answering "how to use it to improve")

The iter-19 failure was in the **use** (as reach-goals), not the **structure**. The iter-23 SE-k lever
uses communities for *temporal abstraction*, and **motion phases are precisely the right structure for
that** — the same structure that was useless for skills is ideal here:

1. **Jump-length (k) selection — the primary use.** Within a motion phase the dynamics are smooth and
   predictable → take a LONG jump (large k). At a phase TRANSITION (foot-strike/contact, the moment the
   limit cycle switches arcs) the dynamics change sharply → take a SHORT jump (small k) for accuracy.
   Motion-phase communities give you the phase-boundary detector **for free**, and boundaries are
   exactly the contact/turning points where the compounding-error tax is worst. So:
   *long k inside a community, short k crossing a community boundary.* Motion phases were the WRONG
   structure for skills but are the RIGHT structure for adaptive temporal abstraction. **This is the
   structural reason iter-23 can succeed where iter-19 failed: same communities, correct use.**

2. **Macro-actions as phase MODULATION, not phase REACHING.** Instead of "reach phase c" (failed),
   define the macro-action as *how to advance/modulate the phase* — speed up, slow down, change
   amplitude, or switch the cycle (e.g. stance→swing earlier). For rhythmic control this is a
   central-pattern-generator / phase-oscillator abstraction: you control the *clock*, not the *target*.

3. **Phase as the macro-clock.** Let one macro-step = one phase advance, so the planner's effective
   horizon (n_macro phases ≈ one+ gait cycles) is *task-meaningful* instead of a fixed step count.

4. **Phase-conditioned jumpy dynamics.** Condition d_k on the (soft) phase/community, so the model knows
   whether it is predicting within-phase (easy) or across-phase (hard) — regularizing the head and
   directly attacking the cross-phase error that limits how far we can jump.

So the honest test for iter-23 isn't "do communities exist" (they do) but "**does community-aware
k-selection beat fixed-k jumpy**" — a different, planner-side claim that the iter-19 skill result does
not pre-empt.

## 4. Forward exploration — can we IMPROVE SimNorm with an SE objective?

Genuinely interesting, and distinct from what we've done. Two framings:

- **(a) SE-regularized SimNorm (encoder loss).** Add a differentiable term that *minimizes the 2-D
  structural entropy of the latent transition graph* within a batch, pushing the encoder toward
  block-structured dynamics (crisp phase communities). This would make the SE-k boundaries sharper and
  k-selection cleaner. **Caution: this is close to what geoglass did — and geoglass was a MIRAGE for
  RETURN on dense tasks.** The difference must be the *target metric*: here SE is in service of
  *planning amenability / k-selection quality*, not raw return — and it should be tested where temporal
  abstraction pays (high-DoF locomotion with clear phases), not on saturated dense tasks. Otherwise it
  risks being Mirage #9.
- **(b) SE-structured SimNorm (architecture).** SimNorm's V groups are *arbitrary fixed slices* of the
  latent dim. An SE objective could instead **learn which dimensions group together** from transition
  structure — a "structured SimNorm" whose simplex groups reflect dynamics communities rather than
  arbitrary partitions. More novel, more speculative.

**Sequencing (important, learned the hard way).** Do NOT conflate *using the SE structure already
present* (iter-23 SE-k primary — cheap, the pre-check says it's there) with *creating more SE structure
via an encoder loss* (the SimNorm-SE idea — which is geoglass-adjacent and has burned us). Order:
1. iter-23: confirm tier-2, then test community-aware **k-selection** vs fixed-k jumpy (uses existing
   structure; no new encoder loss).
2. Only if that wins: ask whether **SE-regularizing SimNorm** *amplifies* the win (a/b above), tested
   on high-DoF phase-rich tasks with the peak+final, CI-separated, pre-registered protocol.

This keeps a clean variable at each step and avoids re-running the geoglass mirage under a new name.

## 5. Status
- tier-1 PASS (proxy); tier-2 running (CheetahRun jumpy ckpt latent dump → offline SE gap on ≥128-node
  k-step graph). CheetahRun chosen first *because* its cyclic gait is the clearest motion-phase case.
- If tier-2 ≳15% gap → build community-aware k-selection (SE-k) as the iter-23 primary.
- SimNorm-SE objective (§4) = explicitly deferred follow-on, gated on SE-k winning first.
